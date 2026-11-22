"""Stage 3 (enrich モード): 対話の構成を変えずに、各発話へフィラー・言い淀みを付加する。

従来の generate_voice.py（full モード）は口語化・相槌挿入・発話分割・話者交代の
再構成まで LLM に任せるため、話者の入れ替わりや continued フラグの誤付与などの
構造破壊が起きる。enrich モードは分業を変える:

  - ターン構造（数・順序・話者）は Stage 2 の出力で確定済みとして一切変えない。
  - LLM には「各発話にフィラー・言い淀み・冒頭の短いリアクション・文末の終助詞を足す」
    ことだけをさせる。
  - 相槌ターンの新設・発話の分割はしない（TTS 後段の refine で MaAI が音声位置に挿入する）。
  - 話者・type は入力からプログラム側でコピーするので、入れ替わりが構造的に起きない。
  - 同一話者の連続ターンには continued: true を機械付与する（retime が間を詰める）。

出力スキーマは full モードと同一（TTS/retime/refine がそのまま使える）。
config の voice.mode: enrich で main.py から選択される。
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI

from cost_tracker import CostTracker
from llm_client import (
    build_async_llm_client,
    build_async_llm_clients,
    endpoint_concurrency,
    emphasize_question_marks,
    extract_json_text,
    run_endpoint_worker_pool,
    sanitize_utterance,
)
from generate_voice import (
    _chat_create,
    add_absolute_timeline,
    build_speaker_profiles,
    normalize_metadata_speakers,
    normalize_turn_speakers,
    parse_speaking_rate,
)

logger = logging.getLogger(__name__)


@dataclass
class EnrichParams:
    """enrich の付与強度。config の `voice.enrich` で上書き（既定は従来挙動を再現）。

    フィラーの最終付与率は実質 filler_cap_ratio（後処理 _thin_fillers の上限）で決まる
    （20B はノルマを超過付与し、上限まで間引かれるため）。filler_ratio はプロンプトの
    下限ノルマ。冒頭リアクションは後処理で削らないので reaction_ratio がほぼそのまま出る。

    sfp_ratio（文末終助詞）の既定 0.12 は、人間コーパス maai-f2f-jp-2025 との実測差から
    決めている（完結文・実質6文字以上で長さを揃えた比較）:
        文末終助詞あり  人間 28.5% / Stage2 20.3%   → 全体で 8pt 不足
        内訳            ね 14.4/13.5（同等）  よ 3.5/4.4（合成が過多）
                        よね 4.2/0.7・な 3.4/0.3（この2語でほぼ不足分すべて）
    つまり増やすべきは総量ではなく「よね」「な」なので、足せる語をこの2つ＋控えめな
    「ね」に限定している（`_ALLOWED_SFP`）。1ターンあたり約1.6文なので、ターン比 12% の
    付与で文比 8pt の不足がほぼ埋まる。
    """
    filler_ratio: float = 0.35          # フィラー付与の目安割合（プロンプトの下限ノルマ）
    filler_cap_ratio: float = 0.5       # フィラー付きターンの上限比率（_thin_fillers）
    reaction_ratio: float = 0.15        # 冒頭リアクション付与の目安割合
    allow_filler_and_reaction: bool = False  # 同一発話にフィラーと冒頭リアクション両方
    max_fillers_per_turn: int = 2       # 長い発話の1発話フィラー上限（短文は常に1）
    monotone_head_cap: int = 3          # 頭「まあ/うーん」の対話内上限
    verify_heads: bool = True           # 冒頭リアクションが直前発話に合うかLLMで検証し剥がす
    sfp_ratio: float = 0.12             # 文末終助詞を足すターンの目安割合（0で無効）
    sfp_cap_ratio: float = 0.18         # 終助詞を足したターンの上限比率（_thin_sfp）

    @classmethod
    def from_config(cls, config: dict) -> "EnrichParams":
        d = ((config.get("voice") or {}).get("enrich")) or {}
        kw = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**kw)


SYSTEM_PROMPT = """\
あなたは日本語の対話台本に、話し言葉らしい「言いよどみ」を加える専門家です。
2人の話者によるテキスト対話が、ターン番号付きで与えられます。
各発話に、フィラー・言い淀み・言い直し、発話冒頭の短いリアクション、および文末の終助詞を、
ピンポイントに（付けすぎず）付加してください。

# 最重要: この工程は「足すだけ」。入力の文字を1文字も削らない・置き換えない

- 入力の発話に書かれている文字は、順番どおり全部そのまま出力に残す。
  やってよいのは、その隙間にフィラー・リアクション語・読点を差し込むことだけ。
- 語の言い換え・要約・短縮・書き直しをしない。冒頭の「はい」「うん」「え」「了解」
  「なるほど」等を別の語に差し替えるのも禁止（削るのではなく、必要なら前に足す）。
  悪い例:「はい、次の項目を追加しましょう。」→「たしかに、次の項目を追加しましょう。」
  （「はい」が消えている。正しくは元の文をそのまま残すか、何も足さない）
- ターンの数・順序・話者・発話の内容（意味・情報）は変えない。
- ターンの追加・削除・結合・分割をしない。相槌ターンを追加しない
  （相槌は後工程で音声に自動挿入するため、台本には不要）。
- 出力の turns は入力と同じターン数で、index には入力のターン番号をそのまま使う。

# 各発話にやってよいこと（これだけ）

1. フィラーを発話の冒頭や途中に入れる。使ってよいフィラーは次の語だけ:
   「えっと」「ええと」「あの」「その」「なんか」「まあ」「うーん」「ん」「こう」
   - フィラーの末尾に延ばし棒「ー」を付けない（「えっとー」「あのー」「んー」は禁止。
     「えっと」「あの」「ん」と書く）。
   - 目安: 全発話の3〜4割程度に入れる。付けすぎない（多くの発話はフィラーなしで残す）。
   - 冒頭だけに付けない。追加分の半分くらいは文の途中（読点「、」の後）に入れる
     （例:「予算はどれくらい、えっと、見込んでいますか？」「それは、なんか、違う気がして」）。
   - 文の途中に入れるのは、意味の切れ目（すでに読点が打てる位置）だけ。
     助詞・助動詞の直前に割り込ませない（悪い例:「素材、えっと、を活用」）。
     名詞の連なり・複合語・修飾語と被修飾語の間にも割り込ませない
     （悪い例:「コンセプト、あのー、決めからですね」「来場者の、んー、関心を引けます」。
     このような位置しか無い発話には、フィラーを冒頭に付けるか、何も付けない）。
   - 文末の「。」「？」の直前にフィラーを置かない。フィラーで発話を終わらせない
     （悪い例:「協力し合いましょう、あの。」「分析が大変になりませんか、なんか？」
     「お会いできるでしょうか、えっと？」）。フィラーは必ず、その後ろに続く言葉の
     前置きとして置く。
   - 文中に入れるときは必ず「、フィラー、」の形にし、後ろの語に癒着させない
     （悪い例:「高さは59センチ、んでしょうか？」。「ん」が「でしょうか」に
     くっついて別の語になっている）。
   - 1つの発話に入れるフィラーは1個まで（頭と途中の両方に付けない）。
   - 同じ語を連発しない（「えっと」ばかり等にしない）。直前の発話と違う語を選び、
     9つの語をまんべんなく使う。特に「まあ」「うーん」を発話の頭に付けるのは
     単調になりやすいので、対話全体で合計3回までにする。
   - 短い反応（10文字以下）には基本入れない。考えて答える場面・話を切り出す場面・
     言いにくいことを言う場面に入れると効果的。
   - 相槌・リアクション語（「うん」「はい」「ええ」「なるほど」「そう」等）は
     フィラーではないので、この「フィラー」としては扱わない。これらを足すときは
     下の 3（冒頭リアクション）の規則に従う。
   - 否定に変わる語（「いや」「ううん」「違う」）は、元の発話がその意味でない限り
     絶対に足さない（意味が反転してしまう）。
2. 言い直し・言いかけを、ところどころ（数ターンに1回程度）入れる:
   「あ、ありがとう」「い、いや、そうじゃなくて」（最初の一音を言いかけて言い直す）、
   「それは、えっと、なんていうか」「あれ、なんだっけ、そう」（途中で詰まる）。
3. 発話の冒頭に、直前の相手の発話への短いリアクション語を、ときどき（ピンポイントに）足す。
   使ってよい語:「そう」「そうそう」「そうなんだ」「ふーん」「なるほど」
   「たしかに」「あー」「そっか」「いいね」。
   - 「へえ」「へぇ」「へー」「ほー」は使わない（TTS で不自然に伸びるうえ、
     多用されやすいため禁止）。驚き・感心を足したいときは「なるほど」「あー」
     「そうなんだ」を使う。
   - この工程では話者が必ず交互（相手→自分→相手…）なので、発話冒頭のリアクションは
     つねに「直前の相手の発話への反応」になる。だから足してよい。
   - ただし発話の中身と矛盾しない語だけを選ぶ。相手に同意・共感していない発話
     （反論・訂正・別意見・質問）の頭に「そう」「なるほど」「たしかに」を足さない。
   - リアクションだけの新しいターンは作らない（ターンは増やさない）。あくまで既存の
     発話の頭に短く足すだけ（相槌ターンの挿入は後工程の MaAI が担当する）。
   - 足す頻度はピンポイント（数ターンに1回程度）。全発話の頭には付けない。
   - 1つの発話に、フィラーと冒頭リアクションを両方は付けない（どちらか一方まで）。
4. 語尾を伸ばす: 「そうですねー」「でさー」。くだけた場面で時々使う。
__SFP_SECTION__6. 読点「、」を、息継ぎ・ためらい・間を作りたい位置に足す
   （TTSが流暢に話しすぎるのを防ぐ）。
7. 何も足すところがない発話は、入力のまま1文字も変えずに出す
   （全発話に手を入れる必要はない。無理に書き換えるより、そのまま返すほうがよい）。

# 変えてはいけないもの

- 入力に書かれている文字列そのもの。削除・言い換え・語順の入れ替えをしない
  （縮約形への直し「している→してる」等も、この工程ではしない）。
- 発話の意味・情報・ニュアンス。内容の追加・削除・要約をしない。
- 話者の口調・speech_style（敬語/タメ口）。敬語をタメ口にしたり、その逆をしない。
- 文末の「？」（質問はそのまま「？」で終える）。文末記号の直前にフィラーを置かない。
- text に使える記号は「、」「。」「？」のみ。ト書き・カッコ書き・空白・
  「…」「！」「〜」等は禁止。

# タイミングの付与

各ターンに、直前のターンが終了してからの相対時間 start_offset（秒）と、
その発話の推定発話長 duration（秒）を付ける。
- start_offset: テンポの良い応答・すぐ返す反応は 0〜0.3。
  考えてから話す・話題が変わる・言いにくいことを言う箇所は 0.5〜1.0。
- duration: 発話の文字数から見積もる（日本語はおおよそ1秒あたり6〜8モーラ）。

# 話者の性別と話速の推定

各話者の性別を、名前・役割・性格・話し方から "male" または "female" で推定する
（判断材料が乏しくても必ずどちらかに決める）。
また対話全体の話す速さ speaking_rate を 1.0〜1.2 の範囲で推定する
（親しい雑談・テンポの良い掛け合い → 1.15〜1.2、一般的な会話 → 1.1 前後、
落ち着いた相談・フォーマル・重い場面 → 1.0〜1.05）。

# 出力形式（JSONのみ。マークダウンのコードブロックは不要）

{
  "speakers": [
    {"name": "話者名", "gender": "male"},
    {"name": "話者名", "gender": "female"}
  ],
  "speaking_rate": 1.15,
  "turns": [
    {"index": 0, "text": "えっと、昨日さ、駅前の新しいカフェ行ったんだけど。", "start_offset": 0.3, "duration": 2.9},
    {"index": 1, "text": "え、まじで？", "start_offset": 0.1, "duration": 0.8}
  ]
}

- turns は入力と同じターン数。index は入力のターン番号（0始まり）をそのまま使う。
- speaker と type は出力しない（入力から自動で引き継ぐ）。
- JSONのみを出力する。
"""

# SYSTEM_PROMPT の項目5に差し込む終助詞ルール（sfp_ratio が 0 なら SFP_OFF に差し替える）。
# 足してよい語を「よね」「な」＋控えめな「ね」に絞るのは EnrichParams の実測比較による。
# 「な」は動詞の言い切りに付くと禁止（「行くな」）になるため、接続先を厳しく限定している。
SFP_SECTION = """\
5. 発話の文末に終助詞を足して、書き言葉的な言い切りを和らげる。
   - 足してよい語は「よね」「な」「ね」だけ。「よ」「かな」「っけ」「わ」「の」「もん」
     「ぞ」「ぜ」は足さない（語気や意味が変わる、または台本に既に足りているため）。
   - 優先して足すのは「よね」と「な」。「ね」はすでに台本に多いので控えめにする。
   - 「よね」＝相手も知っている・同意を求める内容に足す。敬語なら「〜ですよね。」
     「〜ますよね。」、タメ口なら「〜だよね。」「〜たよね。」。すでに「〜よ。」「〜ね。」で
     終わる発話は「〜よね。」にしてよい（元の字を消さずに足すこと）。
   - 「な」＝タメ口の、独り言・感想の発話にだけ足す。直前が「だ」「た」「い」で終わる
     ときだけ（「〜だな。」「〜たな。」「〜いな。」「〜ないな。」）。
     ★動詞の言い切り（「行く」「思う」「見る」等）には絶対に足さない
     （「行くな」「思うな」は禁止の意味になり、発話の意味が反転する）。
     敬語（です・ます・ました）にも足さない（「ですな」「ましたな」は不自然）。
   - 命令・依頼・勧誘（「〜ください」「〜ましょう」「〜しよう」「〜して」）の文末には
     「よね」「な」を足さない。
   - 疑問文（「？」で終わる発話）には足さない。
   - すでに「な」「わ」「かな」「っけ」で終わっている発話には足さない。
   - 目安: 全発話の約__SFP_PCT__%。全部の語尾に手を入れない。
"""

SFP_OFF = """\
5. 発話の文末の終助詞は足さない（語尾は入力のままにする）。
"""


def build_system_prompt(ep: EnrichParams) -> str:
    """付与強度 ep に応じて SYSTEM_PROMPT の目安割合・頻度・同一発話ルールを差し替える。"""
    s = SYSTEM_PROMPT
    fr = int(round(ep.filler_ratio * 100))
    s = s.replace(
        "   - 目安: 全発話の3〜4割程度に入れる。付けすぎない（多くの発話はフィラーなしで残す）。",
        f"   - 目安: 全発話の約{fr}%に入れる"
        + ("（積極的に付ける。ただし入れる場所が無い発話には無理に付けない）。"
           if fr >= 50 else "（付けすぎない。多くの発話はフィラーなしで残す）。"))
    rr = int(round(ep.reaction_ratio * 100))
    s = s.replace(
        "   - 足す頻度はピンポイント（数ターンに1回程度）。全発話の頭には付けない。",
        f"   - 足す頻度は全発話の約{rr}%"
        + ("（多めに。同意・共感できる発話の頭に積極的に付ける）。"
           if rr >= 25 else "（ときどき。全発話の頭には付けない）。"))
    if ep.allow_filler_and_reaction:
        s = s.replace(
            "   - 1つの発話に、フィラーと冒頭リアクションを両方は付けない（どちらか一方まで）。",
            "   - 1つの発話に冒頭リアクションと途中フィラーを両方付けてよい"
            "（例:「なるほど、えっと、具体的には」。頭に短いリアクション、"
            "意味の切れ目にフィラーを入れる）。")
    if ep.max_fillers_per_turn > 1:
        s = s.replace(
            "   - 1つの発話に入れるフィラーは1個まで（頭と途中の両方に付けない）。",
            f"   - 1つの発話に入れるフィラーは最大{ep.max_fillers_per_turn}個まで"
            "（長い発話のみ。短い発話は1個まで）。")
    if ep.sfp_ratio > 0:
        sr = max(1, int(round(ep.sfp_ratio * 100)))
        s = s.replace("__SFP_SECTION__", SFP_SECTION.replace("__SFP_PCT__", str(sr)))
    else:
        s = s.replace("__SFP_SECTION__", SFP_OFF)
    return s


def build_enrich_prompt(metadata: dict, turns: list, ep: EnrichParams) -> str:
    speakers = ", ".join(s["name"] for s in metadata.get("speakers", []))
    lines = [
        "以下の2人の話者によるテキスト対話の各発話に、フィラー・言い淀みを付加してください。",
        "ターンの数・順序・話者・意味は変えないでください。",
        "",
        f"話者: {speakers}",
        f"場面: {metadata.get('setting', '')}",
        f"関係性: {metadata.get('relationship', '')}",
        f"口調: {metadata.get('speech_style', '')}",
        "",
        "【対話（ターン番号: 話者: 発話）】",
    ]
    for i, t in enumerate(turns):
        lines.append(f"{i}: {t['speaker']}: {t['utterance']}")
    lines.append("")
    lines.append(f"全{len(turns)}ターンです。出力の turns も必ず{len(turns)}個にし、"
                 "index 0〜" + str(len(turns) - 1) + " を1つずつ含めてください。")
    # フィラー挿入は比率指示だけだと 30B がほぼ無視する（実測 0〜10%）ため、
    # Stage 2 のターン数ノルマと同様に、具体的な個数ノルマを末尾で要求する。
    # ピンポイント付加（付けすぎない）。フィラーは比率指示だけだと 30B がほぼ無視する
    # ため下限個数のノルマを与えるが、上限は控えめ（半分未満）にして過剰付加を防ぐ。
    n = len(turns)
    quota = max(4, int(n * ep.filler_ratio * 0.85))
    cap = max(quota + 2, int(n * ep.filler_cap_ratio))
    react_quota = max(2, int(n * ep.reaction_ratio))
    lines.append(f"少なくとも{quota}個・多くても{cap}個のターンに、フィラー（えっと、あの、"
                 f"なんか、まあ、うーん等。末尾に「ー」を付けない）または言い直しを"
                 "実際に書き加えてください（これは【フィラーの個数ノルマ】です。"
                 "冒頭リアクション「そう」「なるほど」等はこの数に含めず、"
                 "下の別ノルマで確保するので、ここではフィラー・言い直しの数を必ず満たすこと）。"
                 "付けすぎず、ピンポイントに入れ、"
                 "残りの多くの発話はそのまま出してください。"
                 "入力のテキストをそのまま丸ごと返すだけの出力は不合格です。"
                 "ただし、入力の文字を削ったり別の語に言い換えたりした出力も不合格です"
                 "（入力の文字は順番どおり全部残し、隙間に足すだけにしてください。"
                 "冒頭の「はい」「うん」「え」等を別のリアクション語に差し替えないこと）。"
                 "フィラーは冒頭だけでなく文の途中（読点の後・意味の切れ目）にも入れ、"
                 "助詞の直前（「素材、えっと、を」のような形）には絶対に入れないでください。"
                 "文末の「。」「？」の直前にフィラーを置かないでください"
                 "（「〜しましょう、あの。」「〜ませんか、なんか？」は不合格）。"
                 "文中に入れるときは必ず「、フィラー、」の形にし、"
                 "後ろの語に癒着させないでください（「59センチ、んでしょうか？」は不合格）。"
                 "同じフィラー語ばかり使わず散らし、"
                 "「まあ」「うーん」を発話の頭に付けるのは対話全体で合計3回までにしてください。"
                 "冒頭リアクションは、相手に同意・共感していない発話（反論・質問等）には付けず、"
                 "「いや」「ううん」など意味が反転する語は足さないでください。")
    lines.append(f"さらに冒頭リアクションについて: 全{n}ターンのうち少なくとも{react_quota}個の"
                 "発話の冒頭に、直前の相手発話への短いリアクション語（そう、そうそう、そうなんだ、"
                 "なるほど、たしかに、そっか、あー、いいね等。「へえ」「へー」は使用禁止）を"
                 "実際に書き加えてください。ただし相手に同意・共感していない発話"
                 "（反論・訂正・別意見・質問）の頭には付けないこと。"
                 + ("フィラーと冒頭リアクションは同じ発話に両方付けてもかまいません"
                    "（頭にリアクション、意味の切れ目にフィラーを1個）。"
                    if ep.allow_filler_and_reaction else
                    "1つの発話にはフィラーか冒頭リアクションのどちらか一方までにしてください。"))
    if ep.sfp_ratio > 0:
        sfp_quota = max(1, int(n * ep.sfp_ratio))
        lines.append(f"さらに文末の終助詞について: 全{n}ターンのうち少なくとも{sfp_quota}個の発話の"
                     "文末に終助詞を足してください。足してよいのは「よね」「な」「ね」だけで、"
                     "「よね」と「な」を優先し（「ね」は台本に既に多いので控えめに）、"
                     "「よ」「かな」「っけ」「わ」「の」は足さないでください。"
                     "「な」はタメ口で「〜だ」「〜た」「〜い」で終わる発話にだけ足し、"
                     "動詞の言い切り（「行く」「思う」「見る」等）には絶対に足さないでください"
                     "（「行くな」「思うな」は禁止の意味になり、発話の意味が反転します）。"
                     "敬語（です・ます・ました）にも「な」を足さないでください。"
                     "疑問文（「？」で終わる発話）と、すでに「な」「わ」「かな」「っけ」で"
                     "終わる発話には足さないでください"
                     "（「〜よ。」「〜ね。」を「〜よね。」にするのはかまいません）。"
                     "例:「これは難しい。」→「これは難しいな。」、"
                     "「昨日は疲れた。」→「昨日は疲れたな。」、"
                     "「明日は雨だよ。」→「明日は雨だよね。」、"
                     "「その話は聞きました。」→「その話は聞きましたよね。」。")
    lines.append("加工の例:「最近の新作RPGめっちゃ面白いんだ。」→"
                 "「えっと、最近の新作RPG、めっちゃ面白いんだ。」、"
                 "「具体的に教えて。」→「なるほど、具体的に教えて。」、"
                 "「時間制限緊張するね。」→「時間制限、なんか緊張するね。」。"
                 "このように、数ターンおきに実際に手を入れてください。")
    lines.append("JSONのみを出力してください。")
    return "\n".join(lines)


def _core(s: str) -> str:
    """読点・句点・長音を落とした実質文字列（内容の同一性判定に使う）。"""
    return re.sub(r"[、。？ー\s]", "", s or "")


def _core_len(s: str) -> int:
    return len(_core(s))


# プロンプトが許可しているフィラー語の全体（長い綴りを先に並べる）。
# 以前は平仮名1〜2文字の「あの」「その」「こう」「ん」が漏れており、これらを使った
# 割り込み（「タップだけ、あの、で済むんだね。」）が後処理をすり抜けていた。
# 2026-07-27 に語彙を拡張（人間コーパスに比べて「えっと」偏重だったため）。
# 追加した語も同じ後処理（文末直前の除去・助詞への割り込み除去・個数上限）にかける。
_FILLER_WORDS = (r"えっとー|えっと|ええと|あのう|あのー|あの|そのー|その|"
                 r"なんていうか|なんというか|なんだろう|なんか|"
                 r"まあ|うーん|んー|こう|やっぱり|やっぱ|ちょっと|ほら|ん")

# 「素材、えっと、を活用」のようにフィラーが助詞の直前へ割り込んだ形（20Bで実測）。
# 「、フィラー、」を除去すると元の連結（「素材を活用」）に戻る。
# 助詞の直後が平仮名の場合（「、なんか、はい」等）は助詞ではない可能性が高いので触らない。
_PARTICLE_SPLIT = re.compile(
    rf"、(?:{_FILLER_WORDS})、"
    r"(?=(?:を|が|は|に|で|へ|と|も|の|から|まで|より)(?:[^ぁ-ん]|$))"
)


def fix_particle_splits(text: str) -> str:
    return _PARTICLE_SPLIT.sub("", text)


# フィラー1個ぶんの出現。行頭か読点の直後だけを数える（「だもん、」の「ん、」のような
# 語尾をフィラーと誤認しないため）。前方の読点は消費しないので span は語だけを指す。
_FILLER_ANY = re.compile(rf"(?:^|(?<=、))(?:{_FILLER_WORDS})、")
_HEAD_FILLER = re.compile(rf"^(?:{_FILLER_WORDS})、")
_MID_FILLER = re.compile(rf"、(?:{_FILLER_WORDS})、")
# 「〜だったから、えっと、ね。」のように終助詞の直前へ割り込んで語尾を壊す形
_TAIL_BREAK = re.compile(rf"、(?:{_FILLER_WORDS})、(?=(?:ね|よ|な|さ|わ)[。？]$)")
# 文末記号の直前に置かれたフィラー（「協力し合いましょう、あの。」
# 「分析が大変になりませんか、なんか？」）。発話がフィラーで終わる形は音声にすると
# 必ず不自然なので、読点ごと落として元の文末に戻す。
# 後ろに読点が残る形（「大きいですね、うーん、。」）も同じ扱い。20B が文末に足した
# フィラーを読点で閉じてしまうためで、実測 192,876 ターン中 1,035 件出ていた。
_TAIL_FILLER = re.compile(rf"、(?:{_FILLER_WORDS})、*(?=[。？]$)")
# フィラー除去後などに残る、文末記号の直前の読点（「〜です、。」）
_TAIL_COMMA = re.compile(r"、+(?=[。？]$)")
# 読点で閉じずに後ろの語へ癒着したフィラー（「高さは59センチ、んでしょうか？」）。
# 1モーラの「ん」でのみ実測。閉じの読点が無いので語そのものが別物になる。
_GLUED_FILLER = re.compile(r"、ん(?=[でだにをがはともへ])")
# 冒頭の「へえ」系リアクション（禁止語）。20B が多用し、TTS で不自然に伸びる。
_HEE_HEAD = re.compile(r"^(?:へ+[えぇー]+|ヘ+ー+|ほ+ー+)、?")


# ── 文末終助詞の付加ガード ────────────────────────────────────────────
# 文末（句点の直前）に並ぶ終助詞の連なり。「みんな。」「そうかな。」のように語の一部が
# 引っかかることもあるが、判定は常に「入力からの変化」で行うので誤検出は害にならない。
_SFP_TAIL = re.compile(
    r"(?P<sfp>(?:もんね|かしら|よね|かな|っけ|もん|ね|よ|な|さ|わ|ぞ|ぜ)+)"
    r"ー*(?P<end>[。？]?)$")

# 足してよい終助詞。(元の語尾 → 付加後の語尾) の遷移で許可する。
# 「よ」を足す遷移が無いのは、合成側が人間より「よ」過多だから（EnrichParams 参照）。
_ALLOWED_SFP = {
    "": {"ね", "よね", "な"},   # 終助詞が無い発話に足す
    "よ": {"よね"},             # 「〜よ。」→「〜よね。」（「よ」を消さずに「ね」を足す）
    "ね": {"よね"},             # 「〜ね。」→「〜よね。」
}
# 「な」を足してよい直前の形。動詞の言い切り（「行く」「思う」）に付くと禁止の意味に
# なるため、「〜だ」「〜た」「〜い（〜ない・〜たい）」で終わるときだけに限る。
_NA_STEM_OK = re.compile(r"[だたい]$")
# 敬語の語尾。「ましたな」「ですな」は不自然なので「な」を足させない。
_POLITE_STEM = re.compile(r"(?:です|ます|ました|ません|でした|でしょう|ましょう|ください|なさい)$")
# 「よね」を足せない語尾。
#   前半: 命令・依頼・勧誘・テ形（「教えてくださいよね」は文にならない）。
#   後半: 定型あいさつと一人称の意志表明（実測: 「よろしくお願いしますよね」
#         「提案書を作成しますよね」）。「よね」は相手も知っている事柄に同意を求める語
#         なので、これから自分がすることの表明には付かない。漢語サ変の非過去敬語
#         （「作成します」「提案します」）はこの用法になりやすいのでまとめて弾く
#         （同じ場面で自然なのは「よね」ではなく「ね」で、そちらは許可している）。
_NO_YONE_STEM = re.compile(
    r"(?:ましょう|ください|なさい|よう|ろう|て|で|ろ|え"
    r"|お願いします|いたします|ございます|失礼します"
    r"|[一-龥ァ-ヶ]します|ておきます|てきます|ていきます)$")


def _sfp_of(text: str) -> str:
    """text の文末終助詞（無ければ ""）。"""
    m = _SFP_TAIL.search(text or "")
    return m.group("sfp") if m else ""


def added_sfp(new: str, old: str) -> str | None:
    """enrich が文末終助詞を変えたなら、変更後の終助詞を返す。変えていなければ None。

    「足すだけ」の工程なので、元が「〜よ。」で出力が「〜よね。」なら "よね" を返す。
    """
    sn, so = _sfp_of(new), _sfp_of(old)
    return sn if sn != so else None


def revert_sfp(new: str, old: str) -> str:
    """new の文末終助詞を old の形に戻す（他の位置の付加はそのまま残す）。"""
    m = _SFP_TAIL.search(new or "")
    if not m:
        return new
    so = _sfp_of(old)
    # 足された分と、それを繋いでいた読点（「〜ですね、よね。」の「、」）を落とす
    head = new[:m.start("sfp")].rstrip("、")
    if not head:
        return new  # 発話が終助詞だけ。剥がすと空になるので触らない
    # 元の終助詞が head 側に残っていなければ書き戻す（「〜だ|よね。」→「〜だよ。」）
    if so and not head.endswith(so):
        head += so
    return head + m.group("end")


def fix_added_sfp(new: str, old: str) -> str:
    """文末に足された終助詞を検査し、規則に反するものを元の語尾へ戻す。

    弾く形（いずれも音声にすると意味・語法が壊れる）:
      - 動詞の言い切りへの「な」（「そう思う。」→「そう思うな。」＝禁止の意味）
      - 敬語への「な」（「確認しました。」→「確認しましたな。」）
      - 命令・依頼・勧誘・一人称の意志表明への「よね」
        （「教えてください。」→「教えてくださいよね。」
          「提案書を作成します。」→「提案書を作成しますよね。」）
      - 既存の終助詞に読点を挟んで重ねた形（「効果的ですね。」→「効果的ですね、よね。」）
      - 疑問文への付加（「どこですか？」→「どこですかね？」＝問いの力が変わる）
      - 許可していない語の付加（「よ」「かな」「っけ」「わ」等。意味・語気が変わる）
      - 元の終助詞の消失・別語への差し替え（「〜だね。」→「〜だよ。」）
    """
    sn, so = _sfp_of(new), _sfp_of(old)
    if sn == so:
        return new
    if sn not in _ALLOWED_SFP.get(so, ()):
        return revert_sfp(new, old)
    # 「？」で終わる発話には足さない
    if (new or "").rstrip().endswith("？"):
        return revert_sfp(new, old)
    m = _SFP_TAIL.search(new)
    head = new[:m.start("sfp")]
    # 「〜ですね、よね。」のように読点を挟んで足された形。元の終助詞が前に残ったままで、
    # 拡張（「ね」→「よね」）ではなく重複なので戻す。20B の実測ではこれが最頻の失敗。
    if head.endswith("、"):
        return revert_sfp(new, old)
    stem = head
    added = sn[len(so):] if sn.startswith(so) else sn[:len(sn) - len(so)]
    if added == "な" and not (_NA_STEM_OK.search(stem) and not _POLITE_STEM.search(stem)):
        return revert_sfp(new, old)
    if "よね" in (sn, added) and _NO_YONE_STEM.search(stem):
        return revert_sfp(new, old)
    return new


# ── 冒頭リアクションの整合検証（直前発話に合っているか）─────────────────
# enrich が発話の頭に足しうるリアクション語（stance を持つので直前発話との整合が要る）。
# 長い綴りを先に並べる（re の交替は最初にマッチした枝を採るため、そう < そうそう 等）。
_REACT_WORDS = (r"そうそう|そうなんだ|そうだね|そうですね|そうね|そっか|そうか|そう|"
                r"なるほど|たしかに|確かに|ふーん|いいね|あー|あっ")
_HEAD_REACT = re.compile(rf"^(?:{_REACT_WORDS})、")
# 「純粋な情報の受領・納得」を表す語。どんな疑問文（事実質問でも提案でも）の答えの頭に
# 付いても論理が破綻する（答える側は情報を受け取る立場ではない）ので、直前が疑問文なら
# LLM を待たず無条件で剥がす。
_RECEPTION_ANY = re.compile(r"^(?:なるほど|そうなんだ|ふーん)、")
# 「評価・受領」語。提案・誘い（〜どう？〜てもらえる？）への同意なら自然だが、事実・情報を
# 尋ねる質問（いつ・どこ・いくら・何を…）の答えの頭に付くと不自然。事実質問のときだけ剥がす。
_RECEPTION_FACTUAL = re.compile(r"^(?:いいね|そっか|あー|あっ)、")
# 事実・情報を尋ねる疑問詞（これを含む疑問文は「答える」場面。提案・誘いには含まれない）。
_WH_QUESTION = re.compile(
    r"(?:いつ|どこ|どちら|どれ|いくら|何時|何日|何円|何人|何を|何が|何て|なに|"
    r"どうやって|どうすれ|どの|どんな|どういう|手順)")


def added_head_reaction(text: str, orig: str) -> str | None:
    """text の頭に enrich が“足した”リアクション語（例 "なるほど、"）を返す。無ければ None。

    元の発話が元々そのリアクションで始まっていた場合は追加ではないので触らない（None）。
    """
    m = _HEAD_REACT.match(text or "")
    if not m:
        return None
    if _HEAD_REACT.match(orig or ""):
        return None
    return m.group(0)


def strip_head_reaction(text: str, head: str) -> str:
    """頭のリアクション（"なるほど、"）を剥がして元の発話頭に戻す（純粋な前置きの除去）。"""
    return text[len(head):] if text.startswith(head) else text


def _prev_is_question(prev: str) -> bool:
    return (prev or "").rstrip("。").endswith("？")


def _is_factual_question(prev: str) -> bool:
    """事実・情報を尋ねる疑問文か（提案・誘いではなく「答える」場面か）。"""
    return _prev_is_question(prev) and bool(_WH_QUESTION.search(prev or ""))


def _head_hard_remove(prev: str, cur: str) -> bool:
    """LLM を待たずに機械的に剥がすべき明白な不整合。

    1) 直前が疑問文で、その答えの頭に受領・納得語（なるほど／そうなんだ／ふーん）が付く形
       （「予算はどれくらい？」→「なるほど、2万円」。提案でも事実質問でも答え手の立場と矛盾）。
    2) 直前が事実質問で、その答えの頭に評価・受領語（いいね／そっか／あー）が付く形
       （「早割っていつまで？」→「いいね、来月頭まで」。提案への同意なら自然だが事実回答では不自然）。
    """
    if not _prev_is_question(prev):
        return False
    if _RECEPTION_ANY.match(cur or ""):
        return True
    if _RECEPTION_FACTUAL.match(cur or "") and _is_factual_question(prev):
        return True
    return False


def build_head_verify_prompt(items: list) -> str:
    """冒頭リアクション候補（index/head/prev/cur）を1対話ぶんまとめて判定させるプロンプト。"""
    lines = [
        "日本語の対話について、各発話の冒頭に後から機械的に足された「リアクション語」が、",
        "直前の相手の発話への応答として自然かどうかを判定してください。",
        "",
        "# remove（不自然。剥がすべき）と判定する場合",
        "- 直前が事実を尋ねる質問（いつ・どこ・いくら・何を・どうやって等）で、今の発話がその答えなのに、"
        "頭に『なるほど』『いいね』『そう』『たしかに』等が付いている"
        "（答える側は理解・評価する立場ではないので、事実回答の頭にこれらは不自然）。",
        "- 直前の内容に同意・共感していない（反論・訂正・別の話題・逆質問）のに、"
        "同意・納得のリアクションが付いている。",
        "- リアクションの意味が直前の内容と噛み合わない（新情報でないのに『そうなんだ』で驚く等）。",
        "# keep（自然。残す）と判定する場合",
        "- 直前の相手の発言に素直に頷ける・同意できる・受け止められる内容で、その頭にリアクションが来ている。",
        "- 直前が提案・誘い・依頼（『〜どう？』『〜てもらえる？』『〜しない？』『〜でいい？』等）で、"
        "それに同意する『いいね』『そう』『そうだね』が頭に来ている（提案への同意は自然なので残す）。",
        "",
        "各 index について keep か remove を判定し、JSON配列だけを出力してください。",
        '例: [{"index": 3, "verdict": "remove"}, {"index": 6, "verdict": "keep"}]',
        "",
    ]
    for it in items:
        lines.append(f'index {it["index"]}（冒頭リアクション「{it["head"].rstrip("、")}」）:')
        lines.append(f'  直前の相手の発話: {it["prev"]}')
        lines.append(f'  今の発話: {it["cur"]}')
    lines.append("")
    lines.append("JSON配列のみを出力してください。説明文は不要です。")
    return "\n".join(lines)


def parse_head_verdicts(text: str) -> dict:
    """検証応答（JSON配列）を {index: should_remove(bool)} に変換。壊れていれば空辞書。"""
    try:
        data = json.loads(extract_json_text(text or ""))
    except (json.JSONDecodeError, ValueError):
        return {}
    if isinstance(data, dict):
        data = data.get("verdicts") or data.get("results")
    if not isinstance(data, list):
        return {}
    out = {}
    for e in data:
        if not isinstance(e, dict) or "index" not in e:
            continue
        try:
            i = int(e["index"])
        except (TypeError, ValueError):
            continue
        out[i] = str(e.get("verdict", "")).strip().lower() == "remove"
    return out


async def verify_and_fix_heads(
    client,
    model: str,
    voice_turns: list,
    orig_utterances: list,
    *,
    cost_tracker: "CostTracker | None" = None,
    extra_body: dict | None = None,
    retry: int = 2,
) -> dict:
    """voice_turns の冒頭リアクションを直前発話と照合し、不整合なものを剥がす（in-place）。

    - 明白な不整合（疑問文の答えの頭に理解・評価系リアクション）は LLM を待たず機械的に剥がす。
    - それ以外は LLM に意味判定させる（temperature 0）。LLM 応答が壊れても機械ルール分は必ず反映。
    - 剥がすのは「足した前置き」だけなので、内容・意味は一切変わらない。

    返り値の stats: {"candidates", "removed", "hard", "llm"}。
    """
    items = []
    for i, t in enumerate(voice_turns):
        orig = orig_utterances[i] if i < len(orig_utterances) else ""
        head = added_head_reaction(t["text"], orig)
        if not head:
            continue
        prev = voice_turns[i - 1]["text"] if i > 0 else ""
        items.append({"index": i, "head": head, "prev": prev, "cur": t["text"]})
    stats = {"candidates": len(items), "removed": 0, "hard": 0, "llm": 0}
    if not items:
        return stats

    verdicts: dict = {}
    for attempt in range(retry):
        try:
            resp = await _chat_create(
                client,
                model=model,
                messages=[
                    {"role": "system",
                     "content": "あなたは日本語の対話の自然さを判定する校正者です。"},
                    {"role": "user", "content": build_head_verify_prompt(items)},
                ],
                temperature=0.0,
                max_completion_tokens=2048,
                extra_body=extra_body or {},
            )
            if cost_tracker is not None and resp.usage is not None:
                cost_tracker.add(resp.usage.prompt_tokens, resp.usage.completion_tokens)
            verdicts = parse_head_verdicts(resp.choices[0].message.content or "")
            if verdicts:
                break
        except Exception as e:  # noqa: BLE001 - ローカル推論の一過性失敗は機械ルールで代替
            logger.warning(f"冒頭リアクション検証の呼び出し失敗 (attempt {attempt+1}): {e}")

    for it in items:
        i = it["index"]
        hard = _head_hard_remove(it["prev"], it["cur"])
        llm = bool(verdicts.get(i, False))
        if not (hard or llm):
            continue
        new = strip_head_reaction(it["cur"], it["head"])
        if not new:
            continue
        voice_turns[i]["text"] = new
        voice_turns[i]["duration"] = round(max(0.4, _core_len(new) / 6.5), 2)
        stats["removed"] += 1
        stats["hard"] += int(hard)
        stats["llm"] += int(llm and not hard)
    return stats


def limit_fillers_in_turn(text: str, max_long: int = 2) -> str:
    """1発話内のフィラーを上限個数に間引く（20Bは1発話に2個以上入れることがあり、
    「具体的に、えっと、どんな体験を、なんか、考えているんですか？」のように
    毎文が二度つかえる機械的なリズムになる）。最初の1個（長い発話は max_long 個）を残す。
    """
    cap = 1 if _core_len(text) <= 25 else max_long
    matches = list(_FILLER_ANY.finditer(text))
    if len(matches) <= cap:
        return text
    for m in reversed(matches[cap:]):
        s, e = m.span()
        # 「、F、」は「、」に、頭の「F、」は空に戻す
        if s > 0 and text[s - 1] == "、":
            text = text[:s] + text[e:]
        else:
            text = text[:s] + text[e:] if s == 0 else text
    return text


_MONOTONE_HEAD = re.compile(r"^(?:まあ、|うーん、)")


def _cap_monotone_heads(out: list, stats: dict, cap: int = 3) -> None:
    """頭の「まあ、」「うーん、」を対話全体で cap 回までに制限する。

    プロンプトで「合計3回まで」と指示しても 20B は守らないことがある
    （実測: 1対話で10回）。超過分は頭のフィラーを剥がすだけなので安全。
    """
    seen = 0
    stripped = 0
    for t in out:
        if not _MONOTONE_HEAD.match(t["text"]):
            continue
        seen += 1
        if seen <= cap:
            continue
        t["text"] = _MONOTONE_HEAD.sub("", t["text"])
        t["duration"] = round(max(0.4, _core_len(t["text"]) / 6.5), 2)
        stripped += 1
    stats["monotone_stripped"] = stripped


def _thin_fillers(out: list, stats: dict, cap_ratio: float = 0.5) -> None:
    """フィラー付きターンの比率が cap_ratio を超えたら、短いターンから間引く。

    20B は「多くても◯個」の上限ノルマを守らないことがある（実測93〜95%まで過剰付加）。
    ほぼ全ターンが言い淀む対話は不自然なので、頭のフィラー（次いで文中のフィラー）を
    機械的に剥がして上限内へ戻す。剥がすのは実質文字数が短いターンから
    （短い反応にフィラーは不要、というプロンプト方針と一致）。
    """
    n = len(out)
    cap = int(cap_ratio * n)

    def filled():
        return [i for i, t in enumerate(out) if _FILLER_ANY.search(t["text"])]

    removed = 0
    for i in sorted(filled(), key=lambda i: _core_len(out[i]["text"])):
        if len(filled()) <= cap:
            break
        t = out[i]["text"]
        t2 = _HEAD_FILLER.sub("", t)
        if t2 == t:
            t2 = _MID_FILLER.sub("、", t, count=1)
        if t2 != t and t2:
            out[i]["text"] = t2
            out[i]["duration"] = round(max(0.4, _core_len(t2) / 6.5), 2)
            removed += 1
    stats["thinned"] = removed


def _thin_sfp(out: list, origs: list, stats: dict, cap_ratio: float) -> None:
    """文末終助詞を足したターンの比率が cap_ratio を超えたら、元の語尾へ戻す。

    語尾が毎ターン「〜だよね」「〜だな」になる対話は、フィラー過多と同じくらい単調に
    聞こえる。剥がす順は「ね」を足したターンから（人間コーパス比で「ね」だけは既に
    足りており、不足しているのは「よね」「な」なので、そちらを優先して残す）。
    """
    cap = int(cap_ratio * len(out))
    added = [(i, s) for i, t in enumerate(out)
             if (s := added_sfp(t["text"], origs[i])) is not None]
    removed = 0
    for i, s in sorted(added, key=lambda x: (x[1] != "ね", _core_len(out[x[0]]["text"]))):
        if len(added) - removed <= cap:
            break
        t2 = revert_sfp(out[i]["text"], origs[i])
        if t2 and t2 != out[i]["text"]:
            out[i]["text"] = t2
            out[i]["duration"] = round(max(0.4, _core_len(t2) / 6.5), 2)
            removed += 1
    stats["sfp"] = len(added) - removed
    stats["sfp_thinned"] = removed


def _strip_marks(s: str) -> str:
    return re.sub(r"[、。？\s]", "", s or "")


def _is_subsequence(a: str, b: str) -> bool:
    """a の文字が、順序どおり b の中に（間に何が挟まってもよいので）全部現れるか。"""
    it = iter(b)
    return all(ch in it for ch in a)


def clean_enriched_text(new: str, old: str, max_fillers_per_turn: int = 2) -> tuple[str, bool]:
    """LLM が返した発話を整形し、内容が壊れていれば元のテキストへフォールバックする。

    返り値: (採用テキスト, フォールバックしたか)
    enrich は Stage 2 のテキストに「足すだけ」の工程なので、元の文字が順序どおり
    全部残っていない出力（語の言い換え・要約・冒頭語の差し替え）は採用しない。
    """
    new = emphasize_question_marks(sanitize_utterance(new or ""))
    new = fix_particle_splits(new)
    # 「こう、」を発話の頭に付けるのはフィラーの誤用（20Bで実測）なので剥がす
    new = re.sub(r"^こう、", "", new)
    # 冒頭の「へえ」系リアクションは禁止語。元の発話が元々そう始まっている場合だけ残す。
    if _HEE_HEAD.match(new) and not _HEE_HEAD.match(old):
        new = _HEE_HEAD.sub("", new)
    # 終助詞の直前に割り込んで語尾を壊すフィラー（「〜から、えっと、ね。」）を除去
    new = _TAIL_BREAK.sub("、", new)
    # 文末記号の直前のフィラー（「〜ましょう、あの。」）と、読点で閉じずに後ろの語へ
    # 癒着したフィラー（「59センチ、んでしょうか？」）を除去
    new = _TAIL_FILLER.sub("", new)
    new = _GLUED_FILLER.sub("", new)
    new = _TAIL_COMMA.sub("", new)
    new = limit_fillers_in_turn(new, max_fillers_per_turn)
    if not new:
        return old, True
    if not new.endswith(("。", "？")):
        new = new.rstrip("、") + "。"
    # 文末に足された終助詞を検査（規則違反は元の語尾へ戻す）。文末記号の確定後に行う。
    new = fix_added_sfp(new, old)
    # 内容の捏造（隣のターンの文の混入・言い足し）は不採用
    if _core_len(new) > 2.5 * _core_len(old) + 10:
        return old, True
    # 元が質問（？終わり）なのに疑問が消えた出力も不採用
    if old.rstrip("。").endswith("？") and "？" not in new:
        return old, True
    # 「足すだけ」の徹底: 元の文字が1つでも欠けていたら元の発話に戻す。
    # 読点は足してよいので「、。？」を除いた文字列で判定する。
    # 実測で弾かれる形:「はい、次の項目を…」→「たしかに、次の項目を…」（冒頭語の差し替え）、
    #「予算は1人5万円です。安全面も…」→「うーん、安全面も…」（前半の脱落）、
    #「よろしくお願いします、頑張りましょう。」→「よくやろう、頑張りましょう。」（言い換え）。
    if not _is_subsequence(_strip_marks(old), _strip_marks(new)):
        return old, True
    return new, False


def _strip_speaker_prefix(text: str, names) -> str:
    """入力形式「話者: 発話」につられて text 先頭に混入した話者名を剥がす。

    30B で実測（「サトウそれは大事ね。」のように名前が地の文に直結する形）。
    「ユウ、〜」「ユウさんの〜」のような正当な呼びかけ・言及は剥がさない
    （名前の直後が読点・句点・敬称なら発話の一部とみなす）。
    """
    for name in names:
        if name and text.startswith(name) and len(text) > len(name) + 1:
            rest = text[len(name):]
            if rest.startswith(("、", "。", "？", "さん", "ちゃん", "くん", "君",
                                "さま", "様", "先生", "先輩")):
                continue
            return rest
    return text


def assemble_turns(input_turns: list, edits_by_index: dict, ep: EnrichParams) -> tuple[list, dict]:
    """入力ターン列に LLM のテキスト・タイミングを合流させて voice_turns を作る。

    speaker は入力からコピー、type は全て speech、同一話者の連続ターンには
    continued: true を機械付与する。index が返ってこなかったターンは元テキストのまま。
    """
    stats = {"fallback": 0, "missing": 0, "thinned": 0, "sfp": 0, "sfp_thinned": 0}
    names = {t["speaker"] for t in input_turns}
    out = []
    prev_speaker = None
    for i, t in enumerate(input_turns):
        old = t["utterance"]
        e = edits_by_index.get(i)
        if e is None:
            text, fb = old, False
            stats["missing"] += 1
            offset, duration = 0.3, None
        else:
            new = _strip_speaker_prefix(str(e.get("text") or ""), names)
            text, fb = clean_enriched_text(new, old, ep.max_fillers_per_turn)
            stats["fallback"] += fb
            offset, duration = e.get("start_offset"), e.get("duration")
        try:
            offset = round(float(offset), 2)
        except (TypeError, ValueError):
            offset = 0.3
        # 20B は start_offset を相対時間でなく累積の絶対時刻で返すことがある
        # （実測: 0→2.4→…→143.6。add_absolute_timeline が prev_end+offset を積むため
        # タイムラインが破綻し、retime の LLM フォールバックが数分のギャップを作る）。
        # プロンプトの想定レンジ 0〜1.0 に合わせて全ターンをクランプする。
        offset = min(max(offset, 0.0), 1.2)
        try:
            duration = round(float(duration), 2)
        except (TypeError, ValueError):
            duration = None
        if not duration or duration <= 0:
            duration = round(max(0.4, _core_len(text) / 6.5), 2)
        duration = min(duration, 8.0)
        turn = {
            "speaker": t["speaker"],
            "text": text,
            "type": "speech",
            "start_offset": offset,
            "duration": duration,
        }
        if t.get("pause_before_sec") is not None:
            # ユーザの長い沈黙（Stage2 の dialogue.silence）。generate_voice_passthrough と同じ扱い
            turn["pause_before_sec"] = float(t["pause_before_sec"])
            turn["start_offset"] = turn["pause_before_sec"]
        elif t["speaker"] == prev_speaker:
            turn["continued"] = True
            # 継ぎ足しの間合い: モデルが大きな間を返しても 0.2〜0.6 秒に収める
            turn["start_offset"] = min(max(offset, 0.2), 0.6)
        prev_speaker = t["speaker"]
        out.append(turn)
    _cap_monotone_heads(out, stats, ep.monotone_head_cap)
    _thin_fillers(out, stats, ep.filler_cap_ratio)
    _thin_sfp(out, [t["utterance"] for t in input_turns], stats, ep.sfp_cap_ratio)
    return out, stats


def parse_enrich_response(voice: dict, n_turns: int) -> dict | None:
    """LLM 応答から index→edit の辞書を作る。カバレッジが低すぎれば None（リトライ）。"""
    turns = voice.get("turns") if isinstance(voice, dict) else None
    if not isinstance(turns, list):
        return None
    edits = {}
    for e in turns:
        if not isinstance(e, dict):
            continue
        try:
            i = int(e.get("index"))
        except (TypeError, ValueError):
            continue
        if 0 <= i < n_turns and i not in edits:
            edits[i] = e
    # 7割未満しか返ってこない出力は指示逸脱とみなして作り直す
    if len(edits) < 0.7 * n_turns:
        return None
    return edits


async def convert_single_dialogue(
    client: AsyncOpenAI,
    model: str,
    record: dict,
    cost_tracker: CostTracker,
    semaphore: asyncio.Semaphore,
    retry_max: int = 3,
    retry_delay: float = 1.0,
    request_interval: float = 0.0,
    extra_body: dict | None = None,
    ep: EnrichParams | None = None,
) -> dict | None:
    if ep is None:
        ep = EnrichParams()
    system_prompt = build_system_prompt(ep)
    dialogue_id = record["id"]
    metadata = normalize_metadata_speakers(record["metadata"])
    turns = normalize_turn_speakers(record["dialogue"]["turns"])

    async with semaphore:
        if request_interval > 0:
            await asyncio.sleep(request_interval)
        for attempt in range(retry_max):
            if cost_tracker.exceeded():
                return None
            try:
                response = await _chat_create(
                    client,
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": build_enrich_prompt(metadata, turns, ep)},
                    ],
                    temperature=0.9,
                    max_completion_tokens=16384,
                    extra_body=extra_body or {},
                )
                usage = response.usage
                cost_tracker.add(usage.prompt_tokens, usage.completion_tokens)

                text = extract_json_text(response.choices[0].message.content or "")
                voice = json.loads(text)
                edits = parse_enrich_response(voice, len(turns))
                if edits is None:
                    logger.warning(f"[{dialogue_id}] turns のカバレッジ不足/形式不正。リトライ。")
                    continue

                voice_turns, stats = assemble_turns(turns, edits, ep)
                # モデルが実際に手を入れたターン数（入力から実質文字が変化した数）を数える。
                # 30B は enrich 指示を無視して素通しに近い出力を返すことがある（実測 1/50）。
                # 下限に満たなければ「付加していない」とみなして作り直す（付けすぎ側の上限は
                # プロンプトの cap と _thin_fillers が担保する）。フィラー・言い淀み・冒頭
                # リアクションのいずれも「元にない文字の追加」なのでこの1指標でまとめて計れる。
                _c = lambda s: re.sub(r"[、。？ー\s]", "", s or "")
                n_edited = sum(1 for i, t in enumerate(voice_turns)
                               if _c(t["text"]) != _c(turns[i]["utterance"]))
                # 30B の enrich 付加率は構造的に低い（実測 3〜10%。フィラーは 20B の
                # 仕事）。下限は「ほぼ素通し」だけを弾く控えめな値にし、到達不能な
                # ノルマで無限リトライしない（付加の多寡は最終的にユーザーのモデル選択次第）。
                floor = max(2, len(turns) // 20)
                if (n_edited < floor and attempt < retry_max - 1
                        and not cost_tracker.exceeded()):
                    logger.warning(f"[{dialogue_id}] enrich の付加が {n_edited}/{len(turns)} "
                                   f"ターンのみ（下限{floor}）。素通しとみなしてリトライ。")
                    continue
                # 冒頭リアクションが直前発話に合っているかを検証し、合わないものを剥がす。
                # （生成本体とは別プロンプトの校正パス。剥がすだけなので内容は不変。）
                if ep.verify_heads:
                    vstats = await verify_and_fix_heads(
                        client, model, voice_turns,
                        [t["utterance"] for t in turns],
                        cost_tracker=cost_tracker, extra_body=extra_body or {},
                    )
                    if vstats["removed"]:
                        logger.info(f"[{dialogue_id}] 冒頭リアクション検証: "
                                    f"候補{vstats['candidates']} → 剥がし{vstats['removed']}"
                                    f"（機械{vstats['hard']}/LLM{vstats['llm']}）")
                if stats["fallback"] or stats["missing"] or stats["thinned"] or n_edited:
                    logger.info(f"[{dialogue_id}] enrich 合流: 付加 {n_edited}, "
                                f"元テキストへのフォールバック {stats['fallback']}, "
                                f"index 欠落 {stats['missing']}, "
                                f"フィラー間引き {stats['thinned']}, "
                                f"終助詞 {stats['sfp']}(間引き {stats['sfp_thinned']}) "
                                f"(全{len(turns)}ターン)")
                add_absolute_timeline(voice_turns)
                return {
                    "id": dialogue_id,
                    "metadata": metadata,
                    "speaker_profiles": build_speaker_profiles(
                        metadata, voice.get("speakers")),
                    "speaking_rate": parse_speaking_rate(voice.get("speaking_rate")),
                    "voice_dialogue": {"turns": voice_turns},
                }
            except (json.JSONDecodeError, Exception) as e:
                err_str = str(e)
                if "401" in err_str or "429" in err_str:
                    delay = max(10, retry_delay * (3 ** attempt))
                else:
                    # ローカル推論はレート制限がないため上限30秒に抑える。
                    delay = min(30.0, retry_delay * (2 ** attempt))
                logger.warning(f"[{dialogue_id}] enrich 変換失敗 (attempt {attempt+1}, "
                               f"{delay:.0f}秒待機): {e}")
                if attempt < retry_max - 1:
                    await asyncio.sleep(delay)
        logger.error(f"[{dialogue_id}] 全リトライ失敗")
        return None


async def run_voice_generation(config: dict, cost_tracker: CostTracker) -> int:
    # 複数エンドポイント（同一モデルを別GPUに複数インスタンス）に対応。
    # llm.base_urls を指定するとラウンドロビンで負荷分散する（単数 base_url なら1台）。
    clients = build_async_llm_clients(config)
    num_endpoints = len(clients)
    ep = EnrichParams.from_config(config)
    logger.info(f"enrich 強度: filler_ratio={ep.filler_ratio} cap_ratio={ep.filler_cap_ratio} "
                f"reaction_ratio={ep.reaction_ratio} both={ep.allow_filler_and_reaction} "
                f"max_fillers/turn={ep.max_fillers_per_turn} "
                f"sfp_ratio={ep.sfp_ratio} sfp_cap_ratio={ep.sfp_cap_ratio}")
    input_file = Path(config["dialogue"]["output_file"])
    output_file = Path(config["voice"]["output_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    retry_max = config["retry_max_attempts"]
    retry_delay = config["retry_base_delay"]
    # max_concurrent_requests は「1エンドポイントあたりの並行度」。
    # 全体の同時リクエスト数はエンドポイント数倍にする（各インスタンスを遊ばせない）。
    # llm.ctx_per_request があれば台ごとにサーバの /props から決める（endpoint_concurrency）。
    per_endpoint = endpoint_concurrency(config, clients)
    total_concurrency = sum(per_endpoint)
    semaphore = asyncio.Semaphore(total_concurrency)
    if num_endpoints > 1:
        logger.info(f"エンドポイント {num_endpoints}台でラウンドロビン分散 "
                    f"(各{min(per_endpoint)}〜{max(per_endpoint)}並行 / 合計{total_concurrency}並行): "
                    f"{[str(c[0].base_url) for c in clients]}")

    if not input_file.exists():
        logger.error(f"入力の対話ファイルが見つかりません: {input_file}")
        logger.error("先に Stage 2（対話生成）を実行してください。")
        return 0

    dialogues = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                dialogues.append(json.loads(line))
    total = len(dialogues)

    existing_ids = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_ids.add(json.loads(line)["id"])

    pending = [d for d in dialogues if d.get("id") not in existing_ids]
    logger.info(f"enrich 変換開始: 全{total}件中、変換済み{len(existing_ids)}件、"
                f"未変換{len(pending)}件")

    if not pending:
        logger.info("未変換の対話はありません。")
        return len(existing_ids)

    converted = len(existing_ids)
    done = 0  # このrunで完了処理したタスク数（進捗ログ間引き用）

    # グループバリア（total_concurrency 件を gather で待つ）を廃し、ワーカーが空き次第
    # 次の対話を引く pull 型に変更。遅い対話が他エンドポイント（GPU）を待たせない。
    with open(output_file, "a") as f:
        async def process(client, model, extra_body, record, idx):
            return await convert_single_dialogue(
                client, model, record,
                cost_tracker, semaphore, retry_max, retry_delay,
                request_interval=config.get("request_interval", 0.0),
                extra_body=extra_body, ep=ep,
            )

        async def on_result(result, record, idx):
            nonlocal converted, done
            if result is not None:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                converted += 1
            done += 1
            # 進捗/コストログは total_concurrency 件ごと（従来のグループ相当の頻度）に間引く
            if done % total_concurrency == 0:
                cost_tracker.save_log()
                s = cost_tracker.summary()
                logger.info(f"進捗: {converted}/{total}件 | 入力={s['total_input_tokens']:,}tok "
                            f"出力={s['total_output_tokens']:,}tok | コスト: ${s['total_cost_usd']}")

        await run_endpoint_worker_pool(
            clients, pending, per_endpoint, process, on_result,
            should_stop=lambda: cost_tracker.exceeded(),
        )
        if cost_tracker.exceeded():
            logger.warning(f"コスト上限 (${cost_tracker.max_cost_usd}) に到達。停止しました。")
        cost_tracker.save_log()

    logger.info(f"enrich 変換完了: {converted}件")
    return converted
