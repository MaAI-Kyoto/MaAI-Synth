"""Stage 3: テキスト対話 → 音声対話への変換

Stage 2 で生成したテキスト対話（output/dialogues.jsonl）を、1対話ずつ
自然な音声対話に変換する。LLM に以下を行わせる:

  - 書き言葉的な発話を話し言葉に口語化
  - ターンテイキング（話者交代）の自然化
  - 相槌（backchannel）を短いターンとして挿入
  - フィラー・言い淀み（言いよどみ）を適度に挿入

各発話には発話タイミングを付与する。タイミングは「直前の発話が終了して
からの相対時間（秒）」で表現し、マイナス値はオーバーラップ（かぶせ）を表す。
"""

import asyncio
import json
import logging
import os
import re
from pathlib import Path

from openai import AsyncOpenAI

from cost_tracker import CostTracker
from speaker_gender import infer_gender, normalize_gender
from llm_client import (
    build_async_llm_client,
    emphasize_question_marks,
    extract_json_text,
    sanitize_utterance,
)

logger = logging.getLogger(__name__)

# gpt-5.6-terra 等の新しめモデルは temperature を既定値しか受け付けない。
# 一度でも temperature 非対応エラーが出たら、以後 temperature を送らない。
_TEMP_UNSUPPORTED = False


async def _chat_create(client, **kwargs):
    """temperature 非対応モデルなら自動で temperature を外して再試行する。"""
    global _TEMP_UNSUPPORTED
    if _TEMP_UNSUPPORTED:
        kwargs.pop("temperature", None)
    try:
        return await client.chat.completions.create(**kwargs)
    except Exception as e:
        if "temperature" in str(e).lower() and "temperature" in kwargs:
            _TEMP_UNSUPPORTED = True
            logger.info("モデルが temperature 非対応のため、既定値で続行します。")
            kwargs.pop("temperature", None)
            return await client.chat.completions.create(**kwargs)
        raise


SYSTEM_PROMPT = """\
あなたは日本語の対話を、テキスト対話から自然な「音声対話」へ変換する専門家です。
与えられたテキスト対話（2人の話者）を、実際に声に出して話したときのリアルで
対話的な音声対話に変換してください。単なる読み上げではなく、生きた会話にします。

# 目標: 生きた人間どうしのライブな会話

ラジオの朗読ではなく、対面で人間どうしがテンポよく言葉を交わす会話にする。
相槌・短いリアクション・フィラー・言い淀みを「これでもか」というくらい豊富に入れて、
人間らしい冗長さ・ゆらぎを再現する。ただし元の意味・情報・話題は保つ。

※重要な方針: 相槌の「回数」はとにかく多くする。ただし発話の「かぶせ（オーバーラップ）」は
  全体的に控えめにする。相槌も含め、基本は相手が話し終えた直後に打ち（start_offset は 0 付近）、
  被せる場合もごく短い相槌をわずかに食い気味に入れる程度に留める。「話者どうしの発話の衝突
  （割り込みでターンを奪う・長い発話が真正面からぶつかる）」は避け、きれいなターンテイキングを
  基本とする。

# 変換方針

1. 口語化（積極的に）
   - 書き言葉的・整い過ぎた表現を、実際に喋るときのくだけた話し言葉に直す。
     例:「〜することができます」→「〜できます」、「しかしながら」→「でも」など。
   - 縮約形を積極的に使う:「している→してる」「してしまう→しちゃう」「しておく→しとく」
     「ではない→じゃない」「という→って」「〜だよね」「〜じゃん」「〜かな」「〜っけ」。
   - 敬語の場面でも話し言葉の敬語に:「〜しちゃって」「〜なんですよね」「〜じゃないですか」。
   - 言い差し（言い切らず止める:「いや、それはちょっと。」）・倒置（「食べたよ、もう全部。」）・
     体言止め・聞き返し（「え、明日？」）も自然な範囲で使う。
   - 元の発話に不自然な日本語・翻訳調の言い回しがあれば、意味を保ったまま
     日本人が実際に口にする自然な表現へ直す。
   - 元の発話の意味・情報・話題の流れは保持する（勝手に内容を足し引きしない）。

# 最重要: text は「実際に声に出す言葉」だけにする（ト書き禁止）

各ターンの text には、その話者が実際に口に出して発する言葉だけを書く。
心情・動作・状況を説明するト書き（カッコ書き等の、実際には発話しない要素）は
一切出力しない。これらは後段のTTSがそのまま読み上げてしまうため厳禁。

  禁止する例（これらを text に入れない）:
    - 「（笑いながら）今日はさ、」 ← 「（笑いながら）」は発話しない
    - 「（小声で）ここだけの話なんだけど」「（ため息）」「（沈黙）」「（間）」
    - 「（うれしそうに）」「（驚いて）」「（うなずく）」「（笑）」「※補足」等
    - ト書きだけの独立ターン（例: text が「（二人とも笑う）」だけ）は作らない。

  正しい書き方:
    - 笑いや感嘆を表したいなら、実際の発話音として書く。
      例: 「（笑いながら）多すぎ」→「あはは、多すぎ」／「ふふ、多すぎでしょ」
    - 感情や状況は、あくまで実際のセリフ・口調・フィラー・相槌で表現する
      （カッコ書きの説明に頼らない）。
    - どうしても声にならない「間」を表したいときは、text ではなく
      start_offset（正の値＝ポーズ）で表現する。

  例外: 「あはは」「ふふ」「えー」「うわ」など、実際に口から出る笑い声・感嘆は
  発話なので text に入れてよい（これはト書きではない）。

2. 相槌（backchannel）を極めて密に（最重要・とにかく多く）
   - 聞き手が発する短い反応を、独立した短いターンとして「これでもか」というほど多く入れる。
   - 相槌(type:"backchannel")の text は、句読点を除いて6文字以内の短い語にする。
     語彙例:「うん」「うんうん」「はい」「ええ」「へえ」「ほー」「あー」「えっ」「うわ」
       「なるほど」「たしかに」「そうそう」「だよね」「ですよね」「わかる」「いいね」
       「まじで？」「ほんと？」「そっかー」「あーね」「ふんふん」「ふふ」「あはは」
   - それより長い反応（「へえ、そうなんだ」のような文になっている反応）は相槌にせず、
     type:"speech" の通常ターンとして出す。長い反応もどんどん入れてよいが type は
     必ず speech にする。
   - 相槌は「相手」の発話を聞いている最中・合間の反応である（最重要）。
     - 相槌ターンは、必ず相手の speech 発話の直後（または分割した相手発話の間）に置く。
     - 自分の speech 発話の直前・直後に自分の相槌ターンを置かない。
       返事の頭の「はい」「うん」は相槌ターンにせず、自分の次の speech の文頭に含める
       （例: type:"speech" の「はい、それでは始めますね。」）。
   - 目安: 話し手のまとまった発話（1〜2文ごと）に必ず1回以上、長い発話には2〜3回、
     聞き手の相槌を入れる。短い掛け合いの合間にもこまめに相づちを挟む。
     全ターンのうち4割程度が相槌ターンになるくらい多くてよい。
   - 相槌の「回数」は多く保ちつつ、かぶせ（オーバーラップ）は控えめにする。基本は相手が
     話し終えた直後（start_offset は 0 付近）に打つ。かぶせる場合もごく短い相槌を軽く
     食い気味に入れる程度（-0.2 秒程度まで）に留め、深い被りは作らない。
     （※長い主発話どうしをぶつけるのは避ける＝下の4参照。）
   - 同意・驚き・共感・笑い（「あはは」「ふふ」）・感嘆（「えっ」「うわ」）なども積極的に。
   - 相槌の語をワンパターンにしない（「はい」「うん」ばかりにせず、文脈に合わせて多様に）。

3. フィラー・言い淀み（言いよどみ）を大幅に増やす
   - 「えっとー」「あのー」「そのー」「なんか」「まあ」「こう」「ねえ」などのフィラーを、
     発話の冒頭・途中に高頻度で入れる。目安: 主発話の多く（半分以上）に何らかの
     フィラー・言い淀み・語尾の伸ばしが入るくらい。
   - 言い直し・言いかけ・ためらい・繰り返しを積極的に入れる。
     例:「それは、えっと、なんていうか」「私はね、私はそう思う」「って感じで」
        「あれ、なんだっけ、そう」「いや、っていうか」「でさ、あの、」
        「あ、ありがとうございます」「い、いや、そうじゃなくて」（最初の一音の言い直し）
   - 読点「、」を積極的に打ってポーズを作る。息継ぎ・ためらいの位置に「、」を入れ、
     一息ぶんの長さまで文を区切る（TTSが流暢に話しすぎるのを防ぐ）。
   - 文の途中で詰まる、語尾を伸ばす（「そうですねー」「でさー」）、
     考えながら話す間（「うーん」「んー」）なども入れる。
   - ただし意味・情報は保つ。フィラーや言い淀みで内容を消さない。
   - 聞き手用の感動詞（「へえ」「なるほど」「たしかに」など）を、自分が情報を伝える
     発話の冒頭に付けない。これらは相手の話を聞いたときの反応であり、
     自分の話の頭に付くと不自然（悪い例: 質問に答える側が「へえ、うちの子、熱が」）。
     フィラーとして使ってよい冒頭語は「えっと」「あのー」「なんか」「うーん」「まあ」など。

4. 発話の衝突（割り込み）は控えめに
   - 話者どうしの主発話（speech）を正面からぶつける「割り込み・ターン奪い」は避ける。
     基本は、一方が話し終えてから次の話者が話し始める（きれいなターンテイキング）。
   - 次の話者は、直前の発話の終わり際に「ごく軽く食い気味」に入る程度は自然
     （start_offset は 0 付近〜わずかにマイナス、-0.2 秒程度まで）。長い被りは作らない。
   - ターンを奪うような割り込みは、ここぞという場面でごくまれに使う程度に留める。
   - 被って良いのは「短い相槌」であって、意味のある主発話どうしではない。

5. ターンテイキングと発話の分割（重要）
   - 元のターンの順序と話者は基本的に保つ。
   - 長い発話は、相手の相槌をはさんで複数のターンに分割してよい（次の項参照）。
   - 相槌が入らない発話も、小さな発話単位に分割する（次の項参照）。実際の話し言葉は
     一息で長く話さず、短い塊を間を置きながら継ぎ足していく。

6. 自然さ
   - リズム・密度・メリハリを意識する（速い掛け合いと、ふと置く間の両方）。
   - 話者の口調・関係性・場面（敬語/タメ口など）は元の対話を踏襲する。

# 相槌による先行発話の分割（最重要）

聞き手の相槌が、話し手の発話の「途中」で起きる場合は、その話し手の発話を
相槌が入る位置で2つ（以上）に分割し、間に相槌ターンを挟むこと。
これにより、相槌が発話のどの時点で打たれたかをピンポイントに表現できる。

  悪い例（分割しない）:
    A(speech): 昨日ね、駅前の新しいカフェに行ってきたんだけど、すごく良くて
    B(backchannel): へえ        ← Aのどこで打たれたか分からない

  良い例（相槌位置で分割する）:
    A(speech):       昨日ね、駅前の新しいカフェに行ってきたんだけど     ← A1
    B(backchannel):  へえ                                          ← A1の途中〜終盤にかぶせる
    A(speech, continued): すごく良くてさ、コーヒーも雰囲気も最高だったの ← A2（Aの続き）

分割のルール:
  - 分割後の各セグメントも type は "speech"。Aの続き(A2以降)には "continued": true を付ける。
    最初のセグメント(A1)には continued は付けない（または false）。
  - 話し手は相槌で止まらず話し続けるのが普通。したがって続きのセグメント(A2)は、
    直前の相槌の終了と同時〜わずかに前から始まる程度（start_offset は 0 付近〜ごく軽い
    マイナス）にする。深い被りにはしない。
  - 1つの長い発話に対し、相手が複数回相槌を打つなら、その回数だけ分割してよい
    （A1, 相槌, A2, 相槌, A3 …）。
  - 短い発話（相手が口を挟む余地のない短いもの）は無理に分割しない。
  - 相手が黙って聞いている箇所まで無理に相槌を入れて分割しなくてよい。自然な所に。

相槌が入らない発話の分割（こちらも最重要）:
  - 相槌を挟まない場合でも、長い発話は小さな発話単位に分割する。
    - 2文以上からなる発話は、文ごとに別ターンに分割する。
    - 1文でも長い場合は「、」の意味の切れ目で分割してよい。
    - 目安: 1ターンはおおむね25文字以内の短い塊にする。
  - ただし、切ってよいのは文末（。？）か、意味のまとまりが切れる「、」の位置だけ。
    - 各セグメントは、それ単体で読み上げて自然な「ひとまとまりの発話」にする。
    - 文節の途中や助詞・助動詞の前で切らない。「ね。」「かも。」「、〜」のような
      断片だけのセグメントを作らない（悪い例:「アクセスできる。」+「かも。」）。
    - 8文字未満の断片になるくらいなら、分割せず前のセグメントにつなげたままにする。
    - セグメントの先頭を「、」や助詞から始めない。
  - 続きのセグメントには "continued": true を付ける。話者は同じまま。
  - 続きのセグメントの start_offset は 0.2〜0.6 秒程度の小さな間にする
    （考えながら少しずつ継ぎ足して話す間合い。相槌を挟む分割の 0 付近とは違う）。
    例:
      A(speech):            昨日さ、駅前の新しいカフェ行ったんだけど。
      A(speech, continued): なんか、コーヒーがすごい美味しくて。   ← start_offset 0.4 など
      A(speech, continued): 雰囲気も良かったんだよね。             ← start_offset 0.3 など

# タイミングの表現

各ターンに、直前のターンが終了してからの相対時間 start_offset（秒）を付ける。

  - start_offset > 0 : 直前の発話が終わってから、その秒数だけ間（ポーズ）を置いて話し始める。
  - start_offset = 0 : 直前の発話が終わると同時に話し始める。
  - start_offset < 0 : 直前の発話がまだ終わらないうちに話し始める（オーバーラップ＝かぶせ）。
                       主に相槌、および分割した発話の続き(A2)で話し手が止まらず続ける場合に
                       マイナスになる。

  かぶせ（負の start_offset）の使い分け（オーバーラップは全体的に控えめに）:
  - 相槌は「回数」を多く入れるが、タイミングは基本ゼロ付近（相手が言い終わった直後）に打つ。
    かぶせる場合もごく軽く（-0.2 秒程度まで）に留め、深い被りにはしない。
  - 主発話(speech)どうしは基本かぶせない。次の話者は前の話者が言い終わってから話す
    （start_offset は 0 以上）。入るとしても終わり際にごく軽く（-0.2 秒程度まで）。
  - テンポの良い相づち・同意は 0 に近い値（間を置かず即応）。
  - 話題の切り替わりや考える間では正の値（ポーズ）を置き、メリハリをつける。

また各ターンに、その発話の推定発話長 duration（秒）も付ける。
duration は発話の文字数・モーラ数からおおよそ見積もる（日本語の発話はおおよそ
1秒あたり6〜8モーラ程度。短い相槌は0.3〜0.8秒程度）。

# 話者の性別推定

各話者の性別を、名前・役割・性格・話し方から "male"（男性）または
"female"（女性）で推定し、speakers 配列に含める。これは後段のTTSで
話者ごとに適切な声を割り当てるために使う。判断材料が乏しい場合でも、
最も自然と思われる方を必ずどちらかに決めること。

# 話速（speaking_rate）の推定

この対話全体の話す速さを、内容・雰囲気・関係性・場面から 1.0〜1.2 の範囲で推定し、
speaking_rate として出力する（1.0=標準的な速さ、1.2=やや速め・軽快）。目安:
  - 親しい間柄の雑談・盛り上がる会話・楽しい話題・テンポの良い掛け合い → 1.15〜1.2
  - 一般的な会話 → 1.1 前後
  - 落ち着いた相談・真剣な話・フォーマル・悲しい/重い場面（葬儀など）・年配者 → 1.0〜1.05
必ず 1.0〜1.2 の範囲の数値にすること。

# 出力形式（JSONのみ。マークダウンのコードブロックは不要）

{
  "speakers": [
    {"name": "話者名", "gender": "male"},
    {"name": "話者名", "gender": "female"}
  ],
  "speaking_rate": 1.15,
  "turns": [
    {
      "speaker": "Aさん",
      "text": "昨日ね、えっと、駅前の新しいカフェに行ってきたんだけど",
      "type": "speech",
      "start_offset": 0.3,
      "duration": 2.8
    },
    {
      "speaker": "Bさん",
      "text": "へえ",
      "type": "backchannel",
      "start_offset": -0.1,
      "duration": 0.4
    },
    {
      "speaker": "Aさん",
      "text": "すごく良くてさ、コーヒーも雰囲気も最高だったの",
      "type": "speech",
      "continued": true,
      "start_offset": 0.3,
      "duration": 2.6
    },
    {
      "speaker": "Bさん",
      "text": "いいなあ、今度行ってみよ",
      "type": "speech",
      "start_offset": 0.2,
      "duration": 1.9
    }
  ]
}

要件:
- speaking_rate は 1.0〜1.2 の数値。対話の内容・雰囲気から推定する。
- speakers には与えられた2人の話者それぞれの性別("male"/"female")を入れる。
- type は "speech"（主発話）または "backchannel"（相槌）のいずれか。
- backchannel の text は句読点を除き6文字以内。長い反応は speech にする。
- 分割した発話の続き(2つ目以降のセグメント)には "continued": true を付ける。
- 長い発話は短い塊（1ターンおおむね25文字以内）に分割する。2文以上を1ターンに残さない。
- speaker は与えられた2人の話者の名前のみを使う。
- start_offset・duration は数値（秒）。小数可。
- text は実際に声に出す言葉だけにする。心情・動作・状況を説明するカッコ書き等の
  ト書き（発話しない要素）は一切含めない（TTSが読み上げてしまうため）。
- text に使う記号は「？」だけにする。「…」「!」「！」「〜」「・」やカッコ書き等の
  記号・約物は使わず、区切りは「。」、疑問は「？」を使う（感嘆・伸ばし・間は記号では
  なく実際の発話音や start_offset で表現する）。
- TTSで自然に区切って読めるよう句読点を多めに打つ。意味の切れ目・息継ぎには「、」を
  こまめに入れ、文末には「。」を、質問・確認・問い返しの文末には必ず「？」を付ける
  （短い相槌はこの限りではない）。
- text 内に空白（半角・全角スペース等）を入れない。
- 相手の発話途中で相槌を打つ箇所は、必ず先行発話を分割して相槌を挟む。
- JSONのみを出力する。
"""


def build_voice_prompt(metadata: dict, turns: list) -> str:
    speakers = ", ".join(s["name"] for s in metadata.get("speakers", []))
    lines = [
        "以下は2人の話者によるテキスト対話です。これを自然な音声対話に変換してください。",
        "",
        f"話者: {speakers}",
        f"場面: {metadata.get('setting', '')}",
        f"関係性: {metadata.get('relationship', '')}",
        f"口調: {metadata.get('speech_style', '')}",
        "",
        "【元のテキスト対話】",
    ]
    for t in turns:
        lines.append(f"{t['speaker']}: {t['utterance']}")
    # 相槌の最低個数を数値ノルマとして明示する。システムプロンプトの比率指示だけだと
    # ローカル32Bモデルが相槌ゼロの素通しに近い出力を返すことがあるため、
    # 対話ごとの具体的な数を末尾（モデルが最も従いやすい位置）で要求する。
    min_bc = max(8, len(turns) // 2)
    lines.append("")
    lines.append(f"この対話は元が{len(turns)}ターンです。相手の発話を聞いている最中に打つ"
                 f"聞き手の相槌ターン（type:\"backchannel\"、句読点を除き6文字以内、"
                 f"必ず相手の speech の直後に置く）を{min_bc}個以上追加し、"
                 "相槌が入る位置では先行発話を分割してください。"
                 "自分の speech の直前直後に自分の相槌を置かないでください。")
    lines.append("JSONのみを出力してください。")
    return "\n".join(lines)


def add_absolute_timeline(turns: list) -> list:
    """start_offset と duration から絶対タイムライン（秒）を計算して付与する。

    start_abs[i] = end_abs[i-1] + start_offset[i]
    end_abs[i]   = start_abs[i] + duration[i]
    最初のターンは start_abs = 0 とする（start_offset は無視）。
    """
    prev_end = 0.0
    for i, t in enumerate(turns):
        try:
            offset = float(t.get("start_offset", 0.0))
        except (TypeError, ValueError):
            offset = 0.0
        try:
            duration = float(t.get("duration", 0.0))
        except (TypeError, ValueError):
            duration = 0.0
        start_abs = 0.0 if i == 0 else prev_end + offset
        end_abs = start_abs + duration
        t["start_abs"] = round(start_abs, 2)
        t["end_abs"] = round(end_abs, 2)
        prev_end = end_abs
    return turns


# 話者名の末尾に付くふりがな・役割の注記（例: "美咲 (みさき)" / "佐藤 (営業担当)"）を除去する。
# Stage1 のメタデータにこの注記が焼き込まれると、モデルは各ターンでは素の名前
# （"美咲" / "佐藤"）を書くため、validate_voice_turns の完全一致チェックが全ターンで
# 不一致になり、全リトライ失敗する（両話者ともカッコ付きだと回復不能）。名前は素の形に
# 正規化して、プロンプト・バリデーション・出力の全てで統一する。
_PAREN_RE = re.compile(r"[\s　]*[（(][^）)]*[）)]")


def normalize_speaker_name(name) -> str:
    """話者名からカッコ注記（ふりがな/役割等）を除去して素の名前にする。"""
    if not isinstance(name, str):
        return name
    return _PAREN_RE.sub("", name).strip()


def normalize_metadata_speakers(metadata: dict) -> dict:
    """metadata の speakers[].name を正規化した浅いコピーを返す（元は変更しない）。"""
    new = dict(metadata)
    new["speakers"] = [
        ({**s, "name": normalize_speaker_name(s.get("name"))} if isinstance(s, dict) else s)
        for s in metadata.get("speakers", [])
    ]
    return new


def normalize_turn_speakers(turns: list) -> list:
    """対話ターンの speaker を正規化したコピー（リスト）を返す（元は変更しない）。"""
    return [
        ({**t, "speaker": normalize_speaker_name(t.get("speaker"))} if isinstance(t, dict) else t)
        for t in turns
    ]


def build_speaker_profiles(metadata: dict, speakers_field) -> list:
    """metadata の性別・年代と、Stage 2 が返した性別を結合した話者プロフィール。

    [{"name": ..., "gender": "male"|"female"|None, "age_range": ...}, ...]

    性別は **metadata.speakers[].gender（Stage 1 が決めた値）を最優先** にする。
    Stage 1 は名前・役割・シナリオ本文をその性別として書いているので、名前から
    後付けで推定する Stage 2 の値より確実。Stage 2 の値は、gender を持たない
    古い metadata のためのフォールバックとして残す。
    """
    gender_by_name = {}
    if isinstance(speakers_field, list):
        for s in speakers_field:
            if isinstance(s, dict) and s.get("name"):
                # LLM が speakers 配列でカッコ注記付きの名前を返しても対応付くよう正規化キーで格納。
                gender_by_name[normalize_speaker_name(s["name"])] = normalize_gender(s.get("gender"))

    profiles = []
    for s in metadata.get("speakers", []):
        name = s.get("name")
        profiles.append({
            "name": name,
            "gender": (normalize_gender(s.get("gender"))
                       or gender_by_name.get(name)
                       or infer_gender(name, s.get("role"))),
            "age_range": s.get("age_range"),
        })
    return profiles


def parse_speaking_rate(v, lo: float = 1.0, hi: float = 1.2, default: float = 1.1) -> float:
    """LLM推定の話速を [lo, hi] にクランプする。数値でなければ default。"""
    try:
        r = float(v)
    except (TypeError, ValueError):
        return default
    return round(min(max(r, lo), hi), 3)


def normalize_backchannel_positions(turns: list) -> tuple[list, dict]:
    """相槌の「位置」を決定的に正す。

    相槌は「相手の発話を聞いている最中/合間の短い反応」であるべきだが、LLM は
    - 返事の頭（「はい。」＋続けて自分の speech）を相槌ターンとして出す
    - 自分の発話の直後に自分の相槌を置く（誰の発話への反応でもない自己相槌）
    を大量に混ぜる（実測で相槌の7割超）。これらはリタイム時に相手発話へ食い込む
    配置をされて不自然に聞こえるため、ここで正す:
      - 直後のターンが同一話者 → 相槌テキストを次ターンの文頭に結合（返事の頭に戻す）
      - 直前の非相槌ターンが同一話者（かつ上に該当しない） → 削除（自己相槌）
      - 直前の非相槌ターンが相手 → 正しい相槌としてそのまま残す
    Returns: (新しい turns, 統計dict)
    """
    stats = {"bc_merged_into_reply": 0, "bc_self_dropped": 0, "bc_kept": 0}
    out = list(turns)
    # 後ろから処理: 連鎖（相槌→相槌→speech が同一話者）も順に文頭へ畳み込まれる
    for i in range(len(out) - 2, -1, -1):
        t = out[i]
        if not (isinstance(t, dict) and t.get("type") == "backchannel"):
            continue
        nxt = out[i + 1] if i + 1 < len(out) else None
        if nxt is not None and isinstance(nxt, dict) \
                and nxt.get("speaker") == t.get("speaker"):
            head = (t.get("text") or "").rstrip()
            if head.endswith("。"):
                head = head[:-1] + "、"
            merged = {**nxt, "text": head + (nxt.get("text") or "")}
            if "start_offset" in t:  # 間合いは相槌側の位置ヒントを引き継ぐ
                merged["start_offset"] = t.get("start_offset")
            out[i + 1] = merged
            del out[i]
            stats["bc_merged_into_reply"] += 1
    cleaned = []
    for i, t in enumerate(out):
        if isinstance(t, dict) and t.get("type") == "backchannel":
            prev = next((u for u in reversed(cleaned)
                         if u.get("type") != "backchannel"), None)
            if prev is not None and prev.get("speaker") == t.get("speaker"):
                stats["bc_self_dropped"] += 1
                continue
            stats["bc_kept"] += 1
        cleaned.append(t)
    return cleaned, stats


def validate_voice_turns(turns: list, speaker_names: set) -> bool:
    if not isinstance(turns, list) or not turns:
        return False
    for t in turns:
        if not isinstance(t, dict):
            return False
        if "speaker" not in t or "text" not in t:
            return False
        if t.get("speaker") not in speaker_names:
            return False
    return True


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
) -> dict | None:
    dialogue_id = record["id"]
    # 話者名のカッコ注記（ふりがな/役割）を除去して統一する。プロンプト・バリデーション・
    # 出力レコードの全てで素の名前を使うことで、モデルが素の名前でターンを書いても一致する。
    metadata = normalize_metadata_speakers(record["metadata"])
    turns = normalize_turn_speakers(record["dialogue"]["turns"])
    speaker_names = {s["name"] for s in metadata.get("speakers", [])}

    # 相槌が極端に少ない出力（モデルが変換指示を無視して素通しに近い出力を返す事例が
    # 32Bローカルモデルで散発する）は品質不合格として作り直す。ただし再生成は高コスト
    # なので試行は MIN_BC_RETRIES 回まで、以後はそれまでで最も相槌の多い結果を採用する。
    # 比率は normalize_backchannel_positions 適用後（=正しい位置の相槌のみ）で数える。
    # 最終的な相槌密度は refine（MaAI挿入）が担保するため、ここは素通し検出が目的。
    MIN_BC_RATIO = 0.05
    MIN_BC_RETRIES = 2
    best_result: dict | None = None
    best_ratio = -1.0
    low_bc_attempts = 0

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
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_voice_prompt(metadata, turns)},
                    ],
                    temperature=0.8,
                    # 相槌・発話分割で50-160+ターンに膨らむ。旧metadata由来の最長対話(150-162
                    # ターン)は12288でも finish_reason=length で途中切れ→JSON破損したため16384へ。
                    max_completion_tokens=16384,
                    extra_body=extra_body or {},
                )
                usage = response.usage
                cost_tracker.add(usage.prompt_tokens, usage.completion_tokens)

                text = extract_json_text(response.choices[0].message.content or "")

                voice = json.loads(text)
                voice_turns = voice.get("turns") if isinstance(voice, dict) else None
                # モデル出力側の speaker はカッコ注記を除去、text は「？」以外の記号と
                # 空白を除去する。記号のみで空になったターンは捨てる。
                if isinstance(voice_turns, list):
                    cleaned = []
                    for t in voice_turns:
                        if isinstance(t, dict) and "speaker" in t:
                            t["speaker"] = normalize_speaker_name(t["speaker"])
                        if isinstance(t, dict) and "text" in t:
                            t["text"] = emphasize_question_marks(
                                sanitize_utterance(t["text"]))
                            if not t["text"]:
                                continue
                            # モデルが文末の句読点を落とすことがある（TTSの区切りが
                            # 崩れる）ため、末尾が「。」「？」でなければ「。」を補う。
                            if not t["text"].endswith(("。", "？")):
                                t["text"] = t["text"].rstrip("、") + "。"
                        cleaned.append(t)
                    voice_turns = cleaned
                    # 相槌位置の正規化（返事頭の結合・自己相槌の削除）
                    voice_turns, bc_stats = \
                        normalize_backchannel_positions(voice_turns)
                    if bc_stats["bc_merged_into_reply"] or bc_stats["bc_self_dropped"]:
                        logger.info(f"[{dialogue_id}] 相槌位置の正規化: "
                                    f"返事頭へ結合 {bc_stats['bc_merged_into_reply']}, "
                                    f"自己相槌の削除 {bc_stats['bc_self_dropped']}, "
                                    f"残存 {bc_stats['bc_kept']}")
                if validate_voice_turns(voice_turns, speaker_names):
                    add_absolute_timeline(voice_turns)
                    speaker_profiles = build_speaker_profiles(
                        metadata, voice.get("speakers"))
                    result = {
                        "id": dialogue_id,
                        "metadata": metadata,
                        "speaker_profiles": speaker_profiles,
                        "speaking_rate": parse_speaking_rate(voice.get("speaking_rate")),
                        "voice_dialogue": {"turns": voice_turns},
                    }
                    n_bc = sum(1 for t in voice_turns
                               if t.get("type") == "backchannel")
                    ratio = n_bc / max(len(voice_turns), 1)
                    if ratio >= MIN_BC_RATIO:
                        return result
                    if ratio > best_ratio:
                        best_result, best_ratio = result, ratio
                    if low_bc_attempts >= MIN_BC_RETRIES:
                        logger.warning(f"[{dialogue_id}] 相槌比 {best_ratio:.0%} のまま"
                                       f"再生成上限。最良の結果を採用します。")
                        return best_result
                    low_bc_attempts += 1
                    logger.warning(f"[{dialogue_id}] 相槌が少なすぎます"
                                   f"（{n_bc}/{len(voice_turns)}={ratio:.0%} < "
                                   f"{MIN_BC_RATIO:.0%}）。再生成 {low_bc_attempts}回目。")
                else:
                    logger.warning(f"[{dialogue_id}] 不正な形式。リトライ。")
            except (json.JSONDecodeError, Exception) as e:
                err_str = str(e)
                if "401" in err_str or "429" in err_str:
                    delay = max(10, retry_delay * (3 ** attempt))
                    logger.warning(f"[{dialogue_id}] 音声変換失敗 (attempt {attempt+1}, 認証/レート制限エラー、{delay:.0f}秒待機): {e}")
                else:
                    delay = retry_delay * (2 ** attempt)
                    logger.warning(f"[{dialogue_id}] 音声変換失敗 (attempt {attempt+1}, {delay:.0f}秒待機): {e}")
                if attempt < retry_max - 1:
                    await asyncio.sleep(delay)
        if best_result is not None:  # 相槌不足ながら形式は正しい結果が残っている
            logger.warning(f"[{dialogue_id}] リトライ上限。相槌比 {best_ratio:.0%} の"
                           "最良結果を採用します。")
            return best_result
        logger.error(f"[{dialogue_id}] 全リトライ失敗")
        return None


async def run_voice_generation(config: dict, cost_tracker: CostTracker) -> int:
    client, model, llm_extra = build_async_llm_client(config)
    input_file = Path(config["dialogue"]["output_file"])
    output_file = Path(config["voice"]["output_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    retry_max = config["retry_max_attempts"]
    retry_delay = config["retry_base_delay"]
    semaphore = asyncio.Semaphore(config["max_concurrent_requests"])

    if not input_file.exists():
        logger.error(f"入力の対話ファイルが見つかりません: {input_file}")
        logger.error("先に Stage 2（対話生成）を実行してください。")
        return 0

    # テキスト対話の読み込み
    dialogues = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                dialogues.append(json.loads(line))
    total = len(dialogues)

    # 変換済みIDをチェック（途中再開対応）
    existing_ids = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_ids.add(json.loads(line)["id"])

    pending = [d for d in dialogues if d.get("id") not in existing_ids]
    logger.info(f"音声変換開始: 全{total}件中、変換済み{len(existing_ids)}件、未変換{len(pending)}件")

    if not pending:
        logger.info("未変換の対話はありません。")
        return len(existing_ids)

    converted = len(existing_ids)
    group_size = config["max_concurrent_requests"]

    with open(output_file, "a") as f:
        for group_start in range(0, len(pending), group_size):
            if cost_tracker.exceeded():
                logger.warning(f"コスト上限 (${cost_tracker.max_cost_usd}) に到達。停止します。")
                break

            group_end = min(group_start + group_size, len(pending))
            tasks = [
                convert_single_dialogue(
                    client, model, pending[i],
                    cost_tracker, semaphore, retry_max, retry_delay,
                    request_interval=config.get("request_interval", 0.0),
                    extra_body=llm_extra,
                )
                for i in range(group_start, group_end)
            ]

            results = await asyncio.gather(*tasks)
            for result in results:
                if result is not None:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    converted += 1

            cost_tracker.save_log()
            s = cost_tracker.summary()
            logger.info(f"進捗: {converted}/{total}件 | 入力={s['total_input_tokens']:,}tok 出力={s['total_output_tokens']:,}tok | コスト: ${s['total_cost_usd']}")

    logger.info(f"音声変換完了: {converted}件")
    return converted
