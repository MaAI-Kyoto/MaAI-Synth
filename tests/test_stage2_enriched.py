"""Stage2 enriched モード（S2B）の決定的処理。

Stage3(enrich) と違い、この工程には照合すべき「元テキスト」が無い。壊れた形を
機械的に直すときに、直してよいものだけを直しているか（＝正しい発話を壊していないか）
を検査する。LLM は呼ばない。
"""

import re
import unittest
from pathlib import Path

import yaml

from generate_voice_enrich import _SFP_TAIL
from generate_dialogues import (
    ALL_GATES,
    DEFAULT_PROMPT_FILE,
    ENRICHED_ANCHOR,
    GENERAL_S2,
    S2Params,
    _CASUAL_LEAK_RE,
    _TRUNCATED_COPULA_RE,
    _VERB_DESU_RE,
    _is_polite_style,
    _mid_filler_slots,
    break_long_runs,
    build_dialogue_prompt,
    build_prompt_pack,
    build_system_prompt,
    cap_laugh_heads,
    fix_casual_reactions,
    cap_monotone_heads,
    clean_enriched_utterance,
    enriched_ratios,
    filler_quota,
    fix_na_tail,
    fix_self_intro_sfp,
    fix_vocative_copula,
    inject_fillers,
    thin_fillers,
    thin_ne,
    thin_sfp,
    diversify_sfp_runs,
    fix_ne_misuse,
    fix_vocative_sfp,
    fix_yo_misuse,
    fix_yone_misuse,
    inject_sfp,
    load_gates,
    load_prompt_pack,
    load_self_name_max,
    sfp_candidates,
)
from generate_voice_passthrough import build_voice_turns, convert_record


def _turns(*utterances, speakers=None):
    speakers = speakers or [("A" if i % 2 == 0 else "B") for i in range(len(utterances))]
    return [{"speaker": s, "utterance": u} for s, u in zip(speakers, utterances)]


class FixNaTailTests(unittest.TestCase):
    def test_strips_na_after_verb(self):
        # 「行くな」「思うな」は禁止の意味になり発話の意味が反転する
        self.assertEqual(fix_na_tail("それは違うと思うな。"), "それは違うと思う。")
        self.assertEqual(fix_na_tail("明日は早めに行くな。"), "明日は早めに行く。")
        self.assertEqual(fix_na_tail("資料は先に作るな。"), "資料は先に作る。")

    def test_strips_na_after_polite_ending(self):
        self.assertEqual(fix_na_tail("さっき確認しましたな。"), "さっき確認しました。")
        self.assertEqual(fix_na_tail("それは大変ですな。"), "それは大変です。")

    def test_keeps_correct_usage(self):
        for ok in ("これは難しいな。", "昨日は疲れたな。", "そうだな。", "行きたくないな。"):
            self.assertEqual(fix_na_tail(ok), ok)

    def test_keeps_words_that_merely_end_in_na(self):
        # 平仮名だけの語幹は名詞と区別が付かないので触らない（壊すより残す）
        for ok in ("みんな。", "さかな。", "そうかな。", "またあとでな。"):
            self.assertEqual(fix_na_tail(ok), ok)


class CleanEnrichedUtteranceTests(unittest.TestCase):
    def test_removes_filler_before_sentence_end(self):
        self.assertEqual(clean_enriched_utterance("じゃあ協力しましょう、あの。"),
                         "じゃあ協力しましょう。")
        self.assertEqual(clean_enriched_utterance("分析が大変になりませんか、なんか？"),
                         "分析が大変になりませんか？")

    def test_removes_filler_split_before_particle(self):
        self.assertEqual(clean_enriched_utterance("その素材、えっと、を活用したい。"),
                         "その素材を活用したい。")

    def test_removes_glued_filler(self):
        self.assertEqual(clean_enriched_utterance("高さは59センチ、んでしょうか？"),
                         "高さは59センチでしょうか？")

    def test_removes_banned_head_reaction(self):
        self.assertEqual(clean_enriched_utterance("へえ、それは知らなかった。"),
                         "それは知らなかった。")

    def test_keeps_one_filler_per_turn(self):
        # 1発話に2個入った形は最初の1個だけ残す（囲んでいた読点の片方は間として残る）
        self.assertEqual(
            clean_enriched_utterance("具体的に、えっと、どんな体験を、なんか、考えてる？"),
            "具体的に、えっと、どんな体験を、考えてる？")

    def test_allows_more_fillers_when_budget_is_raised(self):
        # 依頼された形（冒頭＋文中に複数）。max_fillers=1 だと構造的に作れない
        want = ("うーん、実は、その、予算が限られてて、まあ、"
                "どうやって盛り上げようか、悩んでるんです。")
        self.assertEqual(clean_enriched_utterance(want, max_fillers=3), want)
        # 予算が1個なら、短い発話と同じく最初の1個だけ残る
        self.assertNotIn("まあ、", clean_enriched_utterance(want, max_fillers=1))

    def test_strips_a_leading_comma(self):
        # 単体では無害だが、同一話者の連結後に「〜ですか？、なるほど、」になる
        self.assertEqual(clean_enriched_utterance("、なるほど、そこがポイントですね。"),
                         "なるほど、そこがポイントですね。")

    def test_normalizes_small_kana_fillers(self):
        # 「まぁ」はフィラー判定の正規表現に載らないので、個数の上下限から漏れる
        self.assertEqual(clean_enriched_utterance("まぁ、それはそうだね。"),
                         "まあ、それはそうだね。")

    def test_leaves_well_formed_utterance_untouched(self):
        for ok in ("えっと、昨日さ、駅前のカフェ行ったんだよね。",
                   "なるほど、それは知らなかったな。",
                   "予算はどれくらい、なんか、見込んでますか？"):
            self.assertEqual(clean_enriched_utterance(ok), ok)

    def test_always_ends_with_terminator(self):
        self.assertEqual(clean_enriched_utterance("それでね、"), "それでね。")


class ThinningTests(unittest.TestCase):
    def test_thin_fillers_respects_cap(self):
        turns = _turns(*[f"えっと、これは{i}番目の話なんだよね。" for i in range(10)])
        removed = thin_fillers(turns, 0.3)
        fr, _, _, _ = enriched_ratios(turns)
        self.assertEqual(removed, 7)
        self.assertLessEqual(fr, 0.3)

    def test_thin_fillers_removes_from_shortest_first(self):
        turns = _turns("えっと、うん。", "えっと、これはかなり長い説明の発話なんだよね。")
        thin_fillers(turns, 0.5)
        self.assertEqual(turns[0]["utterance"], "うん。")
        self.assertTrue(turns[1]["utterance"].startswith("えっと、"))

    def test_thin_sfp_only_drops_safe_particles(self):
        turns = _turns("これは難しいな。", "そうですよね。", "そうかな。", "あれだよ。",
                       "みんな。", "それでいいと思うな。")
        thin_sfp(turns, 0.0)
        # 落とせるのは「〜だ/た/い」に付いた「な」と「よね」だけ。
        # 「かな」「よ」は語気が変わるので、「みんな」は語が壊れるので触らない。
        # 動詞＋「な」は clean_enriched_utterance が先に直すのでここでは残す。
        self.assertEqual([t["utterance"] for t in turns],
                         ["これは難しい。", "そうです。", "そうかな。", "あれだよ。",
                          "みんな。", "それでいいと思うな。"])

    def test_thin_ne_drops_only_polite_assertions(self):
        """「ね」の間引き（ユーザー指摘 2026-08-01「ねが多すぎます」）。人間コーパスの
        「ね」は発話100文字あたり0.57個で、合成の実測1.34個の半分以下だった。"""
        turns = _turns(
            "では、曲目と機材は来週月曜までにまとめますね。",   # 丁寧形＝落とす
            "布は1300円で、同じ面積でも費用は抑えられますね。",  # 丁寧形＝落とす
            "はい、音楽が完成したらミックスダウンして共有するね。",  # タメ口の終止形＝残す
            "うーん、卒業後の進路がまだ決まらなくてね。",       # 言いさし＝残す
            "はい、詳細はまたご相談くださいね。",              # 依頼の緩和＝残す
            "そうですね。",                                   # 短い＝残す
        )
        self.assertEqual(thin_ne(turns, 0.01), 2)
        self.assertEqual([t["utterance"] for t in turns], [
            "では、曲目と機材は来週月曜までにまとめます。",
            "布は1300円で、同じ面積でも費用は抑えられます。",
            "はい、音楽が完成したらミックスダウンして共有するね。",
            "うーん、卒業後の進路がまだ決まらなくてね。",
            "はい、詳細はまたご相談くださいね。",
            "そうですね。",
        ])

    def test_thin_ne_counts_sentences_inside_a_turn(self):
        """発話の中の文末も対象。ターン末だけ見ると「〜ですね。でも〜ですね。」の
        前半が数え落ちる（ターンの区切り方に左右されない単位にするため）。"""
        turns = _turns("予算は約4万円で収まりますね。"
                       "ただ、造花の数を減らす必要がありますね。")
        self.assertEqual(thin_ne(turns, 0.01), 2)
        self.assertEqual(turns[0]["utterance"],
                         "予算は約4万円で収まります。ただ、造花の数を減らす必要があります。")

    def test_thin_ne_is_off_by_default(self):
        turns = _turns("では、曲目と機材は来週月曜までにまとめますね。")
        self.assertEqual(thin_ne(turns, 0.0), 0)
        self.assertEqual(turns[0]["utterance"],
                         "では、曲目と機材は来週月曜までにまとめますね。")

    def test_thin_ne_keeps_density_above_the_cap_untouched(self):
        """上限に収まっている対話は触らない。"""
        turns = _turns("では、曲目と機材は来週月曜までにまとめますね。",
                       "はい、承知しました。", "こちらこそ、よろしくお願いします。")
        self.assertEqual(thin_ne(turns, 2.0), 0)

    def test_cap_monotone_heads(self):
        turns = _turns(*["まあ、そうだよね。"] * 5)
        stripped = cap_monotone_heads(turns, 3)
        self.assertEqual(stripped, 2)
        self.assertEqual(turns[-1]["utterance"], "そうだよね。")

    def test_enriched_ratios(self):
        turns = _turns("えっと、そうなんだよね。", "うん。", "たしかにそうだな。", "行こう。")
        fr, sr, cm, rr = enriched_ratios(turns)
        self.assertAlmostEqual(fr, 0.25)
        self.assertAlmostEqual(sr, 0.5)
        # 読点1個 / 実質23文字（文末の「。」はターンの終わりなので「間」に数えない）
        self.assertAlmostEqual(cm, 1 / 23 * 100, places=3)
        # 冒頭リアクションは「たしかに、」のように区切りが要る（「たしかにそうだな」は数えない）
        self.assertAlmostEqual(rr, 0.0)


class InjectFillersTests(unittest.TestCase):
    """32B は文中にフィラーを置かないので、節境界へ機械的に差し込む。"""

    SP = S2Params(filler_ratio=0.5, fillers_per_turn=2, filler_inject=True)

    def test_inserts_after_a_clause_final_comma(self):
        turns = _turns("実は、予算が限られてて、どうやって盛り上げようか悩んでるんです。",
                       "はい。")
        self.assertEqual(inject_fillers(turns, self.SP), 1)
        u = turns[0]["utterance"]
        self.assertIn("限られてて、", u)
        self.assertRegex(u, r"限られてて、(?:えっと|あの|その|なんか|まあ|こう|ええと|うーん)、")

    def test_never_inserts_before_a_particle(self):
        # 読点が句の途中に打たれている形。ここに入れると助詞の直前への割り込みになる
        turns = _turns("その素材、を活用したいです。")
        self.assertEqual(inject_fillers(turns, self.SP), 0)
        self.assertEqual(turns[0]["utterance"], "その素材、を活用したいです。")

    def test_skips_short_turns_and_turns_without_a_boundary(self):
        turns = _turns("はい。", "そうですね。", "承知しました。")
        self.assertEqual(inject_fillers(turns, self.SP), 0)

    def test_stops_once_the_target_ratio_is_met(self):
        turns = _turns("えっと、話は変わるんですけど、来週の件どうします？",
                       "なんか、資料が間に合わなくて、焦ってるんですよね。")
        self.assertEqual(inject_fillers(turns, self.SP), 0)

    def test_respects_the_per_turn_budget(self):
        long = ("予算が限られてて、会場も押さえられてなくて、"
                "どうやって盛り上げようか悩んでるんです。")
        turns = _turns(long)
        one = S2Params(filler_ratio=1.0, fillers_per_turn=1, filler_inject=True)
        self.assertEqual(inject_fillers(turns, one), 1)
        self.assertEqual(inject_fillers(turns, one), 0)   # 上限に達したので増えない


class InjectHeadFillersTests(unittest.TestCase):
    """冒頭フィラーの差し込み（人間は冒頭 19.6% / 文中 13.5% で冒頭のほうが多い）。"""

    SP = S2Params(filler_ratio=0.0, head_filler_ratio=0.5, filler_inject=True)
    HEAD = r"^(?:なんか|あの|まあ|えっと|うーん|その|ええと)、"

    def _long(self, n=6):
        return _turns(*[f"来週の展示会の準備、{i}番目の資料を作っています。" for i in range(n)])

    def test_inserts_at_the_head(self):
        turns = self._long(4)
        self.assertGreater(inject_fillers(turns, self.SP), 0)
        heads = [t["utterance"] for t in turns if re.match(self.HEAD, t["utterance"])]
        self.assertTrue(heads)
        for u in heads:
            self.assertIn("来週の展示会の準備", u)      # 本文は削られていない

    def test_never_touches_the_first_turn(self):
        turns = self._long(4)
        inject_fillers(turns, self.SP)
        self.assertNotRegex(turns[0]["utterance"], self.HEAD)

    def test_leaves_two_turns_between_head_fillers(self):
        # 話者は交互なので、間隔を空けないと同じ話者の発話の頭が軒並みそろう
        turns = self._long(9)
        inject_fillers(turns, S2Params(filler_ratio=0.0, head_filler_ratio=1.0,
                                       filler_inject=True))
        idx = [i for i, t in enumerate(turns) if re.match(self.HEAD, t["utterance"])]
        self.assertTrue(all(b - a >= 3 for a, b in zip(idx, idx[1:])), idx)

    def test_skips_turns_that_already_open_with_something(self):
        turns = _turns("そちらの資料はいつまでに必要ですか？",
                       "なるほど、来週の金曜までにお願いしたいです。",
                       "あはは、それはさすがに急ぎすぎでしょう。",
                       "了解、では月曜までに一次稿を出しますね。",
                       "えっと、こちらでも確認しておきますね。")
        inject_fillers(turns, S2Params(filler_ratio=0.0, head_filler_ratio=1.0,
                                       filler_inject=True))
        self.assertTrue(turns[1]["utterance"].startswith("なるほど、"))
        self.assertTrue(turns[2]["utterance"].startswith("あはは、"))
        self.assertTrue(turns[3]["utterance"].startswith("了解、"))
        self.assertTrue(turns[4]["utterance"].startswith("えっと、"))

    def test_skips_short_replies(self):
        turns = _turns("それでいきましょう。", "はい。", "二つ、はい。", "了解です。")
        self.assertEqual(inject_fillers(turns, self.SP), 0)

    def test_off_by_default(self):
        turns = self._long(4)
        inject_fillers(turns, S2Params(filler_ratio=0.0, filler_inject=True))
        self.assertFalse([t for t in turns if re.match(self.HEAD, t["utterance"])])

    def test_is_deterministic_for_the_same_script(self):
        a, b = self._long(6), self._long(6)
        inject_fillers(a, self.SP)
        inject_fillers(b, self.SP)
        self.assertEqual([t["utterance"] for t in a], [t["utterance"] for t in b])


class CasualFillerTests(unittest.TestCase):
    """タメ口専用の語（なんだろう・やっぱ）は敬語の発話には入れない。"""

    SP = S2Params(filler_ratio=1.0, fillers_per_turn=1, filler_inject=True)
    CASUAL = ("なんだろう", "やっぱ")

    def test_never_puts_casual_fillers_into_polite_speech(self):
        polite = ("来週の展示会の準備が間に合わなくて、"
                  "どうやって形にしようか悩んでいるんです。")
        for seed in range(30):     # 語の選択は台本ごとに決まるので複数台本で確かめる
            turns = _turns(f"{seed}日の打ち合わせの件ですが。", polite)
            inject_fillers(turns, self.SP)
            for w in self.CASUAL:
                self.assertNotIn(w, turns[1]["utterance"])

    def test_casual_fillers_stay_out_of_the_head(self):
        casual = "来週の展示会の準備が間に合わなくて、どうやって形にしようか悩んでるんだ。"
        for seed in range(30):
            turns = _turns(f"{seed}日の打ち合わせの件だけどさ。", casual, casual)
            inject_fillers(turns, S2Params(filler_ratio=1.0, head_filler_ratio=1.0,
                                           fillers_per_turn=1, filler_inject=True))
            for t in turns:
                for w in self.CASUAL:
                    self.assertFalse(t["utterance"].startswith(w), t["utterance"])


class LaughTests(unittest.TestCase):
    """冒頭の笑い（人間コーパスでは発話ターンの 2.4%）。"""

    def test_keeps_laughter_up_to_the_cap(self):
        turns = _turns("この前の失敗の話、聞いた？",
                       "あはは、あれはひどかったね。",
                       "だよね、写真も残ってるらしいよ。",
                       "ふふ、それは見たいかも。",
                       "見せてもらおうよ。",
                       "はは、絶対笑うと思う。")
        removed = cap_laugh_heads(turns, 0.35)      # 6ターン → 上限2
        self.assertEqual(removed, 1)
        self.assertTrue(turns[1]["utterance"].startswith("あはは、"))
        self.assertTrue(turns[3]["utterance"].startswith("ふふ、"))
        self.assertTrue(turns[5]["utterance"].startswith("絶対笑う"))

    def test_drops_back_to_back_laughter(self):
        # 笑い合いが続くと芝居がかる（プロンプトで禁じても実測で3組出た）
        turns = _turns("この前の失敗の話、聞いた？",
                       "あはは、あれはひどかったね。",
                       "ふふ、写真も残ってるらしいよ。",
                       "見せてもらおうよ。")
        self.assertEqual(cap_laugh_heads(turns, 0.5), 1)
        self.assertTrue(turns[1]["utterance"].startswith("あはは、"))
        self.assertEqual(turns[2]["utterance"], "写真も残ってるらしいよ。")

    def test_drops_laughter_without_a_reason_in_context(self):
        """淡々とした説明・励ましへの笑い（ユーザー指摘 2026-07-31）。
        「面白い」は機械で判定できないので、笑いの引き金がある場面だけ残す。"""
        turns = _turns("わかりました。検索ワードを工夫してみます。",
                       "検索ワードを絞ると効率が上がります。",
                       "あはは、検索ワードが鍵になるのは本当です。")
        self.assertEqual(cap_laugh_heads(turns, 0.5), 1)
        self.assertEqual(turns[2]["utterance"], "検索ワードが鍵になるのは本当です。")

    def test_drops_laughter_before_encouragement(self):
        turns = _turns("プロジェクトのことは覚えているので、書けそうです。",
                       "ふふ、頑張ってください。")
        self.assertEqual(cap_laugh_heads(turns, 0.5), 1)
        self.assertEqual(turns[1]["utterance"], "頑張ってください。")

    def test_empathy_needs_a_past_experience(self):
        """「私も」だけでは残さない（「僕も今日中に作るから」＝これからの話）。"""
        turns = _turns("素材リスト、今日は午前中に仕上げるよ。",
                       "あはは、僕も今日中にベースメロ作るから、安心してね。")
        self.assertEqual(cap_laugh_heads(turns, 0.5), 1)
        turns = _turns("追加要件がどっさりきて焦ってるんだよね。",
                       "あはは、私も最初は要件爆発で泣いたことあるよ。")
        self.assertEqual(cap_laugh_heads(turns, 0.5), 0)

    def test_drops_laughter_before_a_plain_acknowledgement(self):
        turns = _turns("テンプレート、早めに欲しいんだよね。",
                       "あはは、了解です。明日までに配布しますね。")
        self.assertEqual(cap_laugh_heads(turns, 0.5), 1)
        self.assertEqual(turns[1]["utterance"], "了解です。明日までに配布しますね。")

    def test_drops_laughter_from_the_opening_turn(self):
        turns = _turns("あはは、今日はよろしくお願いします。", "こちらこそ。")
        self.assertEqual(cap_laugh_heads(turns, 0.5), 1)
        self.assertEqual(turns[0]["utterance"], "今日はよろしくお願いします。")

    def test_leaves_laughter_only_turns_alone(self):
        # 剥がすと空になるので触らない（極短ターンの既存ゲートに任せる）
        turns = _turns("それで転んだんだよね。", "あはは。")
        cap_laugh_heads(turns, 0.5)
        self.assertEqual(turns[1]["utterance"], "あはは。")

    def test_removes_written_laughter_marks(self):
        self.assertEqual(clean_enriched_utterance("それは無理でしょ、笑。"), "それは無理でしょ。")
        self.assertEqual(clean_enriched_utterance("完全に忘れてたw"), "完全に忘れてた。")

    def test_keeps_latin_words_that_end_in_w(self):
        # 「w」の除去がラテン文字の語尾を削らないこと（show → sho）
        for ok in ("次のスライドはoverview。", "そこはinterviewですね。"):
            self.assertEqual(clean_enriched_utterance(ok), ok)

    def test_shortens_overlong_laughter(self):
        self.assertEqual(clean_enriched_utterance("あははははは、ひどいな。"),
                         "あはは、ひどいな。")
        self.assertEqual(clean_enriched_utterance("ふふふふふ、いいね。"), "ふふふ、いいね。")

    def test_prompt_has_the_laughter_section_only_when_asked(self):
        self.assertNotIn("## 笑い", build_system_prompt(S2Params()))
        s = build_system_prompt(S2Params(laugh_ratio=0.03))
        self.assertIn("## 笑い", s)
        self.assertIn("全ターンの約3%", s)
        self.assertNotIn("__", s)


class PromptTests(unittest.TestCase):
    def test_default_prompt_is_unchanged(self):
        from generate_dialogues import SYSTEM_PROMPT
        self.assertEqual(build_system_prompt(None), SYSTEM_PROMPT)

    def test_enriched_prompt_substitutes_every_placeholder(self):
        s = build_system_prompt(S2Params())
        self.assertNotIn("__", s)
        self.assertIn("全ターンの約18%に入れる", s)
        self.assertIn("目安: 全ターンの27%", s)
        self.assertIn('"speaking_rate": 1.15', s)

    def test_enriched_prompt_carries_per_turn_filler_budget(self):
        s = build_system_prompt(S2Params(fillers_per_turn=3, monotone_head_cap=5))
        self.assertNotIn("__", s)
        self.assertIn("実質26文字以上 … フィラー3個", s)
        self.assertIn("合計5回まで", s)

    def test_spoken_language_section_comes_first(self):
        # 「最初から話し言葉であること」を強調するため、話し言葉の節を先頭に置く
        for s in (build_system_prompt(None), build_system_prompt(S2Params())):
            self.assertLess(s.index('徹底した「話し言葉」'),
                            s.index("登場する話者は metadata"))
            self.assertIn("★大前提", s[:400])

    def test_length_table_is_graded_and_matches_the_quota(self):
        sp = S2Params(fillers_per_turn=3, filler_len_step=12, comma_span=8)
        s = build_system_prompt(sp)
        self.assertNotIn("__", s)
        for line, length in (("実質15文字以下 … フィラー0個", 15),
                             ("実質16〜27文字 … フィラー1個", 20),
                             ("実質28〜39文字 … フィラー2個", 30),
                             ("実質40文字以上 … フィラー3個", 50)):
            self.assertIn(line, s)
            # 表の数字と後処理の割り当てがずれないこと
            self.assertEqual(filler_quota(length, sp), int(line[-2]))
        # 読点は長い帯ほど多く要求する
        self.assertIn("実質40文字以上 … フィラー3個。読点5個以上。", s)


class PromptPackTests(unittest.TestCase):
    """Stage2 のプロンプト本文はシナリオごとの prompts/stage2_*.yaml が持つ。

    Stage1 の scenarios/*.yaml と同じ考え方で、本文だけをシナリオごとに分け、
    後処理・ゲート・分量の数値はコード側で共有する。
    """

    REPO = Path(__file__).resolve().parent.parent
    # 中身が同一であるべきなのはこの2本。general_gemma は意図的に別物なので入れない。
    PACKS = ("prompts/stage2_general.yaml", "prompts/stage2_first_meeting.yaml")

    def _dict(self, path=PACKS[0]):
        with open(self.REPO / path) as f:
            return yaml.safe_load(f)

    def test_default_pack_is_the_general_one(self):
        self.assertEqual(DEFAULT_PROMPT_FILE, self.PACKS[0])
        self.assertEqual(GENERAL_S2.name, "general")
        self.assertIs(load_prompt_pack(), GENERAL_S2)

    def test_each_config_points_at_its_own_pack(self):
        """ft と main が別のパックを読む（Stage1 の metadata.scenario_file と対）。"""
        for cfg, name in (("configs/main_v1.yaml", "general_gemma"),
                          ("configs/ft_v1.yaml", "first_meeting")):
            with open(self.REPO / cfg) as f:
                path = yaml.safe_load(f)["dialogue"]["prompt_file"]
            self.assertEqual(load_prompt_pack(path).name, name)

    def test_the_two_packs_are_identical_for_now(self):
        """分けた時点では、どちらのパックからも同じプロンプトが出る。

        片方を意図的に書き換えたらこのテストは落ちる。そのときは（分けた目的
        そのものなので）このテストを消すか、意図した差分の検査に書き換える。
        """
        # ★2026-10-03 初対面パックのオウム返しの例だけを意図して変えた（「カフェ。」と言い切る
        #   Gemma の型を減らすため）。その1行を除けば同じであることを確かめる。
        a, b = (load_prompt_pack(path) for path in self.PACKS)
        old = "「カフェ？」「え、五千円？」\n"
        new = ("「カフェですか？」「え、五千円？」（「カフェ。」「カフェですか。」のように相手の語をくり返して"
               "受けるのは、全部合わせて1つの対話で3回まで。ほかは感想や質問で直接受ける）\n")
        for sp in (None, S2Params(), S2Params(laugh_ratio=0.05)):
            for profile in ("qwen", "gemma"):
                for gates, snm in ((ALL_GATES, 0),
                                   (dict(ALL_GATES, long_utterance=False), 1)):
                    pa = build_system_prompt(sp, gates, snm, profile, a)
                    pb = build_system_prompt(sp, gates, snm, profile, b)
                    pa, pb = (x.replace(new, old) for x in (pa, pb))
                    self.assertEqual(pa, pb)

    def test_every_pack_in_the_repo_loads(self):
        """置いてあるパックはすべて読める（断片の食い違いは読み込み時に落ちる）。"""
        found = sorted((self.REPO / "prompts").glob("stage2_*.yaml"))
        self.assertGreaterEqual(len(found), 3)
        for path in found:
            self.assertEqual(load_prompt_pack(str(path)).name, path.stem[len("stage2_"):])

    def test_gemma_pack_asks_for_the_same_number_of_long_turns_twice(self):
        """長め発話の本数は、システムプロンプトと末尾のノルマで必ず同じ数にする。

        食い違うとモデルはどちらかしか満たさず、long_utterance ゲートの棄却が
        戻る（Gemma でこの棄却が突出したのが general_gemma を分けた理由）。
        """
        pack = load_prompt_pack("prompts/stage2_general_gemma.yaml")
        meta = {"scenario": "テスト", "speakers": [{"name": "ア"}, {"name": "イ"}],
                "speech_style": "敬語", "target_turns": 30}
        sys_n = re.search(r"対話全体で(\d+)ターン以上入れる",
                          build_system_prompt(S2Params(), ALL_GATES, 0, "gemma", pack))
        usr_n = re.search(r"長めの発話を(\d+)ターン以上",
                          build_dialogue_prompt(meta, S2Params(), ALL_GATES, "gemma", pack))
        self.assertIsNotNone(sys_n)
        self.assertIsNotNone(usr_n)
        self.assertEqual(sys_n.group(1), usr_n.group(1))

    def test_gemma_pack_counts_length_the_way_the_gate_does(self):
        """ゲートは「、」「。」「？」を除いて31文字を数える。指示も同じ物差しにする。"""
        s = build_system_prompt(S2Params(), ALL_GATES, 0, "gemma",
                                load_prompt_pack("prompts/stage2_general_gemma.yaml"))
        self.assertIn("句読点を除いて", s)
        self.assertIn("長めの発話（句読点を除いて31〜45文字）", s)

    def test_missing_key_is_rejected(self):
        d = self._dict()
        del d["laugh_section"]
        with self.assertRaises(ValueError):
            build_prompt_pack(d, "テスト")

    def test_fragment_that_is_not_in_the_body_is_rejected(self):
        """本文だけ直して断片を直し忘れると、差し替えが黙って空振りする。

        空振りすると「enriched なのに素のスキーマ」「ゲートを切ったのに禁止指示が
        残る」プロンプトがそのまま LLM に飛ぶので、読み込み時に落とす。
        """
        for key in ("schema_plain", "filler_lines_loose", "no_self_name_lines",
                    "long_utt_quota_sys"):
            d = self._dict()
            d[key] = d[key] + "本文には無い行\n"
            with self.assertRaises(ValueError, msg=key):
                build_prompt_pack(d, "テスト")

    def test_missing_enriched_anchor_is_rejected(self):
        d = self._dict()
        d["system_prompt"] = d["system_prompt"].replace(ENRICHED_ANCHOR, "# おわり")
        with self.assertRaises(ValueError):
            build_prompt_pack(d, "テスト")

    def test_profile_patch_needs_old_and_new(self):
        d = self._dict()
        d["profile_patches"]["gemma"]["system_prompt"] = [{"old": "あ"}]
        with self.assertRaises(ValueError):
            build_prompt_pack(d, "テスト")

    def test_profile_patch_that_lost_its_anchor_fails_loudly(self):
        """差し替え元が本文から消えていたら、そのプロファイルの生成は落とす。"""
        d = self._dict()
        d["profile_patches"]["gemma"]["system_prompt"] = [
            {"old": "本文には無い行", "new": "x"}]
        pack = build_prompt_pack(d, "テスト")
        with self.assertRaises(AssertionError):
            build_system_prompt(None, ALL_GATES, 0, "gemma", pack)
        # 差し替えを持たない qwen 側は通る（片方のプロファイルだけ止める）
        build_system_prompt(None, ALL_GATES, 0, "qwen", pack)

    def test_user_prompt_comes_from_the_pack(self):
        """長め発話のノルマ（ユーザープロンプト側）もパックが持つ。"""
        meta = {"scenario": "テスト", "speakers": [{"name": "ア"}, {"name": "イ"}],
                "speech_style": "敬語", "target_turns": 30}
        d = self._dict()
        d["long_utt_quota_user"] = "長め発話をたくさん入れてください。"
        pack = build_prompt_pack(d, "テスト")
        self.assertIn("長め発話をたくさん入れてください。",
                      build_dialogue_prompt(meta, S2Params(), ALL_GATES, "qwen", pack))


class FillerQuotaTests(unittest.TestCase):
    def test_off_by_default_keeps_the_old_two_tier_rule(self):
        sp = S2Params(fillers_per_turn=3)
        self.assertEqual(filler_quota(10, sp), 1)
        self.assertEqual(filler_quota(25, sp), 1)
        self.assertEqual(filler_quota(26, sp), 3)

    def test_grades_with_length_and_stops_at_the_cap(self):
        sp = S2Params(fillers_per_turn=3, filler_len_step=12)
        self.assertEqual(filler_quota(15, sp), 0)     # 短い発話には入れない
        self.assertEqual(filler_quota(16, sp), 1)
        self.assertEqual(filler_quota(27, sp), 1)
        self.assertEqual(filler_quota(28, sp), 2)
        self.assertEqual(filler_quota(40, sp), 3)
        self.assertEqual(filler_quota(90, sp), 3)     # 上限で頭打ち


class HeadSkipTests(unittest.TestCase):
    def test_no_head_filler_before_a_greeting_or_thanks(self):
        # 実測の誤り:「なんか、ありがとうございます、よろしくお願いします。」
        sp = S2Params(head_filler_ratio=1.0, filler_inject=True, monotone_head_cap=9)
        turns = _turns("今日はここまでで大丈夫そうですね、助かりました。",
                       "ありがとうございます、来週までに資料をまとめておきますね。",
                       "それでは、次回の打ち合わせは金曜の午後でどうでしょうか。",
                       "こちらこそ、いろいろ教えていただいて助かりました。")
        inject_fillers(turns, sp)
        for t in turns[1:]:
            self.assertFalse(re.match(r"(?:なんか|あの|まあ|その|えっと|うーん|ええと)、",
                                      t["utterance"]), t["utterance"])


class MidFillerSlotTests(unittest.TestCase):
    def test_never_injects_right_after_a_greeting(self):
        # 「こんにちは」の「は」を節末と見て差し込む実測の誤り
        sp = S2Params(filler_ratio=1.0, filler_inject=True, fillers_per_turn=3,
                      filler_len_step=12)
        turns = _turns("こんにちは、タカシさん。今日はよろしくお願いします。",
                       "ありがとうございます、来週の資料は明日までにまとめますね。")
        inject_fillers(turns, sp)
        self.assertNotIn("こんにちは、こう、", turns[0]["utterance"])
        for t in turns:
            for greet in ("こんにちは、", "ありがとうございます、"):
                head = t["utterance"].partition(greet)[2]
                self.assertFalse(re.match(r"(?:なんか|あの|まあ|その|こう|えっと)、", head),
                                 t["utterance"])

    def test_never_injects_after_a_discourse_marker(self):
        """「それでは」の「は」を主題の助詞と読んで節末と誤判定していた
        （ユーザー指摘:「それでは、なんか、失礼します。」）。"""
        for text in ("はい、ありがとうございます。それでは、失礼します。",
                     "それでは、来週の水曜までに資料を確認しますね。",
                     "では、関係者に共有しておきます。",
                     "うーん、それでは、今日の議題は以上です。"):
            self.assertEqual(_mid_filler_slots(text), [], text)

    def test_never_injects_before_a_closing_phrase(self):
        """定型の締め文句は一息で言うので、直前で割らない。"""
        text = "そうですね、日程は水曜にして、それでは失礼します。"
        self.assertEqual(_mid_filler_slots(text), [], text)

    def test_keeps_topic_marker_slots(self):
        """主題の「は」で終わる節は差し込み先として残す（過剰に弾かないこと）。"""
        text = "この会社では、社員の半分が在宅勤務をしています。"
        self.assertEqual(len(_mid_filler_slots(text)), 1, text)


class BreakLongRunsTests(unittest.TestCase):
    def _one(self, utterance, run_max=14):
        turns = _turns(utterance)
        n = break_long_runs(turns, run_max)
        return turns[0]["utterance"], n

    def test_splits_a_long_run_at_a_clause_boundary(self):
        out, n = self._one("来週から展示会が始まるので準備を前倒しで進めたいんですよね。")
        self.assertEqual(n, 1)
        self.assertEqual(out, "来週から展示会が始まるので、準備を前倒しで進めたいんですよね。")

    def test_does_not_touch_short_runs(self):
        for ok in ("そうなんだ。", "うん、それでいいと思うよ。",
                   "来週から展示会が始まって、準備が追いつかない。"):
            self.assertEqual(self._one(ok), (ok, 0))

    def test_off_when_run_max_is_zero(self):
        u = "来週から展示会が始まるので準備を前倒しで進めたいんですよね。"
        self.assertEqual(self._one(u, run_max=0), (u, 0))

    def test_never_splits_inside_a_word(self):
        # 実測の失敗例。「のです」の「ので」、助数詞の「あたり」、起点の「から」で割らない
        for u in ("そこは本当に大事なところなのですごく気をつけているんです。",
                  "資料の印刷は1部あたり200円くらいで収まると思いますよ。",
                  "駅の東口から会場まで歩いて15分くらいかかりますね。",
                  "受付は朝9時から夕方5時までずっと開いていますよ。"):
            out, _ = self._one(u)
            for bad in ("ので、す", "あたり、", "東口から、", "9時から、"):
                self.assertNotIn(bad, out)

    def test_leaves_the_utterance_intact_apart_from_commas(self):
        u = "先週の金曜に部長に呼ばれて来月の担当を任されることになったんだよね。"
        out, n = self._one(u)
        self.assertGreaterEqual(n, 1)
        self.assertEqual(out.replace("、", ""), u.replace("、", ""))


class VocativeCopulaTests(unittest.TestCase):
    def test_fixes_address_to_the_other_speaker(self):
        # 呼びかけのつもりで名乗りの形になった実測の誤り
        turns = _turns("こんにちは、タカシ先生です。", "こんにちは、ミキさん。",
                       speakers=["ミキ", "タカシ"])
        self.assertEqual(fix_vocative_copula(turns), 1)
        self.assertEqual(turns[0]["utterance"], "こんにちは、タカシ先生。")

    def test_keeps_self_introduction_and_plain_statements(self):
        turns = _turns("担当はタカシ先生です。", "はい、私がタカシです。",
                       "タカシ先生ですよね。", speakers=["ミキ", "タカシ", "ミキ"])
        self.assertEqual(fix_vocative_copula(turns), 0)
        self.assertEqual(turns[0]["utterance"], "担当はタカシ先生です。")
        self.assertEqual(turns[1]["utterance"], "はい、私がタカシです。")
        self.assertEqual(turns[2]["utterance"], "タカシ先生ですよね。")

    def test_user_prompt_has_quota_only_in_enriched_mode(self):
        meta = {"target_turns": 20, "speech_style": "タメ口"}
        self.assertNotIn("言いよどみの分量", build_dialogue_prompt(meta))
        self.assertIn("言いよどみの分量", build_dialogue_prompt(meta, S2Params()))


class PassthroughTests(unittest.TestCase):
    def test_marks_same_speaker_runs_as_continued(self):
        turns = build_voice_turns(_turns("前半だよね。", "で、後半なんだ。", "そうなんだ。",
                                         speakers=["A", "A", "B"]))
        self.assertNotIn("continued", turns[0])
        self.assertTrue(turns[1]["continued"])
        self.assertNotIn("continued", turns[2])

    def test_absolute_timeline_is_monotonic(self):
        turns = build_voice_turns(_turns("あのさ、聞いていい？", "うん、いいよ。",
                                         "ちょっと相談があるんだよね。"))
        self.assertEqual(turns[0]["start_abs"], 0.0)
        for prev, cur in zip(turns, turns[1:]):
            self.assertGreaterEqual(cur["start_abs"], prev["end_abs"])
            self.assertGreater(cur["duration"], 0)

    def test_convert_record_carries_gender_and_rate(self):
        record = {
            "id": "x1",
            "metadata": {"speakers": [{"name": "ユウ", "age_range": "20代"},
                                      {"name": "ケン", "age_range": "30代"}]},
            "dialogue": {
                "speakers": [{"name": "ユウ", "gender": "female"},
                             {"name": "ケン", "gender": "male"}],
                "speaking_rate": 1.15,
                "turns": _turns("えっと、久しぶりだね。", "ほんとだよね。",
                                speakers=["ユウ", "ケン"]),
            },
        }
        out = convert_record(record)
        self.assertEqual(out["speaking_rate"], 1.15)
        self.assertEqual({p["name"]: p["gender"] for p in out["speaker_profiles"]},
                         {"ユウ": "female", "ケン": "male"})
        self.assertEqual([t["type"] for t in out["voice_dialogue"]["turns"]],
                         ["speech", "speech"])

    def test_convert_record_prefers_metadata_gender(self):
        """metadata の gender（Stage 1 が決めた値）が Stage 2 の推定より強い。"""
        record = {
            "id": "x1b",
            "metadata": {"speakers": [{"name": "ユウ", "gender": "male"},
                                      {"name": "ケン", "gender": "male"}]},
            "dialogue": {
                "speakers": [{"name": "ユウ", "gender": "female"},
                             {"name": "ケン", "gender": "male"}],
                "turns": _turns("そうだね。", "うん。", speakers=["ユウ", "ケン"]),
            },
        }
        out = convert_record(record)
        self.assertEqual({p["name"]: p["gender"] for p in out["speaker_profiles"]},
                         {"ユウ": "male", "ケン": "male"})

    def test_convert_record_without_speakers_falls_back(self):
        """どちらの段も性別を付けていなければ名前から推定し、無理なら None。"""
        record = {
            "id": "x2",
            "metadata": {"speakers": [{"name": "ユウ"}, {"name": "ケン"}]},
            "dialogue": {"turns": _turns("そうだね。", "うん。", speakers=["ユウ", "ケン"])},
        }
        out = convert_record(record)
        self.assertEqual({p["name"]: p["gender"] for p in out["speaker_profiles"]},
                         {"ユウ": None, "ケン": "male"})
        self.assertEqual(out["speaking_rate"], 1.1)


if __name__ == "__main__":
    unittest.main()


# --- 文末終助詞の機械付加 ---------------------------------------------------

class SfpInjectTests(unittest.TestCase):
    """inject_sfp / sfp_candidates。「ほとんどの発話に終助詞」を決定的に満たす工程。"""

    def sp(self, **kw):
        return S2Params(sfp_ratio=kw.pop("sfp_ratio", 0.7), sfp_inject=True, **kw)

    def test_da_endings_are_always_filled(self):
        """「〜だ。」「〜んだ。」は書き言葉。ユーザー指摘の最優先項目。"""
        turns = [{"speaker": "A", "utterance": "リモートでも雰囲気大事だ。"},
                 {"speaker": "B", "utterance": "まだ迷ってるんだ。"}]
        inject_sfp(turns, self.sp())
        for t in turns:
            self.assertFalse(re.search(r"(?:んだ|のだ|だ)。$", t["utterance"]),
                             t["utterance"])

    def test_nda_never_takes_ne(self):
        """「〜んだ/んです」に「ね」を足すと、相手のことを推し量る意味になる。"""
        for stem in ("まだ迷ってるんだ", "作ったことがあるんです"):
            self.assertNotIn("ね", sfp_candidates(stem))
            self.assertIn("よ", sfp_candidates(stem))

    def test_greetings_are_left_alone(self):
        for stem in ("よろしくお願いします", "ありがとうございます", "失礼します",
                     "お疲れ様です", "すみません"):
            self.assertEqual(sfp_candidates(stem), [])

    def test_requests_and_invitations_take_only_ne(self):
        """「〜してくださいよ」は押しつけがましく、「〜くださいよね」は文にならない。"""
        for stem in ("資料を送ってください", "一緒に行きましょう", "ちょっと待って"):
            self.assertEqual(sfp_candidates(stem), ["ね"])

    def test_particle_endings_take_only_ne(self):
        """助詞止めの言いさし。「まずはそこからよ」は語法として壊れる。"""
        for stem in ("まずはそこから", "時間もないし", "パレットは柔軟に"):
            self.assertEqual(sfp_candidates(stem), ["ね"])

    def test_na_only_after_da_ta_i_and_never_polite(self):
        self.assertIn("な", sfp_candidates("予算が足りない"))
        self.assertNotIn("な", sfp_candidates("そう思う"))      # 「思うな」＝禁止
        self.assertNotIn("な", sfp_candidates("確認しました"))  # 「ましたな」

    def test_own_ability_does_not_take_yone(self):
        """「調整できますよね」は相手への同意要求になってしまう。"""
        self.assertNotIn("よね", sfp_candidates("その点はグラデーションで調整できます"))

    def test_questions_are_never_touched(self):
        turns = [{"speaker": "A", "utterance": "それ、どこにありますか？"}]
        self.assertEqual(inject_sfp(turns, self.sp()), 0)
        self.assertEqual(turns[0]["utterance"], "それ、どこにありますか？")

    def test_turns_that_already_have_one_are_untouched(self):
        turns = [{"speaker": "A", "utterance": "そうですよね。"}]
        self.assertEqual(inject_sfp(turns, self.sp()), 0)

    def test_avoids_repeating_the_neighbours_particle(self):
        turns = [{"speaker": "A", "utterance": "今日は寒いね。"},
                 {"speaker": "B", "utterance": "本当に寒い。"}]
        inject_sfp(turns, self.sp(sfp_ratio=1.0))
        self.assertFalse(turns[1]["utterance"].endswith("ねね。"))
        self.assertNotEqual(turns[1]["utterance"], "本当に寒いね。")

    def test_reaches_roughly_the_target_ratio(self):
        # 「です」述語（評価・確認）は付加の対象。敬語の動詞述語は対象外なので、
        # 目安に届くかはこちらの形で見る。
        turns = [{"speaker": "A" if i % 2 else "B",
                  "utterance": f"{i}番目の案はかなり現実的です。"}
                 for i in range(20)]
        inject_sfp(turns, self.sp(sfp_ratio=0.7))
        got = sum(1 for t in turns if _SFP_TAIL.search(t["utterance"]))
        self.assertGreaterEqual(got, 12)
        self.assertLessEqual(got, 20)

    def test_polite_verb_predicates_are_left_alone(self):
        """敬語の動詞述語には足さない（「ね」ばかりの対話になるため。ユーザー指摘）。"""
        turns = [{"speaker": "A" if i % 2 else "B",
                  "utterance": f"それは{i}番目の案で問題ないと思います。"}
                 for i in range(20)]
        self.assertEqual(inject_sfp(turns, self.sp(sfp_ratio=0.7)), 0)

    def test_appending_never_produces_a_double_period(self):
        turns = [{"speaker": "A", "utterance": "それは大事だ。"}]
        inject_sfp(turns, self.sp())
        self.assertFalse(turns[0]["utterance"].endswith("。。"))
        self.assertTrue(turns[0]["utterance"].endswith("。"))

    def test_prompt_demands_a_particle_on_most_turns(self):
        p = build_system_prompt(S2Params(sfp_ratio=0.7))
        self.assertIn("ほとんど全部", p)
        self.assertIn("「だ」で言い切らない", p)

    def test_dangling_conditionals_get_nothing(self):
        """「例えば。」「どうしたら。」に足すと「例えばよね。」のように語法が壊れる。"""
        for stem in ("有効です、例えば", "具体的にどうしたら", "具体的にどうすれば"):
            self.assertEqual(sfp_candidates(stem), [])

    def test_conjunctive_endings_take_only_ne(self):
        for stem in ("予算がそこそこかかるけど", "暗闇なので", "資金がたまるかもしれないし"):
            self.assertEqual(sfp_candidates(stem), ["ね"])

    def test_own_ability_takes_only_yo(self):
        """「ご提案できますね」は相手のことのように聞こえ、「できますよね」は同意要求。"""
        self.assertEqual(sfp_candidates("時間管理のコツをご提案できます"), ["よ"])

    def test_casual_na_adjective_stems_get_da_inserted(self):
        """「作れそうよね」は女性語的で不自然。「作れそうだよね」にする。"""
        cands = sfp_candidates("学生団体と協力すればすぐに作れそう")
        self.assertTrue(all(c.startswith("だ") for c in cands), cands)
        self.assertEqual(len(cands), len(set(cands)))
        self.assertIn("だよね", cands)

    def test_polite_na_adjective_stems_are_not_touched(self):
        self.assertIn("ね", sfp_candidates("面白そうです"))

    def test_own_wish_does_not_take_yone(self):
        self.assertNotIn("よね", sfp_candidates("教えていただきたいです"))


class TruncatedCopulaTests(unittest.TestCase):
    """助詞・条件形の直後に「です」が来る出力は文の途中が抜けている（実測）。
    後処理では直せないので、ゲートで検知して作り直させる。"""

    def test_detects_truncated_utterances(self):
        for bad in ("先生、卒業式の準備がやっとこさで、あの、進路に不安がですよね。",
                    "検索ですか、具体的なキーワードはですか？",
                    "週1回の振り返り、具体的にどうすればですよね。",
                    "その資料をですね。"):
            self.assertRegex(bad, _TRUNCATED_COPULA_RE, bad)

    def test_does_not_flag_valid_sentences(self):
        for ok in ("それって、あの、毎日のタスクを3つに絞ることですか？",
                   "えっと、要点は箇条書き、章ごとに1枚、ということですね。",
                   "さすがです。よく気づきましたね。",
                   "資料はメールでお送りしますね。",
                   "覚えやすい言葉です。",
                   "予算はこれで大丈夫ですか？",
                   "私の兄です。介護施設で働いています。",
                   "ご都合はいかがですか？",
                   "C園はいかがですか？"):
            self.assertNotRegex(ok, _TRUNCATED_COPULA_RE, ok)

    def test_prompt_examples_are_complete_sentences(self):
        """良い例を「〜」で省略して書くと、モデルが途中の抜けた文を真似する（実測）。"""
        p = build_system_prompt(S2Params(sfp_ratio=0.7))
        for line in p.splitlines():
            if "良い例" in line:
                self.assertNotIn("「〜", line, line)

    def test_detects_verb_plus_desu(self):
        """動詞終止形＋「です」の破格（LLM点検で検出。列挙の締めで出やすい）。"""
        for bad in ("えっと、まずは受付で名前を書かせる、次に手渡しで渡す、"
                    "最後に一言添えるですよね。",
                    "なんか、メインは緑と黄色がバランス良く見えるですよ。",
                    "①具体的なKPIを示す、②導入前の課題を明確に、"
                    "③導入後の効果を数値で示すです。"):
            self.assertRegex(bad, _VERB_DESU_RE, bad)

    def test_verb_plus_desu_does_not_flag_valid(self):
        """既存192,876ターンで誤検知0件だった形を回帰で守る。"""
        for ok in ("失礼いたします。またよろしくです。",
                   "再査定っていつですか？",
                   "業界を3つですか、えっと、マーケティング、デザインですよ。",
                   "受付後すぐですか、タイミングが分かりますね。",
                   "そういうものです。気にしなくて大丈夫ですよ。",
                   "会議は明日のはずです。",
                   "来週から始めるつもりです。",
                   "雨が降るようです。",
                   "そこが大事なのです。",
                   "実は緊張しているんです。"):
            self.assertNotRegex(ok, _VERB_DESU_RE, ok)


class CasualLeakTests(unittest.TestCase):
    """敬語指定の対話に混ざるタメ口（ユーザー指摘 2026-08-01:
    大学職員が学生に「まあ、そうなんだ。大変ですね」）。"""

    def test_replaces_casual_reaction_heads(self):
        turns = _turns("まあ、そうなんだ。大変ですね、どうされましたか？",
                       "なるほど、そうなんだね。関心領域を教えてください。",
                       "そう、そうなんだ。では行動計画を作りましょうか？",
                       "そっか。では進めましょう。",
                       "いいね、その案でいきましょう。")
        self.assertEqual(fix_casual_reactions(turns), 5)
        self.assertEqual([t["utterance"] for t in turns], [
            "まあ、そうなんですね。大変ですね、どうされましたか？",
            "なるほど、そうなんですね。関心領域を教えてください。",
            "そう、そうなんですね。では行動計画を作りましょうか？",
            "なるほど。では進めましょう。",
            "いいですね、その案でいきましょう。"])

    def test_strips_trailing_echo(self):
        """文末に付く相槌の反響。内容がないので落とす（実測、ゲートは検知するが
        最終試行では通してしまうので機械でも直す）。"""
        turns = _turns("なるほど。その手法なら人手不足の緩和に繋がりますね、そうだよね。",
                       "そのバランスは理想です。再就職支援も必要ですね、そうだな。")
        self.assertEqual(fix_casual_reactions(turns), 2)
        self.assertEqual([t["utterance"] for t in turns], [
            "なるほど。その手法なら人手不足の緩和に繋がりますね。",
            "そのバランスは理想です。再就職支援も必要ですね。"])

    def test_leaves_polite_reactions_alone(self):
        turns = _turns("なるほど。承知しました。", "そうなんですね。大変ですね。",
                       "たしかに、その通りです。")
        self.assertEqual(fix_casual_reactions(turns), 0)

    def test_detects_sentence_final_plain_forms(self):
        for bad in ("時間帯でブロックすると効果的だよ。",
                    "午前は学習、午後は準備といったように決めるんだよね。",
                    "まずは関心がある分野を絞るんだ。データ分析とAIです。",
                    "その点は、週に2日という条件で検索すると見つかりやすいんだな。",
                    "その資料はもう見てる。"):
            self.assertRegex(bad, _CASUAL_LEAK_RE, bad)

    def test_does_not_flag_quotes_or_subordinate_clauses(self):
        """引用・従属節・連体修飾の「だ」「てる」は文末ではないので誤検知しない。"""
        for ok in ("これは重要だと言われました。",
                   "時間があると思いますので、進めます。",
                   "担当する資料は明日までにまとめます。",
                   "まだ。もう少しお待ちください。",
                   "そうなんですね。大変ですね。",
                   "対応が必要だという認識です。",
                   "資料を確認していると連絡がありました。"):
            self.assertNotRegex(ok, _CASUAL_LEAK_RE, ok)

    def test_repairs_sentence_final_plain_forms(self):
        """文末のタメ口は置換で直す（ユーザー指摘 2026-08-01:「実際に利用者数が
        過去5年で30パーセントも増えているんだよ。」が敬語指定の対話に出た）。
        ゲートは検知するが最終試行では通してしまうため、機械でも直す。"""
        turns = _turns("実際に利用者数が過去5年で30パーセントも増えているんだよ。",
                       "長時間かかるのか、時間配分が大事だね。",
                       "まずは保険の見直しポイントが大事だよね。",
                       "記録は紙でもできるんだっけ？",
                       "どこから手を付ければいいんだろう？",
                       "最後に見積もり書をPDFで作成し、学務課に提出することだ。",
                       "その資料はもう見てる。",
                       "名刺を渡すのが遅れて、相手に待たされちゃった。")
        self.assertEqual(fix_casual_reactions(turns), 8)
        self.assertEqual([t["utterance"] for t in turns], [
            "実際に利用者数が過去5年で30パーセントも増えているんですよ。",
            "長時間かかるのか、時間配分が大事ですね。",
            "まずは保険の見直しポイントが大事ですよね。",
            "記録は紙でもできるんでしたっけ？",
            "どこから手を付ければいいんでしょう？",
            "最後に見積もり書をPDFで作成し、学務課に提出することです。",
            "その資料はもう見ています。",
            "名刺を渡すのが遅れて、相手に待たされちゃいました。"])

    def test_keeps_casual_forms_inside_quotes(self):
        """括弧なしの引用の中身はタメ口のままが正しい（146,116ターンの試走で
        53件中3件がこれだった）。文の途中でも直す形は直す。"""
        turns = _turns("相手の都合を確認しながら次はいつが空いてる？と聞くとスムーズです。",
                       "宿題をやってくれないと私も不安になるんだ。と伝えるのですね。",
                       "例えばこのページ、何度も読み返すんだ。はどうでしょう。")
        self.assertEqual(fix_casual_reactions(turns), 0)
        turns = _turns("それは大変だね。まずは書き出してみましょう。")
        self.assertEqual(fix_casual_reactions(turns), 1)
        self.assertEqual(turns[0]["utterance"],
                         "それは大変ですね。まずは書き出してみましょう。")

    def test_register_is_handled_by_post_processing_not_the_prompt(self):
        """プロンプトに指示を足すと他のゲートの棄却率が上がる（実測）。口調の
        使い分けは機械置換に任せ、プロンプトでは指定しない。"""
        p = build_system_prompt(S2Params(reaction_ratio=0.3))
        self.assertIn("後処理が丁寧形に直す", p)
        turns = _turns("そうなんだ。では進めましょう。")
        self.assertEqual(fix_casual_reactions(turns), 1)

    def test_polite_iisashi_is_register_independent(self):
        """混合・タメ口指定の対話でも、その発話が「です／ます」なら敬語の言いさし
        （ユーザー指摘「気になるのですがね」と同じ形が混合指定に残った）。"""
        for stem in ("学生割引、聞いたことないんですが",
                     "働き方のバランスが気になるのですが",
                     "資料は用意しましたけど", "時間が限られていますので"):
            for polite in (True, False):
                self.assertEqual(sfp_candidates(stem, polite=polite), [],
                                 f"{stem} polite={polite}")

    def test_repairs_model_written_adjective_yone(self):
        """付加を止めてもモデルが自力で書く（実測「数字で見るとイメージしやすいですよね」）。
        「ね」に替えれば、自己評価の場合も共有事実の場合も自然になる。"""
        turns = _turns("シミュレーションって、数字で見るとイメージしやすいですよね。",
                       "まあ、10年分？それ、結構大きいですよね。",
                       "あの件、先週でしたよね。",
                       "部屋はとてもきれいですよね。")
        self.assertEqual(fix_yone_misuse(turns), 2)
        self.assertEqual([t["utterance"] for t in turns[:2]],
                         ["シミュレーションって、数字で見るとイメージしやすいですね。",
                          "まあ、10年分？それ、結構大きいですね。"])

    def test_laugh_trigger_ignores_budget_overrun(self):
        """「オーバー」は「予算オーバー」に当たる誤検知だった（実測）。"""
        from generate_dialogues import _LAUGH_TRIGGER
        self.assertNotRegex("予算オーバーです。", _LAUGH_TRIGGER)
        self.assertRegex("ちょっと大げさだけど。", _LAUGH_TRIGGER)

    def test_polite_kara_still_takes_ne(self):
        """「〜ですからね」は理由の提示で自然なので残す。"""
        self.assertIn("ね", sfp_candidates("受付後は法要が始まりますから", polite=True))

    def test_strips_polite_iisashi_ne_in_casual_dialogue(self):
        turns = _turns("学割って使えるのかな？",
                       "学生割引、聞いたことないんですがね。")
        self.assertEqual(fix_ne_misuse(turns, polite=False), 1)
        self.assertEqual(turns[1]["utterance"], "学生割引、聞いたことないんですが。")


class NeMisuseTests(unittest.TestCase):
    """「ね」は相手と共有していることに向ける語。自分の報告に付くと確認要求になる
    （ユーザー指摘 2026-07-31）。"""

    def test_strips_ne_from_own_past_report(self):
        turns = _turns("ロールプレイングは効果的だったんですね。",
                       "はい、受講者の定着率が15パーセント向上しましたね。",
                       "手助けができれば嬉しいですよね。",
                       "先輩の話、すごく参考になりましたね。")
        self.assertEqual(fix_ne_misuse(turns), 2)
        self.assertEqual([t["utterance"] for t in turns[1::2]],
                         ["はい、受講者の定着率が15パーセント向上しました。",
                          "先輩の話、すごく参考になりました。"])

    def test_strips_ne_from_own_feeling(self):
        turns = _turns("来週のミーティングで提案する準備をします。",
                       "あはは、期待していますね。")
        self.assertEqual(fix_ne_misuse(turns), 1)
        self.assertEqual(turns[1]["utterance"], "あはは、期待しています。")

    def test_keeps_sympathy_and_understanding(self):
        """「大変でしたね」（共感）「〜んですね」（相手の話を受ける）は自然。"""
        turns = _turns("先週は徹夜が続いてしまって。", "それは大変でしたね。",
                       "配管が古いと水漏れするんだよ。", "なるほど、それは知らなかったんですね。",
                       "資料は明日までに送ります。", "では明日確認しますね。")
        self.assertEqual(fix_ne_misuse(turns), 0)

    def test_polite_dialogue_does_not_take_ne_on_te_form(self):
        """敬語の対話でテ形の言いさしに「ね」を足すと馴れ馴れしくなる（実測、先生への
        発話で「進路について、不安が拭えなくてね。」）。タメ口の「連絡してね」は自然。"""
        for stem in ("実は、卒業式の準備で忙しく、進路について、不安が拭えなくて",
                     "法事の準備で少し戸惑っておりまして"):
            self.assertEqual(sfp_candidates(stem, polite=True), [], stem)
            self.assertEqual(sfp_candidates(stem), ["ね"], stem)   # タメ口なら従来どおり
        # 敬語でも「〜ですから」（接続助詞）は自然なので残す
        self.assertEqual(sfp_candidates("応募期限の管理、忘れがちですから",
                                        polite=True), ["ね"])

    def test_fix_strips_te_form_ne_in_polite_dialogue(self):
        turns = _turns("こんにちは、どうぞお座りください。",
                       "実は、卒業式の準備で忙しく、不安が拭えなくてね。")
        self.assertEqual(fix_ne_misuse(turns, polite=True), 1)
        self.assertEqual(turns[1]["utterance"],
                         "実は、卒業式の準備で忙しく、不安が拭えなくて。")
        casual = _turns("リストできた？", "うん、あとで送るから安心してね。")
        self.assertEqual(fix_ne_misuse(casual), 0)

    def test_is_polite_style(self):
        self.assertTrue(_is_polite_style({"speech_style": "敬語"}))
        for style in ("敬語とタメ口の混合", "タメ口", "カジュアル", ""):
            self.assertFalse(_is_polite_style({"speech_style": style}), style)

    def test_question_ending_never_takes_yo(self):
        """「〜ですかね」は自然だが「〜ですかよ」は壊れる。"""
        self.assertEqual(sfp_candidates("効果測定の指標は利用者満足度ですか"), ["ね"])


class SfpRunTests(unittest.TestCase):
    """同じ終助詞の連続（実測で「ね」8連続）。プロンプトの「3ターン以上続けない」を
    機械で担保する。"""

    def test_breaks_a_long_run(self):
        turns = _turns("毎朝10時に進捗報告を送るね。", "毎朝10時がベストだね。",
                       "来週のミーティングで最終確認をするね。",
                       "水曜日の午後2時に設定しようね。",
                       "資料の校正は私が手伝うから大丈夫だね。")
        self.assertGreater(diversify_sfp_runs(turns), 0)
        tails = [_SFP_TAIL.search(t["utterance"]).group("sfp")
                 if _SFP_TAIL.search(t["utterance"]) else "" for t in turns]
        for a, b, c in zip(tails, tails[1:], tails[2:]):
            self.assertFalse(a == b == c != "", tails)

    def test_leaves_short_runs_alone(self):
        turns = _turns("そうですね。", "本当にそうですね。", "では準備を進めます。")
        before = [t["utterance"] for t in turns]
        self.assertEqual(diversify_sfp_runs(turns), 0)
        self.assertEqual([t["utterance"] for t in turns], before)

    def test_never_creates_a_da_ending_or_blunt_request(self):
        """「〜だね」を落として「〜だ。」にしない。「〜くださいね」も残す。"""
        turns = _turns("これは大事だね。", "そこは基本だね。", "本当に基本だね。",
                       "では確認してくださいね。", "あわせて共有してくださいね。",
                       "念のため保存してくださいね。")
        diversify_sfp_runs(turns)
        for t in turns:
            self.assertFalse(re.search(r"(?:んだ|のだ|だ)。$", t["utterance"]),
                             t["utterance"])
            if "ください" in t["utterance"]:
                self.assertTrue(t["utterance"].endswith(("くださいね。", "ください。")),
                                t["utterance"])


class YoMisuseTests(unittest.TestCase):
    """「よ」＝相手が知らないこと・相手のためになることを教える語。敬語の動詞述語や
    自分の気持ち・状況に付くと押しつけがましい（ユーザー指摘 2026-07-31）。"""

    def test_polite_verb_does_not_take_yo(self):
        for stem in ("ありがとうございます。頑張ります",
                     "次に志望動機を具体的にまとめます",
                     "実務感覚を早期に育てました"):
            self.assertEqual(sfp_candidates(stem), [], stem)

    def test_advice_and_information_keep_yo(self):
        """助言・情報の「です」述語と可能形はそのまま（「効果的ですよ」）。"""
        for stem in ("週間スケジュールを紙に書き出すと効果的です",
                     "時間管理アプリを活用すると便利です",
                     "近隣の同条件物件は平均で3800万円前後です"):
            self.assertIn("よ", sfp_candidates(stem), stem)
        self.assertEqual(sfp_candidates("時間管理のコツをご提案できます"), ["よ"])

    def test_casual_and_explanatory_keep_yo(self):
        self.assertIn("よ", sfp_candidates("できる部分はサポートする"))
        self.assertIn("よ", sfp_candidates("昔マイボトル持参割引をやったんです"))

    def test_own_state_and_feeling_get_nothing(self):
        for stem in ("卒業式の準備でバタバタしています", "査定が甘いと感じてます",
                     "プライベートの時間が減るのが心配です", "納期が不安です"):
            self.assertEqual(sfp_candidates(stem), [], stem)

    def test_fix_strips_misused_yo(self):
        turns = _turns("それでは、あの、頑張ってくださいね。",
                       "ありがとうございます。頑張りますよ。",
                       "どうぞお座りくださいね。",
                       "なんか、実は、卒業式の準備でバタバタしていますよ。",
                       "デジタルマーケティングが向いているかもしれませんね。",
                       "えっと、プライベートの時間が減るのが心配ですよ。")
        self.assertEqual(fix_yo_misuse(turns), 3)
        self.assertEqual([t["utterance"] for t in turns[1::2]],
                         ["ありがとうございます。頑張ります。",
                          "なんか、実は、卒業式の準備でバタバタしています。",
                          "えっと、プライベートの時間が減るのが心配です。"])

    def test_fix_leaves_natural_yo(self):
        turns = _turns("具体的には、どうすればいいですか？",
                       "うーん、まずは紙に書き出すと効果的ですよ。",
                       "画面共有、ありがとう。",
                       "まずデザインのコード化、俺がやってみるよ。",
                       "高額な装飾はやりたくないんですよね。",
                       "昔、マイボトル持参割引をやったんですよ。",
                       "その点はどうにもなりませんか？",
                       "その、グラデーションで調整できますよ。")
        self.assertEqual(fix_yo_misuse(turns), 0)


class YoneMisuseTests(unittest.TestCase):
    """「よね」＝相手も知っている事柄への同意要求。自分の行動宣言・経験の報告には付かない
    （ユーザー指摘 2026-07-31:「試してみますよね」「やっていましたよね」）。"""

    def test_own_potential_prospect_takes_nothing(self):
        """可能形＋様態は自分がやれる見込みの話（ユーザー指摘「書けそうですよ」）。
        「よ」は押しつけ、「ね」は相手のことを指してしまうので何も付けない。"""
        for stem in ("プロジェクトのことは覚えているので、なんか、書けそうです",
                     "この内容なら作れそうです", "来週なら対応できそうです"):
            self.assertEqual(sfp_candidates(stem), [], stem)

    def test_adjective_prospect_keeps_ne(self):
        """形容詞語幹＋そうは相手の話への評価なので「ね」を残す。"""
        for stem in ("面白そうです", "それは難しそうです", "その案は良さそうです",
                     "はい、そうです"):
            self.assertIn("ね", sfp_candidates(stem), stem)

    def test_want_never_takes_yo(self):
        """自分の願望に「よ」は押しつけになる（ユーザー指摘「リストアップしたいですよ」）。
        「ね」「な」は同じ場面で共感として成立するので残す。"""
        for stem in ("はい、リストアップしたいです", "早く休みたいです",
                     "もう少し詳しく聞きたいんです"):
            self.assertEqual(sfp_candidates(stem), ["ね"], stem)
        self.assertEqual(sfp_candidates("資料をまとめたい"), ["ね", "な"])

    def test_own_ongoing_and_gratitude_take_nothing(self):
        """LLM点検で残った穴。どちらも自分の状況・気持ちなので同意を求める相手がいない
        （実測「省エネ設備を検討中ですよね」「協力させていただき感謝ですよね」）。"""
        for stem in ("キッチンとバスの省エネ設備を検討中です",
                     "見積もりは現在確認中です",
                     "こちらこそ。協力させていただき感謝です",
                     "ご一緒できて光栄です",
                     "ご確認いただければ幸いです"):
            for polite in (True, False):
                self.assertEqual(sfp_candidates(stem, polite=polite), [],
                                 f"{stem} polite={polite}")

    def test_own_action_masu_takes_nothing(self):
        """敬語の動詞述語は付加の対象外（「よ」は押しつけ、「ね」は数が出すぎる）。"""
        for stem in ("応募書類の作成も今週中に行います",
                     "最後は盛り上げ曲で締めくくります",
                     "Googleカレンダーの活用をおすすめします",
                     "インターンを2件ほどピックアップしました"):
            self.assertEqual(sfp_candidates(stem), [], stem)

    def test_masu_ne_whitelist(self):
        """「ます」述語でも予告と帰結には「ね」を戻す（付加可能なターンが44%まで
        落ちて下限ゲートが構造的に届かなくなったため）。"""
        for stem in ("了解、また連絡します", "資料はPDFで共有します",
                     "第2週ですか、確認します", "横3列ですか、試してみます",
                     "合計で25000円ですか、予算内に収まります",
                     "実績を数字で示すとさらに説得力が増します"):
            self.assertEqual(sfp_candidates(stem), ["ね"], stem)

    def test_masu_ne_whitelist_excludes_self_report(self):
        """自分の経験の報告・自分の利得には付けない（「助かりましたね」の指摘）。"""
        for stem in ("去年の卒業式で講演を依頼したことがあります",
                     "チェックリスト、助かります",
                     "先輩の話、すごく参考になりました"):
            self.assertEqual(sfp_candidates(stem), [], stem)

    def test_stative_masu_keeps_model_written_yone(self):
        """「〜ます」に付加はしないが、モデルが書いた「わかりますよね」は残す
        （相手の理解を確かめる用法。逆に「興味がありますよね」は落とす）。"""
        keep = _turns("そこは難しいところです。", "その気持ちはわかりますよね。")
        self.assertEqual(fix_yone_misuse(keep), 0)
        drop = _turns("星野広告はいかがですか？", "星野広告ですか、興味がありますよね。")
        self.assertEqual(fix_yone_misuse(drop), 1)
        self.assertTrue(drop[1]["utterance"].endswith("興味がありますね。"),
                        drop[1]["utterance"])

    def test_own_experience_and_state_do_not_take_yone(self):
        for stem in ("インターンでSNS運用をやっていました", "全てメモしました",
                     "どれがいいか迷っています", "最終確認したいと思います"):
            self.assertNotIn("よね", sfp_candidates(stem), stem)

    def test_shared_past_still_takes_yone(self):
        """「あの件、先週でしたよね」のような共有事実の確認は残す。"""
        self.assertIn("よね", sfp_candidates("あの件は先週でした"))
        self.assertIn("よね", sfp_candidates("あのときは本当に忙しかった"))

    def test_answer_turn_never_takes_yone(self):
        turns = _turns("大学のゼミでのプロジェクト経験はありますか？",
                       "インターンでSNS運用を担当しています")
        inject_sfp(turns, S2Params(sfp_ratio=1.0, sfp_inject=True))
        self.assertNotIn("よね", turns[1]["utterance"])

    def test_past_answer_gets_no_particle_at_all(self):
        """「やっていましたね」も相手に確認を求める形。答えの過去形には足さない。"""
        turns = _turns("大学のゼミでのプロジェクト経験はありますか？",
                       "インターンでSNS運用をやっていました。")
        self.assertEqual(inject_sfp(turns, S2Params(sfp_ratio=1.0, sfp_inject=True)), 0)
        self.assertEqual(turns[1]["utterance"], "インターンでSNS運用をやっていました。")

    def test_fix_replaces_volition_yone_with_ne(self):
        turns = _turns("3ブロックに分けるとバランスが取れますよ。",
                       "なるほど。3ブロックですか、ちょっと試してみますよね。")
        self.assertEqual(fix_yone_misuse(turns), 1)
        self.assertEqual(turns[1]["utterance"],
                         "なるほど。3ブロックですか、ちょっと試してみますね。")

    def test_fix_strips_yone_from_own_experience(self):
        turns = _turns("大学のゼミでのプロジェクト経験はありますか？",
                       "あの、インターンでSNS運用をやっていましたよね。")
        self.assertEqual(fix_yone_misuse(turns), 1)
        self.assertEqual(turns[1]["utterance"], "あの、インターンでSNS運用をやっていました。")

    def test_fix_strips_yone_from_answers(self):
        turns = _turns("どんな配色が子どもに刺さるんですか？",
                       "あの、黄色と水色は目を引くと言われていますよね。")
        self.assertEqual(fix_yone_misuse(turns), 1)
        self.assertTrue(turns[1]["utterance"].endswith("言われています。"))

    def test_formulaic_acceptance_takes_no_particle(self):
        """「了解ですね」「了解ですよね」は受諾の型が崩れる（実測「了解ですよね」）。"""
        self.assertEqual(sfp_candidates("リスト作成、要件まとめ、了解です"), [])
        turns = _turns("では要件をまとめておきますね。", "了解ですよね。")
        self.assertEqual(fix_yone_misuse(turns), 1)
        self.assertEqual(turns[1]["utterance"], "了解です。")

    def test_hedge_kamo_takes_only_ne(self):
        """「向いているかもですよね」はもたつく。「かもですね」にする。"""
        self.assertEqual(sfp_candidates("マーケ系が向いているかもです"), ["ね"])
        turns = _turns("どの業界が向いていますか？", "マーケ系が向いているかもですよね。")
        self.assertEqual(fix_yone_misuse(turns), 1)
        self.assertEqual(turns[1]["utterance"], "マーケ系が向いているかもですね。")

    def test_fix_strips_yone_from_greetings(self):
        turns = _turns("では金曜に伺います。", "はい、よろしくお願いしますよね。")
        self.assertEqual(fix_yone_misuse(turns), 1)
        self.assertEqual(turns[1]["utterance"], "はい、よろしくお願いします。")

    def test_vocative_endings_take_no_particle(self):
        """呼びかけの後ろに終助詞が付くと呼びかけが名詞句に化ける（実測）。"""
        self.assertEqual(sfp_candidates("よろしくお願いします、ミキさん"), [])
        self.assertEqual(sfp_candidates("ぜひお願いします、タカシ先生"), [])
        turns = _turns("では金曜に伺いますね。", "了解です、タカシさんよね。",
                       "ありがとう、ミキさんよ。",
                       speakers=["タカシ", "ミキ", "タカシ"])
        self.assertEqual(fix_vocative_sfp(turns), 2)
        self.assertEqual([t["utterance"] for t in turns[1:]],
                         ["了解です、タカシさん。", "ありがとう、ミキさん。"])

    def test_bare_name_vocative_uses_speaker_names(self):
        """敬称なしの呼びかけは話者名を見ないと分からない（「ありがとう、ミキよ。」）。"""
        turns = _turns("資料、送っておいたよ。", "ありがとう、ミキよね。",
                       speakers=["ミキ", "タカシ"])
        self.assertEqual(fix_vocative_sfp(turns), 1)
        self.assertEqual(turns[1]["utterance"], "ありがとう、ミキ。")

    def test_injection_skips_vocative_endings(self):
        turns = _turns("では金曜に伺いますね。", "よろしくお願いします、ミキさん。",
                       speakers=["タカシ", "ミキ"])
        self.assertEqual(inject_sfp(turns, S2Params(sfp_ratio=1.0, sfp_inject=True)), 0)

    def test_own_interest_and_benefit_do_not_take_yone(self):
        """「興味がありますよね」「参考になりますよね」は同意を求める相手がいない。
        付加側は敬語の動詞述語ごと対象外、修復側は「ね」に替える。"""
        for stem in ("星野広告ですか、興味があります", "具体例がとても参考になります"):
            self.assertEqual(sfp_candidates(stem), [], stem)
        turns = _turns("その例をご覧ください。", "具体例がとても参考になりますよね。")
        self.assertEqual(fix_yone_misuse(turns), 1)
        self.assertTrue(turns[1]["utterance"].endswith("参考になりますね。"),
                        turns[1]["utterance"])

    def test_volitional_takes_only_ne(self):
        """意志形に「だ」を挟むと壊れる（実測「すぐに依頼してみようだな。」）。"""
        for stem in ("すぐに依頼してみよう", "その、助け合いで乗り越えよう"):
            self.assertEqual(sfp_candidates(stem), ["ね"], stem)

    def test_formal_nouns_get_da_inserted(self):
        """「はずよ」「つもりよね」は女性語的。「はずだよ」「つもりだよね」にする。"""
        for stem in ("正直に話せば理解してくれるはず", "基本スペックを決めるつもり"):
            self.assertTrue(all(c.startswith("だ") for c in sfp_candidates(stem)), stem)

    def test_existing_particles_are_not_stacked(self):
        """「かしら」は終助詞。上に「よね」を重ねると壊れる（実測）。"""
        turns = _turns("キッチンは100万円前後ですね。", "予算オーバーかしらよね。")
        self.assertEqual(fix_yone_misuse(turns), 1)
        self.assertEqual(turns[1]["utterance"], "予算オーバーかしら。")
        self.assertEqual(inject_sfp(_turns("予算オーバーかしら。", "そうですね。"),
                                    S2Params(sfp_ratio=1.0, sfp_inject=True)), 0)

    def test_imperative_and_mimetic_endings_get_nothing(self):
        """「頑張れね」は語法が壊れ、「ぐちゃぐちゃよね」は「だ」が要る（実測）。"""
        for stem in ("後で後悔しない分だけ頑張れ", "恋と学業のバランスで頭がぐちゃぐちゃ"):
            self.assertEqual(sfp_candidates(stem), [], stem)

    def test_polite_negative_does_not_take_yone(self):
        """「方法が分かりませんよね」は自分の状態なので同意要求にならない（実測）。

        肯定形と違い否定形は「自分が分からない／相手も分からない」の区別が付かないので、
        「〜ません」はまとめて「ね」に寄せる。
        """
        for stem in ("実は、あの、方法が分かりません", "問題ありません"):
            self.assertEqual(sfp_candidates(stem), [], stem)
        turns = _turns("その手順で進めてください。", "実は、あの、方法が分かりませんよね。")
        self.assertEqual(fix_yone_misuse(turns), 1)
        self.assertTrue(turns[1]["utterance"].endswith("分かりませんね。"),
                        turns[1]["utterance"])

    def test_particle_only_endings_take_ne(self):
        """助詞止め（「上位に景品を」「実は」）に「よね」は付かない（実測）。"""
        for stem in ("上位にちょっとした景品を", "スプリントでやると楽になる、実は"):
            self.assertEqual(sfp_candidates(stem), ["ね"], stem)

    def test_fix_leaves_shared_facts_and_questions(self):
        turns = _turns("先週の会議、長かったですよね。", "あの店、駅の裏でしたよね。",
                       "あのとき忙しかったよね。", "この日程で大丈夫ですよね？")
        self.assertEqual(fix_yone_misuse(turns), 0)


class GateTests(unittest.TestCase):
    """config の dialogue.gates で品質ゲートをシナリオごとに切れること。"""

    def test_unspecified_gates_stay_enabled(self):
        self.assertEqual(load_gates({}), ALL_GATES)
        g = load_gates({"dialogue": {"gates": {"self_name": False}}})
        self.assertFalse(g["self_name"])
        self.assertTrue(g["casual_leak"])       # 書かなかったものは有効のまま

    def test_unknown_gate_name_is_rejected(self):
        """打ち間違いを黙って無視すると、切ったつもりのゲートが効き続ける。"""
        with self.assertRaises(ValueError):
            load_gates({"dialogue": {"gates": {"self_nome": False}}})

    def test_all_gates_enabled_matches_the_default_prompt(self):
        for sp in (None, S2Params()):
            self.assertEqual(build_system_prompt(sp, ALL_GATES),
                             build_system_prompt(sp))

    def test_self_name_max_allows_introductions_in_the_prompt(self):
        """上限を上げたときも、名乗りを許す側の指示に入れ替わる。"""
        s = build_system_prompt(None, ALL_GATES, 1)
        self.assertIn("会話の始めに名乗って自己紹介してよい", s)
        self.assertIn("一度だけ", s)
        self.assertIn("名乗るのは3回までにし",
                      build_system_prompt(None, ALL_GATES, 3))
        # 既定（上限0）は従来どおり名乗り禁止
        self.assertEqual(build_system_prompt(None, ALL_GATES, 0),
                         build_system_prompt(None))

    def test_long_utterance_gate_off_drops_the_quota_from_both_prompts(self):
        """ゲートを切ったら本数のノルマも外す。ばらつきを促す指示自体は残す。"""
        meta = {"scenario": "テスト", "speakers": [{"name": "ア"}, {"name": "イ"}],
                "speech_style": "敬語", "target_turns": 30}
        gates = dict(ALL_GATES, long_utterance=False)
        s = build_system_prompt(S2Params(), gates)
        u = build_dialogue_prompt(meta, S2Params(), gates)
        self.assertNotIn("最低5ターンは入れる", s)
        self.assertNotIn("4ターン以上、対話全体に散らして", u)
        self.assertIn("31〜55文字", s)
        self.assertIn("31〜55文字", u)
        # 有効なら従来どおりノルマが載る
        self.assertIn("最低5ターンは入れる", build_system_prompt(S2Params(), ALL_GATES))
        self.assertIn("4ターン以上、対話全体に散らして",
                      build_dialogue_prompt(meta, S2Params(), ALL_GATES))

    def test_self_name_max_is_read_from_config(self):
        self.assertEqual(load_self_name_max({}), 0)
        self.assertEqual(load_self_name_max({"dialogue": {"self_name_max": 1}}), 1)

    def test_self_name_gate_off_allows_introductions_in_the_prompt(self):
        """ゲートを切ったら、機械で弾かなくなる禁止指示はプロンプトからも外す。"""
        gates = dict(ALL_GATES, self_name=False)
        for sp in (None, S2Params()):
            s = build_system_prompt(sp, gates)
            self.assertNotIn("名乗り・自己紹介（「ねえ、ユウだよ。」）をさせない", s)
            self.assertIn("会話の始めに名乗って自己紹介してよい", s)
            # 相手の名前を本人のセリフに入れる取り違えは、機械で検出できなくなる
            # ぶんプロンプトで禁じ続ける
            self.assertIn("話者の取り違えに注意", s)


class SelfIntroSfpTests(unittest.TestCase):
    """名乗りに付いた終助詞を落とす（inject_sfp が機械的に足してしまう）。"""

    def test_strips_sfp_from_polite_introductions(self):
        turns = _turns("はじめまして、タカシですよね。", "ミユですね。保育士しています。",
                       speakers=["タカシ", "ミユ"])
        self.assertEqual(fix_self_intro_sfp(turns), 2)
        self.assertEqual([t["utterance"] for t in turns],
                         ["はじめまして、タカシです。", "ミユです。保育士しています。"])

    def test_keeps_casual_yo_but_drops_ne(self):
        """タメ口の「ユウだよ。」は自然。自分のことに同意を求める「だね」は落とす。"""
        turns = _turns("ユウだよ。", "ハルカだね。", speakers=["ユウ", "ハルカ"])
        self.assertEqual(fix_self_intro_sfp(turns), 1)
        self.assertEqual([t["utterance"] for t in turns], ["ユウだよ。", "ハルカだ。"])

    def test_leaves_correct_introductions_and_other_names(self):
        # 元から正しい名乗り／相手の名前を含む叙述には触らない
        turns = _turns("はじめまして、ミサキです。", "担当はケンさんですよね。",
                       speakers=["ミサキ", "ケン"])
        self.assertEqual(fix_self_intro_sfp(turns), 0)

    def test_runs_after_injection(self):
        """付加の後に呼ばないと inject_sfp が付け直す（呼ぶ順の回帰）。"""
        turns = _turns("はじめまして、タカシです。", "ミユです。", speakers=["タカシ", "ミユ"])
        inject_sfp(turns, S2Params(sfp_ratio=1.0, sfp_inject=True))
        fix_self_intro_sfp(turns)
        self.assertEqual([t["utterance"] for t in turns],
                         ["はじめまして、タカシです。", "ミユです。"])
