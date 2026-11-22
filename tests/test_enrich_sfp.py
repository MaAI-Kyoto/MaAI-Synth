"""Stage3 enrich の文末終助詞ガード（fix_added_sfp / _thin_sfp）。

「足すだけ」の工程なので終助詞は句点の直前に挿入されるだけだが、接続先を誤ると
意味が反転する（動詞の言い切り＋「な」＝禁止）。ここではその弾き漏らしを検査する。
"""

import unittest

from generate_voice_enrich import (
    EnrichParams,
    added_sfp,
    assemble_turns,
    build_enrich_prompt,
    build_system_prompt,
    clean_enriched_text,
    fix_added_sfp,
    revert_sfp,
)


class AddedSfpTests(unittest.TestCase):
    def test_detects_addition_to_bare_ending(self):
        self.assertEqual(added_sfp("そうだな。", "そうだ。"), "な")

    def test_detects_extension_of_existing_particle(self):
        self.assertEqual(added_sfp("明日は雨だよね。", "明日は雨だよ。"), "よね")
        self.assertEqual(added_sfp("そうだよね。", "そうだね。"), "よね")

    def test_unchanged_ending_is_not_an_addition(self):
        self.assertIsNone(added_sfp("えっと、そうだね。", "そうだね。"))
        # 語の一部が終助詞に見える形（「みんな」）でも、変化が無ければ検出しない
        self.assertIsNone(added_sfp("えっと、みんな。", "みんな。"))

    def test_revert_restores_original_ending(self):
        self.assertEqual(revert_sfp("えっと、行くな。", "行く。"), "えっと、行く。")
        self.assertEqual(revert_sfp("そうだよね。", "そうだよ。"), "そうだよ。")
        self.assertEqual(revert_sfp("えっと、みんなね。", "みんな。"), "えっと、みんな。")


class FixAddedSfpTests(unittest.TestCase):
    def assert_kept(self, new, old):
        self.assertEqual(fix_added_sfp(new, old), new)

    def assert_reverted(self, new, old, expected):
        self.assertEqual(fix_added_sfp(new, old), expected)

    # ── 許可する形 ──────────────────────────────────────────────
    def test_na_after_da_ta_i_is_kept(self):
        self.assert_kept("これは大変だな。", "これは大変だ。")
        self.assert_kept("昨日は疲れたな。", "昨日は疲れた。")
        self.assert_kept("これは難しいな。", "これは難しい。")
        self.assert_kept("よくわからないな。", "よくわからない。")

    def test_yone_is_kept(self):
        self.assert_kept("明日は雨だよね。", "明日は雨だよ。")
        self.assert_kept("その話は聞きましたよね。", "その話は聞きました。")
        self.assert_kept("そこが難しいよね。", "そこが難しい。")

    def test_ne_is_kept_even_after_request_forms(self):
        self.assert_kept("あとで教えてくださいね。", "あとで教えてください。")
        self.assert_kept("そろそろ行きましょうね。", "そろそろ行きましょう。")

    def test_filler_addition_without_sfp_change_is_untouched(self):
        self.assert_kept("えっと、そうだね。", "そうだね。")

    # ── 弾く形 ──────────────────────────────────────────────────
    def test_na_after_plain_verb_is_reverted(self):
        """「行くな」「思うな」は禁止の意味になり、発話の意味が反転する。"""
        self.assert_reverted("私もそう思うな。", "私もそう思う。", "私もそう思う。")
        self.assert_reverted("じゃあ行くな。", "じゃあ行く。", "じゃあ行く。")
        self.assert_reverted("あとで見るな。", "あとで見る。", "あとで見る。")

    def test_na_after_polite_form_is_reverted(self):
        self.assert_reverted("確認しましたな。", "確認しました。", "確認しました。")
        self.assert_reverted("そうでしたな。", "そうでした。", "そうでした。")

    def test_yone_after_request_form_is_reverted(self):
        self.assert_reverted("教えてくださいよね。", "教えてください。", "教えてください。")
        self.assert_reverted("そろそろ行きましょうよね。", "そろそろ行きましょう。",
                             "そろそろ行きましょう。")

    def test_yone_after_first_person_commitment_is_reverted(self):
        """「よね」は相手も知っている事柄に同意を求める語。自分の意志表明には付かない。"""
        self.assert_reverted("次回もよろしくお願いしますよね。", "次回もよろしくお願いします。",
                             "次回もよろしくお願いします。")
        self.assert_reverted("今日の議論を基に提案書を作成しますよね。",
                             "今日の議論を基に提案書を作成します。",
                             "今日の議論を基に提案書を作成します。")
        self.assert_reverted("次回までに全体像を固めておきますよね。",
                             "次回までに全体像を固めておきます。",
                             "次回までに全体像を固めておきます。")
        # 同じ場面でも「ね」は自然なので残す
        self.assert_kept("今日の議論を基に提案書を作成しますね。",
                         "今日の議論を基に提案書を作成します。")
        # 過去形（共有された事実）への「よね」は残す
        self.assert_kept("了解しましたよね。", "了解しました。")

    def test_particle_stacked_after_a_comma_is_reverted(self):
        """「〜ですね、よね。」は拡張ではなく重複。20B の実測で最頻の失敗。"""
        self.assert_reverted("30分短い時間で効果的ですね、よね。",
                             "30分短い時間で効果的ですね。",
                             "30分短い時間で効果的ですね。")
        self.assert_reverted("頑張ってください、応援してますよ、よね。",
                             "頑張ってください、応援してますよ。",
                             "頑張ってください、応援してますよ。")
        self.assert_reverted("えっと、行く、ね。", "行く。", "えっと、行く。")

    def test_disallowed_particles_are_reverted(self):
        # 「よ」は合成側が人間より過多、「かな」「っけ」は意味（推量・想起）が変わる
        self.assert_reverted("それは違うよ。", "それは違う。", "それは違う。")
        self.assert_reverted("明日は晴れるかな。", "明日は晴れる。", "明日は晴れる。")
        self.assert_reverted("先週も来たっけ。", "先週も来た。", "先週も来た。")
        self.assert_reverted("もう帰るわ。", "もう帰る。", "もう帰る。")

    def test_addition_to_question_is_reverted(self):
        self.assert_reverted("予算はいくらですかね？", "予算はいくらですか？",
                             "予算はいくらですか？")

    def test_particle_replacement_is_reverted(self):
        """元の終助詞を別の語に差し替える形（削除を伴う）は元へ戻す。"""
        self.assert_reverted("それは大事だよ。", "それは大事だね。", "それは大事だね。")

    def test_second_particle_on_existing_sfp_is_reverted(self):
        self.assert_reverted("えっと、みんなね。", "みんな。", "えっと、みんな。")


class CleanEnrichedTextSfpTests(unittest.TestCase):
    """clean_enriched_text 経由でもガードが効き、内容保持チェックを壊さないこと。"""

    def test_dangerous_na_is_stripped_and_text_accepted(self):
        text, fb = clean_enriched_text("えっと、私もそう思うな。", "私もそう思う。")
        self.assertEqual(text, "えっと、私もそう思う。")
        self.assertFalse(fb)          # フィラーの付加は残るのでフォールバックではない

    def test_valid_sfp_survives_the_subsequence_guard(self):
        text, fb = clean_enriched_text("えっと、これは難しいな。", "これは難しい。")
        self.assertEqual(text, "えっと、これは難しいな。")
        self.assertFalse(fb)

    def test_trailing_filler_closed_by_a_comma_is_removed(self):
        """「〜ですね、うーん、。」（文末フィラー＋読点）を元の文末へ戻す。"""
        text, _ = clean_enriched_text("トイレのコストは大きいですね、うーん、。",
                                      "トイレのコストは大きいですね。")
        self.assertEqual(text, "トイレのコストは大きいですね。")

    def test_trailing_comma_before_period_is_removed(self):
        text, _ = clean_enriched_text("それは大事です、。", "それは大事です。")
        self.assertEqual(text, "それは大事です。")


class ThinSfpTests(unittest.TestCase):
    def _assemble(self, olds, news, **kw):
        ep = EnrichParams(**kw)
        turns = [{"speaker": "A" if i % 2 == 0 else "B", "utterance": o}
                 for i, o in enumerate(olds)]
        edits = {i: {"text": t, "start_offset": 0.3, "duration": 1.0}
                 for i, t in enumerate(news)}
        return assemble_turns(turns, edits, ep)

    def test_excess_sfp_is_thinned_to_cap(self):
        olds = [f"これは大変だ。" for _ in range(10)]
        news = [f"これは大変だな。" for _ in range(10)]
        out, stats = self._assemble(olds, news, sfp_cap_ratio=0.2, verify_heads=False)
        self.assertEqual(stats["sfp"], 2)
        self.assertEqual(stats["sfp_thinned"], 8)
        self.assertEqual(sum(1 for t in out if t["text"].endswith("だな。")), 2)

    def test_ne_is_thinned_before_yone_and_na(self):
        """不足している「よね」「な」を残し、既に足りている「ね」から剥がす。"""
        olds = ["これは大変だ。", "明日は雨だよ。", "昨日は疲れた。", "そうです。"]
        news = ["これは大変だな。", "明日は雨だよね。", "昨日は疲れたな。", "そうですね。"]
        out, stats = self._assemble(olds, news, sfp_cap_ratio=0.75, verify_heads=False)
        self.assertEqual(stats["sfp_thinned"], 1)
        self.assertEqual(out[3]["text"], "そうです。")     # 「ね」だけが剥がれる
        self.assertEqual(out[1]["text"], "明日は雨だよね。")

    def test_no_thinning_under_cap(self):
        olds = ["これは大変だ。"] + ["そうする。"] * 9
        news = ["これは大変だな。"] + ["そうする。"] * 9
        _, stats = self._assemble(olds, news, sfp_cap_ratio=0.18, verify_heads=False)
        self.assertEqual(stats["sfp"], 1)
        self.assertEqual(stats["sfp_thinned"], 0)


class SfpPromptTests(unittest.TestCase):
    def test_section_is_included_with_ratio(self):
        s = build_system_prompt(EnrichParams(sfp_ratio=0.12))
        self.assertIn("5. 発話の文末に終助詞を足して", s)
        self.assertIn("全発話の約12%", s)
        self.assertNotIn("__SFP_", s)

    def test_section_is_disabled_at_zero(self):
        s = build_system_prompt(EnrichParams(sfp_ratio=0.0))
        self.assertIn("終助詞は足さない", s)
        self.assertNotIn("__SFP_", s)

    def test_user_prompt_states_a_quota(self):
        turns = [{"speaker": "A", "utterance": "そうだ。"} for _ in range(50)]
        p = build_enrich_prompt({"speakers": [{"name": "A"}]}, turns,
                                EnrichParams(sfp_ratio=0.12))
        self.assertIn("少なくとも6個の発話の文末に終助詞", p)
        self.assertNotIn("終助詞", build_enrich_prompt(
            {"speakers": [{"name": "A"}]}, turns, EnrichParams(sfp_ratio=0.0)))


if __name__ == "__main__":
    unittest.main()
