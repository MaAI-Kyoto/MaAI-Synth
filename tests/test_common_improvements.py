"""接客で作った改良のうち、全シナリオに広げたもの（2026-09-26）。

- 「そっか？」「そうか？」「経済学部か？僕は〜」: 受け止め・独り言の「か」を上がり調子にしない
  （llm_client.emphasize_question_marks）。接客の客の「15分か。」→「15分ですか。」は維持
- 読みの注記の共通プロンプト（prompts/yomi_common.yaml）と、接客以外の対話の見せ方
- 数字のかな化: 日常会話の助数詞（つ・匹・軒…）、表に無い助数詞は開かない、きりのいい数は
  1桁ずつ読まない。読み辞書が「寿司」「御祝儀」を巻き込まない
- refine の保護話者 "@staff"
"""

import unittest

from generate_dialogues import (_LAUGH_HEAD, _yomi_user_message, load_prompt_pack,
                                polish_cs_speech, validate_yomi)
from llm_client import emphasize_question_marks as E
from refine_maai import protected_speakers
from tts_reading import to_tts_text


class QuestionMarkTests(unittest.TestCase):
    def test_realize_ka_falls(self):
        self.assertEqual(E("そっか？私は、映画が好き。"), "そっか。私は、映画が好き。")
        self.assertEqual(E("そっか"), "そっか。")
        self.assertEqual(E("あ、そうか？そうすれば、いいんだ。"), "あ、そうか。そうすれば、いいんだ。")
        self.assertEqual(E("なるほど。そういうことか"), "なるほど。そういうことか。")
        self.assertEqual(E("そっか、全部自分でやらなくていいんだ。"), "そっか、全部自分でやらなくていいんだ。")

    def test_volitional_still_question(self):
        self.assertEqual(E("どうしようか"), "どうしようか？")
        self.assertEqual(E("話そうか。"), "話そうか？")

    def test_musing_noun_ka(self):
        self.assertEqual(E("経済学部か？僕は、教育学部なんだ。"), "経済学部か。僕は、教育学部なんだ。")
        self.assertEqual(E("15日か？いいタイミングだね。"), "15日か。いいタイミングだね。")
        # 残りに問いがある・発話の末尾・疑問詞は問いのまま
        self.assertEqual(E("本当か？じゃあ、よく行く？"), "本当か？じゃあ、よく行く？")
        self.assertEqual(E("15分か？"), "15分か？")
        self.assertEqual(E("何か。用事？"), "何か？用事？")

    def test_existing_ack_rules_kept(self):
        self.assertEqual(E("そうでしたか？大変でしたね。"), "そうでしたか。大変でしたね。")
        self.assertEqual(E("それは東京ですか"), "それは東京ですか？")

    def test_cs_customer_musing_still_polite(self):
        meta = {"speech_style": "敬語",
                "speakers": [{"name": "サトウ", "role": "担当者"}, {"name": "ナオコ", "role": "客"}]}
        t = [{"speaker": "サトウ", "utterance": "15分ほどで仕上がります。"},
             {"speaker": "ナオコ", "utterance": E("15分か。それなら、待ちます。")}]
        polish_cs_speech(t, meta)
        self.assertEqual(t[1]["utterance"], "15分ですか。それなら、待ちます。")


class LaughHeadTests(unittest.TestCase):
    def test_period_and_small_tsu_forms_are_laughs(self):
        for u in ("ふふ、そうですね。", "ふふ。そうですね。", "あはは！それは災難。", "ふふっ、本当に。"):
            self.assertTrue(_LAUGH_HEAD.match(u), u)
        self.assertFalse(_LAUGH_HEAD.match("ふふふ"))


class YomiCommonTests(unittest.TestCase):
    def test_every_pack_has_yomi_prompt(self):
        for f in ("prompts/stage2_general.yaml", "prompts/stage2_general_gemma.yaml",
                  "prompts/stage2_first_meeting.yaml"):
            self.assertIn("読み付け係", load_prompt_pack(f).yomi_prompt, f)

    def test_compound_kata_hou_rejected(self):
        turns = [{"utterance": "全然違う方向に、行ってしまうんです。"},
                 {"utterance": "先方に、ご連絡いたします。"},
                 {"utterance": "法人の方が、有利です。"}]
        got, rej = validate_yomi(turns, [
            {"i": 0, "phrase": "違う方向", "surface": "方", "reading": "ほう"},
            {"i": 1, "phrase": "方に", "surface": "方", "reading": "かた"},
            {"i": 2, "phrase": "法人の方が", "surface": "方", "reading": "ほう"},
            {"i": 0, "phrase": "何度", "surface": "何", "reading": "なん"}])
        self.assertEqual(list(got), [2])
        self.assertEqual(len(rej), 3)

    def test_user_message_uses_names_outside_cs(self):
        turns = [{"speaker": "タカシ", "utterance": "どちらの方ですか？"},
                 {"speaker": "ミホ", "utterance": "関西の方です。"}]
        msg = _yomi_user_message(turns, {"scenario": "初対面の二人", "speakers": []}, {0, 1})
        self.assertIn("場面: 初対面の二人", msg)
        self.assertIn("*0 タカシ: どちらの方ですか？", msg)
        cs = _yomi_user_message(turns, {"setting": "窓口", "speakers": [
            {"name": "タカシ", "role": "担当者"}, {"name": "ミホ", "role": "客"}]}, {1})
        self.assertIn(" 0 担当者:", cs)
        self.assertIn("*1 客:", cs)


class ReadingTests(unittest.TestCase):
    def test_everyday_counters(self):
        for src, want in [("1つ", "ひとつ"), ("3つ", "みっつ"), ("1匹", "いっぴき"), ("3匹", "さんびき"),
                          ("1軒", "いっけん"), ("3軒", "さんげん"), ("6冊", "ろくさつ"), ("3品", "さんぴん"),
                          ("6歳", "ろくさい"), ("3分", "さんぷん"), ("1か所", "いっかしょ"),
                          ("1試合", "いっしあい"), ("20代", "にじゅう代")]:
            self.assertEqual(to_tts_text(src), want, src)

    def test_hundreds_multiplier(self):
        self.assertEqual(to_tts_text("2千5百円"), "にせんごひゃくえん")
        self.assertEqual(to_tts_text("3百人"), "さんびゃくにん")

    def test_unknown_counter_left_to_fish(self):
        for src in ("1日だけ", "1ページ", "3連休", "5G", "8000歩"):
            self.assertEqual(to_tts_text(src), src)

    def test_round_numbers_are_quantities(self):
        self.assertEqual(to_tts_text("予算は3500くらい"), "予算はさんぜんごひゃくくらい")
        self.assertEqual(to_tts_text("注文番号は10582です"), "注文番号はいちぜろごおはちにいです")

    def test_dictionary_spares_everyday_words(self):
        for src in ("寿司が好き", "寿命", "長寿", "御祝儀", "一足違いで"):
            self.assertEqual(to_tts_text(src), src)
        self.assertEqual(to_tts_text("寿の熨斗"), "ことぶきののし")


class ProtectedSpeakerTests(unittest.TestCase):
    def test_staff_token(self):
        meta = {"speakers": [{"name": "ハルカ", "role": "担当者"}, {"name": "タロウ", "role": "客"}]}
        self.assertEqual(protected_speakers(["@staff"], meta), {"ハルカ"})
        self.assertEqual(protected_speakers(["サトウ"], None), {"サトウ"})
        self.assertEqual(protected_speakers([], meta), set())


if __name__ == "__main__":
    unittest.main()
