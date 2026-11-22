"""接客の発話の仕上げ（2026-09-23 に 100対話を1ターンずつ読んだレビュー 628件への対処）。

polish_cs_speech / fix_echo_turns / to_sentence_periods の誤分割 / 校閲パスの検証。
"""

import unittest

from generate_dialogues import (
    _proofread_validate,
    fix_echo_turns,
    polish_cs_speech,
    to_sentence_periods,
)

META = {"speech_style": "敬語",
        "speakers": [{"name": "サトウ", "role": "担当者"}, {"name": "ナオコ", "role": "客"}]}
KID = {"speech_style": "混合",
       "speakers": [{"name": "サトウ", "role": "担当者"}, {"name": "ショウタ", "role": "客"}]}


def conv(*utts, start="ナオコ"):
    names = ("ナオコ", "サトウ") if start == "ナオコ" else ("サトウ", "ナオコ")
    return [{"speaker": names[i % 2], "utterance": u} for i, u in enumerate(utts)]


class PolishStaffTests(unittest.TestCase):
    def fix(self, staff_u, prev="あの、少し伺いたいんですけど。", meta=META):
        t = conv(prev, staff_u)
        polish_cs_speech(t, meta)
        return t[1]["utterance"]

    def test_staff_laugh_with_period_is_removed(self):
        # 「ふふ、」だけでなく「ふふ。」も落とす（cap_laugh_heads は読点しか見ていなかった）
        self.assertEqual(self.fix("ふふ。お気になさらないでください。"), "お気になさらないでください。")
        self.assertEqual(self.fix("あはは。無理にとは申しません。"), "無理にとは申しません。")

    def test_staff_fillers_are_removed(self):
        self.assertEqual(self.fix("はい。上の階から順番に見て、その、一番スムーズでございます。"),
                         "はい。上の階から順番に見て、一番スムーズでございます。")
        self.assertEqual(self.fix("まあ、提携外ではございますが、お安いです。"),
                         "提携外ではございますが、お安いです。")
        # 読点の無い指示語は残す
        self.assertEqual(self.fix("その件でしたら、承知いたしました。"), "その件でしたら、承知いたしました。")

    def test_fixed_phrases(self):
        self.assertEqual(self.fix("どういたしまして。どうぞ、ごゆっくり。"), "とんでもございません。どうぞ、ごゆっくり。")
        self.assertEqual(self.fix("ご納得いただけて、安心いたしました。"), "ご納得いただけて、何よりでございます。")
        self.assertEqual(self.fix("合計金額は、2400円になります。"), "合計金額は、2400円でございます。")
        self.assertEqual(self.fix("お待たせいたしました。こちらが、お荷物になります。"),
                         "お待たせいたしました。こちらが、お荷物でございます。")
        self.assertEqual(self.fix("ご確認いたしました。ありがとうございます。"), "確認いたしました。ありがとうございます。")
        self.assertEqual(self.fix("ご登録のご電話番号を、お願いできますか？"), "ご登録のお電話番号を、お願いできますか？")
        self.assertEqual(self.fix("新しいパスワードを、入力していただいてください。"), "新しいパスワードを、入力してください。")
        self.assertEqual(self.fix("大丈夫ですよ。ゆっくりご案内いたしますね。"), "ご安心ください。ゆっくりご案内いたしますね。")
        self.assertEqual(self.fix("よろしいでしょうか？それでは、外のしで準備いたしますね。"),
                         "それでは、外のしで準備いたしますね。")

    def test_naruhodo_and_soudesune(self):
        self.assertEqual(self.fix("なるほど。それでしたら、第2駐車場がおすすめでございます。"),
                         "それでしたら、第2駐車場がおすすめでございます。")
        # 客の問いの直後は考える間、それ以外は同意
        self.assertEqual(self.fix("そうですね。16時を過ぎますと空いてまいります。", prev="空いている時間はありますか？"),
                         "そうでございますね。16時を過ぎますと空いてまいります。")
        self.assertEqual(self.fix("そうですね。それが一番かと存じます。", prev="家族に声をかけてみます。"),
                         "さようでございますね。それが一番かと存じます。")
        # 子ども相手の対話ではやさしい話し言葉のまま
        self.assertEqual(self.fix("そうですね。きっと喜んでくれます。", meta=KID), "そうですね。きっと喜んでくれます。")

    def test_repairs_sentences_split_by_period_insertion(self):
        self.assertEqual(self.fix("こちらこそ。ありがとうございました。失礼いたします。"),
                         "こちらこそ、ありがとうございました。失礼いたします。")
        self.assertEqual(self.fix("変更につきましては、もちろん。可能でございます。"),
                         "変更につきましては、もちろん、可能でございます。")
        self.assertEqual(self.fix("まずは、画面の右上にございます。マイページというボタンを押してください。"),
                         "まずは、画面の右上にございます、マイページというボタンを押してください。")


class PolishCustomerTests(unittest.TestCase):
    def fix(self, cust_u, meta=META):
        t = conv("はい、ご案内いたします。", cust_u, start="サトウ")
        polish_cs_speech(t, meta)
        return t[1]["utterance"]

    def test_customer_side(self):
        self.assertEqual(self.fix("2400円ですね。承知いたしました。"), "2400円ですね。わかりました。")
        self.assertEqual(self.fix("あ、ありました。ここを押せばいいんです。"), "あ、ありました。ここを押せばいいんですね。")
        self.assertEqual(self.fix("控えめな感じ、ということです。"), "控えめな感じ、ということですね。")
        self.assertEqual(self.fix("情報を教えればいいです？"), "情報を教えればいいですか？")
        self.assertEqual(self.fix("丁寧に教えてくれて、ありがとうございました。"), "丁寧に教えていただいて、ありがとうございました。")

    def test_child_customer_is_left_alone(self):
        self.assertEqual(self.fix("これでいいんです。", meta=KID), "これでいいんです。")


class SentencePeriodTests(unittest.TestCase):
    def run1(self, u):
        t = [{"speaker": "A", "utterance": u}]
        to_sentence_periods(t, {"A"})
        return t[0]["utterance"]

    def test_no_longer_splits_fixed_phrases(self):
        for u in ("こちらこそ、ありがとうございます。それでは、失礼いたします。",
                  "配送サイクルの変更につきましては、もちろん、可能でございます。",
                  "まずは、画面の右上にございます、マイページというボタンを押してください。"):
            self.assertEqual(self.run1(u), u)

    def test_still_splits_real_sentence_ends(self):
        self.assertEqual(self.run1("来週月曜に再査定に行きます、結果は火曜までに出せますよね。"),
                         "来週月曜に再査定に行きます。結果は火曜までに出せますよね。")
        self.assertEqual(self.run1("どういたしまして、今週のスケジュールをメールでお送りしますね。"),
                         "どういたしまして。今週のスケジュールをメールでお送りしますね。")


class EchoTurnTests(unittest.TestCase):
    def test_echo_of_staff_checking_action_becomes_a_reply(self):
        t = conv("あの、ポイントの期限を知りたくて。",
                 "トモヤ様ですね。ただいま、履歴をお調べいたしますね。",
                 "履歴。",
                 "はい。お待たせいたしました。現在は、スタンダードプランでございます。",
                 "ありがとうございます。")
        self.assertEqual(fix_echo_turns(t, META), 1)
        self.assertEqual(t[2]["utterance"], "お願いします。")
        self.assertEqual(t[3]["utterance"], "お待たせいたしました。現在は、スタンダードプランでございます。")

    def test_echo_of_own_word_becomes_yes(self):
        t = conv("ゴールド会員のままで、いたいんですよね。",
                 "サユリ様は、現在ゴールド会員でいらっしゃいますね？",
                 "ゴールド会員？",
                 "はい。左様でございます。累計は97万円でございます。",
                 "あと3万円ですね。")
        fix_echo_turns(t, META)
        self.assertEqual(t[2]["utterance"], "はい、そうです。")
        self.assertEqual(t[3]["utterance"], "累計は97万円でございます。")

    def test_natural_surprise_reactions_are_kept(self):
        for staff, cust in (("海老のメニューを、白身魚に変更させていただきますね。", "白身魚ですか？"),
                            ("最短で、明日の午前中にはお届けできる手配をさせていただきます。", "明日ですか？"),
                            ("こちらは、1セット500円でございます。", "500円？")):
            t = conv("あの、お願いがあって。", staff, cust, "はい。", "わかりました。")
            self.assertEqual(fix_echo_turns(t, META), 0, cust)

    def test_child_customer_is_left_alone(self):
        t = conv("ポストカードある？", "ただいま、在庫があるか確認いたしますね。", "在庫？",
                 "はい。お店に、まだ残っているかということですよ。", "あ、そう。")
        self.assertEqual(fix_echo_turns(t, KID), 0)


class ProofreadValidateTests(unittest.TestCase):
    TURNS = conv("あの、それって、結構な金額になりますか？", "さようでございますね。",
                 "まあ、損をするのは嫌なんです。", "お気持ちはよく分かります。")

    def test_accepts_a_reasonable_fix(self):
        ok, rej = _proofread_validate(self.TURNS, [{"i": 1, "text": "はい。本日は、全額で4500円ほどになります"}], META)
        self.assertEqual(ok, {1: "はい。本日は、全額で4500円ほどになります。"})
        self.assertEqual(rej, [])

    def test_rejects_bad_edits(self):
        ok, rej = _proofread_validate(self.TURNS, [
            {"i": 9, "text": "範囲外"},
            {"i": 3, "text": "ふふ、お気持ちはよく分かります。"},       # 担当者の笑い
            {"i": 3, "text": "あ" * 80},                                # 長すぎる
            {"i": "x", "text": "壊れた番号"},
        ], META)
        self.assertEqual(ok, {})
        self.assertEqual(len(rej), 4)

    def test_rejects_everything_when_too_many_edits(self):
        turns = conv(*["客の発話です。", "担当者の発話です。"] * 5)
        edits = [{"i": i, "text": f"直した発話{i}です。"} for i in range(8)]
        ok, rej = _proofread_validate(turns, edits, META)
        self.assertEqual(ok, {})
        self.assertTrue(any("多すぎる" in r for r in rej))


if __name__ == "__main__":
    unittest.main()


class ProofreadRegressionTests(unittest.TestCase):
    """校閲パスの直しすぎを弾く（100対話への実行で実際に出た形）。"""

    def check(self, turns, i, new, meta=META):
        from generate_dialogues import _proofread_regression, _is_polite_style, staff_names
        return _proofread_regression(turns, i, turns[i]["utterance"], new,
                                     staff_names(meta), _is_polite_style(meta))

    def test_keeps_staff_readback_of_customer_words(self):
        t = conv("入場するための、ＱＲコードというものを、出したいんです。", "ＱＲコードですね。")
        self.assertTrue(self.check(t, 1, "はい、承知いたしました。"))

    def test_allows_replacing_a_readback_that_does_not_answer(self):
        t = conv("特別な招待会のようなものはありますか？", "招待会ですね。")
        self.assertEqual(self.check(t, 1, "はい、ございます。12月5日に開催いたします。"), "")

    def test_customer_questions_and_reactions_are_kept(self):
        t = conv("はい。", "18時から、20時までの、2時間制でございます。", "2時間、ですか？")
        self.assertTrue(self.check(t, 2, "はい、お願いします。"))
        t = conv("はい。", "飲み放題は2時間でございます。", "時間は、いつまで飲めるんですか？")
        self.assertTrue(self.check(t, 2, "時間は、いつまで飲めるんですね？"))
        t = conv("はい。", "そのままで大丈夫でございます。", "本当に、助かります。")
        self.assertTrue(self.check(t, 2, "本当に、助かりますね。"))

    def test_allows_please_after_an_announcement(self):
        t = conv("はい。", "ただいま、履歴をお調べいたしますね。", "履歴。")
        self.assertEqual(self.check(t, 2, "お願いします。"), "")

    def test_no_new_keigo_errors_or_stiffness_for_children(self):
        t = conv("よかった。", "それは良かったですね。")
        self.assertTrue(self.check(t, 1, "それは良かったでございますね。"))
        t = conv("やった。", "きっと、喜んでくれると思います。")
        self.assertTrue(self.check(t, 1, "きっと、喜んでくれるかと存じます。", meta=KID))


class ResidualRuleTests(unittest.TestCase):
    """作り直し後のレビュー（2026-09-24）で残った形。"""

    def test_staff_humble_forms(self):
        for u, want in (("まずはお調べしますので、注文番号を教えていただけますか？",
                         "まずはお調べいたしますので、注文番号を教えていただけますか？"),
                        ("この後すぐに、手続きのメールをお送りしますね。",
                         "この後すぐに、手続きのメールをお送りいたしますね。"),
                        ("ご確認できました。現在は、毎月のお届けでございます。",
                         "確認いたしました。現在は、毎月のお届けでございます。"),
                        ("迷惑メールフォルダに入っていないか、確認してください。",
                         "迷惑メールフォルダに入っていないか、ご確認ください。")):
            t = conv("あの、お願いします。", u)
            polish_cs_speech(t, META)
            self.assertEqual(t[1]["utterance"], want)

    def test_own_word_echo_without_staff_mention(self):
        t = conv("あの、配送先を変えたいんです。",
                 "さようでございますか。承知いたしました。",
                 "あ、配送先？",
                 "はい。お買い上げ伝票はお持ちでしょうか？",
                 "あ、こちらです。")
        self.assertEqual(fix_echo_turns(t, META), 1)
        self.assertEqual(t[2]["utterance"], "はい。")

    def test_spoken_placeholder_is_caught(self):
        from generate_dialogues import _PLACEHOLDER_RE
        self.assertTrue(_PLACEHOLDER_RE.search("えみー、まるまる、あっと、じーめいる"))
        self.assertFalse(_PLACEHOLDER_RE.search("まるまる焼きが好きです"))
