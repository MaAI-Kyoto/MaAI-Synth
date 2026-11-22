"""人手評価シート（2026-09-25、100対話中 Bad 37）への対処。

- 担当者の言い方（polish_cs_staff_manner）: 「〜になります」「〜形になります」、上がり調子の
  確認「〜ですね？」、むき出しの問い、「あそこ」「いいえ」「〜ですよ」、業界用語
- 客の気づきの「〜んですね」「そうですよね」を間引かない（thin_ne / thin_sfp の keep_realize）
- fish の読み（tts_reading）: 読み辞書と数字のかな化
"""

import copy
import unittest

from generate_dialogues import polish_cs_speech, polish_cs_staff_manner, thin_ne, thin_sfp
from tts_reading import normalize_numbers, to_tts_text

META = {"speech_style": "敬語",
        "speakers": [{"name": "サトウ", "role": "担当者"}, {"name": "ナオコ", "role": "客"}]}
KID = {"speech_style": "混合",
       "speakers": [{"name": "サトウ", "role": "担当者"}, {"name": "ショウタ", "role": "客"}]}


def fix(u, prev="", polite=True):
    return polish_cs_staff_manner(u, prev, polite)


class NarimasuTests(unittest.TestCase):
    def test_copula_narimasu(self):
        self.assertEqual(fix("合計で、5200円になります。"), "合計で、5200円でございます。")
        self.assertEqual(fix("次回は、火曜日になります。"), "次回は、火曜日でございます。")
        self.assertEqual(fix("メーカー保証の、対象になりますので、ご安心ください。"),
                         "メーカー保証の、対象でございますので、ご安心ください。")

    def test_keep_change_and_honorific(self):
        for u in ("部品が余ると、どうしても、不安になります。", "ご覧になりますか？",
                  "十分綺麗になります。", "新しいパスワードを決める画面になります。",
                  "ご利用いただけるようになります。", "何センチになりますか？"):
            self.assertEqual(fix(u), u)

    def test_katachi(self):
        self.assertEqual(fix("それでは、ご案内する形になります。"), "それでは、ご案内いたします。")
        self.assertEqual(fix("来月の上旬には、店頭に並ぶ形になります。"), "来月の上旬には、店頭に並びます。")
        self.assertEqual(fix("マイページから、入力する形になります。"), "マイページから、入力していただきます。")
        self.assertEqual(fix("当日は、窓口でお支払いする形になります。"), "当日は、窓口でお支払いいただきます。")
        self.assertEqual(fix("こちらで、登録する形になります。"), "こちらで、登録いたします。")
        self.assertEqual(fix("専用の招待状をご提示いただく形になります。"), "専用の招待状をご提示いただきます。")
        self.assertEqual(fix("差額は、毎月500円という形になります。"), "差額は、毎月500円でございます。")
        self.assertEqual(fix("割引が、適用されていた形になります。"), "割引が、適用されておりました。")
        self.assertEqual(fix("支払いが発生しない形になります。"), "支払いが発生しません。")
        self.assertEqual(fix("シルバーの紐で、結ぶモダンな形になります。"),
                         "シルバーの紐で、結ぶモダンな形でございます。")


class ConfirmTests(unittest.TestCase):
    def test_rising_readback_falls(self):
        self.assertEqual(fix("ベージュの財布ですね？"), "ベージュの財布ですね。")
        self.assertEqual(fix("外箱が、潰れていたんですね？"), "外箱が、潰れていたのですね。")
        self.assertEqual(fix("ご利用でございますね？"), "ご利用でございますね。")

    def test_bare_question(self):
        self.assertEqual(fix("短いネジ？"), "短いネジでございますね。")
        self.assertEqual(fix("固定されている？"), "固定されているでしょうか？")
        self.assertEqual(fix("ご予算はおいくらですか？"), "ご予算はおいくらですか？")


class WordingTests(unittest.TestCase):
    def test_words(self):
        self.assertEqual(fix("はい。あそこは、最大料金が1000円です。"), "はい。あちらは、最大料金が1000円です。")
        self.assertEqual(fix("それは良かったです。それでは、お待ちしております。"),
                         "何よりでございます。それでは、お待ちしております。")
        self.assertEqual(fix("焼成スケジュールと、在庫を確認いたします。"), "焼き上がりの予定と、在庫を確認いたします。")
        self.assertEqual(fix("同梱の返送用伝票をお使いください。"), "同封の返送用伝票をお使いください。")

    def test_iie(self):
        self.assertEqual(fix("いいえ。まず、半券をいただけますか？", "本当に申し訳ないです。"),
                         "とんでもございません。まず、半券をいただけますか？")
        self.assertEqual(fix("いいえ、無料でございます。", "有料ですか？"), "いえ、無料でございます。")

    def test_echo_question_and_daijoubu(self):
        self.assertEqual(fix("ガタガタという音ですか？それは、お困りでしたね。",
                             "脱水のときに、ガタガタってすごい音がして。"),
                         "ガタガタという音でございますね。それは、お困りでしたね。")
        self.assertEqual(fix("ご予算はおいくらですか？", "プレゼントを探していて。"), "ご予算はおいくらですか？")
        self.assertEqual(fix("大丈夫でございます。まずは、半券をいただけますか？", "何度もすみません。"),
                         "とんでもございません。まずは、半券をいただけますか？")
        self.assertEqual(fix("はい。こちらを拝見します。"), "はい。こちらを拝見いたします。")
        self.assertEqual(fix("ふふっと笑ってしまい、お気になさらないでください。"), "お気になさらないでください。")

    def test_final_review_rules(self):
        self.assertEqual(fix("歩いて来られたのですね。お疲れ様でございます。"),
                         "歩いて来られたのですね。ご足労いただき、ありがとうございます。")
        self.assertEqual(fix("どのようなご用件でございましたでしょうか？"), "どのようなご用件でしょうか？")
        self.assertEqual(fix("保険の適用がいたしかねます。"), "保険の適用ができかねます。")
        self.assertEqual(fix("ご予約のご件でございますね。"), "ご予約の件でございますね。")
        self.assertEqual(fix("満車の可能性が高いでございます。"), "満車の可能性が高いかと存じます。")
        self.assertEqual(fix("確認いたします。なるほど、破損しておりますね。"), "確認いたします。破損しておりますね。")
        self.assertEqual(fix("お手続きを完了させていただければ、適用されます。"),
                         "お手続きを完了していただければ、適用されます。")
        self.assertEqual(fix("画面を開いておりますか？"), "画面を開いていらっしゃいますか？")
        self.assertEqual(fix("お荷物は届いておりますか？"), "お荷物は届いておりますか？")

    def test_casual_only_when_polite(self):
        self.assertEqual(fix("そういうこと、ありますよね。大丈夫ですよ。"), "そういうこと、ございます。ご安心ください。")
        self.assertEqual(fix("大丈夫ですよ。", polite=False), "大丈夫ですよ。")

    def test_wired_into_polish(self):
        t = [{"speaker": "ナオコ", "utterance": "いくらですか？"},
             {"speaker": "サトウ", "utterance": "合計で、3200円になります。"}]
        polish_cs_speech(t, META)
        self.assertEqual(t[1]["utterance"], "合計で、3200円でございます。")


class KeepRealizeTests(unittest.TestCase):
    TURNS = [{"speaker": "客", "utterance": "あー、そういうのが、あるんですね。"},
             {"speaker": "客", "utterance": "やっぱり、そうですよね。"},
             {"speaker": "客", "utterance": "今日は天気がとてもいいですね。"}] * 3

    def test_thin_ne_keeps_realize(self):
        t = copy.deepcopy(self.TURNS)
        thin_ne(t, 0.01, keep_realize=True)
        self.assertEqual(t[0]["utterance"], "あー、そういうのが、あるんですね。")
        self.assertEqual(t[2]["utterance"], "今日は天気がとてもいいです。")
        t = copy.deepcopy(self.TURNS)
        thin_ne(t, 0.01)                      # 既定（接客以外）は従来どおり落とす
        self.assertEqual(t[0]["utterance"], "あー、そういうのが、あるんです。")

    def test_thin_sfp_keeps_agreement(self):
        t = copy.deepcopy(self.TURNS)
        thin_sfp(t, 0.0, keep_realize=True)
        self.assertEqual(t[1]["utterance"], "やっぱり、そうですよね。")


class ReadingTests(unittest.TestCase):
    def test_dictionary(self):
        self.assertEqual(to_tts_text("表書きは内祝で。"), "おもてがきはうちいわいで。")
        self.assertEqual(to_tts_text("外のしと内のし、ですね。"), "そとのしとうちのし、ですね。")
        self.assertEqual(to_tts_text("たまらない一足です。一足先に。"), "たまらないいっそくです。一足先に。")

    def test_ids_digit_wise(self):
        self.assertEqual(normalize_numbers("090、1234、5678です。"),
                         "ぜろきゅうぜろ、いちにいさんよん、ごおろくななはちです。")
        self.assertEqual(normalize_numbers("10582番です。"), "いちぜろごおはちにいばんです。")

    def test_long_numbers_do_not_crash(self):
        self.assertEqual(normalize_numbers("09012345678円"), "ぜろきゅうぜろいちにいさんよんごおろくななはち円")
        self.assertEqual(normalize_numbers("123456789012345です"), "いちにいさんよんごおろくななはちきゅうぜろいちにいさんよんごおです")

    def test_counters(self):
        self.assertEqual(normalize_numbers("1500円"), "せんごひゃくえん")
        self.assertEqual(normalize_numbers("8000円"), "はっせんえん")
        self.assertEqual(normalize_numbers("30分"), "さんじゅっぷん")
        self.assertEqual(normalize_numbers("4時と9時"), "よじとくじ")
        self.assertEqual(normalize_numbers("3本と6個"), "さんぼんとろっこ")
        self.assertEqual(normalize_numbers("10月1日"), "じゅうがつついたち")
        self.assertEqual(normalize_numbers("2泊3日"), "にはくみっか")
        self.assertEqual(normalize_numbers("3日間"), "みっかかん")
        self.assertEqual(normalize_numbers("5万ポイント"), "ごまんぽいんと")


if __name__ == "__main__":
    unittest.main()


class YomiTests(unittest.TestCase):
    """読みの注記（「上司の方」を fish が「ほう」と読む、2026-09-26）。"""

    def test_validate(self):
        from generate_dialogues import validate_yomi
        turns = [{"utterance": "上司の方に、お渡しします。"}, {"utterance": "こちらの方が安いです。"}]
        got, rej = validate_yomi(turns, [
            {"i": 0, "phrase": "上司の方", "surface": "方", "reading": "かた"},
            {"i": 1, "phrase": "こちらの方", "surface": "方", "reading": "ほう"},
            {"i": 1, "phrase": "あちらの方", "surface": "方", "reading": "ほう"},   # 本文に無い
            {"i": 0, "phrase": "上司の方", "surface": "方", "reading": "kata"},     # かなでない
            {"i": 9, "phrase": "x", "surface": "x", "reading": "x"},
            {"i": 0, "phrase": "お渡し", "surface": "渡し", "reading": "わた"}])  # 送りがな不一致
        self.assertEqual(sorted(got), [0, 1])
        self.assertEqual(len(rej), 4)

    def test_apply(self):
        from tts_reading import apply_yomi
        y = [{"phrase": "上司の方", "surface": "方", "reading": "かた"},
             {"phrase": "こちらの方が", "surface": "方", "reading": "ほう"}]
        self.assertEqual(apply_yomi("上司の方に渡すなら、こちらの方が良いです。", y),
                         "上司のかたに渡すなら、こちらのほうが良いです。")
        self.assertEqual(apply_yomi("本文が書き換わった。", y), "本文が書き換わった。")

    def test_turn_text_follows_config(self):
        import tts_reading
        t = {"text": "上司の方でございますね。",
             "yomi": [{"phrase": "上司の方", "surface": "方", "reading": "かた"}]}
        try:
            tts_reading.configure(False)
            self.assertEqual(tts_reading.turn_text(t), "上司の方でございますね。")
            tts_reading.configure(True)
            self.assertEqual(tts_reading.turn_text(t), "上司のかたでございますね。")
        finally:
            tts_reading.configure(False)

    def test_passthrough_keeps_yomi(self):
        from generate_voice_passthrough import build_voice_turns
        y = [{"phrase": "上司の方", "surface": "方", "reading": "かた"}]
        out = build_voice_turns([{"speaker": "サトウ", "utterance": "上司の方ですね。", "yomi": y},
                                 {"speaker": "ナオコ", "utterance": "はい。"}])
        self.assertEqual(out[0]["yomi"], y)
        self.assertNotIn("yomi", out[1])
