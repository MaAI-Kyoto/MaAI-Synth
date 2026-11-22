"""接客の担当者が用件を復唱するときの「ね」（generate_dialogues.add_readback_ne）。

「配送先の変更でございます。」のような言い切りは、確認ではなく事実の断定になって
耳で聞くと事務的に響く（ユーザー指摘 2026-09-19）。復唱は相手に向けて確かめる発話
なので「〜でございますね。」が正しい。ここでは、足す対象の見分けを検査する。
"""

import unittest

from generate_dialogues import add_customer_echo_ne, add_readback_ne

STAFF_META = {"speakers": [{"name": "サトウ", "role": "担当者"},
                           {"name": "ナオコ", "role": "客"}]}


def run(*utterances, metadata=STAFF_META, window=8, start="ナオコ"):
    """客→担当者→客→… の順に並べたターンを作り、直したあとの発話を返す。"""
    names = ("ナオコ", "サトウ") if start == "ナオコ" else ("サトウ", "ナオコ")
    turns = [{"speaker": names[i % 2], "utterance": u}
             for i, u in enumerate(utterances)]
    n = add_readback_ne(turns, metadata, window)
    return [t["utterance"] for t in turns], n


class ReadbackNeTests(unittest.TestCase):
    def test_adds_ne_to_the_opening_readback(self):
        got, n = run("あの、配送を止めていただきたいんです。",
                     "配送の一時休止をご希望でございます。承知いたしました。")
        self.assertEqual(got[1], "配送の一時休止をご希望でございますね。承知いたしました。")
        self.assertEqual(n, 1)

    def test_adds_ne_to_plain_desu_and_nodesu(self):
        got, _ = run("あの、食洗機を自分で設置したんですけど。",
                     "ご自身で設置されたんです。何かお困りでしょうか？")
        self.assertEqual(got[1], "ご自身で設置されたんですね。何かお困りでしょうか？")
        got, _ = run("あの、アレルギー対応の食事はありますか？",
                     "アレルギー対応のお食事です。かしこまりました。")
        self.assertEqual(got[1], "アレルギー対応のお食事ですね。かしこまりました。")

    def test_incomplete_desu_question_becomes_desune(self):
        # 「です？」は不完全で馴れ馴れしい聞き返し（プロンプトでも禁止している形）。
        got, _ = run("はい。保温機能が、うまく働いていないみたいで。",
                     "保温の調子が、良くないということです？")
        self.assertEqual(got[1], "保温の調子が、良くないということですね？")

    def test_missing_period_is_completed(self):
        got, _ = run("あの、配送先を変えたいんです。", "配送先の変更でございます")
        self.assertEqual(got[1], "配送先の変更でございますね。")

    def test_skips_a_short_lead_sentence(self):
        got, _ = run("あの、プランを変更したいんですけど。",
                     "いらっしゃいませ。プランの変更をご希望でございます。")
        self.assertEqual(got[1], "いらっしゃいませ。プランの変更をご希望でございますね。")

    def test_does_not_touch_affirmations_and_recommendations(self):
        for u in ("さようでございます。19時を過ぎますとご入場できません。",
                  "もちろんです。綺麗な袋に入れてお渡しいたします。",
                  "はい、可能でございます。ご予約のお日にちを伺えますか？",
                  "出張修理がおすすめでございます。",
                  "かしこまりました。すぐにお調べいたします。"):
            got, n = run("あの、少し伺いたいんですけど。", u)
            self.assertEqual(got[1], u)
            self.assertEqual(n, 0)

    def test_does_not_touch_self_introduction(self):
        got, n = run("もしもし。", "サポート窓口のサトウでございます。")
        self.assertEqual(got[1], "サポート窓口のサトウでございます。")
        self.assertEqual(n, 0)

    def test_existing_particle_is_left_alone(self):
        got, n = run("あの、配送先を変えたいんです。", "配送先の変更ですね。承知いたしました。")
        self.assertEqual(n, 0)
        got, n = run("あの、これは使えますか？", "お使いいただけますか。")
        self.assertEqual(n, 0)

    def test_after_the_opening_a_readback_marker_is_required(self):
        # 序盤を過ぎたら、情報提供（値段の案内）には足さない。
        late = ["客の発話。", "はい、お調べいたします。"] * 5
        got, n = run(*(late + ["お値段はいくらですか？",
                               "価格は、税込で、八千五百円でございます。"]))
        self.assertEqual(got[-1], "価格は、税込で、八千五百円でございます。")
        self.assertEqual(n, 0)
        # 復唱の印（「〜についてです」）が付いていれば序盤を過ぎても直す。
        got, n = run(*(late + ["利用できる時間は、どうなってますか？",
                               "利用時間についてです。2時間限定となっております。"]))
        self.assertEqual(got[-1], "利用時間についてですね。2時間限定となっております。")
        self.assertEqual(n, 1)

    def test_own_previous_turn_is_not_a_readback(self):
        turns = [{"speaker": "サトウ", "utterance": "少々お待ちくださいませ。"},
                 {"speaker": "サトウ", "utterance": "配送先の変更でございます。"}]
        self.assertEqual(add_readback_ne(turns, STAFF_META, 8), 0)

    def test_inert_without_a_staff_role(self):
        meta = {"speakers": [{"name": "ナオコ", "role": "相談者"},
                             {"name": "サトウ", "role": "アドバイザー"}]}
        got, n = run("あの、配送先を変えたいんです。", "配送先の変更でございます。",
                     metadata=meta)
        self.assertEqual(n, 0)

    def test_window_zero_disables_the_rule(self):
        got, n = run("あの、配送先を変えたいんです。", "配送先の変更でございます。",
                     window=0)
        self.assertEqual(n, 0)


if __name__ == "__main__":
    unittest.main()


class CustomerEchoNeTests(unittest.TestCase):
    """客が担当者の案内を受け止める言い切り（add_customer_echo_ne）。"""

    def run_pair(self, staff_u, cust_u, metadata=STAFF_META, extra=()):
        turns = [{"speaker": "サトウ", "utterance": staff_u},
                 {"speaker": "ナオコ", "utterance": cust_u}]
        turns += [{"speaker": "ナオコ", "utterance": u} for u in extra]
        n = add_customer_echo_ne(turns, metadata)
        return turns[1]["utterance"], n

    def test_noun_ending_echo_gets_desune(self):
        got, n = self.run_pair("はい。明日、10日の午前中には、到着する予定でございます。",
                               "10日の午前中。")
        self.assertEqual(got, "10日の午前中ですね。")
        self.assertEqual(n, 1)

    def test_echo_is_fixed_before_the_rest_of_the_turn(self):
        got, _ = self.run_pair("宿泊日の前日、15時までにお届けできます。",
                               "前日の15時。助かります。")
        self.assertEqual(got, "前日の15時ですね。助かります。")

    def test_ndesu_echo_gets_ne(self):
        got, _ = self.run_pair("そちらには、12歳以下のお子様向けの、キッズプレートがございます。",
                               "キッズプレートがあるんです。")
        self.assertEqual(got, "キッズプレートがあるんですね。")

    def test_own_situation_is_left_alone(self):
        # 語が重なっていても、自分の希望・事情の説明には付けない。
        got, n = self.run_pair("お部屋の変更を承ります。",
                               "あの、予約している部屋を、広いのに変えたいんです。")
        self.assertEqual(n, 0)
        got, n = self.run_pair("お部屋の広さについて承知いたしました。",
                               "実は、急に友達が増えちゃって、今の部屋だと狭いんです。")
        self.assertEqual(n, 0)

    def test_answer_to_a_question_is_left_alone(self):
        for staff in ("配送停止は、来月の分、1回分ということでよろしいでしょうか？",
                      "来月分の配送を、一時的に停止されたいということですね。",
                      "ご不便をおかけして、それはお困りでしたね。"):
            got, n = self.run_pair(staff, "そうなんです。配送を止めてください。")
            self.assertEqual(n, 0, staff)

    def test_unrelated_turn_is_not_an_echo(self):
        got, n = self.run_pair("お電話ありがとうございます。窓口のサトウが承ります。",
                               "あの、掃除機のバッテリーが弱くなってきまして。")
        self.assertEqual(n, 0)

    def test_casual_speaker_gets_bare_ne(self):
        meta = {"speech_style": "混合",
                "speakers": [{"name": "サトウ", "role": "担当者"},
                             {"name": "ナオコ", "role": "客"}]}
        got, _ = self.run_pair("お値段は、1セット500円でございます。", "500円。",
                               metadata=meta,
                               extra=("うん。", "それ、ちょうだい。", "ありがとう。"))
        self.assertEqual(got, "500円ね。")

    def test_polite_style_keeps_desune_even_for_a_short_speaker(self):
        got, _ = self.run_pair("お値段は、1セット500円でございます。", "500円。")
        self.assertEqual(got, "500円ですね。")

    def test_only_one_sentence_per_turn(self):
        got, n = self.run_pair("紅葉が、今とても綺麗な時期でございます。",
                               "紅葉。やっぱり、今の紅葉の時期がいいんです。")
        self.assertEqual(got, "紅葉ですね。やっぱり、今の紅葉の時期がいいんです。")
        self.assertEqual(n, 1)

    def test_inert_without_a_staff_role(self):
        meta = {"speakers": [{"name": "サトウ", "role": "アドバイザー"},
                             {"name": "ナオコ", "role": "相談者"}]}
        got, n = self.run_pair("明日の午前中に届く予定です。", "明日の午前中。",
                               metadata=meta)
        self.assertEqual(n, 0)

    def test_disabled_by_flag(self):
        turns = [{"speaker": "サトウ", "utterance": "明日の午前中に届く予定でございます。"},
                 {"speaker": "ナオコ", "utterance": "明日の午前中。"}]
        self.assertEqual(add_customer_echo_ne(turns, STAFF_META, False), 0)


class ReadbackEchoTests(unittest.TestCase):
    """序盤（window）を過ぎた復唱を、客の語の繰り返しで見分ける（_is_readback_echo）。

    ユーザー指摘 2026-09-22:「ブライダルコースです。確認いたしました。」がターン8で
    window の外に落ち、復唱の印も無いので言い切りのまま残っていた。
    """

    LATE = ["客の発話。", "はい、お調べいたします。"] * 5      # 10ターン分の前置き

    def run_late(self, cust_u, staff_u):
        got, n = run(*(self.LATE + [cust_u, staff_u]))
        return got[-1], n

    def test_noun_echo_gets_ne_after_the_window(self):
        got, n = self.run_late("えっと、スタンダードの、ブライダルコースをお願いしています。",
                               "ブライダルコースです。確認いたしました。")
        self.assertEqual(got, "ブライダルコースですね。確認いたしました。")
        self.assertEqual(n, 1)

    def test_numbers_dates_and_counts_are_readbacks(self):
        for cust, staff, want in (
                ("えっと、090に、1234、5678です。",
                 "090の、1234、5678でございます。少々お待ちください。",
                 "090の、1234、5678でございますね。少々お待ちください。"),
                ("えっと、来週の、土曜日の午後で、お願いしたいです。",
                 "承知いたしました。来週の土曜日、午後です。お席を確保いたします。",
                 "承知いたしました。来週の土曜日、午後ですね。お席を確保いたします。"),
                # 依頼の「お願いできますか？」は問いではないので、その次は復唱
                ("あの、定番のカレーパンを、あと5個ほど、お願いできますか？",
                 "カレーパンを5個です。承知いたしました。",
                 "カレーパンを5個ですね。承知いたしました。"),
                ("一人で来ました。",
                 "お一人でいらしたのです。では、ルートをご案内いたします。",
                 "お一人でいらしたのですね。では、ルートをご案内いたします。")):
            got, _ = self.run_late(cust, staff)
            self.assertEqual(got, want)

    def test_answers_to_questions_are_left_alone(self):
        for cust, staff in (
                ("メールですか？", "はい。件名に、購入完了という言葉が入っているメールでございます。"),
                ("来月5日？", "はい。来月5日の予定でございます。"),
                # 客の短い体言止めは聞き返し。その次は答え。
                ("ログインボタン。", "はい。画面の、下の方にある、大きなボタンでございます。"),
                ("お店の名前を教えていただけますか？", "お店の名前は、菓子処はるかという名前でございます。")):
            got, n = self.run_late(cust, staff)
            self.assertEqual(got, staff)
            self.assertEqual(n, 0)

    def test_evaluations_and_reports_are_left_alone(self):
        # 語が重なっても、形容詞・「はず」「みたい」の述語は評価・報告
        for cust, staff in (
                ("あはは。お小遣い、全部、使っちゃうよ。", "大切なお小遣いで、プレゼントを選ぶなんて、本当に、優しいです。"),
                ("あー、よかった。本当に、安心しました。", "本当によかったです。見つかって私も安心いたしました。"),
                ("伝票、ついてましたっけ。", "はい。箱の中に、ピンク色の伝票が入っているはずです。"),
                ("ベージュがいいです。", "ベージュの在庫を確認したところ、まだ確保可能でございます。")):
            got, n = self.run_late(cust, staff)
            self.assertEqual(got, staff)
            self.assertEqual(n, 0)

    def test_unrelated_statement_is_not_an_echo(self):
        got, n = self.run_late("あの、少し伺いたいんですけど。",
                               "価格は、税込で、八千五百円でございます。")
        self.assertEqual(n, 0)


class ThinNeProtectTests(unittest.TestCase):
    """thin_ne は担当者の復唱の「ね」を落とさない（readback_sites を protect に渡す）。

    2026-09-22 の実測: モデルは復唱をほぼ「〜ですね」で書くのに、thin_ne が上限超過ぶんと
    して落とし（raw の復唱39件中15件）、add_readback_ne が序盤ぶんしか戻せなかった。
    """

    def test_readback_ne_survives_thinning(self):
        from generate_dialogues import readback_sites, thin_ne
        turns = [{"speaker": "ナオコ", "utterance": "あの、来月の披露宴の、コース料理のことで、相談したいんですけど。"},
                 {"speaker": "サトウ", "utterance": "お料理の内容についてでございますね。承知いたしました。"},
                 {"speaker": "ナオコ", "utterance": "えっと、スタンダードの、ブライダルコースをお願いしています。"},
                 {"speaker": "サトウ", "utterance": "ブライダルコースでございますね。確認いたしました。"},
                 {"speaker": "ナオコ", "utterance": "海老なんですけど、かなり重いアレルギーの方で。"},
                 {"speaker": "サトウ", "utterance": "白身魚に変更すると、お味のバランスも崩れずにお楽しみいただけますね。"}]
        protect = {(i, j) for i, j, *_ in readback_sites(turns, STAFF_META, 8)}
        self.assertEqual(protect, {(1, 0), (3, 0)})
        removed = thin_ne(turns, 0.01, False, protect)       # 上限0＝落とせるだけ落とす
        self.assertEqual(removed, 1)
        self.assertEqual(turns[1]["utterance"], "お料理の内容についてでございますね。承知いたしました。")
        self.assertEqual(turns[3]["utterance"], "ブライダルコースでございますね。確認いたしました。")
        self.assertEqual(turns[5]["utterance"], "白身魚に変更すると、お味のバランスも崩れずにお楽しみいただけます。")

    def test_without_protect_the_readback_is_thinned(self):
        from generate_dialogues import thin_ne
        turns = [{"speaker": "ナオコ", "utterance": "えっと、スタンダードの、ブライダルコースをお願いしています。"},
                 {"speaker": "サトウ", "utterance": "ブライダルコースでございますね。確認いたしました。"}]
        self.assertEqual(thin_ne(turns, 0.01, False), 1)
        self.assertEqual(turns[1]["utterance"], "ブライダルコースでございます。確認いたしました。")


class StaffTailTests(unittest.TestCase):
    """担当者の文末（2026-09-22 2回目の指摘）。

    thin_ne は接客では客の文だけを間引く（担当者の「ね」は復唱・受け止め・予告の機能語）。
    「〜です？」は「〜ですか？」に、「〜くださいね。」は「〜ください。」に直す。
    """

    def test_thin_ne_only_customer_sentences(self):
        from generate_dialogues import sentence_ids, staff_names, thin_ne
        turns = [{"speaker": "ナオコ", "utterance": "先週も、同じ商品を、こちらで買ったんですけどね。"},
                 {"speaker": "サトウ", "utterance": "12345番のシオリ様ですね。ただいま、履歴をお調べいたしますね。"},
                 {"speaker": "ナオコ", "utterance": "紙で、ファイルにまとめて、保管したいタイプなんですよね。"},
                 {"speaker": "サトウ", "utterance": "几帳面でいらっしゃるんですね。お気持ちは、よく分かります。"},
                 {"speaker": "ナオコ", "utterance": "それなら、来週の水曜日に、また伺いますね。"}]
        staff = staff_names(STAFF_META)
        only = sentence_ids(turns, {"ナオコ"})
        self.assertEqual(staff, {"サトウ"})
        removed = thin_ne(turns, 0.01, False, set(), only)     # 上限0＝落とせるだけ落とす
        self.assertGreaterEqual(removed, 1)
        self.assertEqual(turns[1]["utterance"], "12345番のシオリ様ですね。ただいま、履歴をお調べいたしますね。")
        self.assertEqual(turns[3]["utterance"], "几帳面でいらっしゃるんですね。お気持ちは、よく分かります。")
        self.assertEqual(turns[4]["utterance"], "それなら、来週の水曜日に、また伺います。")

    def test_name_and_number_readback_is_protected(self):
        # 「12345番」を1語に切ると客の「12345で」と一致しなかった（語は文字種ごとに切る）
        from generate_dialogues import readback_sites
        late = ["客の発話。", "はい、お調べいたします。"] * 5
        turns = [{"speaker": ("ナオコ", "サトウ")[i % 2], "utterance": u}
                 for i, u in enumerate(late + ["えっと、番号は、12345で、名前はシオリです。",
                                              "12345番のシオリ様ですね。ただいま、履歴をお調べいたしますね。"])]
        sites = readback_sites(turns, STAFF_META, 8)
        self.assertEqual([(i, j, ne) for i, j, _, _, ne in sites], [(11, 0, True)])

    def test_staff_desu_question_and_kudasai_ne(self):
        from generate_dialogues import fix_staff_tails
        turns = [{"speaker": "ナオコ", "utterance": "トモキです。こちらが、会員証になります。"},
                 {"speaker": "サトウ", "utterance": "ありがとうございます。現在は、スタンダードプランをご利用です？"},
                 {"speaker": "ナオコ", "utterance": "紙の明細は、もう来ないです？"},
                 {"speaker": "サトウ", "utterance": "ご不明な点がございましたら、お気軽にご連絡くださいね。"},
                 {"speaker": "サトウ", "utterance": "本日ですね？少々お待ちくださいませ。よろしいですか？"}]
        n = fix_staff_tails(turns, STAFF_META)
        self.assertEqual(n, 2)
        self.assertEqual(turns[1]["utterance"], "ありがとうございます。現在は、スタンダードプランをご利用ですか？")
        self.assertEqual(turns[2]["utterance"], "紙の明細は、もう来ないです？")      # 客は触らない
        self.assertEqual(turns[3]["utterance"], "ご不明な点がございましたら、お気軽にご連絡ください。")
        self.assertEqual(turns[4]["utterance"], "本日ですね？少々お待ちくださいませ。よろしいですか？")

    def test_inert_without_a_staff_role(self):
        from generate_dialogues import fix_staff_tails
        meta = {"speakers": [{"name": "ナオコ", "role": "相談者"}, {"name": "サトウ", "role": "アドバイザー"}]}
        turns = [{"speaker": "サトウ", "utterance": "お気軽にご連絡くださいね。"}]
        self.assertEqual(fix_staff_tails(turns, meta), 0)
