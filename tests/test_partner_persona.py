"""metadata.partner_persona（ユーザ側の人物カード）のテスト。

カードが無い（config に無い・メタデータに無い）ときは Stage1 / Stage2 のプロンプトを変えない。
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generate_dialogues import build_dialogue_prompt                         # noqa: E402
from generate_metadata import DiversityTracker, attach_partner_card, build_user_prompt  # noqa: E402
from partner_persona import (TRAIT_PLAIN, TRAIT_TEXT, PartnerPool, _occupation_line,     # noqa: E402
                             talk_style, trait_percentiles, usable_facts)

MS = {"name": "イノウエ", "gender": "male", "age_range": "30代", "role": "大学の研究者"}


def _card(i, gender="female", dec="40代"):
    return {"card_id": f"c{gender}{dec}{i}", "gender": gender, "age": int(dec[:2]) + 3,
            "age_range": dec, "surname": "岡田", "prefecture": "宮城県", "marital": "既婚 (子供あり)",
            "occupation": "小売業 中堅", "hobbies": ["編み物", "釣り"],
            "talk_style": ["口数は少なめで、自分から話題を広げることはあまりない。答えは短めになりがち"],
            "trait_levels": {}, "facts": ["猫が好きである", "散歩が習慣である"],
            "source": {"nemotron_uuid": "u", "rpc_interlocutor_id": "AA"}}


def _pool(tmp_path, n=30):
    p = tmp_path / "pool.jsonl"
    with open(p, "w") as f:
        for g in ("male", "female"):
            for d in ("20代", "30代", "40代", "50代", "60代", "70代"):
                for i in range(n):
                    f.write(json.dumps(_card(i, g, d), ensure_ascii=False) + "\n")
    return str(p)


def _meta(card=None):
    m = {"id": "x", "scenario": "イノウエと岡田が初めて会う",
         "speakers": [{"name": "イノウエ", "gender": "male", "role": "大学の研究者", "personality": "穏やか"},
                      {"name": "オカダ", "gender": "female", "role": "スーパーの店員",
                       "personality": "控えめ", "age_range": "40代"}],
         "target_turns": 20, "main_speaker_index": 0}
    if card:
        m["speakers"][1]["persona_card"] = card
    return m


# --- 話し方の傾向・事実の選別 -------------------------------------------------

def _people(vals):
    keys = list(TRAIT_TEXT)
    return [{"personality": {k: v for k in keys}} for v in vals]


def test_talk_style_uses_only_extremes_and_caps_at_three():
    people = _people([1, 2, 3, 4, 5, 6, 7, 4, 4, 4])
    pct = trait_percentiles(people)
    lines, levels = talk_style({k: 1.0 for k in TRAIT_TEXT}, pct)
    assert len(lines) == 3 and all(v == "low" for v in levels.values())
    assert lines[0] in {t[0] for t in TRAIT_TEXT.values()}
    lines, levels = talk_style({k: 4.0 for k in TRAIT_TEXT}, pct)
    assert lines == [TRAIT_PLAIN] and set(levels.values()) == {"mid"}


def test_usable_facts_drops_identity_sentences():
    s = ["私は学生である。", "埼玉県出身である。", "猫が好きだ。", "夫と二人暮らし。", "45歳です。",
         "休みの日はできるだけ何もしたくない。"]
    assert usable_facts(s) == ["猫が好きだ。", "休みの日はできるだけ何もしたくない。"]


@pytest.mark.parametrize("occ,want", [
    ("小売業 中堅", "小売業の中堅企業で働いている"),
    ("小売業 中堅 (現在は引退)", "今は引退している（以前は小売業の中堅企業で働いていた）"),
    ("宿泊業 経営", "宿泊業の事業を経営している"),
    ("病院 中堅 経営", "病院の中堅企業の経営に携わっている"),
    ("卸売業 中小 経営 (現在は引退)", "今は引退している（以前は卸売業の中小企業の経営に携わっていた）"),
    ("郵便業 (現在は離職)", "今は仕事をしていない（以前は郵便業の仕事をしていた）"),
    ("学生", "学生"),
])
def test_occupation_line(occ, want):
    assert _occupation_line(occ) == want


def test_occupation_line_drops_exec_label_for_young_cards():
    assert _occupation_line("卸売業 中堅 経営", 21) == "卸売業の中堅企業で働いている"
    assert _occupation_line("卸売業 中堅 経営", 55) == "卸売業の中堅企業の経営に携わっている"


# --- PartnerPool ------------------------------------------------------------

def test_pool_respects_gender_weights_and_never_repeats(tmp_path):
    pool = PartnerPool({"pool_file": _pool(tmp_path),
                        "age_mix": {"male": {"20代": 1.0, "70代": 0.5}, "female": {"40代": 1.0}}})
    got = [pool.next_card("male") for _ in range(30)]
    assert all(c["gender"] == "male" for c in got)
    assert Counter(c["age_range"] for c in got) == {"20代": 20, "70代": 10}
    assert len({c["card_id"] for c in got}) == 30
    assert {pool.next_card("female")["age_range"] for _ in range(5)} == {"40代"}


def test_pool_skips_cards_already_in_metadata(tmp_path):
    pool = PartnerPool({"pool_file": _pool(tmp_path, n=2), "age_mix": {"female": {"40代": 1.0}}})
    pool.add_used(_card(0, "female", "40代"))
    assert pool.next_card("female")["card_id"] == "cfemale40代1"
    with pytest.raises(ValueError):
        pool.next_card("female")


# --- Stage1 -----------------------------------------------------------------

def _stage1(partner_persona=None, seed=0):
    random.seed(seed)
    tr = DiversityTracker(pair_mix={"男女": 1.0}, main_speaker=MS,
                          partner_roles={"会社員": {"weight": 1, "examples": "営業職"}},
                          partner_persona=partner_persona)
    cards = []
    return build_user_prompt(1, 16, 24, tr, num_speakers=2, card_out=cards), cards


def test_stage1_unchanged_without_partner_persona():
    random.seed(0)
    tr = DiversityTracker(pair_mix={"男女": 1.0}, main_speaker=MS,
                          partner_roles={"会社員": {"weight": 1, "examples": "営業職"}})
    old = build_user_prompt(1, 16, 24, tr, num_speakers=2)
    new, cards = _stage1()
    assert new == old and cards == []
    assert "2人目の role は「会社員」" in new


def test_stage1_card_replaces_partner_role(tmp_path):
    p, cards = _stage1({"pool_file": _pool(tmp_path)})
    assert len(cards) == 1 and cards[0]["gender"] == "female"   # 男女の組・主話者が男性
    assert "2人目は次の人物にしてください" in p
    assert "姓「岡田」の読みをカタカナ" in p
    assert f"age_range は \"{cards[0]['age_range']}\"" in p
    assert "2人目の role は「会社員」" not in p
    assert "年代・性格は自由に多様に" not in p


def test_attach_partner_card_fixes_age_and_drops_wrong_gender():
    card = _card(0, "female", "60代")
    ok, wrong = _meta(), _meta()
    wrong["speakers"][1]["gender"] = "male"
    out, n_age, n_drop = attach_partner_card([ok, wrong], card)
    assert out == [ok] and (n_age, n_drop) == (1, 1)
    assert ok["speakers"][1]["age_range"] == "60代"
    assert ok["speakers"][1]["persona_card"] is card


# --- Stage2 -----------------------------------------------------------------

def test_stage2_card_block_without_raw_card():
    p = build_dialogue_prompt(_meta(_card(0)), persona="- 性格・人柄: 穏やか")
    assert "persona_card" not in p and "rpc_interlocutor_id" not in p
    i_main = p.index("【イノウエの人物設定】")
    i_part = p.index("【オカダの人物設定】")
    i_quota = p.index("発話は1ターン1文だけにし")
    assert i_main < i_part < i_quota
    assert "「猫が好きである」" in p and "口数は少なめ" in p
    # カードを外した部分は、カードの無いメタデータのプロンプトと同じ
    base = build_dialogue_prompt(_meta(), persona="- 性格・人柄: 穏やか")
    block_start = p.index("\n\n【オカダの人物設定】")
    block_end = p.index("\n\n発話は1ターン1文だけにし")
    assert p[:block_start] + p[block_end:] == base


def test_stage2_does_not_mutate_metadata():
    m = _meta(_card(0))
    build_dialogue_prompt(m)
    assert "persona_card" in m["speakers"][1]


def test_stage1_same_generation_card_is_always_keigo(tmp_path):
    pool = _pool(tmp_path)
    random.seed(0)
    tr = DiversityTracker(pair_mix={"男女": 1.0}, main_speaker=MS, style_mix={"混合": 1.0},
                          partner_persona={"pool_file": pool,
                                           "age_mix": {"female": {"40代": 1.0, "70代": 1.0}}})
    got = {}
    for _ in range(4):
        cards = []
        p = build_user_prompt(1, 16, 24, tr, num_speakers=2, card_out=cards)
        got.setdefault(cards[0]["age_range"], set()).add(
            "敬語" if "speech_style は「敬語」" in p else "混合" if "speech_style は「混合」" in p else "?")
    assert got == {"40代": {"敬語"}, "70代": {"混合"}}
    assert tr.style_issued == Counter({"混合": 2})     # 敬語の固定は発注数に数えない
