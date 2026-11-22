"""metadata.main_speaker.role（接客の「担当者」固定）のテスト。

role を書かない既存の inoue 版（初対面・汎用）の挙動は変えない。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate_metadata import (                                            # noqa: E402
    DiversityTracker,
    build_user_prompt,
    enforce_main_speaker,
)

SPEC = {"name": "イノウエ", "gender": "male", "age_range": "30代", "role": "担当者"}


def _meta(*speakers, scenario="", flow=None):
    return {"scenario": scenario, "conversation_flow": list(flow or []),
            "speakers": [dict(s) for s in speakers]}


def _agent(name="サトウ", gender="male", age="30代", role="担当者"):
    return {"name": name, "gender": gender, "age_range": age, "role": role}


def _customer(name="タナカ", gender="female", age="50代"):
    return {"name": name, "gender": gender, "age_range": age, "role": "客"}


# --- enforce_main_speaker ------------------------------------------------------

def test_agent_is_moved_to_index0_and_renamed_in_text():
    m = _meta(_customer(), _agent(),
              scenario="サトウが電話に出て、タナカの用件を聞く",
              flow=["サトウが電話に出て用件を尋ね、タナカが配送日の変更を伝える"])
    out, n_swap, n_age, n_name, n_drop = enforce_main_speaker([m], SPEC)
    assert (n_swap, n_age, n_name, n_drop) == (1, 0, 1, 0)
    assert out[0]["speakers"][0] == _agent("イノウエ")
    assert out[0]["speakers"][1]["role"] == "客"
    assert out[0]["main_speaker_index"] == 0
    assert out[0]["scenario"] == "イノウエが電話に出て、タナカの用件を聞く"
    assert out[0]["conversation_flow"][0].startswith("イノウエが電話に出て")


def test_role_wins_over_a_customer_that_matches_gender_and_age():
    """role 無しなら「30代男性の客」が主話者に選ばれてしまう組。担当者を採る。"""
    m = _meta(_customer("ケンタ", "male", "30代"), _agent("ミサキ", "female", "30代"))
    out, *_ = enforce_main_speaker([m], {k: v for k, v in SPEC.items() if k != "role"})
    assert out[0]["speakers"][0]["name"] == "イノウエ" and out[0]["speakers"][0]["role"] == "客"
    # role 付きだと担当者の性別が合わないので破棄（客を担当者に読み替えない）
    m = _meta(_customer("ケンタ", "male", "30代"), _agent("ミサキ", "female", "30代"))
    out, n_swap, n_age, n_name, n_drop = enforce_main_speaker([m], SPEC)
    assert out == [] and n_drop == 1


def test_agent_age_is_rewritten_but_gender_mismatch_drops():
    m = _meta(_agent("サトウ", "male", "50代"), _customer())
    out, n_swap, n_age, n_name, n_drop = enforce_main_speaker([m], SPEC)
    assert (n_swap, n_age, n_drop) == (0, 1, 0)
    assert out[0]["speakers"][0]["age_range"] == "30代"

    m = _meta(_agent("ミサキ", "female", "30代"), _customer("ケンタ", "male", "30代"))
    out, *_, n_drop = enforce_main_speaker([m], SPEC)
    assert out == [] and n_drop == 1


def test_missing_or_duplicated_role_drops():
    m = _meta(_agent(role="受付担当"), _customer())          # 表記揺れは一致しない
    assert enforce_main_speaker([m], SPEC)[0] == []
    m = _meta(_agent("サトウ"), _agent("スズキ"))              # 担当者が2人
    assert enforce_main_speaker([m], SPEC)[0] == []


def test_customer_name_overlapping_the_persona_drops():
    m = _meta(_agent("サトウ"), _customer("イノウエダ"))
    out, *_, n_drop = enforce_main_speaker([m], SPEC)
    assert out == [] and n_drop == 1


# --- build_user_prompt --------------------------------------------------------

def _tail(spec):
    t = DiversityTracker(main_speaker=spec,
                         pair_mix={"男女": 0.5, "男性同士": 0.5, "女性同士": 0.0})
    return build_user_prompt(1, 12, 30, t, num_speakers=2)


def test_prompt_fixes_the_role_and_varies_everything_else():
    p = _tail(SPEC)
    assert 'role を "担当者"、gender を "male"、age_range を "30代"' in p
    assert 'name は必ず "イノウエ"' in p
    assert "勤め先（setting）" in p
    assert "1人目の役割・性格・関係性・話し方は" not in p       # 役割は変えさせない
    assert "年代・性格は自由に多様に" in p and "年代・役割・性格は自由" not in p


def test_prompt_without_role_is_unchanged():
    p = _tail({k: v for k, v in SPEC.items() if k != "role"})
    assert "role を" not in p
    assert ("ただし1人目の役割・性格・関係性・話し方は、これらの制約とは無関係に"
            "毎回まったく違うものにしてください（同じような人物像を繰り返さない）。\n"
            "2人目の性別は上の組み合わせの指定に従い、年代・役割・性格は自由に多様にしてください。\n\n"
            "この対話は speakers の1人目から話し始めてください。") in p
