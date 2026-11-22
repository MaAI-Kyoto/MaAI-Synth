"""metadata.main_speaker.persona_file（主話者のペルソナ）のテスト。

ペルソナが無い・空のときは Stage1 / Stage2 のプロンプトを一字も変えない。
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generate_dialogues import build_dialogue_prompt                        # noqa: E402
from generate_metadata import DiversityTracker, build_user_prompt           # noqa: E402
from persona import load_persona, render_persona                            # noqa: E402

SPEC = {"name": "イノウエ", "gender": "male", "age_range": "30代", "role": "担当者"}
FILLED = "personality: 落ち着いていて面倒見がよい。\nspeech_style: |\n  結論から短く言う。\n  語尾はやわらかめ。\n"


def _write(tmp_path, text):
    p = tmp_path / "p.yaml"
    p.write_text(text)
    return str(p)


def _meta():
    return {"id": "x", "scenario": "イノウエが電話で応対する",
            "speakers": [{"name": "イノウエ", "gender": "male", "role": "担当者",
                          "personality": "明るい"},
                         {"name": "タナカ", "gender": "female", "role": "客"}],
            "target_turns": 20, "main_speaker_index": 0}


def _stage1(ms, seed=0):
    import random
    random.seed(seed)
    tr = DiversityTracker(pair_mix={"男女": 1.0}, main_speaker=ms)
    return build_user_prompt(1, 16, 24, tr, num_speakers=2)


# --- persona.py -------------------------------------------------------------

def test_shipped_personas_load():
    # 同梱のファイルは読めること（未知の項目・YAML の崩れが無い）
    files = sorted((ROOT / "personas").glob("*.yaml"))
    assert {f.stem for f in files} >= {"inoue_ft_v5", "inoue_ft_v6_002", "inoue_qa_1"}
    for f in files:
        load_persona({"persona_file": str(f)})


def test_empty_template_means_no_persona(tmp_path):
    # 記入前のひな形はペルソナ無しと同じ（config に書いてあってもプロンプトを変えない）
    tpl = "\n".join(f"{k}:" for k in ("personality", "speech_style", "background",
                                       "habits", "avoid"))
    assert load_persona({"persona_file": _write(tmp_path, tpl)}) == ""


def test_render_joins_lines_and_skips_empty(tmp_path):
    out = load_persona({"persona_file": _write(tmp_path, FILLED + "avoid:\n")})
    assert out == ("- 性格・人柄: 落ち着いていて面倒見がよい。\n"
                   "- 話し方・口調: 結論から短く言う。語尾はやわらかめ。")


def test_ascii_words_across_lines_keep_a_space():
    assert render_persona({"habits": "uses VAP\nmodels"}) == "- よく使う言い回し・癖: uses VAP models"


def test_unknown_key_is_an_error():
    with pytest.raises(ValueError):
        render_persona({"personaliti": "typo"})


def test_missing_file_is_an_error():
    with pytest.raises(FileNotFoundError):
        load_persona({"persona_file": "personas/__nope__.yaml"})


# --- Stage1 -----------------------------------------------------------------

def test_stage1_unchanged_when_persona_empty(tmp_path):
    empty = dict(SPEC, persona_file=_write(tmp_path, "personality:\n"))
    assert _stage1(empty) == _stage1(SPEC)


def test_stage1_replaces_vary_line_with_persona(tmp_path):
    base = _stage1(SPEC)
    p = _stage1(dict(SPEC, persona_file=_write(tmp_path, FILLED)))
    assert "性格・話し方・勤め先（setting）・相手との関係性は" in base
    assert "性格・話し方・勤め先（setting）・相手との関係性は" not in p
    assert "【1人目の人物設定】\n- 性格・人柄: 落ち着いていて面倒見がよい。" in p
    assert "勤め先（setting）・相手との関係性・用件は、毎回まったく違う" in p
    # 開始者の指定・JSON 指示は末尾のまま
    assert p.endswith("JSON配列のみを出力してください（マークダウンのコードブロックは不要です）。")


def test_stage1_without_role_keeps_role_varying(tmp_path):
    ms = {k: v for k, v in SPEC.items() if k != "role"}
    p = _stage1(dict(ms, persona_file=_write(tmp_path, FILLED)))
    assert "役割・場面・相手との関係性は、毎回まったく違う" in p


# --- Stage2 -----------------------------------------------------------------

def test_stage2_unchanged_without_persona():
    assert build_dialogue_prompt(_meta(), persona="") == build_dialogue_prompt(_meta())


def test_stage2_persona_after_metadata_before_quotas():
    p = build_dialogue_prompt(_meta(), persona="- 性格・人柄: 穏やか")
    i_json = p.index('"target_turns"')
    i_per = p.index("【イノウエの人物設定】")
    i_quota = p.index("発話は1ターン1文だけにし")
    assert i_json < i_per < i_quota
    assert "- 性格・人柄: 穏やか" in p
    assert "main_speaker_index" not in p


def test_stage2_ignores_persona_without_main_speaker_index():
    m = _meta()
    m.pop("main_speaker_index")
    assert build_dialogue_prompt(m, persona="- 性格・人柄: 穏やか") == build_dialogue_prompt(m)


def test_stage1_persona_vary_override(tmp_path):
    ms = dict(SPEC, role="大学の研究者", persona_file=_write(tmp_path, FILLED),
              persona_vary="出会った場面・相手・話題")
    p = _stage1(ms)
    assert 'role を "大学の研究者"' in p
    assert "ただし出会った場面・相手・話題は、毎回まったく違う" in p
    assert "勤め先（setting）" not in p

