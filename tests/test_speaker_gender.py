"""話者の性別の記録と、声の割り当てへの反映のテスト。"""

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speaker_gender import infer_gender, normalize_gender, resolve_gender  # noqa: E402
from generate_metadata import (                                           # noqa: E402
    PAIR_MIX_TARGET,
    DiversityTracker,
    build_user_prompt,
    fill_speaker_genders,
)
from generate_voice import build_speaker_profiles                         # noqa: E402
from generate_dialogues import fix_gendered_first_person                   # noqa: E402


# --- 正規化 -----------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("male", "male"), ("Female", "female"), ("  MALE ", "male"),
    ("男性", "male"), ("女", "female"),
    ("unknown", None), ("", None), (None, None), (1, None), (["male"], None),
])
def test_normalize_gender(raw, expected):
    assert normalize_gender(raw) == expected


# --- 推定（フォールバック）--------------------------------------------------

def test_infers_from_role_keywords():
    assert infer_gender("ヨシ", "シングルマザー") == "female"
    assert infer_gender("カツヤ", "3人の子の父") == "male"


def test_infers_from_katakana_name_suffix():
    # このパイプラインの話者名はカタカナなので、語尾で見る必要がある
    assert infer_gender("ユウコ") == "female"
    assert infer_gender("ミホ") == "female"
    assert infer_gender("コウジ") == "male"
    assert infer_gender("ユウタ") == "male"


def test_returns_none_for_ambiguous_names():
    """両性に散らばる語尾は None にする（誤った性別は声を確実に外すため）。"""
    for name in ("ミキ", "ユウキ", "タカシ", "ヨシ", "ユウ", "サヤ", "カツヤ", "ナオト"):
        assert infer_gender(name) is None, name


def test_ambiguous_role_is_not_decided():
    """「祖父母」は母にも父にも当たる。片方だけ見ると female になってしまう。"""
    assert infer_gender("サトシ", "祖父母") is None
    assert infer_gender("サトシ", "両親の介護をしている夫") == "male"


def test_role_wins_over_name_suffix():
    # 語尾だけなら female になる名前でも、役割がはっきりしていればそちら
    assert infer_gender("ミノル", "2児の父") == "male"


def test_resolve_gender_prefers_explicit_value():
    assert resolve_gender("female", "コウジ", None) == "female"
    assert resolve_gender("わからない", "コウジ", None) == "male"


# --- Stage 1: 記録 ----------------------------------------------------------

def _meta(*speakers):
    return {"speakers": [dict(s) for s in speakers]}


def test_fill_speaker_genders_normalizes_and_reports():
    metas = [_meta({"name": "ミキ", "gender": "Female", "role": "学生"},
                   {"name": "タカシ", "gender": "男性", "role": "大学職員"})]
    n_bad, n_unknown = fill_speaker_genders(metas)
    assert [s["gender"] for s in metas[0]["speakers"]] == ["female", "male"]
    assert (n_bad, n_unknown) == (0, 0)


def test_fill_speaker_genders_fills_missing_from_name_and_role():
    metas = [_meta({"name": "ユウコ", "role": "大学院生"},
                   {"name": "ヨシ", "gender": None, "role": "シングルマザー"})]
    n_bad, n_unknown = fill_speaker_genders(metas)
    assert [s["gender"] for s in metas[0]["speakers"]] == ["female", "female"]
    assert (n_bad, n_unknown) == (2, 0)


def test_fill_speaker_genders_leaves_none_when_undecidable():
    metas = [_meta({"name": "ミキ", "role": "学生"})]
    n_bad, n_unknown = fill_speaker_genders(metas)
    # キーは必ず存在させる（下流が meta.get("gender") を見るため）
    assert metas[0]["speakers"][0]["gender"] is None
    assert (n_bad, n_unknown) == (1, 1)


def test_fill_speaker_genders_tolerates_broken_entries():
    metas = [{"speakers": ["ミキ", None]}, "こわれた行", {}]
    assert fill_speaker_genders(metas) == (0, 0)


# --- Stage 1: 性別の組み合わせの散らし方 ------------------------------------

def _ask(tracker, batch_size=1):
    """build_user_prompt が指定してくる組み合わせのラベルを返す。"""
    prompt = build_user_prompt(batch_size, 24, 38, tracker, num_speakers=2)
    return [label for label in PAIR_MIX_TARGET if label in prompt.rsplit("\n\n", 2)[-2]]


def test_pair_quota_converges_to_the_target_mix():
    t = DiversityTracker()
    got = Counter()
    for _ in range(200):
        label = next(iter(t.pair_quota(1)))
        got[label] += 1
        t.pair_pending[label] -= 1          # 発注ぶんを実績に置き換える
        t.pair_counts[label] += 1
    for label, ratio in PAIR_MIX_TARGET.items():
        assert got[label] == pytest.approx(200 * ratio, abs=1)


def test_pair_quota_does_not_repeat_while_requests_are_in_flight():
    """並行リクエストに同じ組み合わせを配らない（発注時に pending に積む）。"""
    t = DiversityTracker()
    labels = [next(iter(t.pair_quota(1))) for _ in range(4)]
    assert len(set(labels)) == 3            # 3種すべてが出る（男女が2件）


def test_pair_quota_corrects_a_skewed_history():
    t = DiversityTracker()
    t.pair_counts["男女"] = 100             # 既存データが男女に偏っている
    assert next(iter(t.pair_quota(1))) != "男女"


def test_pair_quota_is_written_into_the_prompt():
    t = DiversityTracker()
    assert _ask(t, batch_size=1)            # 1件なら組み合わせを1つ指定
    spec = build_user_prompt(4, 24, 38, DiversityTracker(), num_speakers=2)
    assert "4件のうち" in spec


def test_pair_quota_is_skipped_when_not_two_speakers():
    prompt = build_user_prompt(1, 24, 38, DiversityTracker(), num_speakers=3)
    assert not any(label in prompt for label in PAIR_MIX_TARGET)


def test_tracker_counts_pairs_from_existing_metadata():
    t = DiversityTracker()
    t.add(_meta({"name": "ア", "gender": "male"}, {"name": "イ", "gender": "male"}))
    t.add(_meta({"name": "ウ", "gender": "female"}, {"name": "エ", "gender": "male"}))
    t.add(_meta({"name": "オ", "gender": None}, {"name": "カ", "gender": "male"}))
    assert dict(t.pair_counts) == {"男性同士": 1, "男女": 1}


# --- Stage 3: プロフィール --------------------------------------------------

def test_profiles_prefer_metadata_gender_over_stage2_guess():
    meta = _meta({"name": "ヨシ", "gender": "female", "role": "シングルマザー", "age_range": "50代"})
    # Stage 2 が名前から male と誤推定しても metadata が勝つ
    profiles = build_speaker_profiles(meta, [{"name": "ヨシ", "gender": "male"}])
    assert profiles == [{"name": "ヨシ", "gender": "female", "age_range": "50代"}]


def test_profiles_fall_back_to_stage2_for_old_metadata():
    meta = _meta({"name": "ミキ", "role": "学生", "age_range": "20代前半"})
    profiles = build_speaker_profiles(meta, [{"name": "ミキ", "gender": "female"}])
    assert profiles[0]["gender"] == "female"


def test_profiles_fall_back_to_inference_when_nothing_is_given():
    meta = _meta({"name": "ユウコ", "role": "大学院生", "age_range": "30代"})
    assert build_speaker_profiles(meta, None)[0]["gender"] == "female"


# --- Stage 2: 一人称の性別合わせ --------------------------------------------

def _turns(*pairs):
    return [{"speaker": s, "utterance": u} for s, u in pairs]


def test_rewrites_male_first_person_for_a_female_speaker():
    turns = _turns(("サラ", "PDFは俺が作業するから、コード化だけお願いできる？"),
                   ("サラ", "細かいフォントサイズとかは僕が調整するね。"),
                   ("サラ", "おれも助かる。"))
    fixed = fix_gendered_first_person(turns, {"サラ": "female"})
    assert [t["utterance"] for t in turns] == [
        "PDFは私が作業するから、コード化だけお願いできる？",
        "細かいフォントサイズとかは私が調整するね。",
        "私も助かる。"]
    assert len(fixed) == 3


def test_rewrites_female_first_person_for_a_male_speaker():
    turns = _turns(("ケンジ", "あたしもそう思います。"))
    fix_gendered_first_person(turns, {"ケンジ": "male"})
    assert turns[0]["utterance"] == "私もそう思います。"


def test_keeps_first_person_matching_the_speaker():
    turns = _turns(("ケンジ", "俺が行くよ。"), ("サラ", "あたしが行くね。"),
                   ("ミナ", "私が行きます。"))
    assert fix_gendered_first_person(
        turns, {"ケンジ": "male", "サラ": "female", "ミナ": "female"}) == []


def test_does_not_touch_quoted_speech():
    turns = _turns(("サラ", "彼が「俺がやる」って言ってたよ。"))
    assert fix_gendered_first_person(turns, {"サラ": "female"}) == []


def test_does_not_touch_words_that_merely_contain_the_kana():
    """「おれ」「ぼく」は他の語に埋まる。助詞・文末が続くときだけ拾う。"""
    turns = _turns(("サラ", "おれいを言わないと。"), ("サラ", "枝が折れそう。"),
                   ("サラ", "気持ちが倒れる寸前です。"))
    before = [t["utterance"] for t in turns]
    assert fix_gendered_first_person(turns, {"サラ": "female"}) == []
    assert [t["utterance"] for t in turns] == before


def test_skips_speakers_with_unknown_gender():
    turns = _turns(("ミキ", "俺が行くよ。"))
    assert fix_gendered_first_person(turns, {}) == []
    assert turns[0]["utterance"] == "俺が行くよ。"


# --- Stage 4: 割り当て ------------------------------------------------------

VOICES = [
    {"id": "f20", "gender": "female", "age_range": "20代"},
    {"id": "f50", "gender": "female", "age_range": "50代"},
    {"id": "m20", "gender": "male", "age_range": "20代"},
    {"id": "m50", "gender": "male", "age_range": "50代"},
]


def _attrs(record, names):
    from generate_tts import resolve_speaker_attrs
    return resolve_speaker_attrs(record, names)


def test_stage4_uses_metadata_gender():
    record = {
        "id": "d1",
        "metadata": _meta({"name": "ヨシ", "gender": "female", "role": "住職", "age_range": "50代"},
                          {"name": "ユウ", "gender": "male", "role": "檀家", "age_range": "20代"}),
        # speaker_profiles が誤っていても metadata が勝つ
        "speaker_profiles": [{"name": "ヨシ", "gender": "male", "age_range": "50代"},
                             {"name": "ユウ", "gender": "female", "age_range": "20代"}],
    }
    attrs = _attrs(record, ["ヨシ", "ユウ"])
    assert attrs["ヨシ"] == {"gender": "female", "age": 50, "main": False}
    assert attrs["ユウ"] == {"gender": "male", "age": 20, "main": False}

    from generate_tts import assign_voices
    got = assign_voices("d1", ["ヨシ", "ユウ"], VOICES, attrs)
    assert got["ヨシ"]["id"] == "f50"
    assert got["ユウ"]["id"] == "m20"


def test_stage4_still_reads_profiles_for_old_metadata():
    record = {
        "id": "d2",
        "metadata": _meta({"name": "ミキ", "role": "学生", "age_range": "20代"},
                          {"name": "タカシ", "role": "職員", "age_range": "50代"}),
        "speaker_profiles": [{"name": "ミキ", "gender": "female", "age_range": "20代"},
                             {"name": "タカシ", "gender": "male", "age_range": "50代"}],
    }
    from generate_tts import assign_voices
    attrs = _attrs(record, ["ミキ", "タカシ"])
    got = assign_voices("d2", ["ミキ", "タカシ"], VOICES, attrs)
    assert (got["ミキ"]["id"], got["タカシ"]["id"]) == ("f20", "m50")


def test_stage4_warns_when_gender_is_unknown(caplog):
    record = {
        "id": "d3",
        "metadata": _meta({"name": "ミキ", "role": "学生", "age_range": "20代"}),
        "speaker_profiles": [],
    }
    from generate_tts import assign_voices
    attrs = _attrs(record, ["ミキ"])
    assert attrs["ミキ"]["gender"] is None
    with caplog.at_level("WARNING"):
        assign_voices("d3", ["ミキ"], VOICES, attrs)
    assert "性別が不明" in caplog.text


def test_stage4_warns_when_same_gender_voices_are_exhausted(caplog):
    voices = [{"id": "f20", "gender": "female", "age_range": "20代"},
              {"id": "m20", "gender": "male", "age_range": "20代"}]
    record = {
        "id": "d4",
        "metadata": _meta({"name": "ミサ", "gender": "female", "role": "主婦", "age_range": "20代"},
                          {"name": "ユリ", "gender": "female", "role": "主婦", "age_range": "20代"}),
    }
    from generate_tts import assign_voices
    attrs = _attrs(record, ["ミサ", "ユリ"])
    with caplog.at_level("WARNING"):
        got = assign_voices("d4", ["ミサ", "ユリ"], voices, attrs)
    assert sorted(v["id"] for v in got.values()) == ["f20", "m20"]
    assert "同性の声が残っていません" in caplog.text
