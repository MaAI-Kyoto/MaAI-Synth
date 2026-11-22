"""ユーザ側の人物カードの母集団（personas/partner_pool.jsonl）を作る。

Nemotron-Personas-Japan から性別×年代（20〜70代）のセルごとに --per-cell 人を引き、
それぞれに RealPersonaChat の1人（同じ性別・年代が近いほど選ばれやすい）の性格検査と
ペルソナ文2文を組み合わせる。中身と使い方は partner_persona.py の冒頭を参照。

  # Nemotron-Personas-Japan (CC BY 4.0): https://huggingface.co/datasets/nvidia/Nemotron-Personas-Japan
  # RealPersonaChat (CC BY-SA 4.0):       https://github.com/nu-dialogue/real-persona-chat
  python tools/build_partner_personas.py \
      --nemotron third_party/data/nemotron_personas_japan \
      --rpc third_party/data/real-persona-chat \
      --out personas/partner_pool.jsonl
  (see readme/personas.md; the resulting pool is a derivative of both datasets, CC BY-SA 4.0)
"""

import argparse
import ast
import json
import random
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "pipeline"))  # pipeline modules

from partner_persona import DECADES, age_range_of, talk_style, trait_percentiles, usable_facts

_NEMO_COLS = ["uuid", "persona", "sex", "age", "marital_status", "occupation", "prefecture",
              "hobbies_and_interests_list"]
# 紹介文の書き出し「岡田 咲弥は」から姓を取る（約97%で取れる。取れない行は姓なし）。
_NAME_RE = re.compile(r"^([一-龥々ヶ]{1,4}) [^\sは、（(]{1,5}は")
# 主話者（イノウエ）と同じ読みの姓は使わない。
_SURNAME_NG = re.compile(r"井上|井ノ上|井之上|井の上|猪上|猪股井上")
_RPC_AGE = {"-19": 15, "20-29": 25, "30-39": 35, "40-49": 45, "50-59": 55, "60-69": 65}
_RPC_GENDER = {"Male": "male", "Female": "female"}


def _hobbies(v) -> list[str]:
    if isinstance(v, list):
        return [str(x) for x in v]
    try:
        return [str(x) for x in ast.literal_eval(str(v))]
    except (ValueError, SyntaxError):
        return []


def _surname(persona) -> str:
    m = _NAME_RE.match(str(persona))
    return m.group(1) if m and not _SURNAME_NG.search(m.group(1)) else ""


def load_nemotron(root: Path, per_cell: int, rng: random.Random) -> tuple[list[dict], Counter]:
    """性別×年代のセルごとに per_cell 人。全シャードから一様に引く（セルの中は国勢調査どおり）。

    2つめの返り値は全行（100万人）の姓の出現数（珍しい姓の差し替えに使う）。
    """
    import pandas as pd
    df = pd.concat([pq.read_table(p, columns=_NEMO_COLS).to_pandas()
                    for p in sorted(root.rglob("*.parquet"))], ignore_index=True)
    surnames = Counter(s for s in df.persona.map(_surname) if s)
    df = df[(df.age >= 20) & (df.age < 80)].copy()
    df["gender"] = df.sex.map({"男": "male", "女": "female"})
    df["age_range"] = df.age.map(age_range_of)
    out = []
    for g in ("male", "female"):
        for d in DECADES:
            cell = df[(df.gender == g) & (df.age_range == d)]
            n = min(per_cell, len(cell))
            out.extend(cell.sample(n, random_state=rng.randrange(2**31)).to_dict("records"))
    return out, surnames


def load_rpc(root: Path) -> list[dict]:
    with open(root / "real_persona_chat" / "interlocutors.json") as f:
        d = json.load(f)
    return list(d.values()) if isinstance(d, dict) else list(d)


def pick_rpc(people: list[dict], gender: str, age: int, rng: random.Random) -> dict:
    """同じ性別の人から、年代が近いほど選ばれやすく1人。

    RPC は50代以上が18人しかいないので、重みは緩くする（距離の2乗で割ると男性60代の1割が同じ1人になった）。
    """
    cand = [p for p in people
            if _RPC_GENDER.get(p["demographic_information"]["gender"]) == gender
            and len(usable_facts(p["persona"])) >= 2]
    w = [1.0 / (1 + abs(_RPC_AGE.get(p["demographic_information"]["age"], 35) - age) / 10)
         for p in cand]
    return rng.choices(cand, weights=w, k=1)[0]


def build(nemotron: Path, rpc: Path, per_cell: int, seed: int, min_surname: int = 30) -> list[dict]:
    rng = random.Random(seed)
    people = load_rpc(rpc)
    pct = trait_percentiles(people)
    rows, surnames = load_nemotron(nemotron, per_cell, rng)
    # ★珍しい姓は LLM が読みを誤る（試走: 蝶名林→「チョウメイリン」）。100万人中 min_surname 人
    #   未満の姓と、姓の取れない行は、よくある姓（出現数の重み）に差し替える。
    common = [(k, v) for k, v in surnames.items() if v >= min_surname]
    cards = []
    for r in rows:
        p = pick_rpc(people, r["gender"], int(r["age"]), rng)
        style, levels = talk_style(p["personality"], pct)
        hob = _hobbies(r["hobbies_and_interests_list"])
        surname = _surname(r["persona"])
        if surnames[surname] < min_surname:
            surname = rng.choices([k for k, _ in common], weights=[v for _, v in common], k=1)[0]
        cards.append({
            "card_id": f"nemo:{r['uuid']}",
            "gender": r["gender"],
            "age": int(r["age"]),
            "age_range": r["age_range"],
            "surname": surname,
            "prefecture": r["prefecture"],
            "marital": r["marital_status"],
            "occupation": r["occupation"],
            "hobbies": rng.sample(hob, min(len(hob), rng.choice((2, 3)))),
            "talk_style": style,
            "trait_levels": levels,
            "facts": rng.sample(usable_facts(p["persona"]), 2),
            "source": {"nemotron_uuid": r["uuid"], "rpc_interlocutor_id": p["interlocutor_id"]},
        })
    return cards


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nemotron", type=Path, default=Path("third_party/data/nemotron_personas_japan"),
                    help="directory with the Nemotron-Personas-Japan parquet files")
    ap.add_argument("--rpc", type=Path,
                    default=Path("third_party/data/real-persona-chat"),
                    help="clone of https://github.com/nu-dialogue/real-persona-chat (or its release archive)")
    ap.add_argument("--out", type=Path, default=Path("personas/partner_pool.jsonl"))
    ap.add_argument("--per-cell", type=int, default=2000, help="性別×年代のセルごとの人数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-surname", type=int, default=30, help="これ未満しか出ない姓はよくある姓に差し替える")
    a = ap.parse_args()
    cards = build(a.nemotron, a.rpc, a.per_cell, a.seed, a.min_surname)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for c in cards:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    cell = Counter((c["gender"], c["age_range"]) for c in cards)
    rpc_use = Counter(c["source"]["rpc_interlocutor_id"] for c in cards)
    print(f"{len(cards)}枚 → {a.out}")
    print("セル:", dict(sorted(cell.items())))
    print(f"姓あり {sum(1 for c in cards if c['surname']) / len(cards):.1%}、"
          f"姓 {len({c['surname'] for c in cards})}種、RPC の話者 {len(rpc_use)}人（最多 {rpc_use.most_common(1)}）")
    print("話し方の記述:", Counter(s for c in cards for s in c["talk_style"]).most_common())


if __name__ == "__main__":
    main()
