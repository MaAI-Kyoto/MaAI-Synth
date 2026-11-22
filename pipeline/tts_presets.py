"""TTS 方式の名前（tts.engine: pocket-ja-nc-v1 など）を、tts_presets/<名前>.yaml の中身に展開する（2026-11-15）。

config の extends を全部解いたあとで呼ぶ（generate_tts.load_config / main.load_config の最後）。tts.engine が tts_presets/ にある名前なら、
その yaml を tts に deep-merge で重ね（方式の中身が勝つ）、tts.method にその名前を残す。yaml の中の engine（fish_speech など）が
実際に動く合成器になる。名前が無ければ何もしない（従来の engine: fish_speech などはそのまま）。
"""
from __future__ import annotations

import copy
import functools
from pathlib import Path

import yaml

PRESET_DIR = Path(__file__).resolve().parent.parent / "tts_presets"


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@functools.lru_cache(maxsize=None)
def _load(name: str) -> dict | None:
    p = PRESET_DIR / f"{name}.yaml"
    if not p.exists():
        return None
    return yaml.safe_load(p.read_text()) or {}


def names() -> list[str]:
    return sorted(p.stem for p in PRESET_DIR.glob("*.yaml"))


def apply(cfg: dict) -> dict:
    tts = cfg.get("tts")
    if not isinstance(tts, dict):
        return cfg
    name = str(tts.get("engine") or "")
    preset = _load(name) if name else None
    if preset is None:
        return cfg
    new = _merge(tts, copy.deepcopy(preset))
    new["method"] = name
    cfg = dict(cfg)
    cfg["tts"] = new
    return cfg
