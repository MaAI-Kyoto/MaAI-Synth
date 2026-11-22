"""ボイスバンク(参照音声)管理ツール — Stage 4 (generate_tts.py) 用

参照音声(wav)とその文字起こし・性別・年代をまとめた CSV ボイスバンクを、
スキャン生成・検証する。CSV列: id, gender, age_range, audio_path, text

使い方:
    # voices/wav 配下の *.wav から CSV 雛形を生成/追記（既存行は保持）
    python tools/manage_voices.py scan --dir voices/<bank>/wav

    # ボイスバンクCSVを検証（音声の長さ・ch・SR、文字起こし、性別/年代、分布）
    python tools/manage_voices.py validate
    python tools/manage_voices.py validate --voices voices/<bank>/voices.csv

CSVの id/audio_path は scan が埋める。gender・age_range・text は手で記入する。
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import yaml

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "pipeline"))  # pipeline modules

from generate_tts import VOICE_CSV_FIELDS, load_voices_csv, _parse_age

# 検証の基準
MIN_DURATION_SEC = 3.0       # これ未満はエラー
RECOMMENDED_DURATION_SEC = 5.0  # これ未満は警告（音色のみなら3秒でも可）
MAX_DURATION_SEC = 15.0      # これ超は警告（長すぎる参照は生成が破綻しやすい。推奨3〜10秒）
MIN_SAMPLE_RATE = 16000      # これ未満は警告
SENTENCE_END = "。！？.!?"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def voices_csv_path(config: dict, override: str = None) -> Path:
    if override:
        return Path(override)
    vf = config.get("tts", {}).get("voices_file")
    if not vf:
        print("config の tts.voices_file が設定されていません。", file=sys.stderr)
        sys.exit(1)
    return Path(vf)


# --------------------------------------------------------------------------
# scan: wav ディレクトリから CSV 雛形を生成/追記
# --------------------------------------------------------------------------

def cmd_scan(config: dict, wav_dir: str):
    csv_path = voices_csv_path(config)
    wav_root = Path(wav_dir)
    if not wav_root.exists():
        print(f"wav ディレクトリが見つかりません: {wav_root}", file=sys.stderr)
        sys.exit(1)

    # 既存CSVを読み込み（id重複を避けるため）
    existing_rows = []
    existing_ids = set()
    if csv_path.exists():
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                existing_rows.append(row)
                if (row.get("id") or "").strip():
                    existing_ids.add(row["id"].strip())

    # trim_reference.py が生成する短縮版(*_trimNs.wav)は元音声と重複するので拾わない
    import re
    trim_pat = re.compile(r"_trim\d+s$")

    wavs = sorted(p for p in wav_root.rglob("*.wav"))
    added = skipped = 0
    for p in wavs:
        vid = p.stem
        if trim_pat.search(vid):
            skipped += 1
            continue
        if vid in existing_ids:
            continue
        existing_rows.append({
            "id": vid,
            "gender": "",
            "age_range": "",
            "audio_path": str(p),
            "text": "",
        })
        existing_ids.add(vid)
        added += 1

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=VOICE_CSV_FIELDS)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow({k: row.get(k, "") for k in VOICE_CSV_FIELDS})

    print(f"スキャン完了: {wav_root} 内 wav {len(wavs)}件 / 新規追加 {added}件"
          + (f" / 短縮版スキップ {skipped}件" if skipped else ""))
    print(f"CSV: {csv_path}（合計 {len(existing_rows)}行）")
    if added:
        print("→ gender / age_range / text の空欄を記入してください。")


# --------------------------------------------------------------------------
# validate: ボイスバンクCSVを検証
# --------------------------------------------------------------------------

def _probe_audio(path: str):
    """soundfile で音声情報を取得。(samplerate, channels, duration) または None。"""
    import soundfile as sf
    info = sf.info(path)
    return info.samplerate, info.channels, info.duration


def cmd_validate(config: dict, voices: str = None):
    csv_path = voices_csv_path(config, voices)
    if not csv_path.exists():
        print(f"ボイスバンクCSVがありません: {csv_path}", file=sys.stderr)
        print("先に `python manage_voices.py scan --dir <wav dir>` を実行してください。", file=sys.stderr)
        sys.exit(1)

    voices = load_voices_csv(csv_path)
    if not voices:
        print(f"有効な行がありません: {csv_path}", file=sys.stderr)
        sys.exit(1)

    errors = 0
    warnings = 0
    seen_ids = set()
    gender_counter = Counter()
    age_band_counter = Counter()

    print(f"検証対象: {csv_path}（{len(voices)}声）\n")
    for v in voices:
        vid = v["id"]
        msgs = []

        # id 重複
        if vid in seen_ids:
            msgs.append(("ERROR", f"id が重複: {vid}"))
        seen_ids.add(vid)

        # 音声ファイル
        ap = v["audio_path"]
        if not ap:
            msgs.append(("ERROR", "audio_path が空"))
        elif not Path(ap).exists():
            msgs.append(("ERROR", f"音声が見つからない: {ap}"))
        else:
            try:
                sr, ch, dur = _probe_audio(ap)
                if dur < MIN_DURATION_SEC:
                    msgs.append(("ERROR", f"短すぎる ({dur:.1f}s < {MIN_DURATION_SEC}s)"))
                elif dur < RECOMMENDED_DURATION_SEC:
                    msgs.append(("WARN", f"やや短い ({dur:.1f}s, 推奨{RECOMMENDED_DURATION_SEC}s+)"))
                elif dur > MAX_DURATION_SEC:
                    msgs.append(("WARN", f"長すぎる ({dur:.1f}s > {MAX_DURATION_SEC}s)。"
                                         f"参照は3〜10秒推奨。長いと生成が破綻し別言語のような音声になりやすい"))
                if ch != 1:
                    msgs.append(("WARN", f"mono推奨だが {ch}ch"))
                if sr < MIN_SAMPLE_RATE:
                    msgs.append(("WARN", f"サンプルレート低い ({sr}Hz < {MIN_SAMPLE_RATE}Hz)"))
            except Exception as e:
                msgs.append(("ERROR", f"音声読み込み失敗: {e}"))

        # 文字起こし
        text = v["text"] or ""
        if not text:
            msgs.append(("ERROR", "text(文字起こし)が空"))
        elif text[-1] not in SENTENCE_END:
            msgs.append(("WARN", "text が句読点(。！？)で終わっていない"))

        # 性別
        g = v["gender"]
        if g not in ("male", "female"):
            msgs.append(("ERROR", f"gender は male/female のいずれか (現在: {g!r})"))
        else:
            gender_counter[g] += 1

        # 年代
        age = _parse_age(v["age_range"])
        if age is None:
            msgs.append(("WARN", "age_range が未設定/解釈不可（年代マッチが効かない）"))
        elif g in ("male", "female"):
            band = f"{g}/{age//10*10}代"
            age_band_counter[band] += 1

        # 出力
        has_err = any(level == "ERROR" for level, _ in msgs)
        has_warn = any(level == "WARN" for level, _ in msgs)
        mark = "✗" if has_err else ("△" if has_warn else "✓")
        print(f"  {mark} {vid}")
        for level, m in msgs:
            print(f"      [{level}] {m}")
            if level == "ERROR":
                errors += 1
            else:
                warnings += 1

    # 分布サマリ
    print("\n--- 分布 ---")
    print(f"  性別: " + ", ".join(f"{k}={c}" for k, c in sorted(gender_counter.items())) or "  (なし)")
    if age_band_counter:
        print("  性別×年代: " + ", ".join(f"{k}={c}" for k, c in sorted(age_band_counter.items())))
    for g in ("male", "female"):
        if gender_counter.get(g, 0) < 2:
            print(f"  [WARN] {g} の声が {gender_counter.get(g,0)}件。"
                  f"同性2話者の対話で声を重複なく割り当てるには各性別2声以上が望ましい。")
            warnings += 1

    print(f"\n結果: {len(voices)}声中 ERROR {errors}件 / WARN {warnings}件")
    if errors:
        print("ERROR を解消してください。", file=sys.stderr)
        sys.exit(1)
    print("OK: 本番(TTS)に進めます。")


def main():
    parser = argparse.ArgumentParser(description="ボイスバンク管理ツール (Stage 4 用)")
    parser.add_argument("--config", default="configs/base.yaml", help="設定ファイルパス")
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="wavディレクトリからCSV雛形を生成/追記")
    p_scan.add_argument("--dir", required=True, help="wavが入ったディレクトリ")

    p_val = sub.add_parser("validate", help="ボイスバンクCSVを検証")
    p_val.add_argument("--voices", default=None,
                       help="検証するCSV（既定は config の tts.voices_file）。"
                            "config 以外のバンク（voices/<bank>/voices.csv など）を見るときに使う")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "scan":
        cmd_scan(config, args.dir)
    elif args.command == "validate":
        cmd_validate(config, args.voices)


if __name__ == "__main__":
    main()
