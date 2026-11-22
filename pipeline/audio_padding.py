"""対話音声の冒頭・末尾に無音（ルームトーン）を足す。合成時にも後付けにも使う。

なぜ:
    学習データの対話が**例外なく 0.000 秒ちょうど**に音で始まると（output_ft_v1_gemma_inoue_010
    の1万対話で実測 100%）、これを学習した対話モデルは「開始と同時に喋り出す」挙動を覚える
    （ユーザ報告 2026-10-28）。末尾も同じで、締めの挨拶の直後にファイルが終わっていると
    「締めたあとは黙っている」という例がデータに1つも無い。

    対話ごとに違う長さの間を前後に置く。長さは id から対数一様で引く（中央値 ≒ sqrt(lo*hi)）。
    一様分布にしないのは、短い間（1秒前後）が実際の会話でいちばん多く、長い間は少数で足りる
    ため。0 を含めないのは「0秒ちょうど」の型を残さないため。

    足す区間はデジタル無音にしない。合成時は無音区間にコンフォートノイズ（ピンクノイズ・
    既定 -80 dBFS）を敷いているので、後から足した区間だけが無音だと「開始直後はノイズが無い」
    という手がかりをモデルに与える。ここでも同じノイズを同じ水準で敷き、発話との境目は
    fade_ms でクロスフェードする（実測: 010 のギャップのノイズフロアは -80.5/-81.5 dBFS）。

使い方（後付け。既存の Stage4 成果物を書き換える）:
    # 数本だけ試す（--dry-run は何も書かない）
    python audio_padding.py --config config_ft_v1_gemma_inoue.yaml --limit 3 --dry-run
    python audio_padding.py --config config_ft_v1_gemma_inoue.yaml --limit 3 --verify
    # 全件（wav → jsonl の2段。中断しても再開できる）
    python audio_padding.py --config config_ft_v1_gemma_inoue.yaml --workers 8
    # wav の段をファイルサーバ ファイルサーバ の上で回す（work_20261028/pad_run.sh がこの形）
    python audio_padding.py --config ... --emit-jobs jobs.jsonl          # 手元
    python3 audio_padding.py --run-jobs jobs.jsonl --applied-log a.jsonl \
        --path-map /mnt/data=/export/data                                # ファイルサーバ 上
    python audio_padding.py --config ... --meta-only --applied-log a.jsonl  # 手元
    # 2026-10-29 以前に後付けした分の <id>.json を wav に揃える（何度流してもよい）
    python audio_padding.py --config ... --sidecars [--emit-jobs j.jsonl → ファイルサーバ で --run-jobs j.jsonl --sidecars]
    # 取り消し（wav の頭と尾を削って時刻を戻す）
    python audio_padding.py --config config_ft_v1_gemma_inoue.yaml --undo

可逆性:
    wav は「頭に lead 秒・尾に tail 秒を足す」だけなので、同じ秒数を削れば元に戻る
    （PCM_16 で読み書きするので、元のサンプル値はビット単位で保たれる）。
    秒数は manifest の lead_in_sec / tail_pad_sec に記録する（これが後付け済みの目印でもある）。
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

LEAD_DEFAULT = (0.4, 6.0)
TAIL_DEFAULT = (1.0, 6.0)



def _dlg_audio(d, rid):
    """manifest に audio_path が無いときの <d>/<rid>.wav か .flac（2026-11-17）。"""
    from pathlib import Path as _P
    p = _P(d) / f"{rid}.flac"
    return p if p.exists() else _P(d) / f"{rid}.wav"

def _draw(key: str, lo: float, hi: float) -> float:
    """[lo, hi] の対数一様。key から決まるので何度引いても同じ値になる。"""
    lo, hi = max(0.01, float(lo)), max(0.02, float(hi))
    if hi <= lo:
        return round(lo, 3)
    u = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return round(lo * (hi / lo) ** u, 3)


def lead_in_sec(dialogue_id: str, cfg: dict | None) -> float:
    """対話の冒頭に置く無音（秒）。cfg が無い・enabled: false なら 0.0。"""
    if not cfg or not cfg.get("enabled", True):
        return 0.0
    lo, hi = (list(cfg.get("sec") or LEAD_DEFAULT) + [0.0, 0.0])[:2]
    return _draw(f"lead:{dialogue_id}", lo, hi)


def tail_pad_sec(dialogue_id: str, cfg: dict | None) -> float:
    """対話の末尾に置く無音（秒）。cfg が無い・enabled: false なら 0.0。

    冒頭より少し長めの範囲を既定にしている。冒頭の無音が教えるのは「始まる前に待つ」
    だが、末尾が教えるのは「締めたあとは黙ったまま」なので、短いと例として弱い。
    """
    if not cfg or not cfg.get("enabled", True):
        return 0.0
    lo, hi = (list(cfg.get("sec") or TAIL_DEFAULT) + [0.0, 0.0])[:2]
    return _draw(f"tail:{dialogue_id}", lo, hi)


def padding_marks(dialogue_id: str, lead_cfg: dict | None,
                  tail_cfg: dict | None) -> dict:
    """manifest に記録する印。後付けスクリプトの二重適用防止と復元に使う。

    合成の段で無音を入れた対話にもこの印を付けておけば、後付け（audio_padding.py）が
    「もう入っている」と判断して飛ばす。どちらの経路で入れても同じキーになる。
    """
    lead = lead_in_sec(dialogue_id, lead_cfg)
    tail = tail_pad_sec(dialogue_id, tail_cfg)
    return {"lead_in_sec": lead, "tail_pad_sec": tail} if (lead or tail) else {}


def room_tone(n: int, ch: int, level_dbfs: float, color: str, seed_key: str) -> np.ndarray:
    """ルームトーン（合成時の noise_fill と同じピンクノイズ・同じ水準）。"""
    from generate_tts import _generate_noise          # 生成式を1つに保つ
    rng = np.random.default_rng(
        int(hashlib.md5(f"tone:{seed_key}".encode()).hexdigest()[:8], 16))
    amp = 10.0 ** (float(level_dbfs) / 20.0)
    out = np.zeros((n, ch), dtype=np.float32)
    for c in range(ch):
        out[:, c] = _generate_noise(n, color, rng) * amp
    return out


def pad_wav(path: Path, lead: float, tail: float, noise: dict,
            seed_key: str) -> tuple[float, float]:
    """wav の前後に無音を足して書き戻す。(新しい秒数, 追加区間の実測 RMS dBFS)。"""
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    n_lead, n_tail = int(round(lead * sr)), int(round(tail * sr))
    ch = x.shape[1]
    out = np.zeros((n_lead + x.shape[0] + n_tail, ch), dtype=np.float32)
    out[n_lead:n_lead + x.shape[0]] = x
    level = float(noise.get("level_dbfs", -80.0))
    color = str(noise.get("color", "pink"))
    fade = max(0, int(float(noise.get("fade_ms", 30.0)) / 1000.0 * sr))
    if n_lead > 0:
        tone = room_tone(n_lead, ch, level, color, f"{seed_key}:lead")
        k = min(fade, n_lead)
        if k > 0:                      # 発話に入る直前で絞る（合成時と同じ扱い）
            tone[-k:] *= np.linspace(1.0, 0.0, k, dtype=np.float32)[:, None]
        out[:n_lead] = tone
    if n_tail > 0:
        tone = room_tone(n_tail, ch, level, color, f"{seed_key}:tail")
        k = min(fade, n_tail)
        if k > 0:                      # 最後の発話が終わった直後に立ち上げる
            tone[:k] *= np.linspace(0.0, 1.0, k, dtype=np.float32)[:, None]
        out[-n_tail:] = tone
    measured = -999.0
    if n_lead > fade * 2:
        seg = out[:max(1, n_lead - fade)]
        measured = round(float(20 * np.log10(max(1e-12, np.sqrt((seg ** 2).mean())))), 1)
    # 2026-11-17 flac もそのままの形式で書き戻す（一時名が .wav だと flac の名前に wav の中身が入っていた）
    tmp = path.with_name(path.name + ".padtmp" + path.suffix)
    sf.write(str(tmp), out, sr, format=path.suffix.lstrip(".").upper(), subtype="PCM_16")
    os.replace(tmp, path)
    return round(out.shape[0] / sr, 3), measured


def trim_wav(path: Path, lead: float, tail: float) -> float:
    """pad_wav の逆。頭 lead 秒・尾 tail 秒を削って書き戻す。"""
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    a = int(round(lead * sr))
    b = x.shape[0] - int(round(tail * sr))
    out = x[a:max(a, b)]
    # 2026-11-17 flac もそのままの形式で書き戻す（一時名が .wav だと flac の名前に wav の中身が入っていた）
    tmp = path.with_name(path.name + ".padtmp" + path.suffix)
    sf.write(str(tmp), out, sr, format=path.suffix.lstrip(".").upper(), subtype="PCM_16")
    os.replace(tmp, path)
    return round(out.shape[0] / sr, 3)


# ── jsonl の時刻をずらす ───────────────────────────────────────────────
# ずらすのは「音の先頭からの絶対時刻」だけ。相対値（start_offset・duration・
# decided_offset 等）は触らない。
_ABS_KEYS_TURN = ("start_sec", "end_sec", "start_abs", "end_abs")
_ABS_KEYS_RETIME = ("anchor_sec", "llm_start_sec")
_ABS_KEYS_SEG = ("start", "end")


def shift_record(rec: dict, lead: float, tail: float, mark: bool) -> bool:
    """1レコードの絶対時刻を lead ぶん後ろへずらす（manifest / transcripts / refined 共通）。"""
    if lead == 0 and tail == 0:
        return False
    if "duration_sec" in rec and isinstance(rec["duration_sec"], (int, float)):
        rec["duration_sec"] = round(rec["duration_sec"] + lead + tail, 3)
    for t in rec.get("turns") or []:
        if not isinstance(t, dict):
            continue
        for k in _ABS_KEYS_TURN:
            if isinstance(t.get(k), (int, float)):
                t[k] = round(t[k] + lead, 3)
        rt = t.get("retime")
        if isinstance(rt, dict):
            for k in _ABS_KEYS_RETIME:
                if isinstance(rt.get(k), (int, float)):
                    rt[k] = round(rt[k] + lead, 3)
    for s in rec.get("segments") or []:
        if isinstance(s, dict):
            for k in _ABS_KEYS_SEG:
                if isinstance(s.get(k), (int, float)):
                    s[k] = round(s[k] + lead, 3)
    vd = rec.get("voice_dialogue")
    if isinstance(vd, dict):
        for t in vd.get("turns") or []:
            if isinstance(t, dict):
                for k in _ABS_KEYS_TURN:
                    if isinstance(t.get(k), (int, float)):
                        t[k] = round(t[k] + lead, 3)
    if mark:
        rec["lead_in_sec"] = lead
        rec["tail_pad_sec"] = tail
    return True


def fix_sidecar(path: Path, lead: float, tail: float, target_dur: float) -> str:
    """音声の隣の <id>.json（J-Moshi の前処理 align_mfa.py が発話の切り出しに使う）を
    wav に揃える。'ok'（揃っていた）/ 'shifted' / 'mismatch:...' / 'error:...' を返す。

    ★2026-10-29 まで後付けは jsonl だけをずらし、<id>.json は元の時刻のままだった
    （1万対話あたり約97%）。揃っているかは duration_sec で見る: target_dur と同じなら
    触らない（合成時に無音を入れた対話もこれ）。lead + tail を足すと target_dur になる
    ときだけずらす。どちらでもなければ触らずに報告する。取り消しは lead/tail を負で渡す。
    """
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        d = float(meta.get("duration_sec") or 0.0)
        if abs(d - target_dur) < 0.002:
            return "ok"
        if abs(d + lead + tail - target_dur) >= 0.002:
            return f"mismatch:{d}+{lead}+{tail}!={target_dur}"
        shift_record(meta, lead, tail, mark=lead > 0)
        if lead < 0:
            meta.pop("lead_in_sec", None)
            meta.pop("tail_pad_sec", None)
        tmp = path.with_name(path.name + ".padtmp")
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)                   # generate_tts.write_transcript と同じ書式
        return "shifted"
    except Exception as e:
        return f"error:{type(e).__name__}:{e}"[:160]


def _sidecar_job(args):
    return args[0], fix_sidecar(Path(args[1]), args[2], args[3], args[4])


def _run_sidecars(jobs: list, workers: int) -> None:
    """jobs = [(id, <id>.json のパス, lead, tail, 揃えたあとの duration_sec)]。"""
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor
    cnt, bad = Counter(), []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:   # 小さい json の読み書きだけ
        for k, (rid, st) in enumerate(ex.map(_sidecar_job, jobs), 1):
            cnt[st.split(":")[0]] += 1
            if st not in ("ok", "shifted"):
                bad.append((rid, st))
            if k % 2000 == 0:
                logger.info(f"  <id>.json {k}/{len(jobs)}")
    logger.info(f"<id>.json: ずらした {cnt['shifted']}件 / 揃っていた {cnt['ok']}件"
                f" / 食い違い {cnt['mismatch']}件 / 失敗 {cnt['error']}件")
    for rid, st in bad[:10]:
        logger.error(f"  {rid}: {st}")


# ── 後付けの CLI ──────────────────────────────────────────────────────

def _targets(layout, variant: str):
    """(wav ディレクトリ, 時刻を持つ jsonl の canonical パス) を返す。"""
    if variant == "ref":
        return layout.ref_audio_dir, [layout.ref_manifest, layout.ref_transcripts,
                                      layout.refined_records]
    return layout.base_audio_dir, [layout.base_manifest, layout.base_transcripts]


def _pad_job(args):
    """1対話ぶんの wav の書き換え（ProcessPoolExecutor から呼ぶので関数は top-level）。"""
    rid, path, lead, tail, noise, undo = args
    import soundfile as sf
    p = Path(path)
    try:
        info = sf.info(str(p))
        if undo:
            if info.duration <= lead + tail:
                return rid, "skip", info.duration, -999.0
            return rid, "ok", trim_wav(p, lead, tail), -999.0
        dur, rms = pad_wav(p, lead, tail, noise, rid)
        return rid, "ok", dur, rms
    except Exception as e:                      # 1本の失敗で全体を止めない
        return rid, f"error:{type(e).__name__}:{e}"[:160], 0.0, -999.0


def _run_pool(todo: list, workers: int, applied_log: Path, undo: bool):
    """todo の wav を並列に書き換える。(足した対話 {id: (lead, tail)}, 失敗, 実測 RMS)。"""
    changed: dict[str, tuple[float, float]] = {}
    errs, rmss = [], []
    applied_log.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as ex, \
            open(applied_log, "a", encoding="utf-8") as alog:
        futs = {ex.submit(_pad_job, t): t for t in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            rid, st, dur, rms = fut.result()
            t = futs[fut]
            if st == "ok":
                changed[rid] = (t[2], t[3])
                if not undo:            # 足した直後に記録（flush まで含めて1件ずつ）
                    alog.write(json.dumps({"id": rid, "lead": t[2], "tail": t[3]}) + "\n")
                    alog.flush()
                if rms > -900:
                    rmss.append(rms)
            elif st != "skip":
                errs.append((rid, st))
            if k % 500 == 0:
                logger.info(f"  wav {k}/{len(todo)}（失敗 {len(errs)}）")
    return changed, errs, rmss


def run_jobs(a) -> None:
    """--emit-jobs の書き出しを読んで wav だけ書き換える（ファイルサーバ ファイルサーバ の上で回す）。

    NFS 越しだと ファイルサーバ の 1Gb 回線が詰まって1対話 約1.2秒かかるが、ファイルサーバ の上なら
    約0.2秒（2026-10-29 実測。出力は手元で回したものと md5 一致）。ファイルサーバ からは
    リポジトリが見えないので config は読まない。記録（--applied-log）は --audio-only と
    同じ書式なので、jsonl のシフトは手元で --meta-only を通す。記録済みの id は飛ばす。
    """
    src, dst = a.path_map.split("=", 1) if a.path_map else ("", "")
    if a.sidecars:                     # <id>.json を揃えるジョブ（記録は要らない。何度流してもよい）
        jobs = []
        for line in open(a.run_jobs, encoding="utf-8"):
            if line.strip():
                j = json.loads(line)
                p = j["path"]
                if src and p.startswith(src):
                    p = dst + p[len(src):]
                jobs.append((j["id"], p, float(j["lead"]), float(j["tail"]), float(j["duration"])))
        logger.info(f"{a.run_jobs}: <id>.json {len(jobs)}件")
        _run_sidecars(jobs, a.workers)
        return
    applied_log = Path(a.applied_log)
    done = set()
    if applied_log.exists():
        done = {json.loads(l)["id"] for l in open(applied_log, encoding="utf-8") if l.strip()}
    todo = []
    for line in open(a.run_jobs, encoding="utf-8"):
        if not line.strip():
            continue
        j = json.loads(line)
        if j["id"] in done:
            continue
        p = j["path"]
        if src and p.startswith(src):
            p = dst + p[len(src):]
        todo.append((j["id"], p, float(j["lead"]), float(j["tail"]), j["noise"], False))
    logger.info(f"{a.run_jobs}: 済み {len(done)}件 / これから {len(todo)}件")
    changed, errs, rmss = _run_pool(todo, a.workers, applied_log, undo=False)
    logger.info(f"wav 完了: {len(changed)}件 / 失敗 {len(errs)}件")
    if rmss:
        logger.info(f"足した区間のノイズ: 中央値 {sorted(rmss)[len(rmss)//2]:.1f} dBFS")
    for rid, st in errs[:10]:
        logger.error(f"  {rid}: {st}")


def main():
    ap = argparse.ArgumentParser(
        description="既存の Stage4 成果物に対話の冒頭・末尾の無音を後付けする（可逆）")
    ap.add_argument("--config", help="--run-jobs 以外では必須")
    ap.add_argument("--variant", default="ref", choices=["ref", "base"])
    ap.add_argument("--lead", help="冒頭の無音の範囲（既定は config の tts.lead_in.sec）")
    ap.add_argument("--tail", help="末尾の無音の範囲（既定は config の tts.tail_pad.sec）")
    ap.add_argument("--limit", type=int, help="先頭 N 対話だけ")
    ap.add_argument("--ids-file", help="対象 id を1行1件で書いたファイル")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="何も書かずに引いた秒数だけ出す")
    ap.add_argument("--undo", action="store_true", help="後付けを取り消す")
    ap.add_argument("--audio-only", action="store_true",
                    help="wav だけ直す（jsonl は触らない）。複数マシンで分担するときに使う")
    ap.add_argument("--meta-only", action="store_true",
                    help="jsonl だけ直す（--applied-log に書かれた対話が対象。カンマ区切りで複数）")
    ap.add_argument("--applied-log",
                    help="足した秒数の記録（既定 work_20261028/pad_applied_<出力先>.jsonl）")
    ap.add_argument("--verify", action="store_true",
                    help="書き換えた wav を開き直して、足した区間のノイズを実測する")
    ap.add_argument("--emit-jobs",
                    help="wav は触らず、対象（id・実パス・秒数・ノイズ設定）を jsonl に書いて終わる")
    ap.add_argument("--run-jobs",
                    help="--emit-jobs の jsonl の wav だけ書き換える（config 不要。ファイルサーバ 上で使う）")
    ap.add_argument("--path-map", help="--run-jobs でパスの頭を置き換える（例 /mnt/data=/export/data）")
    ap.add_argument("--sidecars", action="store_true",
                    help="後付け済みの対話の <id>.json だけを wav に揃える（2026-10-29 以前の後付けの修正。"
                         "--emit-jobs / --run-jobs と組み合わせられる）")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if a.run_jobs:
        if not a.applied_log and not a.sidecars:
            ap.error("--run-jobs には --applied-log が要る")
        run_jobs(a)
        return
    if not a.config:
        ap.error("--config が要る")
    if a.emit_jobs and a.undo:
        ap.error("--emit-jobs は --undo と併用できない")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from audit_pipeline import read_variant_records, resolve_layout, variant_files
    from main import load_config

    config = load_config(a.config)
    layout = resolve_layout(config)
    tts_cfg = config.get("tts") or {}
    noise = dict(tts_cfg.get("noise_fill") or {})
    lead_cfg = {"sec": [float(x) for x in a.lead.split(",")]} if a.lead else \
        (tts_cfg.get("lead_in") or {"sec": list(LEAD_DEFAULT)})
    tail_cfg = {"sec": [float(x) for x in a.tail.split(",")]} if a.tail else \
        (tts_cfg.get("tail_pad") or {"sec": list(TAIL_DEFAULT)})
    wav_dir, jsonls = _targets(layout, a.variant)
    man = jsonls[0]

    # 途中で落ちても二重に足さないための記録。wav を足した時点で1行ずつ append し、
    # jsonl のシフトはこの記録に対して行う（印が manifest に入るのは最後なので、
    # 「wav は足したが manifest は未シフト」の状態を跨いで再開できるようにする）。
    logs = [Path(x) for x in (a.applied_log.split(",") if a.applied_log
                              else [str(Path("work_20261028")
                                         / f"pad_applied_{man.parent.name}.jsonl")])]
    applied_log = logs[0]
    applied: dict[str, tuple[float, float]] = {}
    for lg in logs:
        if lg.exists():
            for line in open(lg, encoding="utf-8"):
                line = line.strip()
                if line:
                    r = json.loads(line)
                    applied[r["id"]] = (r["lead"], r["tail"])

    recs = read_variant_records(man, keys=("duration_sec", "lead_in_sec", "tail_pad_sec",
                                           "audio_path"))
    ids = list(recs)
    if a.ids_file:
        want = {l.strip() for l in open(a.ids_file) if l.strip()}
        ids = [i for i in ids if i in want]
    if a.limit:
        ids = ids[:a.limit]
    logger.info(f"{man.name}: {len(recs)}件 → 対象 {len(ids)}件 / variant={a.variant}")

    def sidecar_of(rid: str) -> str:
        return str(Path(recs[rid].get("audio_path") or _dlg_audio(wav_dir, rid)).with_suffix(".json"))

    if a.sidecars:                     # manifest の印がある対話の <id>.json を wav に揃える
        jobs = [(rid, os.path.realpath(sidecar_of(rid)), float(recs[rid]["lead_in_sec"]),
                 float(recs[rid].get("tail_pad_sec") or 0.0), float(recs[rid]["duration_sec"]))
                for rid in ids if isinstance(recs[rid].get("lead_in_sec"), (int, float))]
        logger.info(f"後付け済み {len(jobs)}件の <id>.json を見ます")
        if a.emit_jobs:
            with open(a.emit_jobs, "w", encoding="utf-8") as f:
                for rid, p, lead, tail, dur in jobs:
                    f.write(json.dumps({"id": rid, "path": p, "lead": lead, "tail": tail,
                                        "duration": dur}) + "\n")
            logger.info(f"対象を書き出しました: {a.emit_jobs}（{len(jobs)}件）")
        elif not a.dry_run:
            _run_sidecars(jobs, a.workers * 2)
        return

    # すでに後付け済みか（manifest の印）で仕分ける
    todo, done = [], 0
    for rid in ids:
        r = recs[rid]
        has = isinstance(r.get("lead_in_sec"), (int, float))
        if a.undo and not has:
            done += 1
            continue
        if not a.undo and has:
            done += 1
            continue
        if not a.undo and rid in applied:
            continue                   # wav は足し済み。jsonl のシフトだけ下で行う
        lead = (r.get("lead_in_sec") if a.undo else lead_in_sec(rid, lead_cfg))
        tail = (r.get("tail_pad_sec") if a.undo else tail_pad_sec(rid, tail_cfg))
        p = r.get("audio_path") or str(_dlg_audio(wav_dir, rid))
        todo.append((rid, p, float(lead or 0.0), float(tail or 0.0), noise, a.undo))
    logger.info(f"済み {done}件 / これから {len(todo)}件"
                + ("（--dry-run なので書きません）" if a.dry_run else ""))
    if a.dry_run:
        for rid, p, lead, tail, *_ in todo[:10]:
            logger.info(f"  {rid[:8]} lead={lead:.3f}s tail={tail:.3f}s {Path(p).name}")
        return
    if a.emit_jobs:                    # 実パスで書く（シンボリックリンクは ファイルサーバ から辿れない）
        with open(a.emit_jobs, "w", encoding="utf-8") as f:
            for rid, p, lead, tail, nz, _ in todo:
                f.write(json.dumps({"id": rid, "path": os.path.realpath(p), "lead": lead,
                                    "tail": tail, "noise": nz}, ensure_ascii=False) + "\n")
        logger.info(f"対象を書き出しました: {a.emit_jobs}（{len(todo)}件）")
        return
    if a.meta_only:
        todo = []
    # 1段目: wav
    # 「wav は足したが jsonl は未シフト」の対話（記録にあって manifest に印が無い）は、
    # 音声を触らず jsonl だけ直す。★ここを todo の有無だけで打ち切ると、音声の段が
    # 済んだあとのメタデータの段が丸ごと空振りする（2026-10-29 に実際に起きた）。
    changed: dict[str, tuple[float, float]] = {}
    if not a.undo:
        changed.update({i: v for i, v in applied.items() if i in recs
                        and not isinstance(recs[i].get("lead_in_sec"), (int, float))})
        if changed:
            logger.info(f"前回 wav まで済んでいた {len(changed)}件は jsonl だけ直します")
    if not todo and not changed:
        return
    done_now, errs, rmss = _run_pool(todo, a.workers, applied_log, a.undo)
    changed.update(done_now)
    logger.info(f"wav 完了: {len(changed)}件（うち今回 {len(todo) - len(errs)}件）"
                f" / 失敗 {len(errs)}件")
    if rmss:
        logger.info(f"足した区間のノイズ: 中央値 {sorted(rmss)[len(rmss)//2]:.1f} dBFS"
                    f"（合成時の設定 {noise.get('level_dbfs', -80.0)} dBFS）")
    for rid, st in errs[:10]:
        logger.error(f"  {rid}: {st}")
    if not changed:
        return
    if a.audio_only:
        logger.info(f"--audio-only なので jsonl は触りません（記録: {applied_log}）")
        return

    # 2段目: jsonl（canonical と part の全部。1行ずつ読んで書き換え、原子的に差し替える）
    for canonical in jsonls:
        for path in variant_files(canonical):
            if not path.exists():
                continue
            n = 0
            tmp = path.with_suffix(path.suffix + ".padtmp")
            with open(path, encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as dst:
                for line in src:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    rid = rec.get("id")
                    if rid in changed:
                        lead, tail = changed[rid]
                        if a.undo:
                            shift_record(rec, -lead, -tail, mark=False)
                            rec.pop("lead_in_sec", None)
                            rec.pop("tail_pad_sec", None)
                        else:
                            shift_record(rec, lead, tail, mark=True)
                        n += 1
                    dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
            os.replace(tmp, path)
            if n:
                logger.info(f"  {path.name}: {n}行")
    logger.info("jsonl 完了")
    # 音声の隣の <id>.json も揃える（recs は jsonl をずらす前の値）
    sgn = -1.0 if a.undo else 1.0
    _run_sidecars([(rid, sidecar_of(rid), sgn * lead, sgn * tail,
                    float(recs[rid]["duration_sec"]) + sgn * (lead + tail))
                   for rid, (lead, tail) in changed.items()
                   if rid in recs and isinstance(recs[rid].get("duration_sec"), (int, float))],
                  a.workers * 2)
    if a.undo and applied_log.exists():      # 取り消したものは記録からも外す
        keep = [l for l in open(applied_log, encoding="utf-8")
                if l.strip() and json.loads(l)["id"] not in changed]
        with open(applied_log, "w", encoding="utf-8") as f:
            f.writelines(keep)

    if a.verify:
        import soundfile as sf
        for rid, (lead, tail) in list(changed.items())[:5]:
            p = recs[rid].get("audio_path") or str(_dlg_audio(wav_dir, rid))
            x, sr = sf.read(p, dtype="float32", always_2d=True)
            h = x[:max(1, int(lead * sr) - int(0.03 * sr))]
            db = 20 * np.log10(max(1e-12, float(np.sqrt((h ** 2).mean()))))
            logger.info(f"  {rid[:8]} 秒数 {x.shape[0]/sr:.3f} / 冒頭 {lead:.3f}秒の"
                        f"ノイズ {db:.1f} dBFS / 先頭サンプル最大 {float(np.abs(x[:100]).max()):.2e}")


if __name__ == "__main__":
    main()
