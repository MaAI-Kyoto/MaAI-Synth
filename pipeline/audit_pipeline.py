"""Stage4 完了後の全数検査と、必要な Stage からの「その対話だけ」作り直し。

★2026-11-20 旧 Stage3（passthrough 変換）は廃止して Stage2 の最後に組み込み、TTS（旧 Stage4）を
Stage3 と呼ぶことにした。このファイルの Stage 名（--force-stage の stage1〜stage4・repair_history）は
互換のため旧番号のまま: stage3 = voice_dialogues の行だけ捨てて変換し直す、stage4 = TTS の成果物だけ捨てる。

背景:
    各 Stage は自前のゲートを持っているが、取りこぼしが Stage4 まで生き延びると
    そこで**黙って**落ちる。実例（output_ft_v1_gemma_001 / 2026-08-27）:

      Stage2 が1ターンだけ speaker を `ヨシノリ` → `ヨシリ` と打ち間違えた
        → Stage3 のゲート（generate_voice_passthrough.validate_voice_record の
           「話者不明のターン」）は今なら弾くが、その対話は**ゲート追加前**の
           Stage3 実行で voice_dialogues.jsonl に入っており、再実行時は
           existing_ids で素通しされるため二度と検査されなかった
        → Stage4 --retime は `len(speaker_order) < 2` としか見ないので通過し、
           3ch の wav（幽霊話者に女性声が割り当たった）を生成
        → Stage4 --refine は `len(speaker_order) != 2` で skipped += 1 して continue。
           ログには件数しか出ないので、5000 のはずが 4999 になった理由が分からない
        → ref wav が出来ない＝常に pending なので、何度回しても 4999 のまま

    つまり必要なのは「新規レコードだけでなく**既存の全レコードを現行ルールで
    検査し直す**」ことと、「壊れた対話を、原因の Stage まで戻して作り直す」こと。
    本スクリプトはその2つをやる。

検査（--pre なら 1〜4 のテキスト系だけ・Stage4 前の事前確認に使える）:
    1. 件数の欠け        metadata → dialogues → voice_dialogues の取りこぼし
    2. Stage3 ゲート     validate_voice_record を全レコードに掛け直す
    3. 話者数            話者がちょうど2人でない（★refine が黙って落とす条件）
    4. 話者名            turn の speaker が metadata.speakers に無い
    5. base(retime)      manifest 行・wav の欠け / channels≠2 / duration≤0
    6. refine            refined レコード・manifest 行・wav の欠け / turns 空
    7. --deep-audio      wav のヘッダを実際に開いて manifest と突き合わせる

作り直し（--repair）:
    所見ごとに「戻るべき Stage」を決め、同じ id に複数あれば**いちばん手前**を採る。

      stage1  メタデータごと捨てて作り直す（新しい uuid になる）
              ＝ 同じメタデータで一度 stage2 に差し戻して再発した対話
      stage2  対話だけ捨ててメタデータは残す（Stage2 が同じ id で引き直す）
              ＝ 話者名・話者数・空テキストなど、台本そのものの問題
      stage3  voice_dialogues の行だけ捨てる（Stage2 の最後の変換がやり直す）
      stage4  Stage4 の成果物だけ捨てる（テキストは無傷。TTS を回し直すだけ）

    どの stage でも Stage4 の成果物は必ず退避する。resume がファイル存在で効くので、
    古い wav が残っていると作り直しが素通しされるため。

    捨てるものは消さずに退避する:
      - JSONL のレコード  → `<stem>_discarded.jsonl`（topup_failed_dialogues と同じ）
      - 音声・キャッシュ  → `<出力ルート>/_audit_quarantine/<日時>/…`（--purge delete で削除）

使い方:
    # 検査だけ（既定・何も壊さない）
    .venv-qwen/bin/python audit_pipeline.py --config config_ft_v1_gemma.yaml

    # Stage4 を回す前の事前確認（テキスト系のみ・数秒で終わる）
    .venv-qwen/bin/python audit_pipeline.py --config config_ft_v1_gemma.yaml --pre

    # 検査 → 該当 Stage まで戻して破棄 → Stage1-3 を回して埋め戻す
    .venv-qwen/bin/python audit_pipeline.py --config config_ft_v1_gemma.yaml \
        --repair --run-stages
    # そのあと Stage4 を回し直す（欠けた分だけが pending になる）
    ./run_tts_sharded.sh

    # 特定 id だけを Stage2 から作り直す
    .venv-qwen/bin/python audit_pipeline.py --config config_ft_v1_gemma.yaml \
        --repair --id 383e231d-9707-442b-b4ab-f3f8df2342eb --force-stage stage2

注意:
    - --repair は破壊的。既定は検査のみ。まず素で走らせてレポートを見ること。
    - Stage4 は GPU・fish サーバ・シャード構成が要るので本スクリプトからは回さない。
      --run-stages は Stage1-3 まで。Stage4 は run_tts_sharded.sh を使う。
    - 検査は maai/torch を要求しない。.venv-qwen で動く（--deep-audio も soundfile だけ）。
"""

import argparse
import asyncio
import json
import logging
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from audio_io import audio_index

from cost_tracker import CostTracker
from generate_tts import (
    _SHARD_PART_RE,
    dialogue_speaker_order,
    engine_output_paths, base_variant_name, ref_variant_name,
    transcripts_path,
)
from generate_voice_passthrough import skip_file_for, validate_voice_record
from main import load_config, setup_logging
from refine_maai import QualityParams as _BCQualityParams, RefineConfig, _bc_dur_limit
from retime_maai import RetimeConfig
from topup_failed_dialogues import _atomic_write_lines, _retire_records

logger = logging.getLogger(__name__)

# 戻るべき Stage。数字が小さいほど手前＝同じ id に複数の所見があればこちらを採る。
STAGE_RANK = {"stage1": 1, "stage2": 2, "stage3": 3, "stage4": 4}


@dataclass
class Finding:
    """1対話・1件の所見。"""
    id: str
    code: str            # 所見の種類（下の CHECKS のキー）
    stage: str           # 戻るべき Stage
    detail: str          # 人が読む説明

    def as_dict(self) -> dict:
        return {"id": self.id, "code": self.code, "stage": self.stage,
                "detail": self.detail}


@dataclass
class Layout:
    """config から解決した入出力パス一式。命名規則は各 Stage の実装と同じ関数で導出する。"""
    output_root: Path
    metadata_file: Path
    dialogue_file: Path
    voice_file: Path
    dialogue_skip: Path
    voice_skip: Path
    # Stage4: base（--retime）
    base_audio_dir: Path
    base_manifest: Path
    base_transcripts: Path
    # Stage4: refine（--refine）
    ref_audio_dir: Path
    ref_manifest: Path
    ref_transcripts: Path
    refined_records: Path
    # キャッシュ類
    turn_cache_dir: Path          # <audio>/plain/turns/<id>/（★2026-11-22 旧 <audio>/<engine>/turns）
    ref_turn_cache_dir: Path      # <audio>/<variant_ref>/turns/<id>/
    ops_dir: Path                 # refine_ops/<variant_ref>/<id>.<hash>.json
    repair_history: Path


def resolve_layout(config: dict) -> Layout:
    """generate_tts / refine_maai と同じ導出でパスを組み立てる（ハードコードしない）。"""
    tts_cfg = config["tts"]
    engine = tts_cfg.get("engine", "gemini")
    rt = RetimeConfig.from_tts_cfg(tts_cfg)
    rf = RefineConfig.from_tts_cfg(tts_cfg)

    # ★2026-11-22 バリアント名からエンジン名を外した（旧 f"{engine}_{rt.tag}" / f"{engine}_{rt.tag}_{rf.tag}"）
    base_variant = base_variant_name(rt.tag)
    ref_variant = ref_variant_name(rt.tag, rf.tag)
    base_audio_dir, base_manifest = engine_output_paths(tts_cfg, engine, base_variant)
    ref_audio_dir, ref_manifest = engine_output_paths(tts_cfg, engine, ref_variant)

    rp = Path(rf.refined_records_file)
    voice_file = Path(config["voice"]["output_file"])
    return Layout(
        output_root=voice_file.parent,
        metadata_file=Path(config["metadata"]["output_file"]),
        dialogue_file=Path(config["dialogue"]["output_file"]),
        voice_file=voice_file,
        dialogue_skip=skip_file_for(Path(config["dialogue"]["output_file"])),
        voice_skip=skip_file_for(voice_file),
        base_audio_dir=base_audio_dir,
        base_manifest=base_manifest,
        base_transcripts=transcripts_path(base_manifest),
        ref_audio_dir=ref_audio_dir,
        ref_manifest=ref_manifest,
        ref_transcripts=transcripts_path(ref_manifest),
        refined_records=rp.with_name(f"{rp.stem}_{ref_variant}{rp.suffix}"),
        turn_cache_dir=engine_output_paths(tts_cfg, engine)[0] / "turns",
        ref_turn_cache_dir=ref_audio_dir / "turns",
        ops_dir=Path(rf.ops_dir) / ref_variant,
        repair_history=voice_file.parent / "audit_repair_history.jsonl",
    )


# --------------------------------------------------------------------------
# JSONL の読み書き（追記系は canonical と per-shard part を同じものとして扱う）
# --------------------------------------------------------------------------

def variant_files(canonical: Path) -> list[Path]:
    """canonical と、その書き手ごとの分割（*.part-IofN.* / *.part-w-<host>-<pid>.*）を返す。

    generate_tts.part_output_path が作る分割のこと（シャード方式・クレーム方式の両方）。
    マージ前は part にしか無く、マージ後は両方に有る。検査も破棄も両方を見る必要がある。

    ★判定は merge_shard_outputs と同じ正規表現で行う。glob だけだと
    `<stem>.part-1of2_discarded.jsonl`（_retire_records が作る退避先）まで
    part とみなしてしまい、破棄したはずの行が次の検査で生き返る。
    """
    out = [canonical] if canonical.exists() else []
    for p in sorted(canonical.parent.glob(f"{canonical.stem}.part-*{canonical.suffix}")):
        m = _SHARD_PART_RE.match(p.name)
        if m and m.group("base") + m.group("suffix") == canonical.name:
            out.append(p)
    return out


def read_ids(path: Path) -> set[str]:
    ids = set()
    if not path.exists():
        return ids
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line).get("id"))
    return ids


def read_records(path: Path) -> dict[str, dict]:
    """id → レコード。同じ id が複数あれば後勝ち（merge_shard_outputs と同じ規則）。"""
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("id"):
                out[rec["id"]] = rec
    return out


def read_variant_records(canonical: Path, keys: tuple[str, ...] | None = None) -> dict[str, dict]:
    """canonical + part をまとめて id → レコード。keys 指定で必要な項目だけ保持（省メモリ）。"""
    out: dict[str, dict] = {}
    for p in variant_files(canonical):
        for rid, rec in read_records(p).items():
            out[rid] = {k: rec.get(k) for k in keys} if keys else rec
    return out


def read_skip_reasons(path: Path) -> dict[str, str]:
    out = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec.get("id")] = rec.get("reason", "")
    return out


# --------------------------------------------------------------------------
# 検査
# --------------------------------------------------------------------------

# code → (戻るべき Stage, 説明)。レポートの見出しにも使う。
CHECKS = {
    "missing_dialogue":  ("stage2", "metadata にあるが対話が無い"),
    "dialogue_skipped":  ("stage1", "Stage2 が恒久スキップした（メタデータごと引き直す）"),
    "missing_voice":     ("stage3", "対話はあるが voice_dialogues に無い"),
    "voice_skipped":     ("stage2", "Stage3 が恒久スキップした（台本の問題）"),
    "bad_voice_record":  ("stage2", "Stage3 のゲートに今かけると弾かれる"),
    "speaker_count":     ("stage2", "話者がちょうど2人でない（refine が黙って落とす）"),
    "speaker_unknown":   ("stage2", "turn の speaker が metadata.speakers に無い"),
    "missing_base":      ("stage4", "base manifest に行が無い"),
    "missing_base_wav":  ("stage4", "base の wav が無い"),
    "bad_base_channels": ("stage4", "base の channels が2でない"),
    "bad_base_duration": ("stage4", "base の duration が0以下"),
    "missing_refined":   ("stage4", "refined レコードが無い（refine がスキップ/失敗）"),
    "missing_ref":       ("stage4", "refine manifest に行が無い"),
    "missing_ref_wav":   ("stage4", "refine の wav が無い"),
    "empty_refined":     ("stage4", "refined レコードの turns が空"),
    "audio_mismatch":    ("stage4", "wav の実体が manifest と食い違う（--deep-audio）"),
    "overlong_bc":       ("stage4", "挿入相槌の長さがテキストに対して異常（話者の取り違え）"),
}


# 挿入相槌の長さの上限に使うパラメータ（refine 側の既定値をそのまま使う）
_BC_Q = _BCQualityParams()


def _finding(rid: str, code: str, detail: str) -> Finding:
    return Finding(id=rid, code=code, stage=CHECKS[code][0], detail=detail)


def audit(layout: Layout, config: dict, pre: bool = False, deep_audio: bool = False,
          only_ids: set[str] | None = None, check_backlog: bool = False) -> list[Finding]:
    """全数検査して所見のリストを返す。何も変更しない。"""
    findings: list[Finding] = []

    meta_ids = read_ids(layout.metadata_file)
    dial_ids = read_ids(layout.dialogue_file)
    voice_recs = read_records(layout.voice_file)
    dial_skip = read_skip_reasons(layout.dialogue_skip)
    voice_skip = read_skip_reasons(layout.voice_skip)

    def keep(rid: str) -> bool:
        return only_ids is None or rid in only_ids

    target = config.get("target_dialogues")
    logger.info("--- 件数 ---")
    logger.info(f"  metadata        {len(meta_ids)}件"
                + (f"（target_dialogues={target}）" if target else ""))
    logger.info(f"  dialogues       {len(dial_ids)}件")
    logger.info(f"  voice_dialogues {len(voice_recs)}件"
                + (f" / うち id 指定 {len(only_ids)}件" if only_ids else ""))

    # --- 1. 件数の欠け ---------------------------------------------------
    # Stage2 は metadata に対する増分方式なので、まだ作っていない分（積み残し）は
    # 欠損ではない。ここを所見にすると生成途中のコーパスが真っ赤になるので、
    # 既定では件数だけ報告する（--check-backlog で所見に格上げできる）。
    backlog = 0
    for rid in sorted(meta_ids - dial_ids):
        if rid in dial_skip:
            if keep(rid):
                findings.append(_finding(rid, "dialogue_skipped", dial_skip[rid]))
        elif check_backlog:
            if keep(rid):
                findings.append(_finding(rid, "missing_dialogue", "Stage2 未生成"))
        else:
            backlog += 1
    if backlog:
        logger.info(f"  Stage2 の積み残し {backlog}件（メタデータはあるが未生成。"
                    "欠損とは区別しています。所見にするなら --check-backlog）")
    for rid in sorted(dial_ids - set(voice_recs)):
        if not keep(rid):
            continue
        if rid in voice_skip:
            findings.append(_finding(rid, "voice_skipped", voice_skip[rid]))
        else:
            findings.append(_finding(rid, "missing_voice", "Stage3 未変換"))

    # --- 2〜4. voice_dialogues のレコードを現行ルールで検査し直す ----------
    # ★ここが肝。Stage3 は existing_ids で既存を素通しするので、ゲートを後から
    #   足しても過去のレコードは検査されない。全件に掛け直す。
    for rid, rec in voice_recs.items():
        if not keep(rid):
            continue
        reason = validate_voice_record(rec)
        if reason:
            findings.append(_finding(rid, "bad_voice_record", reason))

        turns = (rec.get("voice_dialogue") or {}).get("turns") or []
        order = dialogue_speaker_order(turns) if turns else []
        if len(order) != 2:
            findings.append(_finding(
                rid, "speaker_count", f"話者{len(order)}人: {order}"))

        known = {s.get("name") for s in (rec.get("metadata") or {}).get("speakers") or []}
        unknown = sorted({t.get("speaker") for t in turns} - known) if known else []
        if unknown:
            findings.append(_finding(
                rid, "speaker_unknown",
                f"metadata.speakers={sorted(known)} に無い: {unknown}"))

    if pre:
        return findings

    # --- 5〜6. Stage4 の成果物 -------------------------------------------
    base = read_variant_records(layout.base_manifest,
                                keys=("channels", "duration_sec", "channel_speakers"))
    ref = read_variant_records(layout.ref_manifest,
                               keys=("channels", "duration_sec", "turns"))
    refined = read_variant_records(layout.refined_records, keys=("voice_dialogue",))
    logger.info(f"Stage4 成果物: base manifest {len(base)}件 / "
                f"refine manifest {len(ref)}件 / refined レコード {len(refined)}件")

    # 音声の有無は dir を 1 回読んで引く（1 本ずつ exists() は NFS 越しで 1 dir 約 35 分・2026-11-17）
    base_idx, ref_idx = audio_index(layout.base_audio_dir), audio_index(layout.ref_audio_dir)
    for rid in voice_recs:
        if not keep(rid):
            continue
        b = base.get(rid)
        if b is None:
            findings.append(_finding(rid, "missing_base", "manifest に id が無い"))
        else:
            ch = b.get("channels")
            if ch != 2:
                cs = b.get("channel_speakers") or {}
                findings.append(_finding(
                    rid, "bad_base_channels", f"channels={ch} channel_speakers={cs}"))
            if not (b.get("duration_sec") or 0) > 0:
                findings.append(_finding(
                    rid, "bad_base_duration", f"duration_sec={b.get('duration_sec')}"))
            if rid not in base_idx:      # wav / flac どちらでも（2026-11-17）
                findings.append(_finding(rid, "missing_base_wav",
                                         str(layout.base_audio_dir / f"{rid}.wav|.flac")))

        r = refined.get(rid)
        if r is None:
            findings.append(_finding(rid, "missing_refined",
                                     "refine がスキップまたは失敗している"))
        elif not ((r.get("voice_dialogue") or {}).get("turns") or []):
            findings.append(_finding(rid, "empty_refined", "turns が空"))

        rr = ref.get(rid)
        if rr is None:
            findings.append(_finding(rid, "missing_ref", "manifest に id が無い"))
        else:
            if rid not in ref_idx:
                findings.append(_finding(rid, "missing_ref_wav",
                                         str(layout.ref_audio_dir / f"{rid}.wav|.flac")))
            # 挿入相槌の wav がテキストに対して長すぎる＝文脈の発話を自分の声で読み上げた
            # 合成異常（ユーザ報告 2026-10-28「同じ発話が2回続き、2回目が相手のチャネル」）。
            # 010 の実測: 挿入相槌 91,050件中243件、対話数では2.4%。refine_maai 側に
            # 引き直しと破棄を入れたので、新しい run では出ない。
            for t in (rr.get("turns") or []):
                if t.get("type") != "backchannel":
                    continue
                lim = _bc_dur_limit(t.get("text", ""), _BC_Q)
                got = float(t.get("duration_sec") or 0.0)
                if lim > 0 and got > lim:
                    findings.append(_finding(
                        rid, "overlong_bc",
                        f"turn{t.get('index')}「{t.get('text')}」{got:.2f}秒"
                        f"（上限{lim:.2f}秒）"))

    # --- 7. wav の実体を開いて manifest と突き合わせる ---------------------
    if deep_audio:
        findings += _audit_audio_files(layout, base, ref, keep)

    return findings


def _audit_audio_files(layout: Layout, base: dict, ref: dict, keep) -> list[Finding]:
    """wav のヘッダだけ読んで channels / duration が manifest と一致するか見る。"""
    try:
        import soundfile as sf
    except ImportError:
        logger.warning("soundfile が無いので --deep-audio をスキップします。")
        return []

    out: list[Finding] = []
    for audio_dir, manifest in ((layout.base_audio_dir, base),
                                (layout.ref_audio_dir, ref)):
        label = audio_dir.name
        idx = audio_index(audio_dir)
        for rid, rec in manifest.items():
            if not keep(rid):
                continue
            p = idx.get(rid)
            if p is None:
                continue  # 欠けは上のチェックが拾っている
            try:
                info = sf.info(str(p))
            except Exception as e:            # 壊れた wav
                out.append(_finding(rid, "audio_mismatch",
                                    f"{label}: 読めない ({type(e).__name__}: {e})"))
                continue
            want_ch = rec.get("channels")
            want_dur = rec.get("duration_sec") or 0.0
            if want_ch is not None and info.channels != want_ch:
                out.append(_finding(rid, "audio_mismatch",
                                    f"{label}: channels 実体{info.channels} "
                                    f"≠ manifest {want_ch}"))
            if want_dur and abs(info.duration - want_dur) > 0.5:
                out.append(_finding(rid, "audio_mismatch",
                                    f"{label}: duration 実体{info.duration:.2f}s "
                                    f"≠ manifest {want_dur:.2f}s"))
    return out


# --------------------------------------------------------------------------
# 所見 → 戻るべき Stage の確定
# --------------------------------------------------------------------------

def plan_stages(findings: list[Finding], layout: Layout,
                force_stage: str | None = None) -> dict[str, str]:
    """id → 戻るべき Stage。同じ id に複数の所見があればいちばん手前を採る。

    さらに、過去に stage2 へ差し戻したのに再発した id は stage1 へ格上げする
    （そのメタデータ自体が地雷とみなす。topup_failed_dialogues と同じ判断）。
    """
    plan: dict[str, str] = {}
    for f in findings:
        cur = plan.get(f.id)
        if cur is None or STAGE_RANK[f.stage] < STAGE_RANK[cur]:
            plan[f.id] = f.stage

    if force_stage:
        return {rid: force_stage for rid in plan}

    past = defaultdict(set)
    if layout.repair_history.exists():
        with open(layout.repair_history) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    past[rec.get("id")].add(rec.get("stage"))
    for rid, stage in list(plan.items()):
        if stage == "stage2" and "stage2" in past[rid]:
            logger.info(f"[{rid}] 前回 stage2 に差し戻したのに再発 → stage1 に格上げ")
            plan[rid] = "stage1"
    return plan


# --------------------------------------------------------------------------
# 破棄（作り直しの前準備）
# --------------------------------------------------------------------------

def _quarantine(path: Path, qdir: Path, root: Path, purge: str) -> bool:
    """path を退避（または削除）する。動かしたら True。"""
    if not path.exists():
        return False
    if purge == "delete":
        shutil.rmtree(path) if path.is_dir() else path.unlink()
        return True
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = Path(path.name)
    dest = qdir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(dest))
    return True


def purge_stage4(layout: Layout, ids: set[str], stamp: str, purge: str) -> dict[str, int]:
    """Stage4 の成果物を id 単位で消す。

    resume はファイル存在で効くので、ここを消さないと作り直しが素通しされる。
    manifest / transcripts / refined は canonical と per-shard part の両方から行を抜く。
    """
    counts = Counter()
    qdir = layout.output_root / "_audit_quarantine" / stamp

    # 音声・キャッシュ・ops
    for rid in sorted(ids):
        targets = [
            layout.base_audio_dir / f"{rid}.wav",
            layout.base_audio_dir / f"{rid}.flac",
            layout.base_audio_dir / f"{rid}.json",
            layout.ref_audio_dir / f"{rid}.wav",
            layout.ref_audio_dir / f"{rid}.flac",
            layout.ref_audio_dir / f"{rid}.json",
            layout.turn_cache_dir / rid,          # 発話単位wavキャッシュ（ディレクトリ）
            layout.ref_turn_cache_dir / rid,
        ]
        targets += sorted(layout.ops_dir.glob(f"{rid}.*.json"))  # refine ops キャッシュ
        for t in targets:
            if _quarantine(t, qdir, layout.output_root, purge):
                counts["files"] += 1
        # クレーム（run_tts_pool.sh の .claims/<id>）。完了・異常終了したクレームは印として
        # 残り、TTL（融合は90分）が切れるまで「他プロセスが処理中」と見なされて、wav を
        # 消しても再実行で飛ばされる。中身は owner.json だけなので退避せず消す。
        for audio_dir in (layout.base_audio_dir, layout.ref_audio_dir):
            for sub in (".claims", ".claims_recompose"):
                c = audio_dir / sub / rid
                if c.is_dir():
                    shutil.rmtree(c, ignore_errors=True)
                    counts["claims"] += 1

    # 追記系 JSONL から該当行を抜く（canonical と part の両方）
    for canonical in (layout.base_manifest, layout.base_transcripts,
                      layout.ref_manifest, layout.ref_transcripts,
                      layout.refined_records):
        for p in variant_files(canonical):
            counts["lines"] += _retire_records(p, ids, stamp)
    return dict(counts)


def _drop_from_skip(skip_file: Path, ids: set[str], stamp: str) -> int:
    """skip ファイルから該当 id を抜いて履歴へ退避する（作り直しを可能にする）。"""
    if not ids or not skip_file.exists():
        return 0
    kept, moved = [], []
    with open(skip_file) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            (moved if json.loads(s).get("id") in ids else kept).append(s)
    if moved:
        with open(skip_file.parent / f"{skip_file.stem}_history.jsonl", "a") as f:
            for s in moved:
                rec = json.loads(s)
                rec["_reset_at"] = stamp
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _atomic_write_lines(skip_file, kept)
    return len(moved)


def repair(layout: Layout, plan: dict[str, str], purge: str) -> dict:
    """plan に従って破棄する。Stage1-3 の再生成そのものは呼び出し側（--run-stages）。"""
    stamp = datetime.now().isoformat(timespec="seconds")
    by_stage = defaultdict(set)
    for rid, stage in plan.items():
        by_stage[stage].add(rid)

    all_ids = set(plan)
    # どの Stage から戻る場合も Stage4 の成果物は必ず捨てる
    s4 = purge_stage4(layout, all_ids, stamp, purge)
    logger.info(f"Stage4 成果物を破棄: ファイル{s4.get('files', 0)}件 / "
                f"JSONL {s4.get('lines', 0)}行 / クレーム{s4.get('claims', 0)}件"
                + (f" → {layout.output_root / '_audit_quarantine' / stamp}"
                   if purge == "move" and s4.get("files") else ""))

    # stage3 以前は voice_dialogues の行も捨てる（Stage2 の最後の変換がやり直す）
    voice_ids = by_stage["stage1"] | by_stage["stage2"] | by_stage["stage3"]
    n_voice = _retire_records(layout.voice_file, voice_ids, stamp)

    # stage2 以前は対話も捨てる（Stage2 が同じ id で引き直す）
    dial_ids = by_stage["stage1"] | by_stage["stage2"]
    n_dial = _retire_records(layout.dialogue_file, dial_ids, stamp)

    # stage1 はメタデータごと捨てる（Stage1 が新しい uuid を補充する）
    n_meta = _retire_records(layout.metadata_file, by_stage["stage1"], stamp)

    # 恒久スキップの記録が残っていると作り直しても素通しされるので抜いておく
    n_skip = (_drop_from_skip(layout.dialogue_skip, all_ids, stamp)
              + _drop_from_skip(layout.voice_skip, all_ids, stamp))

    with open(layout.repair_history, "a") as f:
        for rid, stage in sorted(plan.items()):
            f.write(json.dumps({"id": rid, "stage": stage, "at": stamp},
                               ensure_ascii=False) + "\n")

    logger.info(f"破棄: メタデータ{n_meta}件 / 対話{n_dial}件 / 音声対話{n_voice}件 / "
                f"恒久スキップ記録{n_skip}件")
    return {"stamp": stamp, "stage4": s4, "metadata": n_meta, "dialogue": n_dial,
            "voice": n_voice, "skip": n_skip,
            "by_stage": {k: len(v) for k, v in sorted(by_stage.items())}}


# --------------------------------------------------------------------------
# レポート
# --------------------------------------------------------------------------

def report(findings: list[Finding], plan: dict[str, str], max_ids: int) -> None:
    if not findings:
        logger.info("所見なし。すべての対話が Stage4 まで健全です。")
        return

    by_code = defaultdict(list)
    for f in findings:
        by_code[f.code].append(f)

    logger.info(f"--- 所見 {len(findings)}件 / 対象 {len(plan)}対話 ---")
    for code in CHECKS:                      # CHECKS の順（＝手前の Stage から）
        fs = by_code.get(code)
        if not fs:
            continue
        stage, desc = CHECKS[code]
        logger.info(f"[{stage}] {code}: {len(fs)}件 — {desc}")
        for f in fs[:max_ids]:
            logger.info(f"    {f.id}  {f.detail}")
        if len(fs) > max_ids:
            logger.info(f"    … 他 {len(fs) - max_ids}件（--max-ids で増やせます）")

    counts = Counter(plan.values())
    logger.info("--- 戻るべき Stage ---")
    for stage in sorted(counts, key=lambda s: STAGE_RANK[s]):
        logger.info(f"  {stage}: {counts[stage]}対話")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

async def _run_stages(config: dict, max_iters: int) -> None:
    """Stage1-2（＋変換）を回して埋め戻す（topup_failed_dialogues の収束ループをそのまま使う）。"""
    from topup_failed_dialogues import run as topup_run
    cost_tracker = CostTracker(
        pricing=config["pricing"],
        max_cost_usd=config["max_cost_usd"],
        cost_log_file=config["cost_log_file"],
    )
    await topup_run(config, cost_tracker, max_iters, with_voice=True)
    cost_tracker.save_log()


def main() -> int:
    p = argparse.ArgumentParser(
        description="Stage4 後の全数検査と、必要な Stage からの部分作り直し")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--pre", action="store_true",
                   help="テキスト系の検査だけ（Stage4 を回す前の事前確認に使う）")
    p.add_argument("--deep-audio", action="store_true",
                   help="wav のヘッダを開いて manifest と突き合わせる（遅い）")
    p.add_argument("--check-backlog", action="store_true",
                   help="Stage2 の積み残し（metadata はあるが未生成）も所見に含める")
    p.add_argument("--id", action="append", default=[],
                   help="この id だけを対象にする（複数可）")
    p.add_argument("--ids-file", help="対象 id を1行1件で書いたファイル")
    p.add_argument("--repair", action="store_true",
                   help="★破壊的。所見のある対話を該当 Stage まで戻して破棄する")
    p.add_argument("--force-stage", choices=sorted(STAGE_RANK),
                   help="所見によらず戻る Stage を固定する（--repair と併用）")
    p.add_argument("--run-stages", action="store_true",
                   help="--repair のあと Stage1-3 を回して埋め戻す（Stage4 は別途）")
    p.add_argument("--max-iters", type=int, default=5,
                   help="--run-stages の収束ループ上限（default 5）")
    p.add_argument("--purge", choices=("move", "delete"), default="move",
                   help="Stage4 成果物の処分。move=_audit_quarantine へ退避（既定）")
    p.add_argument("--report", help="所見を JSON で書き出すパス")
    p.add_argument("--max-ids", type=int, default=20,
                   help="所見ごとに表示する id の件数（default 20）")
    p.add_argument("--log-file", help="ログの追記先（既定は標準出力のみ）")
    args = p.parse_args()

    if args.force_stage and not args.repair:
        p.error("--force-stage は --repair と一緒に使ってください。")
    if args.run_stages and not args.repair:
        p.error("--run-stages は --repair と一緒に使ってください。")

    config = load_config(args.config)
    if args.log_file:
        setup_logging(args.log_file)
    else:
        logging.basicConfig(
            level=logging.INFO, stream=sys.stdout,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    layout = resolve_layout(config)
    only_ids = set(args.id) or None
    if args.ids_file:
        only_ids = (only_ids or set()) | {
            ln.strip() for ln in Path(args.ids_file).read_text().splitlines() if ln.strip()}

    logger.info(f"config={args.config} / 出力ルート={layout.output_root}")
    findings = audit(layout, config, pre=args.pre, deep_audio=args.deep_audio,
                     only_ids=only_ids, check_backlog=args.check_backlog)
    plan = plan_stages(findings, layout, force_stage=args.force_stage)
    report(findings, plan, args.max_ids)

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(
            {"config": args.config,
             "generated_at": datetime.now().isoformat(timespec="seconds"),
             "findings": [f.as_dict() for f in findings],
             "plan": plan}, ensure_ascii=False, indent=2))
        logger.info(f"レポート: {args.report}")

    if not findings:
        return 0
    if not args.repair:
        logger.info("検査のみ（既定）。作り直すなら --repair を付けてください。")
        return 1

    result = repair(layout, plan, args.purge)
    logger.info(f"作り直し対象: {result['by_stage']}")

    if args.run_stages:
        logger.info("--- Stage1-3 を回して埋め戻します ---")
        asyncio.run(_run_stages(config, args.max_iters))
    else:
        logger.info("次: Stage1-3 を回して埋め戻してください")
        logger.info(f"    .venv-qwen/bin/python topup_failed_dialogues.py --config {args.config}")

    logger.info("次: Stage4 を回し直してください（欠けた分だけが pending になります）")
    logger.info(f"    CONFIG={args.config} ./run_tts_sharded.sh")
    logger.info("そのあともう一度この検査を回して、所見が0になれば完了です。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
