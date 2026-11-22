"""対話単位の動的クレーム（generate_tts.DialogueClaims）と part ファイル名。

シャードの静的分割に代えて、各プロセスが未処理の対話を先着順に取る方式の要件:

  - 同じ対話を2プロセスが同時に掴まない（mkdir の原子性）
  - 取得後に完了（wav）が見つかったら処理しない ＝ NFS の属性キャッシュ越しでも
    二重合成しない
  - 失敗・スキップで手放した対話は、他プロセスが即座に取り直せる
  - 保持プロセスが落ちて更新の止まったクレームは TTL 経過後に引き取れる
    （これが無いと「クレームだけ残って永久に未処理」の対話ができる）
  - 書き手ごとの part ファイルがマージ側の正規表現と噛み合う。特に
    `*_discarded.jsonl`（audit_pipeline の退避先）を part と誤認しないこと
"""

import contextlib
import json
import os
import signal
import tempfile
import time
import unittest
from pathlib import Path

from generate_tts import (
    DialogueClaims, _SHARD_PART_RE, _exit_on_sigterm, merge_shard_outputs,
    part_output_path, shard_tag, worker_tag,
)

DID = "0000aaaa-1111-2222-3333-444455556666"


class DialogueClaimsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "audio" / ".claims"
        self.audio = Path(self._tmp.name) / "audio"
        self.audio.mkdir(parents=True, exist_ok=True)
        self._open = []

    def tearDown(self):
        for c in self._open:
            c.close()
        self._tmp.cleanup()

    def claims(self, ttl_sec=1800.0):
        # heartbeat は長めにして、テスト中に mtime が更新されないようにする
        c = DialogueClaims(self.root, ttl_sec=ttl_sec, heartbeat_sec=3600.0)
        self._open.append(c)
        return c

    def wav(self, did=DID):
        return self.audio / f"{did}.wav"

    def test_同じ対話を掴めるのは1プロセスだけ(self):
        a, b = self.claims(), self.claims()
        self.assertTrue(a.take(DID, self.wav()))
        self.assertFalse(b.take(DID, self.wav()))
        self.assertEqual(b.busy, 1)

    def test_取得後に完了済みなら処理しない(self):
        a = self.claims()
        self.wav().write_bytes(b"")           # 別プロセスが完了させた状態
        self.assertFalse(a.take(DID, self.wav()))
        self.assertEqual(a.done_elsewhere, 1)
        # 着手済みの印としてクレームは残す（他プロセスに掴み直させない）
        self.assertTrue((self.root / DID).is_dir())

    def test_未完了で手放した対話は他プロセスが取れる(self):
        a, b = self.claims(), self.claims()
        self.assertTrue(a.take(DID, self.wav()))
        self.assertFalse(b.take(DID, self.wav()))
        a.release(DID)                        # 失敗・スキップで手放す
        self.assertFalse((self.root / DID).exists())
        self.assertTrue(b.take(DID, self.wav()))

    def test_次の対話を取ると前の未完了は自動で解放される(self):
        # 呼び出し側は本体の途中で continue するだけでよい、という前提の担保。
        a, b = self.claims(), self.claims()
        other = "9999bbbb-1111-2222-3333-444455556666"
        self.assertTrue(a.take(DID, self.wav()))
        self.assertTrue(a.take(other, self.wav(other)))
        self.assertTrue(b.take(DID, self.wav()))          # 前の分は解放済み

    def test_完了した対話のクレームは残る(self):
        a, b = self.claims(), self.claims()
        self.assertTrue(a.take(DID, self.wav()))
        self.wav().write_bytes(b"")
        a.finish(DID)
        a.take("9999bbbb-1111-2222-3333-444455556666", None)   # 別の対話へ
        self.assertTrue((self.root / DID).is_dir())
        self.assertFalse(b.take(DID, self.wav()))              # 完了として素通り

    def test_close_は未完了のクレームを解放する(self):
        a, b = self.claims(), self.claims()
        self.assertTrue(a.take(DID, self.wav()))
        a.close()
        self.assertTrue(b.take(DID, self.wav()))

    def test_TTL切れのクレームは引き取れる(self):
        a = self.claims()
        b = self.claims(ttl_sec=60.0)
        self.assertTrue(a.take(DID, self.wav()))
        # 保持プロセスが落ちて heartbeat が止まった状態を mtime で作る
        old = time.time() - 3600
        os.utime(self.root / DID, (old, old))
        self.assertTrue(b.take(DID, self.wav()))
        self.assertEqual(b.stolen, 1)
        # 引き取り後は自分のものになっている（作り直されて mtime も新しい）
        self.assertGreater((self.root / DID).stat().st_mtime, old + 1800)

    def test_TTL内のクレームは引き取らない(self):
        a = self.claims()
        b = self.claims(ttl_sec=3600.0)
        self.assertTrue(a.take(DID, self.wav()))
        old = time.time() - 600            # 10分前 < TTL 60分
        os.utime(self.root / DID, (old, old))
        self.assertFalse(b.take(DID, self.wav()))

    def test_引き取った対話が完了済みなら処理しない(self):
        # 前回 run で完了した対話のクレームは TTL 切れになる。奪えてしまうが、
        # 取得後の wav 確認で弾かれる（＝二重合成しない）ことを確かめる。
        a = self.claims()
        b = self.claims(ttl_sec=60.0)
        self.assertTrue(a.take(DID, self.wav()))
        a.finish(DID)
        self.wav().write_bytes(b"")
        old = time.time() - 3600
        os.utime(self.root / DID, (old, old))
        self.assertFalse(b.take(DID, self.wav()))
        self.assertEqual(b.done_elsewhere, 1)


class InterruptReleasesClaimTest(unittest.TestCase):
    """Ctrl+C（KeyboardInterrupt）で抜けても、握っていたクレームが解放されること。

    解放されないと mtime が新しいまま残り、すぐ再実行してもその対話は TTL
    （融合モードの既定は90分）が切れるまで「他プロセスが処理中」と見なされて
    飛ばされる。run() / run_refine() が contextlib.closing で包んで保証している。
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "audio" / ".claims"
        self.audio = Path(self._tmp.name) / "audio"
        self.audio.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self._tmp.cleanup()

    def _claims(self):
        return DialogueClaims(self.root, ttl_sec=1800.0, heartbeat_sec=3600.0)

    def _run_loop(self, raise_at: str | None):
        """closing() で包んだ処理ループ（run/run_refine と同じ形）を模す。"""
        claims = self._claims()
        with contextlib.closing(claims):
            for did in ("d1", "d2"):
                if not claims.take(did, self.audio / f"{did}.wav"):
                    continue
                if did == raise_at:
                    raise KeyboardInterrupt
                (self.audio / f"{did}.wav").write_bytes(b"")
                claims.finish(did)

    def test_中断した対話のクレームは残らない(self):
        with self.assertRaises(KeyboardInterrupt):
            self._run_loop(raise_at="d2")
        self.assertTrue((self.root / "d1").exists(), "完了分の印は残る")
        self.assertFalse((self.root / "d2").exists(), "中断分は解放されるべき")

    def test_中断後にすぐ再実行すると続きから処理できる(self):
        with self.assertRaises(KeyboardInterrupt):
            self._run_loop(raise_at="d2")
        # 再実行: d1 は完了済みで飛ばし、d2 は取れる（TTL を待たない）
        claims = self._claims()
        with contextlib.closing(claims):
            self.assertFalse(claims.take("d1", self.audio / "d1.wav"))
            self.assertTrue(claims.take("d2", self.audio / "d2.wav"))

    def test_closing_無しだと中断分がTTLまで掴まれたままになる(self):
        """回帰の番人: 素の close() 頼みに戻すとこうなる、という対比。"""
        claims = self._claims()
        self.assertTrue(claims.take("d2", self.audio / "d2.wav"))
        del claims                             # close() を通さずに落ちた状態
        again = self._claims()
        with contextlib.closing(again):
            self.assertFalse(again.take("d2", self.audio / "d2.wav"))

    def test_SIGTERM_は_SystemExit_に変換される(self):
        """pkill（SIGTERM）でも巻き戻し（finally / closing）が走るようにするため。

        既定の SIGTERM はプロセスを即座に落とすので、この変換が無いと
        `pkill -f generate_tts` でクレームが解放されない。
        """
        with self.assertRaises(SystemExit):
            _exit_on_sigterm(signal.SIGTERM, None)


class PartPathTest(unittest.TestCase):
    def test_タグ無しならパスは変わらない(self):
        p = Path("output/audio_manifest_maai.jsonl")
        self.assertEqual(part_output_path(p, None), p)
        self.assertIsNone(shard_tag(None))
        self.assertIsNone(shard_tag((1, 1)))      # 1分割は分けない

    def test_シャードとワーカーの両方の名前がpartとして認識される(self):
        p = Path("output/audio_manifest_maai.jsonl")
        for tag in (shard_tag((2, 6)), worker_tag()):
            name = part_output_path(p, tag).name
            m = _SHARD_PART_RE.match(name)
            self.assertIsNotNone(m, name)
            self.assertEqual(m.group("base") + m.group("suffix"), p.name)

    def test_ワーカータグは英数字とハイフンだけ(self):
        # `_` を許すと *_discarded.jsonl を part と誤認する（下のテスト）
        self.assertRegex(worker_tag(), r"^w-[A-Za-z0-9-]+$")

    def test_破棄ファイルはpartとみなさない(self):
        for tag in ("2of6", "w-host-1234"):
            name = f"audio_manifest_x.part-{tag}_discarded.jsonl"
            self.assertIsNone(_SHARD_PART_RE.match(name), name)


class MergePartsTest(unittest.TestCase):
    def test_シャード名とワーカー名のpartをまとめて統合する(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "audio_manifest_fish_maai.jsonl"
            (root / "audio_manifest_fish_maai.part-1of2.jsonl").write_text(
                json.dumps({"id": "a", "v": 1}) + "\n")
            (root / "audio_manifest_fish_maai.part-w-node22-1234.jsonl").write_text(
                json.dumps({"id": "b", "v": 1}) + "\n")
            # 同じ id が複数の part にあれば後勝ち（マージの既定規則）
            (root / "audio_manifest_fish_maai.part-w-node23-9.jsonl").write_text(
                json.dumps({"id": "a", "v": 2}) + "\n")
            # 退避ファイルは取り込まない
            (root / "audio_manifest_fish_maai.part-1of2_discarded.jsonl").write_text(
                json.dumps({"id": "z", "v": 1}) + "\n")

            merge_shard_outputs(root)
            got = {json.loads(ln)["id"]: json.loads(ln)["v"]
                   for ln in canonical.read_text().splitlines() if ln.strip()}
            self.assertEqual(got, {"a": 2, "b": 1})

    def test_下のフォルダのpartは統合しない(self):
        # audio_fish/_stage4/ に退避した旧 Stage4 の part を、本番の統合で触らない（2026-11-16）
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub = root / "audio_fish" / "_stage4"
            sub.mkdir(parents=True)
            (sub / "audio_manifest_fish_maai.part-w-node22-1.jsonl").write_text(
                json.dumps({"id": "old", "v": 1}) + "\n")
            (root / "audio_manifest_fish_maai.part-w-node22-2.jsonl").write_text(
                json.dumps({"id": "new", "v": 1}) + "\n")
            self.assertEqual(merge_shard_outputs(root), 1)
            self.assertFalse((sub / "audio_manifest_fish_maai.jsonl").exists())
            got = [json.loads(ln)["id"] for ln in
                   (root / "audio_manifest_fish_maai.jsonl").read_text().splitlines()]
            self.assertEqual(got, ["new"])

    def test_extra_dirsの直下も統合する(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, other = Path(tmp) / "a", Path(tmp) / "b"
            root.mkdir(); other.mkdir()
            (other / "voice_dialogues_refined_x.part-w-h-1.jsonl").write_text(
                json.dumps({"id": "r"}) + "\n")
            self.assertEqual(merge_shard_outputs(root, [other, root]), 1)
            self.assertTrue((other / "voice_dialogues_refined_x.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
