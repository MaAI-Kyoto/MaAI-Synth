"""backend="maaibatch"（MaAI 公式リポジトリの MaaiBatch）の契約テスト。★2026-11-22

vapbatch の契約（tests/test_vapbatch_contract.py）と同じ「外すと黙って結果がずれる」ものを
MaaiBatchScorer でも固定する。数値の等価性は compare_vap_backends.py --tests test4 で測る。

★torch / transformers と MaAI リポジトリ（MaAI/src）と HF キャッシュの重みが要る。
  pip 版 maai と混ざらないよう、このファイルは単独で走らせること:

      export HF_HOME=<your HF cache> HF_HUB_OFFLINE=1
      .venv-maai-cu12/bin/python -m unittest tests.test_maaibatch_contract -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

from retime_maai import MaaiBatchParams, RetimeConfig, make_scorer
from retime_maaibatch import DEFAULT_REPO_SRC

CHUNK = 1280        # 16000 / 12.5

try:
    import torch  # noqa: F401
    from transformers import MimiModel  # noqa: F401
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

_HF = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
HAVE_MODEL = (HAVE_TORCH and Path(DEFAULT_REPO_SRC, "maai", "batch.py").exists()
              and (_HF / "models--maai-kyoto--vap_jp").is_dir()
              and (_HF / "models--maai-kyoto--vap_bc_jp").is_dir())
_SKIP = "torch/transformers・MaAI/src・HF キャッシュのどれかが無い（Stage3 の venv と HF_HOME で実行する）"


def _cfg(device="cpu", mc=False):
    cfg = RetimeConfig(backend="maaibatch", device=device, frame_rate=12.5,
                       model_type="normal-ver2", maaibatch=MaaiBatchParams(mc=mc))
    cfg.vap.context_len_sec = 20
    cfg.bc.context_len_sec = 20
    return cfg


class TestConfig(unittest.TestCase):
    def test_backend_parsed_from_config(self):
        rc = RetimeConfig.from_tts_cfg({"retime": {"backend": "maaibatch",
                                                   "maaibatch": {"mc": True, "batch_size": 4}}})
        self.assertEqual(rc.backend, "maaibatch")
        self.assertTrue(rc.maaibatch.mc)
        self.assertEqual(rc.maaibatch.batch_size, 4)

    def test_typo_in_maaibatch_block_is_rejected(self):
        with self.assertRaises(ValueError):
            RetimeConfig.from_tts_cfg({"retime": {"backend": "maaibatch",
                                                  "maaibatch": {"mcc": True}}})

    def test_summary_records_repo_and_variant(self):
        s = RetimeConfig(backend="maaibatch", maaibatch=MaaiBatchParams(mc=True)).summary()
        self.assertEqual(s["backend"], "maaibatch")
        self.assertTrue(s["maaibatch"]["vap_mc"])
        self.assertIn("repo_commit", s["maaibatch"])

    def test_vap_mc_mode_means_mc(self):
        rc = RetimeConfig(backend="maaibatch")
        rc.vap.mode = "vap_mc"
        self.assertTrue(rc.summary()["maaibatch"]["vap_mc"])


@unittest.skipUnless(HAVE_MODEL, _SKIP)
class TestScorerContract(unittest.TestCase):
    """MaaiScorer / BatchVapScorer と同じ契約。"""

    @classmethod
    def setUpClass(cls):
        cls.vap = make_scorer(_cfg(), "vap")
        cls.bc = make_scorer(_cfg(), "bc")
        rng = np.random.default_rng(0)
        cls.k = 30
        n = cls.k * CHUNK
        cls.a = (rng.standard_normal(n) * 0.05).astype(np.float32)
        cls.b = (rng.standard_normal(n) * 0.05).astype(np.float32)

    def test_uses_repo_maai_not_pip(self):
        import maai
        self.assertIn(str(Path(DEFAULT_REPO_SRC).resolve()), str(Path(maai.__file__).resolve()))

    def test_standard_vap_by_default(self):
        """mc を書かなければ標準 VAP（新しい Maai の既定 mc=True に引きずられない）。"""
        self.assertEqual(self.vap.variant, "vap")
        self.assertEqual(self.bc.variant, "bc")

    def test_drops_first_frame_like_maai(self):
        for sc in (self.vap, self.bc):
            r = sc.score(self.a, self.b)
            self.assertEqual(len(r), self.k - 1, sc.mode)
            self.assertEqual(r[0][0], 2 * CHUNK, sc.mode)
            self.assertEqual(r[-1][0], self.k * CHUNK, sc.mode)

    def test_end_samples_are_a_uniform_grid(self):
        es = np.array([e for e, _ in self.vap.score(self.a, self.b)])
        self.assertTrue(np.all(np.diff(es) == CHUNK))

    def test_result_keys_match_maai(self):
        rv = self.vap.score(self.a, self.b)[0][1]
        self.assertEqual(set(rv), {"p_now", "p_future", "vad"})
        for k in ("p_now", "p_future", "vad"):
            self.assertEqual(len(rv[k]), 2, k)
        rb = self.bc.score(self.a, self.b)[0][1]
        self.assertEqual(set(rb), {"p_bc", "p_bc_detect"})
        self.assertIsInstance(rb["p_bc"], float)

    def test_zero_pads_to_frame_multiple(self):
        """末尾の半端は切り上げる（MaaiBatch 単体は捨てるので、ラッパで MaaiScorer に揃える）。"""
        r = self.vap.score(self.a[: 10 * CHUNK + 137], self.b[: 10 * CHUNK + 137])
        self.assertEqual(len(r), 10)
        self.assertEqual(r[-1][0], 11 * CHUNK)

    def test_short_input_returns_empty(self):
        self.assertEqual(self.vap.score(self.a[:CHUNK], self.b[:CHUNK]), [])

    def test_vap_channel_order(self):
        n = 30 * CHUNK
        loud = (np.random.default_rng(1).standard_normal(n) * 0.3).astype(np.float32)
        quiet = np.zeros(n, dtype=np.float32)
        p_a = np.mean([fr["p_now"][0] for _, fr in self.vap.score(loud, quiet)][-10:])
        p_b = np.mean([fr["p_now"][0] for _, fr in self.vap.score(quiet, loud)][-10:])
        self.assertGreater(p_a, p_b)

    def test_bc_channel_swap(self):
        n = 30 * CHUNK
        loud = (np.random.default_rng(2).standard_normal(n) * 0.3).astype(np.float32)
        quiet = np.zeros(n, dtype=np.float32)
        p_ok = np.mean([fr["p_bc"] for _, fr in self.bc.score(loud, quiet)][-10:])
        p_rev = np.mean([fr["p_bc"] for _, fr in self.bc.score(quiet, loud)][-10:])
        self.assertNotAlmostEqual(p_ok, p_rev, places=3)

    def test_right_padding_does_not_change_earlier_frames(self):
        r1 = self.vap.score(self.a, self.b)
        pad = np.zeros(10 * CHUNK, dtype=np.float32)
        r2 = self.vap.score(np.concatenate([self.a, pad]), np.concatenate([self.b, pad]))
        self.assertGreater(len(r2), len(r1))
        for (e1, f1), (e2, f2) in zip(r1, r2[: len(r1)]):
            self.assertEqual(e1, e2)
            for ch in (0, 1):
                self.assertAlmostEqual(f1["p_now"][ch], f2["p_now"][ch], places=5)

    def test_score_long_many_matches_individual_calls(self):
        both = self.bc.score_long_many([(self.a, self.b), (self.b, self.a)])
        for r, xs in zip(both, [(self.a, self.b), (self.b, self.a)]):
            single = self.bc.score(*xs)
            self.assertEqual([e for e, _ in r], [e for e, _ in single])
            for (_, f1), (_, f2) in zip(r, single):
                self.assertAlmostEqual(f1["p_bc"], f2["p_bc"], places=5)
        self.assertNotAlmostEqual(both[0][-1][1]["p_bc"], both[1][-1][1]["p_bc"], places=3)

    def test_stats_are_recorded(self):
        sc = make_scorer(_cfg(), "vap")
        self.assertEqual(sc.stats.calls, 0)
        sc.score(self.a, self.b)
        self.assertEqual(sc.stats.calls, 1)
        self.assertEqual(sc.stats.frames, self.k)


class TestImportGuard(unittest.TestCase):
    def test_foreign_maai_is_rejected(self):
        """別の場所の maai（pip 版）が先に import されていたら落とす。"""
        import types
        from retime_maaibatch import import_maai_from_repo
        if not Path(DEFAULT_REPO_SRC, "maai", "batch.py").exists():
            self.skipTest("MaAI/src が無い")
        saved = sys.modules.get("maai")
        fake = types.ModuleType("maai")
        fake.__file__ = "/somewhere/site-packages/maai/__init__.py"
        sys.modules["maai"] = fake
        try:
            with self.assertRaises(RuntimeError):
                import_maai_from_repo(DEFAULT_REPO_SRC)
        finally:
            if saved is None:
                sys.modules.pop("maai", None)
            else:
                sys.modules["maai"] = saved


if __name__ == "__main__":
    unittest.main()
