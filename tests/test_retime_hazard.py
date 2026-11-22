"""R2 別案（ハザード方式）のテスト。

閾値クロス後に毎フレームくじを引き、当たった時刻 + N(0.3, sd) を開始点にする。
既存の潜時モデル（LatencyParams）は無効化せずに残してあり、hazard.enabled が
立っているときだけ「クロスした枝」が置き換わる、というのが前提。
"""

import unittest

import numpy as np

from retime_maai import (HazardParams, LatencyParams, MaaiRetimer, RetimeConfig,
                         ScoreTimeline)
from test_retime_latency import FakeScorer, make_retimer

FRAME = 1.0 / 12.5
ANCHOR = 25.0
# 250フレーム分は閾値未満、その後ずっと 0.9（＝窓の中でクロスする）
P_SEQ = [0.1] * 250 + [0.9] * 40


def make(hazard=None, latency=None):
    cfg = make_retimer(**(latency or {"enabled": True}))
    if hazard is not None:
        cfg.hazard = HazardParams(**hazard)
    r = MaaiRetimer(cfg)
    r._vap = FakeScorer(P_SEQ, FRAME)
    return cfg, r


def decide(r, cfg, dialogue_id="d", p_seq=None):
    if p_seq is not None:
        r._vap = FakeScorer(p_seq, FRAME)
    r._seed_rng(dialogue_id)
    timeline = ScoreTimeline(num_ch=2, sr_engine=16000)
    return r._decide_vap(1, timeline, ANCHOR, 0, set(), ANCHOR, 0.3, FRAME)


class HazardBasicTests(unittest.TestCase):

    def test_method_and_detail_are_recorded(self):
        cfg, r = make(hazard={"enabled": True})
        start, method, p, anchor = decide(r, cfg)
        self.assertEqual(method, "vap-hazard")
        d = r._vap_detail
        for k in ("tc_offset", "fire_offset", "hazard_wait", "hazard_rolls",
                  "start_offset"):
            self.assertIn(k, d)
        # 記録から実配置がそのまま再現できる（発火時刻 + オフセット）
        self.assertAlmostEqual(start - anchor,
                               d["fire_offset"] + d["start_offset"], places=3)  # 記録は小数4桁丸め
        # クロス確認フレームは、副フレーム補間した tc より後ろにある
        self.assertGreaterEqual(d["fire_offset"] - d["hazard_wait"], d["tc_offset"])

    def test_offset_is_within_bounds_and_centered(self):
        cfg, r = make(hazard={"enabled": True, "offset_mean": 0.3, "offset_sd": 0.1})
        r._seed_rng("off")
        vals = np.array([r._sample_start_offset() for _ in range(20000)])
        self.assertGreaterEqual(vals.min(), 0.0)
        self.assertLessEqual(vals.max(), 0.6)
        self.assertAlmostEqual(vals.mean(), 0.3, delta=0.01)
        self.assertAlmostEqual(vals.std(), 0.1, delta=0.01)

    def test_high_p_fires_almost_immediately(self):
        """p=0.9 をそのまま確率に使うと、ほとんど最初の1回で当たる。"""
        cfg, r = make(hazard={"enabled": True})
        rolls = []
        for i in range(300):
            decide(r, cfg, dialogue_id=f"d{i}")
            rolls.append(r._vap_detail["hazard_rolls"])
        self.assertGreater(np.mean(np.array(rolls) == 1), 0.8)

    def test_lower_scale_makes_the_wait_longer(self):
        """scale を下げると待ちフレーム数が増える（時間スケールのつまみ）。"""
        waits = {}
        for scale in (1.0, 0.1):
            cfg, r = make(hazard={"enabled": True, "scale": scale})
            w = []
            for i in range(300):
                decide(r, cfg, dialogue_id=f"s{i}")
                w.append(r._vap_detail["hazard_wait"])
            waits[scale] = float(np.mean(w))
        self.assertGreater(waits[0.1], waits[1.0] + 0.2)

    def test_timeout_when_the_die_never_fires(self):
        """窓内で一度も当たらなければ timeout として記録される。"""
        cfg, r = make(hazard={"enabled": True, "scale": 0.0, "max_prob": 0.0})
        start, method, _, anchor = decide(r, cfg)
        self.assertEqual(method, "vap-hazard-timeout")
        # 窓の右端（anchor + probe_sec）+ オフセットあたりに落ちる
        self.assertGreater(start - anchor, cfg.probe_sec - 0.2)

    def test_reproducible_for_the_same_dialogue_id(self):
        cfg, r = make(hazard={"enabled": True})
        a = decide(r, cfg, dialogue_id="same")[0]
        b = decide(r, cfg, dialogue_id="same")[0]
        self.assertEqual(a, b)

    def test_result_stays_inside_the_allowed_window(self):
        cfg, r = make(hazard={"enabled": True, "scale": 0.2})
        for i in range(200):
            start, _, _, anchor = decide(r, cfg, dialogue_id=f"w{i}")
            self.assertGreaterEqual(start - anchor, -cfg.vap.max_overlap - 1e-9)
            self.assertLessEqual(start - anchor, cfg.vap.max_gap + 1e-9)


class HazardDoesNotDisturbTheOldPathTests(unittest.TestCase):
    """hazard.enabled=false なら既存の挙動が1ビットも変わらないこと。"""

    def test_latency_path_is_unchanged(self):
        cfg_a, ra = make(hazard={"enabled": False})
        cfg_b = make_retimer(enabled=True)          # hazard フィールドは既定（無効）
        rb = MaaiRetimer(cfg_b)
        rb._vap = FakeScorer(P_SEQ, FRAME)
        a = decide(ra, cfg_a, dialogue_id="x")
        b = decide(rb, cfg_b, dialogue_id="x")
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[1], "vap")

    def test_fallback_branch_is_untouched_by_hazard(self):
        """クロスしない交替は hazard を有効にしても R4（fallback-latency）のまま。"""
        cfg, r = make(hazard={"enabled": True})
        start, method, p, anchor = decide(r, cfg, p_seq=[0.1] * 300)
        self.assertEqual(method, "fallback-latency")
        self.assertIsNone(p)


if __name__ == "__main__":
    unittest.main()
