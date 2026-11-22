import unittest

import numpy as np

from retime_maai import LatencyParams, MaaiRetimer, RetimeConfig, ScoreTimeline


class FakeScorer:
    """一定間隔のフレームで p_now を返すダミースコアラ。

    p_seq は「next_ch 側の p_now」の並び。frame_dur ごとに 1 フレーム進む。
    """

    def __init__(self, p_seq, frame_dur):
        self.p_seq = p_seq
        self.frame_dur = frame_dur

    def score(self, x0, x1, scan_from=0):
        step = int(round(self.frame_dur * 16000))
        for k, p in enumerate(self.p_seq):
            yield (k + 1) * step, {"p_now": [p, p]}


def make_retimer(**latency_kwargs):
    cfg = RetimeConfig(frame_rate=12.5, probe_sec=2.0,
                       latency=LatencyParams(**latency_kwargs))
    cfg.vap.threshold = 0.55
    cfg.vap.min_consecutive_frames = 2
    cfg.vap.context_len_sec = 20
    cfg.vap.max_overlap = 0.5
    cfg.vap.min_gap = 0.10
    return cfg


class SubframeInterpolationTests(unittest.TestCase):
    """R1: 閾値クロスの副フレーム補間でフレーム格子の櫛が消えること。"""

    def _decide(self, subframe, p_seq, anchor=25.0):
        cfg = make_retimer(enabled=False, subframe=subframe)
        r = MaaiRetimer(cfg)
        r._vap = FakeScorer(p_seq, 1.0 / cfg.frame_rate)
        r._seed_rng("t")
        timeline = ScoreTimeline(num_ch=2, sr_engine=16000)
        return r._decide_vap(next_ch=1, timeline=timeline, frontier=anchor,
                             frontier_idx=0, failed_turns=set(),
                             prev_placed_end=anchor, llm_off=0.3,
                             frame_dur=1.0 / cfg.frame_rate)

    def test_offsets_land_on_frame_grid_without_subframe(self):
        """補間なしだと、p の値が違っても開始時刻は同じ格子点に落ちる。"""
        offs = set()
        for p in (0.60, 0.75, 0.90):
            start, method, _, anchor = self._decide(False, [0.1] * 250 + [p, p] + [0.9] * 20)
            self.assertEqual(method, "vap")
            offs.add(round(start - anchor, 6))
        self.assertEqual(len(offs), 1, "補間なしでは1つの格子点に潰れるはず")

    def test_subframe_spreads_offsets_between_frames(self):
        """補間ありなら、閾値の超え方に応じて連続的にずれる。"""
        offs = []
        for p in (0.60, 0.75, 0.90):
            start, method, _, anchor = self._decide(True, [0.1] * 250 + [p, p] + [0.9] * 20)
            self.assertEqual(method, "vap")
            offs.append(start - anchor)
        self.assertEqual(len(set(np.round(offs, 6))), 3, "3通りの連続値になるはず")
        # p が高いほど閾値を早く跨ぐ = 開始が早い
        self.assertTrue(offs[0] > offs[1] > offs[2])
        # 補間結果は隣接2フレームの間に必ず収まる
        span = 1.0 / 12.5
        self.assertLess(max(offs) - min(offs), span)


class LatencyModelTests(unittest.TestCase):
    """R2/R3/R4: 潜時サンプリング・min_gap 抑止・フォールバック。"""

    def make(self, **kw):
        cfg = make_retimer(enabled=True, **kw)
        r = MaaiRetimer(cfg)
        r._vap = FakeScorer([0.1] * 250 + [0.9] * 20, 1.0 / cfg.frame_rate)
        return cfg, r

    def test_latency_is_added_and_recorded(self):
        cfg, r = self.make()
        r._seed_rng("d1")
        timeline = ScoreTimeline(num_ch=2, sr_engine=16000)
        start, method, _, anchor = r._decide_vap(1, timeline, 25.0, 0, set(), 25.0, 0.3,
                                                 1.0 / cfg.frame_rate)
        self.assertEqual(method, "vap")
        d = r._vap_detail
        self.assertIn("tc_offset", d)
        self.assertIsNotNone(d["latency"])
        # 記録された tc_offset + latency が実際の配置と一致する
        self.assertAlmostEqual(start - anchor, d["tc_offset"] + d["latency"], places=3)

    def test_same_dialogue_id_is_reproducible(self):
        cfg, r = self.make()
        out = []
        for _ in range(2):
            r._seed_rng("same-id")
            out.append([r._sample_latency(0.0)[1] for _ in range(20)])
        self.assertEqual(out[0], out[1])
        r._seed_rng("other-id")
        self.assertNotEqual(out[0], [r._sample_latency(0.0)[1] for _ in range(20)])

    def test_sampled_offsets_respect_bounds(self):
        cfg, r = self.make(max_overlap=0.4, max_gap=2.0)
        r._seed_rng("bounds")
        vals = [r._sample_latency(0.0)[0] for _ in range(20000)]
        self.assertGreaterEqual(min(vals), -0.4)
        self.assertLessEqual(max(vals), 2.0)

    def test_no_min_gap_spike_when_latency_enabled(self):
        """R3: min_gap のクランプが効くと +min_gap に人工スパイクが立つ。"""
        cfg, r = self.make()
        r._seed_rng("spike")
        vals = np.array([r._sample_latency(0.0)[0] for _ in range(20000)])
        in_forbidden = ((vals > 0.0) & (vals < cfg.vap.min_gap)).mean()
        at_min_gap = (np.abs(vals - cfg.vap.min_gap) < 1e-9).mean()
        self.assertGreater(in_forbidden, 0.01, "(0, min_gap) が空になってはいけない")
        self.assertLess(at_min_gap, 1e-3, "min_gap ちょうどに質量が溜まってはいけない")

    def test_fallback_uses_latency_not_llm_offset(self):
        """R4: 閾値クロス無しでも LLM の定型オフセットを使わない。"""
        cfg, r = self.make()
        r._vap = FakeScorer([0.1] * 60, 1.0 / cfg.frame_rate)   # 一度も閾値を超えない
        r._seed_rng("fb")
        timeline = ScoreTimeline(num_ch=2, sr_engine=16000)
        start, method, p, anchor = r._decide_vap(1, timeline, 25.0, 0, set(), 25.0, 1.2,
                                                 1.0 / cfg.frame_rate)
        self.assertEqual(method, "fallback-latency")
        self.assertIsNone(p)
        self.assertNotAlmostEqual(start - anchor, 1.2, places=3)

    def test_fallback_keeps_llm_when_disabled(self):
        cfg = make_retimer(enabled=False)
        r = MaaiRetimer(cfg)
        r._vap = FakeScorer([0.1] * 60, 1.0 / cfg.frame_rate)
        r._seed_rng("fb2")
        timeline = ScoreTimeline(num_ch=2, sr_engine=16000)
        start, method, _, anchor = r._decide_vap(1, timeline, 25.0, 0, set(), 25.0, 1.2,
                                                 1.0 / cfg.frame_rate)
        self.assertEqual(method, "fallback-llm")
        self.assertAlmostEqual(start - 25.0, 1.2, places=3)


class DisabledIsUnchangedTests(unittest.TestCase):
    """latency.enabled=false のとき、従来の挙動が完全に保たれること。"""

    def test_min_gap_clamp_still_applies(self):
        cfg = make_retimer(enabled=False, subframe=False)
        r = MaaiRetimer(cfg)
        # 閾値超えがちょうどアンカー直後 → (0, min_gap) に入り min_gap に持ち上がる
        r._vap = FakeScorer([0.1] * 251 + [0.9] * 20, 1.0 / cfg.frame_rate)
        r._seed_rng("x")
        timeline = ScoreTimeline(num_ch=2, sr_engine=16000)
        anchor = 25.0 - 0.5 - 20.0 + 251 * (1.0 / 12.5) - 0.01   # クロスが anchor の直後に来る
        start, method, _, a = r._decide_vap(1, timeline, anchor + 20.5, 0, set(),
                                            anchor + 20.5, 0.3, 1.0 / cfg.frame_rate)
        off = start - a
        self.assertTrue(off <= -1e-9 or off >= cfg.vap.min_gap - 1e-9,
                        f"(0, min_gap) に落ちてはいけない: {off}")
        self.assertEqual(r._vap_detail.get("latency"), None)


if __name__ == "__main__":
    unittest.main()
