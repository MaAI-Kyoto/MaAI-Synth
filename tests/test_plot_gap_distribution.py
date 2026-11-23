"""tools/plot_gap_distribution.py: how gaps are counted (speaker changes only, backchannels left out)."""
import numpy as np

from plot_gap_distribution import dialogue_gaps, summarize


def test_gaps_merge_same_speaker_and_skip_backchannels():
    segs = [
        {"start": 0.0, "end": 1.0, "channel": 0, "type": "speech"},
        {"start": 1.1, "end": 2.0, "channel": 0, "type": "speech"},       # same speaker: one turn 0.0-2.0
        {"start": 1.5, "end": 1.8, "channel": 1, "type": "backchannel"},  # not a turn
        {"start": 2.3, "end": 3.0, "channel": 1, "type": "speech"},       # gap +0.3
        {"start": 2.8, "end": 4.0, "channel": 0, "type": "speech"},       # overlap -0.2
    ]
    assert np.allclose(dialogue_gaps(segs), [0.3, -0.2])


def test_gaps_are_order_independent():
    segs = [
        {"start": 2.3, "end": 3.0, "channel": 1, "type": "speech"},
        {"start": 0.0, "end": 2.0, "channel": 0, "type": "speech"},
    ]
    assert np.allclose(dialogue_gaps(segs), [0.3])


def test_summarize_rates():
    s = summarize(np.array([-0.2, 0.1, 0.3, 0.7]), 1)
    assert s["speaker_changes"] == 4
    assert s["overlap_rate"] == 0.25
    assert s["over_0.5s_rate"] == 0.25
