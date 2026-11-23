#!/usr/bin/env python3
"""Plot the distribution of turn-taking gaps in generated dialogues (the figure in the README).

    python tools/plot_gap_distribution.py data/ft_v2/transcripts_maai_ref.jsonl \
        --out docs/img/gap_distribution.png --dark-out docs/img/gap_distribution_dark.png

gap = start of the next speaker's speech - end of the previous speaker's speech (negative = overlap).
Backchannels are left out and consecutive utterances of the same speaker are merged into one turn, so every
gap is a change of speaker. Several transcript files are pooled. The statistics are printed as JSON.
Requires matplotlib.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

THEMES = {  # colors match GitHub's light / dark page so the PNG sits on the README without a frame
    "light": dict(bg="#ffffff", ink="#1f2328", ink2="#59636e", muted="#818b98", grid="#e6e8eb",
                  band="#f3f4f6", bar="#2a78d6"),
    "dark": dict(bg="#0d1117", ink="#f0f6fc", ink2="#9198a1", muted="#7d8590", grid="#262c36",
                 band="#161b22", bar="#3987e5"),
}


def dialogue_gaps(segments: list[dict]) -> list[float]:
    """Gaps between speaker changes of one dialogue (`segments` of a transcripts_*.jsonl record)."""
    speech = sorted((s for s in segments if s.get("type", "speech") == "speech"), key=lambda s: float(s["start"]))
    turns: list[list] = []  # [channel, start, end]
    for s in speech:
        ch, st, en = int(s["channel"]), float(s["start"]), float(s["end"])
        if turns and turns[-1][0] == ch:
            turns[-1][2] = max(turns[-1][2], en)
        else:
            turns.append([ch, st, en])
    return [b[1] - a[2] for a, b in zip(turns, turns[1:])]


def load_gaps(paths: list[Path]) -> tuple[np.ndarray, int]:
    gaps: list[float] = []
    n_dialogues = 0
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    gaps.extend(dialogue_gaps(json.loads(line)["segments"]))
                    n_dialogues += 1
    return np.asarray(gaps, dtype=float), n_dialogues


def summarize(g: np.ndarray, n_dialogues: int) -> dict:
    q = np.percentile(g, [5, 25, 50, 75, 95])
    return {
        "dialogues": n_dialogues, "speaker_changes": int(g.size),
        "mean": round(float(g.mean()), 3), "sd": round(float(g.std()), 3),
        "p05": round(float(q[0]), 3), "p25": round(float(q[1]), 3), "median": round(float(q[2]), 3),
        "p75": round(float(q[3]), 3), "p95": round(float(q[4]), 3),
        "overlap_rate": round(float(np.mean(g < 0)), 4),
        "over_0.5s_rate": round(float(np.mean(g > 0.5)), 4),
        "over_1.0s_rate": round(float(np.mean(g > 1.0)), 4),
    }


def plot(g: np.ndarray, stats: dict, out: Path, theme: str, subtitle: str,
         lo: float, hi: float, bin_sec: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = THEMES[theme]
    edges = np.arange(lo, hi + bin_sec / 2, bin_sec)
    counts, _ = np.histogram(g, bins=edges)
    share = counts / g.size * 100
    outside = float(np.mean((g < lo) | (g > hi)))
    ymax = share.max() * 1.18

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": True})
    fig = plt.figure(figsize=(8, 3.9), dpi=200, facecolor=c["bg"])
    ax = fig.add_axes((0.085, 0.2, 0.89, 0.57), facecolor=c["bg"])
    fig.text(0.02, 0.94, "Turn-taking gaps in generated dialogues", fontsize=13, weight="bold", color=c["ink"])
    fig.text(0.02, 0.875, subtitle, fontsize=8.5, color=c["ink2"])

    ax.axvspan(lo, 0, color=c["band"], zorder=0, lw=0)
    ax.bar(edges[:-1], share, width=bin_sec * 0.84, align="edge", color=c["bar"], lw=0, zorder=2)
    ax.axvline(0, color=c["muted"], lw=0.8, zorder=1)

    med = stats["median"]
    ax.plot([med, med], [0, ymax * 0.9], color=c["ink2"], lw=0.9, ls=(0, (3, 2)), zorder=3)
    ax.text(med + 0.02, ymax * 0.9, f"median {med:.2f} s", color=c["ink"], fontsize=8.5, va="center")
    ax.text(lo + 0.03, ymax * 0.9, "overlap", color=c["ink"], fontsize=8.5, weight="bold", va="center")
    ax.text(lo + 0.03, ymax * 0.78, f"{stats['overlap_rate'] * 100:.1f}% of speaker changes\n"
            "start before the other\nspeaker has finished", color=c["ink2"], fontsize=7.5, va="top", linespacing=1.3)
    ax.text(0.62, share.max() * 0.32, f"longer than 0.5 s: {stats['over_0.5s_rate'] * 100:.1f}%",
            color=c["ink2"], fontsize=7.5, va="bottom")

    ax.set_xlim(lo, hi)
    ax.set_ylim(0, ymax)
    ax.set_xticks(np.arange(np.ceil(lo / 0.2) * 0.2, hi + 1e-9, 0.2))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.1f}".replace("-", "\u2212") if abs(v) > 1e-9 else "0"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.set_xlabel("gap [s]  =  start of the next speaker  −  end of the previous speaker", fontsize=8.5, color=c["ink2"])
    ax.set_ylabel(f"share per {bin_sec * 1000:.0f} ms bin", fontsize=8.5, color=c["ink2"])
    ax.tick_params(colors=c["muted"], labelcolor=c["ink2"], labelsize=8, length=3, width=0.6)
    ax.grid(axis="y", color=c["grid"], lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(c["muted"])
    ax.spines["bottom"].set_linewidth(0.8)
    fig.text(0.02, 0.025, "Backchannels are left out; consecutive utterances of one speaker are merged. "
             f"{outside * 100:.1f}% of the gaps fall outside the plotted range.", fontsize=7, color=c["muted"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=c["bg"])
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("transcripts", nargs="+", type=Path, help="transcripts_*.jsonl written by Stage 3")
    ap.add_argument("--out", type=Path, default=Path("docs/img/gap_distribution.png"))
    ap.add_argument("--dark-out", type=Path, default=None, help="also draw a version for GitHub's dark theme")
    ap.add_argument("--subtitle", default=None, help="second line of the figure (default: counts)")
    ap.add_argument("--range", nargs=2, type=float, default=(-0.6, 1.2), metavar=("LO", "HI"))
    ap.add_argument("--bin", type=float, default=0.025, help="bin width in seconds")
    args = ap.parse_args()

    g, n = load_gaps(args.transcripts)
    stats = summarize(g, n)
    print(json.dumps(stats, ensure_ascii=False))
    subtitle = args.subtitle or f"{n:,} dialogues, {g.size:,} speaker changes"
    plot(g, stats, args.out, "light", subtitle, *args.range, args.bin)
    if args.dark_out:
        plot(g, stats, args.dark_out, "dark", subtitle, *args.range, args.bin)


if __name__ == "__main__":
    main()
