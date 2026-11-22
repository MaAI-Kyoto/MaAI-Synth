#!/usr/bin/env python3
"""Make short demo videos (MP4) of synthesized dialogues: timeline + moving playhead + subtitles + stereo audio.

GitHub cannot play audio files inside a README, but it plays videos that are uploaded as attachments
(drag the MP4 into the README editor on github.com; it becomes https://github.com/user-attachments/assets/...).

    python tools/make_demo_videos.py --out data/videos --sec 60 \
        --pick ft_v1_04=data/ft_v1 --pick rp_v1_01=data/rp_v1

`--pick KEY=DIR` takes dialogue number NN of KEY "<scenario>_<NN>" from the run directory DIR, in the same order as
tools/build_demo_page.py (the order of voice_dialogues.jsonl). Requires ffmpeg (libx264, libass) and matplotlib.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_demo_page import COL_SYS, COL_USR, TITLES, load_records  # noqa: E402

W, H = 1280, 400
AX = (0.12, 0.40, 0.85, 0.34)          # axes position in figure fractions (left, bottom, width, height)


def draw_background(rec: dict, png: Path, sec: float, title: str, sys_name: str, usr_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.text(0.5, 0.93, title, ha="center", va="center", fontsize=17, weight="bold", color="#1f2328")
    fig.text(0.5, 0.855, f"left channel: system ({sys_name})   ·   right channel: user ({usr_name})",
             ha="center", va="center", fontsize=11, color="#656d76")
    ax = fig.add_axes(AX)
    lanes = {0: 1.0, 1: 0.0}
    for g in rec["segments"]:
        s, e = float(g["start"]), float(g["end"])
        if s >= sec:
            continue
        ch = int(g["channel"])
        bc = g.get("type") == "backchannel"
        h = 0.36 if bc else 0.7
        ax.broken_barh([(s, min(e, sec) - s)], (lanes.get(ch, 0.0) - h / 2, h),
                       facecolors=COL_SYS if ch == 0 else COL_USR, alpha=0.55 if bc else 0.95, edgecolor="none")
    ax.set_xlim(0, sec)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels([f"System\n({sys_name})", f"User\n({usr_name})"], fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlabel("time [s]", fontsize=9)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.legend(handles=[Patch(color=COL_SYS, label="system speech"), Patch(color=COL_USR, label="user speech"),
                       Patch(color="#888888", alpha=0.55, label="backchannel (thin bar)")],
              loc="lower right", fontsize=8, ncol=3, frameon=False, bbox_to_anchor=(1.0, 1.0))
    fig.savefig(png, facecolor="white")
    plt.close(fig)


def ass_time(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def bgr(hex_rgb: str) -> str:
    r, g, b = hex_rgb[1:3], hex_rgb[3:5], hex_rgb[5:7]
    return f"&H00{b}{g}{r}".upper()


def write_ass(rec: dict, path: Path, sec: float, font: str, max_chars: int) -> None:
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sys,{font},25,{bgr(COL_SYS)},&H000000FF,&H00FFFFFF,&H00FFFFFF,1,0,0,0,100,100,0,0,1,0,0,2,40,40,64,1
Style: Usr,{font},25,{bgr(COL_USR)},&H000000FF,&H00FFFFFF,&H00FFFFFF,1,0,0,0,100,100,0,0,1,0,0,2,40,40,22,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    for g in rec["segments"]:
        s, e = float(g["start"]), min(float(g["end"]), sec)
        if s >= sec:
            continue
        text = str(g["text"]).replace("\n", " ")
        if len(text) > max_chars:
            text = text[: max_chars - 1] + "…"
        style = "Sys" if int(g["channel"]) == 0 else "Usr"
        lines.append(f"Dialogue: 0,{ass_time(s)},{ass_time(max(e, s + 0.6))},{style},,0,0,0,,{g['speaker']}：{text}")
    path.write_text(head + "\n".join(lines) + "\n", encoding="utf-8")


def make_video(rec: dict, out: Path, sec: float, title: str, sys_name: str, usr_name: str, font: str,
               max_chars: int) -> None:
    sec = min(sec, float(rec["duration_sec"])) if sec > 0 else float(rec["duration_sec"])
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        draw_background(rec, td / "bg.png", sec, title, sys_name, usr_name)
        write_ass(rec, td / "subs.ass", sec, font, max_chars)
        x0, x1 = AX[0] * W, (AX[0] + AX[2]) * W
        y0, ah = (1 - AX[1] - AX[3]) * H, AX[3] * H
        fade = min(1.5, sec / 4)
        fc = (f"[0:v][1:v]overlay=x='{x0:.1f}+({x1 - x0:.1f})*t/{sec:.3f}':y={y0:.1f}[v];"
              f"[v]subtitles=subs.ass[vs]")
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
               "-loop", "1", "-framerate", "25", "-i", "bg.png",
               "-f", "lavfi", "-i", f"color=c=0xE03C31:s=3x{int(ah)}:r=25",
               "-i", str(rec["_audio_path"].resolve()),
               "-filter_complex", fc, "-map", "[vs]", "-map", "2:a",
               "-t", f"{sec:.3f}", "-af", f"afade=t=out:st={sec - fade:.3f}:d={fade:.3f}",
               "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p", "-crf", "24",
               "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out.resolve())]
        subprocess.run(cmd, check=True, cwd=td)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pick", action="append", required=True, metavar="SCENARIO_NN=DIR")
    ap.add_argument("--out", default="data/videos")
    ap.add_argument("--sec", type=float, default=60.0, help="length of each video (0 = whole dialogue)")
    ap.add_argument("--sys-name", default="Inoue")
    ap.add_argument("--usr-name", default="Mori")
    ap.add_argument("--font", default="Noto Sans CJK JP", help="font for the Japanese subtitles")
    ap.add_argument("--max-chars", type=int, default=44, help="cut longer subtitle lines")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for spec in args.pick:
        key, _, d = spec.partition("=")
        scen, _, nn = key.rpartition("_")
        rec = load_records(Path(d))[int(nn) - 1]
        title = f"{TITLES.get(scen, (scen, ''))[0]}  ({scen}, MaAI-Synth sample)"
        mp4 = out / f"{key}.mp4"
        make_video(rec, mp4, args.sec, title, args.sys_name, args.usr_name, args.font, args.max_chars)
        print(f"{mp4}  {mp4.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
