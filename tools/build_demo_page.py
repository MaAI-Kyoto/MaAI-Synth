#!/usr/bin/env python3
"""Build a static listening page (GitHub Pages) from Stage 3 outputs.

For every run directory (the `data/<scenario>/` of a config) it takes the final dialogues
(`transcripts_maai_ref.jsonl` + `audio/maai_ref/*.flac`), converts the audio to MP3, draws a
two-lane timeline of who speaks when, and writes `index.html` with players and clickable transcripts.

    python tools/build_demo_page.py --out docs \
        --run ft_v1=data/ft_v1 --run qa_v1=data/qa_v1 --max-per-run 5

Requires ffmpeg (MP3) and matplotlib (timelines). The page has no external dependencies.
"""
from __future__ import annotations

import argparse
import html
import json
import subprocess
from pathlib import Path

TITLES = {  # scenario name -> (English title, Japanese title)
    "ft_v1": ("First meeting", "初対面の雑談"),
    "ft_v2": ("First meeting v2 (persona cards)", "初対面の雑談 v2（ユーザ側の人物カードあり）"),
    "ft_sil_v1": ("First meeting + long user silences", "初対面の雑談＋ユーザ側の長い沈黙"),
    "ft_sil_v2": ("First meeting + long user silences v2", "初対面の雑談＋ユーザ側の長い沈黙 v2"),
    "ft_5min_v1": ("First meeting, 5 minutes", "初対面の雑談（約5分）"),
    "al_v1": ("Attentive listening", "傾聴（ユーザが話し、システムが聞き役）"),
    "rp_v1": ("Overlapping speech", "声が重なる場面（割り込み・同時発話など）"),
    "qa_v1": ("Question answering", "質問応答（ユーザが質問し、システムが答える）"),
    "cs_v1": ("Customer service", "接客"),
    "main_v1": ("General conversation", "いろいろな場面の会話"),
    "tri_ft_v1": ("Three-party first meeting", "三人の初対面"),
}
COL_SYS, COL_USR = "#2f6db5", "#d9822b"


def fmt(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def find_transcripts(run: Path) -> Path:
    for name in ("transcripts_maai_ref.jsonl", "transcripts_fish_speech_maai_ref.jsonl"):
        if (run / name).exists():
            return run / name
    cands = sorted(run.glob("transcripts_*_ref.jsonl"))
    cands = [c for c in cands if ".part-" not in c.name]
    if not cands:
        raise SystemExit(f"{run}: no transcripts_*_ref.jsonl (run generate_tts.py --retime-refine, then --merge-parts)")
    return cands[0]


def audio_dir_for(run: Path, transcripts: Path) -> Path:
    variant = transcripts.stem.replace("transcripts_", "")
    return run / "audio" / variant


def load_records(run: Path) -> list[dict]:
    tr = find_transcripts(run)
    recs = {}
    for line in open(tr, encoding="utf-8"):
        r = json.loads(line)
        recs[r["id"]] = r
    order = []
    vd = run / "voice_dialogues.jsonl"
    if vd.exists():
        order = [json.loads(line)["id"] for line in open(vd, encoding="utf-8")]
    ids = [i for i in order if i in recs] + [i for i in recs if i not in order]
    adir = audio_dir_for(run, tr)
    out = []
    for i in ids:
        r = recs[i]
        r["_audio_path"] = adir / r["audio"]
        if r["_audio_path"].exists():
            out.append(r)
    return out


def draw_timeline(rec: dict, png: Path, span: float | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    dur = float(rec["duration_sec"])
    end = min(dur, span) if span else dur
    fig, ax = plt.subplots(figsize=(12, 1.9), dpi=110)
    lanes = {0: 1.0, 1: 0.0}
    for g in rec["segments"]:
        s, e = float(g["start"]), float(g["end"])
        if s > end:
            continue
        ch = int(g["channel"])
        bc = g.get("type") == "backchannel"
        h = 0.36 if bc else 0.7
        y = lanes.get(ch, 0.0) - h / 2
        ax.broken_barh([(s, min(e, end) - s)], (y, h), facecolors=COL_SYS if ch == 0 else COL_USR,
                       alpha=0.55 if bc else 0.95, edgecolor="none")
    ax.set_xlim(0, end)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels(["System (ch0)", "User (ch1)"], fontsize=9)
    ax.set_xlabel("time [s]", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.legend(handles=[Patch(color=COL_SYS, label="system speech"), Patch(color=COL_USR, label="user speech"),
                       Patch(color="#888888", alpha=0.55, label="backchannel (thin bar)")],
              loc="upper right", fontsize=7, ncol=3, frameon=False, bbox_to_anchor=(1.0, 1.32))
    fig.tight_layout()
    fig.savefig(png, transparent=False, facecolor="white")
    plt.close(fig)


def to_mp3(src: Path, dst: Path, bitrate: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
                    "-c:a", "libmp3lame", "-b:a", bitrate, str(dst)], check=True)


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root{{--bg:#ffffff;--fg:#1f2328;--mut:#656d76;--card:#f6f8fa;--line:#d0d7de;--sys:{sys};--usr:{usr};--hl:#fff3c4}}
@media (prefers-color-scheme: dark){{:root{{--bg:#0d1117;--fg:#e6edf3;--mut:#9198a1;--card:#161b22;--line:#30363d;--hl:#3b3420}}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,-apple-system,"Segoe UI","Hiragino Sans","Noto Sans JP",sans-serif}}
main{{max-width:1040px;margin:0 auto;padding:24px 16px 64px}}
h1{{font-size:1.7rem;margin:.2em 0}} h2{{font-size:1.25rem;margin:2em 0 .3em;padding-top:.6em;border-top:1px solid var(--line)}}
.sub{{color:var(--mut)}} .ja{{color:var(--mut);font-size:.92em}}
.legend span{{display:inline-block;width:.9em;height:.9em;border-radius:2px;vertical-align:-.1em;margin:0 .3em 0 .8em}}
nav a{{margin-right:.8em;white-space:nowrap}} a{{color:var(--sys)}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:12px 0}}
.card h3{{margin:0 0 6px;font-size:1rem}} .card img{{width:100%;height:auto;border-radius:6px;background:#fff;cursor:pointer}}
audio{{width:100%;margin:6px 0}} details summary{{cursor:pointer;color:var(--mut)}}
.tr{{max-height:340px;overflow:auto;border-top:1px solid var(--line);margin-top:6px}}
.tr div{{display:grid;grid-template-columns:4.2em 6.5em 1fr;gap:.5em;padding:2px 4px;cursor:pointer;border-radius:4px}}
.tr div:hover{{background:var(--line)}} .tr div.on{{background:var(--hl)}}
.tr .t{{color:var(--mut);font-variant-numeric:tabular-nums}} .tr .s0{{color:var(--sys)}} .tr .s1{{color:var(--usr)}}
.tr .bc{{opacity:.75;font-size:.92em}}
</style></head><body><main>
<h1>{title}</h1>
<p class="sub">{subtitle}</p>
<p class="legend">Stereo: <span style="background:var(--sys)"></span>left = ch0 system ({sys_name})
<span style="background:var(--usr)"></span>right = ch1 user ({usr_name}) · thin bars = backchannels · click a line to jump there.</p>
<nav>{nav}</nav>
{sections}
<p class="sub" style="margin-top:3em">Generated by <code>tools/build_demo_page.py</code>. Generated dialogues follow the licenses and terms of the LLM, TTS model and voices used to make them.</p>
</main>
<script>
document.querySelectorAll('.card').forEach(card=>{{
  const a=card.querySelector('audio'), rows=[...card.querySelectorAll('.tr div')], img=card.querySelector('img');
  rows.forEach(r=>r.addEventListener('click',()=>{{a.currentTime=+r.dataset.s;a.play();}}));
  if(img) img.addEventListener('click',e=>{{const b=img.getBoundingClientRect();a.currentTime=(e.clientX-b.left)/b.width*(+img.dataset.span);a.play();}});
  a.addEventListener('timeupdate',()=>{{const t=a.currentTime;let last=null;
    rows.forEach(r=>{{const on=t>=+r.dataset.s&&t<+r.dataset.e;r.classList.toggle('on',on);if(on)last=r;}});
    if(last&&card.querySelector('details').open){{const box=last.parentElement;const y=last.offsetTop-box.offsetTop;
      if(y<box.scrollTop||y>box.scrollTop+box.clientHeight-30)box.scrollTop=y-60;}}}});
}});
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=DIR",
                    help="scenario name and its output directory (repeatable, page order)")
    ap.add_argument("--out", default="docs", help="output directory (default docs/)")
    ap.add_argument("--max-per-run", type=int, default=5)
    ap.add_argument("--bitrate", default="96k", help="MP3 bitrate (default 96k)")
    ap.add_argument("--timeline-sec", type=float, default=0.0,
                    help="draw only the first N seconds (0 = whole dialogue)")
    ap.add_argument("--title", default="MaAI-Synth — synthesized spoken dialogues")
    ap.add_argument("--subtitle", default="Two-party spoken dialogues generated end-to-end: LLM scripts → TTS → turn-taking and backchannel timing by MaAI.")
    ap.add_argument("--sys-name", default="Inoue")
    ap.add_argument("--usr-name", default="Mori")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    (out / "img").mkdir(parents=True, exist_ok=True)
    nav, sections, index = [], [], []
    for spec in args.run:
        name, _, d = spec.partition("=")
        recs = load_records(Path(d))[: args.max_per_run]
        en, ja = TITLES.get(name, (name, ""))
        nav.append(f'<a href="#{name}">{html.escape(en)}</a>')
        cards = []
        for n, r in enumerate(recs, 1):
            key = f"{name}_{n:02d}"
            to_mp3(r["_audio_path"], out / "samples" / f"{key}.mp3", args.bitrate)
            span = min(float(r["duration_sec"]), args.timeline_sec) if args.timeline_sec else float(r["duration_sec"])
            draw_timeline(r, out / "img" / f"{key}.png", span)
            rows = []
            for g in r["segments"]:
                bc = g.get("type") == "backchannel"
                rows.append(
                    f'<div data-s="{float(g["start"]):.2f}" data-e="{float(g["end"]):.2f}"{" class=bc" if bc else ""}>'
                    f'<span class="t">{fmt(float(g["start"]))}</span>'
                    f'<span class="s{int(g["channel"])}">{html.escape(str(g["speaker"]))}{"（相槌）" if bc else ""}</span>'
                    f'<span>{html.escape(str(g["text"]))}</span></div>')
            cards.append(
                f'<div class="card" id="{key}"><h3>{html.escape(en)} #{n:02d} '
                f'<span class="sub">· {fmt(float(r["duration_sec"]))}</span></h3>'
                f'<img src="img/{key}.png" alt="timeline of {key}" data-span="{span:.2f}" loading="lazy">'
                f'<audio controls preload="none" src="samples/{key}.mp3"></audio>'
                f'<details><summary>Transcript (with backchannels)</summary><div class="tr">{"".join(rows)}</div></details></div>')
            n_bc = sum(1 for g in r["segments"] if g.get("type") == "backchannel")
            index.append({"key": key, "scenario": name, "id": r["id"], "duration_sec": r["duration_sec"],
                          "segments": len(r["segments"]), "backchannels": n_bc})
        sections.append(f'<h2 id="{name}">{html.escape(en)} <span class="ja">{html.escape(ja)}</span></h2>'
                        f'<p class="sub">config: <code>configs/{name}.yaml</code></p>' + "".join(cards))
        print(f"{name}: {len(recs)} dialogues")
    (out / "index.html").write_text(PAGE.format(
        title=html.escape(args.title), subtitle=html.escape(args.subtitle), sys=COL_SYS, usr=COL_USR,
        sys_name=html.escape(args.sys_name), usr_name=html.escape(args.usr_name),
        nav=" ".join(nav), sections="".join(sections)), encoding="utf-8")
    (out / "samples" / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / ".nojekyll").write_text("")
    print(f"wrote {out / 'index.html'}")


if __name__ == "__main__":
    main()
