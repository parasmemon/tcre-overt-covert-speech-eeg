"""
Task-paradigm figure and timing table, computed from the BIDS events files (cue markers).

  (a) session: 4 runs x 50 trials, one condition per run; run orders observed across participants
  (b) single trial (protocol timing): picture for 3.5 s (marker at picture onset), named aloud or silently,
      then a 2.5-s grey-screen rest; baseline (-0.5 to 0 s) and analysis window (0 to 2.5 s) of the
      cleaned-epochs derivative

Outputs (in OUT_DIR):
  task_timing_per_subject.tsv   run order, median cue-to-cue interval, break durations
  fig_task_paradigm.png         600-dpi figure

Usage: python make_paradigm_figure.py /path/to/bids_root /path/to/output_dir [classic|bold|outline|mono|all]
"""
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

TRIALS_PER_RUN = 50
PICTURE_S = 3.5        # picture on screen, response period; the marker is at picture onset (protocol)
GREY_S = 2.5           # grey-screen inter-stimulus interval, rest (recording protocol)
COLORS = {"O": "#2a78d6", "C": "#eb6834"}          # same pair as the other figures (validated)
INK, INK_2, GRID, FILL = "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"


def timing(root):
    rows = []
    for f in sorted(Path(root).glob("sub-*/eeg/*_task-OvertCovert_events.tsv")):
        e = pd.read_csv(f, sep="\t")
        on, tt = e.onset.to_numpy(), e.trial_type.to_numpy()
        runs = np.arange(len(on)) // TRIALS_PER_RUN
        order = "".join(tt[runs == r][0].split("/")[1][0] for r in range(4))
        iti = np.concatenate([np.diff(on[runs == r]) for r in range(4)])
        gaps = [on[runs == r + 1][0] - on[runs == r][-1] for r in range(3)]
        rows.append({"participant_id": f.name.split("_")[0], "n_trials": len(on), "run_order": order,
                     "cue_interval_median_s": round(float(np.median(iti)), 2),
                     "run_duration_median_s": round(float(np.median([on[runs == r][-1] - on[runs == r][0]
                                                                     for r in range(4)])), 1),
                     "break_1_s": round(gaps[0]), "break_2_s": round(gaps[1]), "break_3_s": round(gaps[2])})
    return pd.DataFrame(rows)


THEMES = {
    # classic: light grey boxes, light-blue response period (default)
    "classic": dict(run_fc=FILL, run_ec=INK_2, run_lw=0.8, run_txt=INK, run_sub=INK_2,
                    resp_fc="#cfe0f6", resp_ec="none", resp_lw=0, resp_txt=INK, resp_hatch=None,
                    isi_fc=FILL, isi_ec="none", isi_lw=0, isi_ls="-", isi_txt=INK_2,
                    letters=COLORS, key="bars"),
    # bold: dark filled blocks with white text, high contrast
    "bold": dict(run_fc="#3a3a38", run_ec="#3a3a38", run_lw=0.8, run_txt="#ffffff", run_sub="#e4e3df",
                 resp_fc="#1c5cab", resp_ec="none", resp_lw=0, resp_txt="#ffffff", resp_hatch=None,
                 isi_fc="#d9d8d3", isi_ec="none", isi_lw=0, isi_ls="-", isi_txt=INK,
                 letters=COLORS, key="bars"),
    # outline: no fills, thin outlines; response period outlined in blue
    "outline": dict(run_fc="#ffffff", run_ec=INK, run_lw=0.9, run_txt=INK, run_sub=INK_2,
                    resp_fc="#ffffff", resp_ec=COLORS["O"], resp_lw=1.6, resp_txt=INK, resp_hatch=None,
                    isi_fc="#ffffff", isi_ec=INK_2, isi_lw=0.8, isi_ls="--", isi_txt=INK_2,
                    letters=COLORS, key="bars"),
    # mono: greyscale only (prints in black and white); response period hatched
    "mono": dict(run_fc="#f0efec", run_ec=INK, run_lw=0.8, run_txt=INK, run_sub=INK_2,
                 resp_fc="#d9d8d3", resp_ec=INK, resp_lw=0.8, resp_txt=INK, resp_hatch="////",
                 isi_fc="#ffffff", isi_ec=INK, isi_lw=0.8, isi_ls="-", isi_txt=INK_2,
                 letters={"O": INK, "C": INK_2}, key="text"),
}


def make_figure(df, path, theme="classic"):
    T = THEMES[theme]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "hatch.color": "#8a8983",
                         "hatch.linewidth": 0.5})
    iti = float(df.cue_interval_median_s.median())
    run_len = float(df.run_duration_median_s.median()) + iti
    brks = df[["break_1_s", "break_2_s", "break_3_s"]].to_numpy().astype(float)
    brk = float(np.median(brks))
    brk_min, brk_max = max(1, round(brks.min() / 60)), round(brks.max() / 60)   # nearest minute
    fig = plt.figure(figsize=(7.2, 4.4))
    ax_a = fig.add_axes([0.03, 0.56, 0.94, 0.36])
    ax_b = fig.add_axes([0.07, 0.10, 0.88, 0.30])

    # ---------------- (a) session structure (drawn in axes-fraction coordinates)
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.axis("off")
    ax_a.text(0, 1.02, "(a) Session: 4 runs of 50 trials, one condition per run", fontsize=9, color=INK,
              va="bottom")
    span = 0.78                                         # width used by the 4 runs + 3 breaks
    unit = span / (4 * run_len + 3 * brk)
    x = 0.0
    for r in range(4):
        w = run_len * unit
        ax_a.add_patch(Rectangle((x, 0.55), w, 0.30, fc=T["run_fc"], ec=T["run_ec"], lw=T["run_lw"]))
        ax_a.text(x + w / 2, 0.74, f"Run {r + 1}", ha="center", va="center", fontsize=8, color=T["run_txt"])
        ax_a.text(x + w / 2, 0.63, "50 trials", ha="center", va="center", fontsize=7, color=T["run_sub"])
        x += w
        if r < 3:
            wb = brk * unit
            ax_a.text(x + wb / 2, 0.70, "break", ha="center", va="center", fontsize=7, color=INK_2)
            x += wb
    # condition key (identity by colour + text, or text only in the greyscale theme)
    kx = span + 0.03
    for y, cond, txt in ((0.78, "O", "Overt run: name aloud"),
                         (0.56, "C", "Covert run: name silently")):
        if T["key"] == "bars":
            ax_a.add_patch(Rectangle((kx, y - 0.06), 0.008, 0.12, fc=T["letters"][cond], lw=0))
            ax_a.text(kx + 0.016, y, txt, fontsize=7, color=INK, va="center")
        else:
            ax_a.text(kx, y, cond, fontsize=9, fontweight="bold", color=T["letters"][cond], va="center")
            ax_a.text(kx + 0.018, y, txt, fontsize=7, color=INK, va="center")
    # observed run orders
    counts = Counter(df.run_order)
    ax_a.text(0, 0.36, "Run order across participants (O = overt run, C = covert run):",
              fontsize=7.5, color=INK, va="center")
    groups = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    gw = 0.98 / len(groups)
    for i, (order, n) in enumerate(groups):
        gx = i * gw
        for j, ch in enumerate(order):
            ax_a.text(gx + j * 0.022, 0.14, ch, fontsize=9, fontweight="bold", color=T["letters"][ch],
                      va="center")
        ax_a.text(gx + 4 * 0.022 + 0.006, 0.14, f"n = {n}", fontsize=7.5, color=INK_2, va="center")

    # ---------------- (b) single trial
    nxt = PICTURE_S + GREY_S                              # nominal next picture (protocol)
    ax_b.set_xlim(-0.8, nxt + 0.7)
    ax_b.set_ylim(0, 1)
    for sp in ("left", "right", "top"):
        ax_b.spines[sp].set_visible(False)
    ax_b.spines["bottom"].set_color(INK_2)
    ax_b.set_yticks([])
    ax_b.set_xticks([-0.5, 0, 1, 2, 2.5, 3, 3.5, 4, 5, 6])
    ax_b.set_xticklabels(["−0.5", "0", "1", "2", "2.5", "3", "3.5", "4", "5", "6"], color=INK_2)
    ax_b.set_xlabel("Time from marker (s)", color=INK)
    ax_b.text(-0.8, 1.30, "(b) Single trial", fontsize=9, color=INK, va="bottom")
    y0, h = 0.47, 0.36
    grey = dict(fc=T["isi_fc"], ec=T["isi_ec"], lw=T["isi_lw"], ls=T["isi_ls"])
    ax_b.add_patch(Rectangle((-0.8, y0), 0.8, h, **grey))
    ax_b.text(-0.4, y0 + h / 2, "rest", ha="center", va="center", fontsize=7, color=T["isi_txt"])
    ax_b.add_patch(Rectangle((0, y0), PICTURE_S, h, fc=T["resp_fc"], ec=T["resp_ec"], lw=T["resp_lw"],
                             hatch=T["resp_hatch"]))
    ax_b.text(PICTURE_S / 2, y0 + h / 2, f"picture ({PICTURE_S:.1f} s)\novert: name aloud\ncovert: name silently", ha="center",
              va="center", fontsize=7, linespacing=1.3,
              color=T["resp_txt"], bbox=dict(fc=T["resp_fc"], ec="none", pad=1.5) if T["resp_hatch"] else None)
    ax_b.add_patch(Rectangle((PICTURE_S, y0), GREY_S, h, **grey))
    ax_b.text(PICTURE_S + GREY_S / 2, y0 + h / 2,
              f"grey screen ({GREY_S:.1f} s)\nrest", ha="center",
              va="center", fontsize=7, color=T["isi_txt"], linespacing=1.3)
    ax_b.add_patch(Rectangle((nxt, y0), 0.7, h, fc=T["resp_fc"], ec=T["resp_ec"], lw=T["resp_lw"],
                             hatch=T["resp_hatch"]))
    for t0, lab in ((0, "marker\n(picture onset)"), (nxt, "next\npicture")):
        ax_b.annotate("", xy=(t0, y0 + h), xytext=(t0, y0 + h + 0.28),
                      arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0))
        ax_b.text(t0, y0 + h + 0.31, lab, ha="center", va="bottom", fontsize=7, color=INK, linespacing=1.1)
    # epoch definitions of the cleaned-epochs derivative (brackets)
    yb = 0.32
    for (t1, t2, lab) in ((-0.5, 0.0, "baseline"), (0.0, 2.5, "epoch analysis window (0–2.5 s)")):
        ax_b.plot([t1, t1, t2, t2], [yb + 0.05, yb, yb, yb + 0.05], color=INK_2, lw=0.8)
        ax_b.text((t1 + t2) / 2, yb - 0.04, lab, ha="center", va="top", fontsize=6.5, color=INK_2)
    fig.savefig(path, dpi=600, facecolor="white")
    plt.close(fig)


def main(root, out_dir, theme="classic"):
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = timing(Path(root).expanduser())
    df.to_csv(out_dir / "task_timing_per_subject.tsv", sep="\t", index=False)
    if theme == "all":
        for t in THEMES:
            make_figure(df, out_dir / f"fig_task_paradigm_{t}.png", t)
    else:
        make_figure(df, out_dir / "fig_task_paradigm.png", theme)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "paradigm_output",
         sys.argv[3] if len(sys.argv) > 3 else "classic")
