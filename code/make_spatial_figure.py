"""
Spatial-selectivity figure: TCRE vs disc.

  (a) inter-electrode correlation matrices of one example participant (all trials), TCRE and disc,
      electrodes grouped by site (F4, P6, F3, P5). Default example: the participant whose
      disc-minus-TCRE within-site correlation is closest to the group median (not hand-picked).
  (b) within-site correlation for every participant, disc -> TCRE, overt and covert

Inputs: cleaned-epochs derivative (for panel a) and spatial_selectivity_per_subject.tsv written by
spatial_selectivity.py (for panel b). Correlation per epoch, Fisher-z averaged over epochs.
Array channels only (channel_sets.py). Read-only.

Usage:
    python make_spatial_figure.py /path/to/bids_root /path/to/output_dir [example_subject | auto]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from channel_sets import SITES, array_electrodes, load_epochs   # noqa: E402
from spatial_selectivity import fisher_mean_corr               # noqa: E402

TASK = "OvertCovert"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# diverging map (reference palette blue <-> red, neutral gray midpoint): correlations can be negative
CMAP = LinearSegmentedColormap.from_list(
    "div_r", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f2a9a8", "#e34948", "#8f1f1f"])
CMAP.set_bad("#ffffff")


def example_matrices(root, sub):
    f = root / "derivatives" / "cleaned-epochs" / sub / "eeg" / f"{sub}_task-{TASK}_desc-cleaned_epo.fif"
    ep = load_epochs(f, 250.0)
    el = array_electrodes(sub, len(ep.ch_names))
    x = ep.get_data()
    sites = [e[1] for e in el]
    mats = {}
    for mod, pos in (("TCRE", 2), ("Disc", 3)):
        r = fisher_mean_corr(x[:, [e[pos] for e in el], :])
        np.fill_diagonal(r, np.nan)
        mats[mod] = r
    return mats, sites


def make_figure(mats, sites, ex_sub, per, path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK_2,
                         "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2})
    fig = plt.figure(figsize=(7.2, 3.5))
    axm = [fig.add_axes([0.035, 0.20, 0.235, 0.64]), fig.add_axes([0.28, 0.20, 0.235, 0.64])]
    cax = fig.add_axes([0.53, 0.20, 0.012, 0.64])
    axp = fig.add_axes([0.71, 0.20, 0.275, 0.64])

    # (a) matrices
    n = len(sites)
    bounds = [i for i in range(1, n) if sites[i] != sites[i - 1]]
    centers = []
    start = 0
    for b in bounds + [n]:
        centers.append(((start + b - 1) / 2, sites[start]))
        start = b
    for ax, mod in zip(axm, ("Disc", "TCRE")):
        im = ax.imshow(mats[mod], cmap=CMAP, vmin=-1, vmax=1, interpolation="nearest")
        for b in bounds:
            ax.axhline(b - 0.5, color=INK_2, lw=0.6)
            ax.axvline(b - 0.5, color=INK_2, lw=0.6)
        ax.set_xticks([c for c, _ in centers])
        ax.set_xticklabels([s for _, s in centers])
        ax.set_yticks([c for c, _ in centers])
        ax.set_yticklabels([s for _, s in centers] if mod == "Disc" else [])
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(mod, fontsize=8.5, color=INK)
    fig.text(0.035, 0.93, f"(a) Inter-electrode correlation ({ex_sub}, all trials)",
             fontsize=9, color=INK)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Correlation (r)", color=INK)
    cb.outline.set_visible(False)

    # (b) paired within-site correlation
    fig.text(0.635, 0.93, "(b) Within-site r, all participants", fontsize=9, color=INK)
    d = per[per.modality.isin(["tripolar", "disc"])]
    xpos = {"Overt": (0.0, 1.0), "Covert": (2.0, 3.0)}
    for cond, (xd, xt) in xpos.items():
        g = d[d.condition == cond].pivot(index="participant_id", columns="modality", values="within_site_r")
        for _, row in g.iterrows():
            axp.plot([xd, xt], [row.disc, row.tripolar], color=GRID, lw=0.9, zorder=1)
            axp.scatter([xd, xt], [row.disc, row.tripolar], s=9, color=INK_2, zorder=2, lw=0)
        axp.plot([xd, xt], [g.disc.median(), g.tripolar.median()], color=INK, lw=2.0, zorder=3)
        lower = int((g.tripolar < g.disc).sum())
        axp.text((xd + xt) / 2, -0.13, cond, ha="center", va="top", fontsize=8, color=INK,
                 transform=axp.get_xaxis_transform())
        axp.text((xd + xt) / 2, -0.23, f"TCRE lower in {lower}/{len(g)}", ha="center", va="top", fontsize=6.5,
                 color=INK_2, transform=axp.get_xaxis_transform())
    axp.set_xticks([0, 1, 2, 3])
    axp.set_xticklabels(["Disc", "TCRE", "Disc", "TCRE"])
    axp.set_xlim(-0.4, 3.4)
    axp.set_ylim(0, 1.0)
    axp.set_ylabel("Within-site r")
    axp.grid(True, axis="y", color=GRID, lw=0.5)
    axp.spines[["top", "right"]].set_visible(False)
    axp.text(3.4, 0.02, "thin: participants\nthick: median", ha="right", va="bottom", fontsize=6.5, color=INK_2)
    fig.savefig(path, dpi=600, facecolor="white")
    plt.close(fig)


def representative(per):
    """Participant whose disc-minus-TCRE within-site r (mean of both conditions) is closest to the group median."""
    d = per[per.modality.isin(["tripolar", "disc"])]
    w = d.pivot_table(index="participant_id", columns="modality", values="within_site_r", aggfunc="mean")
    diff = w.disc - w.tripolar
    return (diff - diff.median()).abs().idxmin()


def main(root, out_dir, ex_sub="auto"):
    root, out_dir = Path(root).expanduser(), Path(out_dir).expanduser()
    per = pd.read_csv(out_dir / "spatial_selectivity_per_subject.tsv", sep="\t")
    if ex_sub == "auto":
        ex_sub = representative(per)
        print(f"Example participant (closest to the group median difference): {ex_sub}")
    mats, sites = example_matrices(root, ex_sub)
    make_figure(mats, sites, ex_sub, per, out_dir / "fig_spatial_selectivity.png")
    for mod, r in mats.items():
        same = np.array(sites)[:, None] == np.array(sites)[None, :]
        print(f"{ex_sub} {mod}: within-site r = {np.nanmean(r[same]):.2f}, between-site r = {np.nanmean(r[~same]):.2f}")
    print(f"Saved {out_dir / 'fig_spatial_selectivity.png'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "auto")
