"""
Event-related potentials (ERP): average response after the cue, tripolar (TCRE) vs disc,
overt vs covert speech, per scalp site (F3, F4, P5, P6).

Uses the cleaned-epochs derivative (1-40 Hz, -0.5 to 2.5 s, baseline -0.5 to 0 s), resampled to
250 Hz. For every participant, modality, condition and channel:
  ERP = average over trials, divided by the channel's single-trial standard deviation in the
        pre-cue window (-0.4 to 0 s, both conditions pooled)
  -> amplitude in "baseline SD units". This removes the unknown amplification of the tripolar
     output and differences between channels, so TCRE and disc can be shown on one scale and
     participants can be averaged. Overt and covert share the same scale factor.
Then averaged over the array electrodes of each site, and over participants (mean +/- SEM).

Channel convention: odd channels = tripolar, even channels = disc. Array channels only
(channel_sets.py). All 16 participants. Nothing in the dataset is modified.
Per-participant results are cached in OUT_DIR/cache.

Outputs (in OUT_DIR):
  erp_group.npz             times, group mean and SEM per modality/condition/site, per-subject ERPs
  erp_peak_per_subject.tsv  largest absolute ERP deflection 0-1 s (baseline SD units) and its latency
  erp_summary.tsv           group mean of the peak per modality/condition/site
  fig_erp_tripolar_disc.png 600-dpi figure (2 rows x 4 sites)

Usage:
    python erp_analysis.py /path/to/bids_root /path/to/output_dir
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from channel_sets import array_electrodes, cached, load_epochs   # noqa: E402

TASK = "OvertCovert"
CONDITIONS = ["Overt", "Covert"]
MODALITIES = ["tripolar", "disc"]
SITES_PLOT = ["F3", "F4", "P5", "P6"]          # left/right frontal, left/right parietal
TARGET_SFREQ = 250.0
BASE = (-0.4, 0.0)
PEAK_WIN = (0.0, 1.0)

COLORS = {"Overt": "#2a78d6", "Covert": "#eb6834"}   # validated categorical pair (light surface)
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def subject_erp(fpath, sub):
    ep = load_epochs(fpath, TARGET_SFREQ)
    t = ep.times
    el = array_electrodes(sub, len(ep.ch_names))
    x = ep.get_data()                                            # epochs x channels x times
    is_overt = ep.events[:, 2] == ep.event_id["Stimulus/Overt"]
    b = (t >= BASE[0]) & (t < BASE[1])
    sd = x[:, :, b].transpose(1, 0, 2).reshape(x.shape[1], -1).std(axis=1)   # per channel
    sd[sd == 0] = np.nan
    out = {}
    for cond, m in (("Overt", is_overt), ("Covert", ~is_overt)):
        erp = x[m].mean(axis=0) / sd[:, None]                     # channels x times, baseline-SD units
        for mod, pos in (("tripolar", 2), ("disc", 3)):
            for site in SITES_PLOT:
                idx = [e[pos] for e in el if e[1] == site]
                out[(mod, cond, site)] = np.nanmean(erp[idx], axis=0)
    return t, out


def make_figure(t, group, n, path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK_2,
                         "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2})
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9), sharex=True, sharey=True, constrained_layout=True)
    letters = iter("abcdefgh")
    for r, mod in enumerate(MODALITIES):
        for c, site in enumerate(SITES_PLOT):
            ax = axes[r, c]
            ax.axhline(0, color=GRID, lw=0.8, zorder=0)
            ax.axvline(0, color=INK_2, lw=0.8, ls="--", zorder=1)
            for cond in CONDITIONS:
                mean, sem = group[(mod, cond, site)]
                ax.fill_between(t, mean - sem, mean + sem, color=COLORS[cond], alpha=0.18, lw=0)
                ax.plot(t, mean, color=COLORS[cond], lw=1.5, label=cond)
            ax.set_title(f"({next(letters)}) {'TCRE' if mod == 'tripolar' else 'Disc'} - {site}",
                         loc="left", fontsize=8.5, color=INK)
            ax.grid(True, axis="y", color=GRID, lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            if r == 1:
                ax.set_xlabel("Time from picture onset (s)")
            if c == 0:
                ax.set_ylabel("Amplitude\n(baseline SD units)")
    axes[0, 3].legend(frameon=False, loc="upper right", fontsize=7.5)
    axes[0, 0].set_xlim(t[0], t[-1])
    fig.suptitle(f"Group-mean ERP (mean ± SEM, n = {n}); each channel scaled by its pre-stimulus SD",
                 fontsize=9, color=INK_2, x=0.01, ha="left")
    fig.savefig(path, dpi=600, facecolor="white")
    plt.close(fig)


def main(root, out_dir):
    root, out_dir = Path(root).expanduser(), Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted((root / "derivatives" / "cleaned-epochs").glob(f"sub-*/eeg/*_task-{TASK}_desc-cleaned_epo.fif"))
    if not files:
        sys.exit("No cleaned-epochs files found. Run make_cleaned_epochs.py first.")

    acc, rows, t = {}, [], None
    for f in files:
        sub = f.name.split("_")[0]
        t, erp = cached(out_dir, "erp", sub, lambda: subject_erp(f, sub))
        w = (t >= PEAK_WIN[0]) & (t <= PEAK_WIN[1])
        for k, v in erp.items():
            acc.setdefault(k, []).append(v)
            i = np.nanargmax(np.abs(v[w]))
            rows.append({"participant_id": sub, "modality": k[0], "condition": k[1], "site": k[2],
                         "peak_abs_sd_units": round(float(abs(v[w][i])), 3),
                         "peak_latency_s": round(float(t[w][i]), 3)})
        print(f"{sub}: done", flush=True)

    group = {k: (np.nanmean(v, axis=0), np.nanstd(v, axis=0, ddof=1) / np.sqrt(len(v))) for k, v in acc.items()}
    np.savez(out_dir / "erp_group.npz", times=t,
             **{f"mean_{m}_{c}_{s}": g[0] for (m, c, s), g in group.items()},
             **{f"sem_{m}_{c}_{s}": g[1] for (m, c, s), g in group.items()},
             **{f"subjects_{m}_{c}_{s}": np.array(v) for (m, c, s), v in acc.items()})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "erp_peak_per_subject.tsv", sep="\t", index=False)
    summ = (df.groupby(["modality", "condition", "site"])
              .agg(n=("peak_abs_sd_units", "size"),
                   peak_mean=("peak_abs_sd_units", "mean"), peak_sd=("peak_abs_sd_units", "std"),
                   latency_median_s=("peak_latency_s", "median"))
              .round(3).reset_index())
    summ.to_csv(out_dir / "erp_summary.tsv", sep="\t", index=False)
    make_figure(t, group, len(files), out_dir / "fig_erp_tripolar_disc.png")
    print("\n" + summ.to_string(index=False))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "erp_output")
