"""
Time-frequency representation (TFR) and ERD/ERS for tripolar (TCRE) and disc signals.

Uses the cleaned-epochs derivative (1-40 Hz, -0.5 to 2.5 s).
  - Morlet wavelets, 3-40 Hz (1 Hz steps), n_cycles = max(f/2, 2)
  - each epoch is mirror-padded by 1 s on both sides before the wavelet transform, then cropped
    back, so the -0.5 to 0 s baseline is not contaminated by edge artefacts
  - power averaged over epochs per channel, expressed as % change from baseline (-0.4 to 0 s;
    the first 0.1 s of the epoch is skipped and 0.1 s is trimmed from both ends of the maps):
        ERD = power decrease (negative %), ERS = power increase (positive %)
  - then averaged over the channels of each modality -> one map per participant/modality/condition
% change is a ratio, so the result does not depend on the unknown amplification of the tripolar output.

Channel convention: odd channels = tripolar, even channels = disc. Array channels only: non-array
channels (EOG) are excluded as listed in channel_sets.py. All 16 participants
included. Nothing in the dataset is modified. Per-participant maps are cached in OUT_DIR/cache,
so a stopped run resumes where it stopped.

Outputs (in OUT_DIR):
  tfr_band_erd_per_subject.tsv   mean % change in alpha (8-13 Hz) and beta (13-30 Hz), 0-2.5 s
  tfr_band_erd_stats.tsv         Wilcoxon: ERD vs 0, and tripolar vs disc
  tfr_group_maps.npz             group-median maps (freqs, times, 4 maps) for re-plotting
  fig_tfr_erd_ers.png            600-dpi figure, 2 x 2 (tripolar/disc x overt/covert)

Usage:
    python tfr_erd_ers.py /path/to/bids_root /path/to/output_dir
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mne
import numpy as np
import pandas as pd
from mne.time_frequency import tfr_array_morlet
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from channel_sets import array_channels, cached, load_epochs   # noqa: E402

TASK = "OvertCovert"
CONDITIONS = ["Overt", "Covert"]
MODALITIES = ["tripolar", "disc"]
FREQS = np.arange(3.0, 41.0, 1.0)
N_CYCLES = np.maximum(FREQS / 2.0, 2.0)
TARGET_SFREQ = 250.0
DECIM = 5                       # TFR output at 50 Hz time resolution
PAD_S = 1.0                     # mirror padding (s) on each side
BASELINE = (-0.4, 0.0)          # first 0.1 s skipped: wavelet edge region
EDGE_TRIM = 0.1                 # s trimmed from both ends of the maps
WINDOW = (0.0, 2.5)             # window for band summaries; adjust to the speech period if needed
BANDS = {"alpha": (8, 13), "beta": (13, 30)}
CHUNK = 20                      # epochs per wavelet batch (keeps memory low)

# Diverging map: blue = ERD (decrease), neutral gray midpoint, red = ERS (increase)
CMAP = LinearSegmentedColormap.from_list(
    "erd_ers", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f2a9a8", "#e34948", "#8f1f1f"])
INK, INK_2 = "#0b0b0b", "#52514e"


def mean_power(x, sf):
    """x: epochs x channels x times -> mean power over epochs (channels x freqs x times_decim)."""
    pad = int(PAD_S * sf)
    acc, n = None, 0
    for i in range(0, len(x), CHUNK):
        xp = np.pad(x[i:i + CHUNK], ((0, 0), (0, 0), (pad, pad)), mode="reflect")
        p = tfr_array_morlet(xp, sf, FREQS, n_cycles=N_CYCLES, output="power", decim=DECIM, verbose=False)
        p = p.sum(axis=0)
        acc = p if acc is None else acc + p
        n += len(xp)
    acc /= n
    cut = pad // DECIM
    return acc[..., cut:acc.shape[-1] - cut]


def subject_maps(fpath, sub):
    ep = load_epochs(fpath, TARGET_SFREQ)
    sf = ep.info["sfreq"]
    tri, disc = array_channels(sub, len(ep.ch_names))
    picks = np.concatenate([tri, disc])            # only array channels go through the wavelets
    n_tri = len(tri)
    times = ep.times[::DECIM]
    maps = {}
    for cond in CONDITIONS:
        x = ep[f"Stimulus/{cond}"].get_data()[:, picks]
        p = mean_power(x, sf)[..., :len(times)]                         # channels x freqs x times
        b = (times >= BASELINE[0]) & (times <= BASELINE[1])
        base = p[..., b].mean(axis=-1, keepdims=True)
        pct = (p - base) / base * 100.0
        keep = (times >= times[0] + EDGE_TRIM - 1e-9) & (times <= times[-1] - EDGE_TRIM + 1e-9)
        maps[("tripolar", cond)] = pct[:n_tri].mean(axis=0)[:, keep]
        maps[("disc", cond)] = pct[n_tri:].mean(axis=0)[:, keep]
    return times[keep], maps


def band_values(times, m):
    w = (times >= WINDOW[0]) & (times <= WINDOW[1])
    return {b: float(m[(FREQS >= lo) & (FREQS <= hi)][:, w].mean()) for b, (lo, hi) in BANDS.items()}


def stats_table(df):
    rows = []
    for b in BANDS:
        for mod in MODALITIES:
            for cond in CONDITIONS:
                v = df.query("modality == @mod and condition == @cond")[b]
                rows.append({"band": b, "test": f"{mod} {cond}: change vs 0", "n": len(v),
                             "mean_pct": round(v.mean(), 1), "sd_pct": round(v.std(), 1),
                             "negative_in": f"{int((v < 0).sum())}/{len(v)}",
                             "wilcoxon_p": round(wilcoxon(v).pvalue, 4)})
        for cond in CONDITIONS:
            t = df.query("modality == 'tripolar' and condition == @cond").set_index("participant_id")[b]
            d = df.query("modality == 'disc' and condition == @cond").set_index("participant_id")[b]
            t, d = t.align(d, join="inner")
            rows.append({"band": b, "test": f"{cond}: tripolar vs disc", "n": len(t),
                         "mean_pct": f"{t.mean():.1f} vs {d.mean():.1f}", "sd_pct": "",
                         "negative_in": "", "wilcoxon_p": round(wilcoxon(t, d).pvalue, 4)})
    return pd.DataFrame(rows)


def make_figure(times, group, n, path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK_2,
                         "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2})
    lim = 100.0   # power cannot fall below -100 %; increases above +100 % saturate (colorbar arrow)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True, sharey=True, constrained_layout=True)
    letters = iter("abcd")
    for r, mod in enumerate(MODALITIES):
        for c, cond in enumerate(CONDITIONS):
            ax = axes[r, c]
            im = ax.pcolormesh(times, FREQS, group[(mod, cond)], cmap=CMAP, vmin=-lim, vmax=lim, shading="auto")
            ax.axvline(0, color=INK, lw=0.8, ls="--")
            ax.set_title(f"({next(letters)}) {'Tripolar (TCRE)' if mod == 'tripolar' else 'Disc'} - {cond}",
                         loc="left", fontsize=9, color=INK)
            for lo, hi in BANDS.values():
                ax.axhline(hi, color="white", lw=0.5, alpha=0.6)
            if r == 1:
                ax.set_xlabel("Time from picture onset (s)")
            if c == 0:
                ax.set_ylabel("Frequency (Hz)")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, extend="max")
    cb.set_label("Power change from baseline (%)\nblue = ERD, red = ERS", color=INK)
    cb.outline.set_edgecolor(INK_2)
    fig.suptitle(f"Group-median ERD/ERS (baseline -0.4 to 0 s, n = {n})", fontsize=9, color=INK_2, x=0.01, ha="left")
    fig.savefig(path, dpi=600, facecolor="white")
    plt.close(fig)


def main(root, out_dir):
    root, out_dir = Path(root).expanduser(), Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted((root / "derivatives" / "cleaned-epochs").glob(f"sub-*/eeg/*_task-{TASK}_desc-cleaned_epo.fif"))
    if not files:
        sys.exit("No cleaned-epochs files found. Run make_cleaned_epochs.py first.")

    acc = {(m, c): [] for m in MODALITIES for c in CONDITIONS}
    rows, times = [], None
    for f in files:
        sub = f.name.split("_")[0]
        times, maps = cached(out_dir, "tfr", sub, lambda: subject_maps(f, sub))
        for k, m in maps.items():
            acc[k].append(m)
            rows.append({"participant_id": sub, "modality": k[0], "condition": k[1], **band_values(times, m)})
        print(f"{sub}: done", flush=True)

    group = {k: np.median(v, axis=0) for k, v in acc.items()}   # median: robust to single-subject artefacts
    np.savez(out_dir / "tfr_group_maps.npz", freqs=FREQS, times=times,
             **{f"{m}_{c}": g for (m, c), g in group.items()},
             **{f"subjects_{m}_{c}": np.array(v) for (m, c), v in acc.items()})
    df = pd.DataFrame(rows)
    df.round(2).to_csv(out_dir / "tfr_band_erd_per_subject.tsv", sep="\t", index=False)
    st = stats_table(df)
    st.to_csv(out_dir / "tfr_band_erd_stats.tsv", sep="\t", index=False)
    make_figure(times, group, len(files), out_dir / "fig_tfr_erd_ers.png")
    print("\n" + st.to_string(index=False))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "tfr_output")
