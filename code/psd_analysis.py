"""
Power spectral density (PSD) of tripolar (TCRE) vs disc signals, overt vs covert speech.

Uses the cleaned-epochs derivative (1-40 Hz). For every participant, modality and condition:
  - Welch PSD of the post-stimulus window (0 to 2.5 s), 1-s segments (1 Hz resolution), 50% overlap
  - averaged over epochs and channels
  - converted to RELATIVE power (divided by total 1-40 Hz power), so the result does not depend
    on the unknown amplification of the tripolar output
Band relative power: delta 1-4, theta 4-8, alpha 8-13, beta 13-30, gamma 30-40 Hz.

Channel convention: odd channels (1, 3, 5, ...) = tripolar, even channels (2, 4, 6, ...) = disc.
Array channels only: non-array channels (EOG) are excluded as listed in
channel_sets.py. All 16 participants are included. Nothing in the dataset is modified.
Per-participant results are cached in OUT_DIR/cache, so a stopped run resumes where it stopped.

Outputs (in OUT_DIR):
  psd_group_mean.tsv            group mean +/- SEM relative PSD per frequency (table view of the figure)
  psd_band_power_per_subject.tsv
  psd_band_stats.tsv            Wilcoxon signed-rank tests on RELATIVE power: overt vs covert
                                (per modality) and tripolar vs disc (per condition), per band
  psd_overt_covert_db_per_subject.tsv / psd_overt_covert_db_stats.tsv
                                overt/covert ABSOLUTE power ratio in dB per band, within modality
                                (gain cancels); tests vs 0 and tripolar vs disc. A larger ratio
                                in beta/gamma means more speech-related muscle (EMG) pickup.
  fig_psd_tripolar_disc.png     600-dpi figure

Usage:
    python psd_analysis.py /path/to/bids_root /path/to/output_dir
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from channel_sets import array_channels, cached, load_epochs   # noqa: E402

TASK = "OvertCovert"
CONDITIONS = ["Overt", "Covert"]
MODALITIES = ["tripolar", "disc"]
TMIN, TMAX = 0.0, 2.5
FMIN, FMAX = 1.0, 40.0
TARGET_SFREQ = 250.0
trapz = getattr(np, "trapezoid", None) or np.trapz   # works with numpy 1.x and 2.x
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}

# Reference categorical palette (slot 1 blue, slot 2 orange) and recessive ink for axes/text
COLORS = {"Overt": "#2a78d6", "Covert": "#eb6834"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def subject_psd(fpath, sub):
    ep = load_epochs(fpath, TARGET_SFREQ)
    sf = ep.info["sfreq"]
    tri, disc = array_channels(sub, len(ep.ch_names))
    out = {}
    for cond in CONDITIONS:
        e = ep[f"Stimulus/{cond}"].copy().crop(TMIN, TMAX)
        spec = e.compute_psd(method="welch", fmin=FMIN, fmax=FMAX, n_fft=int(sf),
                             n_per_seg=int(sf), n_overlap=int(sf / 2), verbose=False)
        p, freqs = spec.get_data(return_freqs=True)          # epochs x channels x freqs
        p = p.mean(axis=0)                                   # mean over epochs
        for mod, idx in (("tripolar", tri), ("disc", disc)):
            m = p[idx].mean(axis=0)                          # mean over array channels
            out[(mod, cond)] = m / trapz(m, freqs)    # relative power (1/Hz)
            out[("abs", mod, cond)] = m               # absolute power (same channels, same gain)
    return freqs, out


def overt_covert_db(freqs, rel):
    """Overt/covert power ratio in dB per band, within each modality.
    Same channels and same gain in both conditions, so the unknown gain cancels."""
    res = []
    for mod in MODALITIES:
        o, c = rel[("abs", mod, "Overt")], rel[("abs", mod, "Covert")]
        row = {"modality": mod}
        for band, (lo, hi) in BANDS.items():
            idx = (freqs >= lo) & (freqs <= hi)
            row[band] = float(10 * np.log10(o[idx].mean() / c[idx].mean()))
        res.append(row)
    return res


def ratio_stats(r):
    rows = []
    for band in BANDS:
        for mod in MODALITIES:
            v = r[r.modality == mod].set_index("participant_id")[band]
            rows.append({"band": band, "test": f"{mod}: overt/covert dB vs 0", "n": len(v),
                         "mean_dB": round(v.mean(), 2), "sd_dB": round(v.std(), 2),
                         "positive_in": f"{int((v > 0).sum())}/{len(v)}",
                         "wilcoxon_p": round(wilcoxon(v).pvalue, 4)})
        t = r[r.modality == "tripolar"].set_index("participant_id")[band]
        d = r[r.modality == "disc"].set_index("participant_id")[band]
        t, d = t.align(d, join="inner")
        rows.append({"band": band, "test": "overt/covert dB: tripolar vs disc", "n": len(t),
                     "mean_dB": f"{t.mean():.2f} vs {d.mean():.2f}", "sd_dB": "",
                     "positive_in": f"tripolar smaller in {int((t < d).sum())}/{len(t)}",
                     "wilcoxon_p": round(wilcoxon(t, d).pvalue, 4)})
    return pd.DataFrame(rows)


def band_power(freqs, rel):
    res = {}
    for b, (lo, hi) in BANDS.items():
        idx = (freqs >= lo) & (freqs <= hi)
        res[b] = float(trapz(rel[idx], freqs[idx]) * 100)   # % of 1-40 Hz power
    return res


def stats_table(bp):
    rows = []
    for b in BANDS:
        for mod in MODALITIES:          # overt vs covert within modality
            o = bp.query("modality == @mod and condition == 'Overt'").set_index("participant_id")[b]
            c = bp.query("modality == @mod and condition == 'Covert'").set_index("participant_id")[b]
            o, c = o.align(c, join="inner")
            rows.append({"band": b, "comparison": f"{mod}: overt vs covert", "n": len(o),
                         "mean_A": round(o.mean(), 2), "mean_B": round(c.mean(), 2),
                         "A_greater_in": f"{int((o > c).sum())}/{len(o)}",
                         "wilcoxon_p": round(wilcoxon(o, c).pvalue, 4)})
        for cond in CONDITIONS:         # tripolar vs disc within condition
            t = bp.query("modality == 'tripolar' and condition == @cond").set_index("participant_id")[b]
            d = bp.query("modality == 'disc' and condition == @cond").set_index("participant_id")[b]
            t, d = t.align(d, join="inner")
            rows.append({"band": b, "comparison": f"{cond}: tripolar vs disc", "n": len(t),
                         "mean_A": round(t.mean(), 2), "mean_B": round(d.mean(), 2),
                         "A_greater_in": f"{int((t > d).sum())}/{len(t)}",
                         "wilcoxon_p": round(wilcoxon(t, d).pvalue, 4)})
    return pd.DataFrame(rows)


def make_figure(freqs, group, n, path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK_2,
                         "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True, constrained_layout=True)
    for ax, mod, letter in zip(axes, MODALITIES, "ab"):
        ends = {}
        for cond in CONDITIONS:
            mean, sem = group[(mod, cond)]
            ax.fill_between(freqs, mean - sem, mean + sem, color=COLORS[cond], alpha=0.18, lw=0)
            ax.plot(freqs, mean, color=COLORS[cond], lw=2, label=cond)
            ends[cond] = np.log10(mean[-1])
        # direct labels at the line ends, nudged apart so they never overlap (log axis)
        lo_c, hi_c = sorted(CONDITIONS, key=lambda c: ends[c])
        gap = 0.14
        if ends[hi_c] - ends[lo_c] < gap:
            mid = (ends[hi_c] + ends[lo_c]) / 2
            ends[lo_c], ends[hi_c] = mid - gap / 2, mid + gap / 2
        for cond in CONDITIONS:
            ax.text(freqs[-1] + 0.6, 10 ** ends[cond], cond, color=INK, fontsize=7, va="center")
        ax.set_yscale("log")
        ax.set_xlim(FMIN, FMAX)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(f"({letter}) {'Tripolar (TCRE)' if mod == 'tripolar' else 'Disc'}",
                     loc="left", fontsize=9, color=INK)
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        for lo, hi in BANDS.values():
            ax.axvline(lo, color=GRID, lw=0.6, zorder=0)
    axes[0].set_ylabel("Relative power (1/Hz)")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle(f"Group-mean relative PSD, 0-2.5 s after picture onset (mean ± SEM, n = {n})",
                 fontsize=9, color=INK_2, x=0.01, ha="left")
    fig.savefig(path, dpi=600, facecolor="white")
    plt.close(fig)


def main(root, out_dir):
    root, out_dir = Path(root).expanduser(), Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted((root / "derivatives" / "cleaned-epochs").glob(f"sub-*/eeg/*_task-{TASK}_desc-cleaned_epo.fif"))
    if not files:
        sys.exit("No cleaned-epochs files found. Run make_cleaned_epochs.py first.")

    all_rel, bp_rows, ratio_rows, freqs = {k: [] for k in [(m, c) for m in MODALITIES for c in CONDITIONS]}, [], [], None
    for f in files:
        sub = f.name.split("_")[0]
        freqs, rel = cached(out_dir, "psd", sub, lambda: subject_psd(f, sub))
        for k in all_rel:
            all_rel[k].append(rel[k])
            bp_rows.append({"participant_id": sub, "modality": k[0], "condition": k[1], **band_power(freqs, rel[k])})
        ratio_rows += [{"participant_id": sub, **x} for x in overt_covert_db(freqs, rel)]
        print(f"{sub}: done", flush=True)

    n = len(files)
    group = {k: (np.mean(v, axis=0), np.std(v, axis=0, ddof=1) / np.sqrt(len(v))) for k, v in all_rel.items()}
    gm = pd.DataFrame({"frequency_Hz": freqs})
    for (mod, cond), (m, s) in group.items():
        gm[f"{mod}_{cond}_mean"] = m
        gm[f"{mod}_{cond}_sem"] = s
    gm.to_csv(out_dir / "psd_group_mean.tsv", sep="\t", index=False, float_format="%.6g")

    bp = pd.DataFrame(bp_rows)
    bp.round(2).to_csv(out_dir / "psd_band_power_per_subject.tsv", sep="\t", index=False)
    st = stats_table(bp)
    st.to_csv(out_dir / "psd_band_stats.tsv", sep="\t", index=False)
    ratio = pd.DataFrame(ratio_rows)
    ratio.round(3).to_csv(out_dir / "psd_overt_covert_db_per_subject.tsv", sep="\t", index=False)
    rs = ratio_stats(ratio)
    rs.to_csv(out_dir / "psd_overt_covert_db_stats.tsv", sep="\t", index=False)
    print("\nOvert/covert absolute power ratio (dB):\n" + rs.to_string(index=False))
    make_figure(freqs, group, n, out_dir / "fig_psd_tripolar_disc.png")
    print("\n" + st.to_string(index=False))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "psd_output")
