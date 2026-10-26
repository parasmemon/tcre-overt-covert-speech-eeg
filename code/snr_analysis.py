"""
Signal-to-noise ratio (SNR) of tripolar (TCRE) vs disc signals.

Evoked-response SNR, computed per channel from the cleaned-epochs derivative (1-40 Hz):
    SNR_dB = 20 * log10( RMS(ERP, post-cue window) / RMS(ERP, pre-cue window) )
where ERP = average over trials. The pre-cue RMS of the averaged ERP is the residual noise
level, so SNR_dB tells how far the time-locked response stands out of the noise.
It is a ratio, so it does not depend on the unknown amplification of the tripolar output.
Per participant, the median over channels of each modality is reported.

Windows: pre-cue -0.4 to 0 s, post-cue 0 to 1.0 s (edit PRE/POST below if your response window differs).
Channel convention: odd channels = tripolar, even channels = disc. Array channels only: non-array
channels (EOG) are excluded as listed in channel_sets.py. All 16 participants
included. Nothing in the dataset is modified. Per-participant results are cached in OUT_DIR/cache.

Outputs (in OUT_DIR):
  snr_per_subject.tsv
  snr_stats.tsv        Wilcoxon signed-rank: tripolar vs disc (per condition)

Usage:
    python snr_analysis.py /path/to/bids_root /path/to/output_dir
"""

import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from channel_sets import array_channels, cached   # noqa: E402

TASK = "OvertCovert"
CONDITIONS = ["Overt", "Covert"]
PRE = (-0.4, 0.0)
POST = (0.0, 1.0)


def rms(x, axis=-1):
    return np.sqrt(np.mean(x ** 2, axis=axis))


def subject_snr(fpath, sub):
    ep = mne.read_epochs(fpath, preload=False, verbose=False)     # ERP computed epoch by epoch (low memory)
    t = ep.times
    tri, disc = array_channels(sub, len(ep.ch_names))
    pre, post = (t >= PRE[0]) & (t < PRE[1]), (t >= POST[0]) & (t <= POST[1])
    rows = []
    for cond in CONDITIONS:
        erp = ep[f"Stimulus/{cond}"].average(picks="all").data       # channels x times (full rate)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = 20 * np.log10(rms(erp[:, post]) / rms(erp[:, pre]))
        snr = np.where(np.isfinite(snr), snr, np.nan)                  # flat channels -> ignored
        for mod, idx in (("tripolar", tri), ("disc", disc)):
            v = snr[idx]
            rows.append({"condition": cond, "modality": mod, "snr_db_median": round(float(np.nanmedian(v)), 2),
                         "snr_db_iqr": round(float(np.nanpercentile(v, 75) - np.nanpercentile(v, 25)), 2),
                         "n_channels": int(np.isfinite(v).sum())})
    return rows


def main(root, out_dir):
    root, out_dir = Path(root).expanduser(), Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted((root / "derivatives" / "cleaned-epochs").glob(f"sub-*/eeg/*_task-{TASK}_desc-cleaned_epo.fif"))
    if not files:
        sys.exit("No cleaned-epochs files found. Run make_cleaned_epochs.py first.")

    rows = []
    for f in files:
        sub = f.name.split("_")[0]
        rows += [{"participant_id": sub, **r} for r in cached(out_dir, "snr", sub, lambda: subject_snr(f, sub))]
        print(f"{sub}: done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "snr_per_subject.tsv", sep="\t", index=False)

    st = []
    for cond in CONDITIONS:
        t = df.query("modality == 'tripolar' and condition == @cond").set_index("participant_id")["snr_db_median"]
        d = df.query("modality == 'disc' and condition == @cond").set_index("participant_id")["snr_db_median"]
        t, d = t.align(d, join="inner")
        st.append({"condition": cond, "n": len(t),
                   "tripolar_mean_db": round(t.mean(), 2), "tripolar_sd_db": round(t.std(), 2),
                   "disc_mean_db": round(d.mean(), 2), "disc_sd_db": round(d.std(), 2),
                   "tripolar_higher_in": f"{int((t > d).sum())}/{len(t)}",
                   "wilcoxon_p": round(wilcoxon(t, d).pvalue, 4)})
    st = pd.DataFrame(st)
    st.to_csv(out_dir / "snr_stats.tsv", sep="\t", index=False)
    print("\n" + st.to_string(index=False))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "snr_output")
