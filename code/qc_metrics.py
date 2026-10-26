"""
Descriptive quality-control (QC) metrics for the TCRE overt/covert speech EEG dataset.

Nothing is rejected, flagged or modified. The script only DESCRIBES the data so that
users can choose their own channel/epoch rejection criteria.

Outputs (written to OUT_DIR):
  qc_recording.tsv  one row per participant x electrode type (tripolar / disc), from the raw data:
                    n_channels, duration, sampling rate, trial counts,
                    channel RMS (median, min, max) and 1st-99th percentile range (median),
                    60 Hz line-noise ratio (dB above neighbouring 55-58 / 62-65 Hz bands)
  qc_epochs.tsv     one row per participant x electrode type x condition, from the
                    cleaned-epochs derivative: peak-to-peak (PTP) amplitude per epoch
                    (max over channels) summarised as median / 95th percentile / max,
                    plus the % of epochs that WOULD exceed example thresholds
                    (100, 200, 500, 1000 uV). No epochs are removed.

Usage:
    python qc_metrics.py /path/to/bids_root /path/to/output_dir
"""

import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids
sys.path.insert(0, str(Path(__file__).resolve().parent))

TASK = "OvertCovert"
LINE_FREQ = 60.0
HIGHPASS = 0.5                     # Hz, removes DC offset before amplitude statistics
CONDITIONS = ["Stimulus/Overt", "Stimulus/Covert"]
EXAMPLE_THRESHOLDS_UV = [100, 200, 500, 1000]


def split_groups(ch_names):
    """Odd channels (1, 3, 5, ...) = tripolar; even channels (2, 4, 6, ...) = disc
    (emulated from the outer ring of the same electrode)."""
    return {"tripolar": list(ch_names[0::2]), "disc": list(ch_names[1::2])}


CHUNK = 8   # channels loaded at a time (keeps memory low for long 64-channel recordings)


def channel_stats(raw_lazy, chs):
    """Per-channel RMS, 1-99th percentile range (uV) and 60 Hz noise ratio (dB), read in chunks."""
    rms, rng, noise = [], [], []
    for i in range(0, len(chs), CHUNK):
        part = raw_lazy.copy().pick(chs[i:i + CHUNK]).load_data(verbose=False)
        psd = part.compute_psd(method="welch", fmin=50, fmax=70,
                               n_fft=int(part.info["sfreq"] * 2), verbose=False)
        p, f = psd.get_data(return_freqs=True)
        peak = (f >= LINE_FREQ - 1) & (f <= LINE_FREQ + 1)
        side = ((f >= LINE_FREQ - 5) & (f <= LINE_FREQ - 2)) | ((f >= LINE_FREQ + 2) & (f <= LINE_FREQ + 5))
        noise += list(10 * np.log10(p[:, peak].mean(1) / p[:, side].mean(1)))
        part.filter(l_freq=HIGHPASS, h_freq=None, verbose=False)
        x = part.get_data() * 1e6
        rms += list(np.sqrt((x ** 2).mean(axis=1)))
        rng += list(np.percentile(x, 99, axis=1) - np.percentile(x, 1, axis=1))
        del part, x
    return np.array(rms), np.array(rng), np.array(noise)


def recording_qc(root, sub):
    bp = BIDSPath(subject=sub, task=TASK, datatype="eeg", root=root)
    raw = read_raw_bids(bp, verbose=False)          # not preloaded
    raw.pick("eeg")
    groups = split_groups(raw.ch_names)
    counts = pd.Series(raw.annotations.description).value_counts()

    rows = []
    for g, chs in groups.items():
        if not chs:
            continue
        rms, rng, noise = channel_stats(raw, chs)
        rows.append({
            "participant_id": f"sub-{sub}", "electrode": g, "n_channels": len(chs),
            "sfreq_Hz": raw.info["sfreq"], "duration_min": round(raw.times[-1] / 60, 1),
            "n_overt": int(counts.get(CONDITIONS[0], 0)), "n_covert": int(counts.get(CONDITIONS[1], 0)),
            "rms_median_uV": round(float(np.median(rms)), 2),
            "rms_min_uV": round(float(rms.min()), 2),
            "rms_max_uV": round(float(rms.max()), 2),
            "p1_p99_range_median_uV": round(float(np.median(rng)), 2),
            "line_noise_60Hz_dB": round(float(np.median(noise)), 1),
        })
    return rows


def epoch_qc(root, sub):
    f = (Path(root) / "derivatives" / "cleaned-epochs" / f"sub-{sub}" / "eeg"
         / f"sub-{sub}_task-{TASK}_desc-cleaned_epo.fif")
    if not f.exists():
        return []
    ep = mne.read_epochs(f, preload=True, verbose=False)
    rows = []
    for g, chs in split_groups(ep.ch_names).items():
        if not chs:
            continue
        for cond in CONDITIONS:
            x = ep[cond].get_data(picks=chs) * 1e6               # epochs x channels x times, uV
            ptp = (x.max(axis=2) - x.min(axis=2)).max(axis=1)    # worst channel per epoch
            row = {"participant_id": f"sub-{sub}", "electrode": g,
                   "condition": cond.split("/")[-1], "n_epochs": len(ptp),
                   "ptp_median_uV": round(float(np.median(ptp)), 1),
                   "ptp_p95_uV": round(float(np.percentile(ptp, 95)), 1),
                   "ptp_max_uV": round(float(ptp.max()), 1)}
            for t in EXAMPLE_THRESHOLDS_UV:
                row[f"pct_over_{t}uV"] = round(100 * float((ptp > t).mean()), 1)
            rows.append(row)
    return rows


def main(root, out_dir, subjects=None):
    """Per-participant results are cached (OUT_DIR/cache), so a stopped run resumes where it stopped."""
    from channel_sets import cached
    root, out_dir = Path(root).expanduser(), Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects = subjects or sorted(p.name[4:] for p in root.glob("sub-*") if p.is_dir())
    rec, epo = [], []
    for sub in subjects:
        print(f"sub-{sub} ...", flush=True)
        r, e = cached(out_dir, "qc", f"sub-{sub}", lambda: (recording_qc(root, sub), epoch_qc(root, sub)))
        rec += r; epo += e
    pd.DataFrame(rec).to_csv(out_dir / "qc_recording.tsv", sep="\t", index=False)
    pd.DataFrame(epo).to_csv(out_dir / "qc_epochs.tsv", sep="\t", index=False)
    print("Done:", out_dir)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "qc_output", sys.argv[3:] or None)
