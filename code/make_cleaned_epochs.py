"""
Generate derivatives/cleaned-epochs from the raw BIDS dataset.

Array channels only (EOG excluded); 60 Hz notch, 1-40 Hz zero-phase FIR band-pass,
epochs -0.5 to 2.5 s around each picture onset, baseline -0.5 to 0 s; no rejection,
no re-referencing, no resampling. Channel layouts are defined in channel_sets.py.

Usage:
    python make_cleaned_epochs.py /path/to/bids_root [01 02 ...]   # all or selected participants
    python make_cleaned_epochs.py /path/to/bids_root --finalize     # write dataset_description, README, qc_summary.tsv
"""
import json
import sys
from pathlib import Path

import mne
import mne_bids
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids

TASK = "OvertCovert"
NOTCH_HZ = 60.0
L_FREQ, H_FREQ = 1.0, 40.0
TMIN, TMAX = -0.5, 2.5
BASELINE = (-0.5, 0.0)
EVENT_ID = {"Stimulus/Overt": 1, "Stimulus/Covert": 2}
PIPELINE = "cleaned-epochs"
MAX_SAMPLES = 30_000_000                   
SOURCE_DATASET_URL = "https://openneuro.org/datasets/ds008895"
CODE_URL = "https://github.com/parasmemon/tcre-overt-covert-speech-eeg"
# -------------------------------------------------------------------------


def out_path(out_root, sub):
    return BIDSPath(subject=sub, task=TASK, datatype="eeg", root=out_root, description="cleaned",
                    suffix="epo", extension=".fif", check=False)


def process_subject(root, out_root, sub, fpath=None):
    raw = read_raw_bids(BIDSPath(subject=sub, task=TASK, datatype="eeg", root=root), verbose=False)
    picks = mne.pick_types(raw.info, eeg=True, eog=False)          # array channels only
    sf = raw.info["sfreq"]
    events, _ = mne.events_from_annotations(raw, event_id=EVENT_ID, verbose=False)
    events = events[np.unique(events[:, 0], return_index=True)[1]]
    s0, s1 = int(round(TMIN * sf)), int(round(TMAX * sf))
    idx = events[:, 0][:, None] + np.arange(s0, s1 + 1)[None, :]
    data = np.empty((len(events), len(picks), idx.shape[1]))
    group = max(1, min(8, MAX_SAMPLES // raw.n_times))
    for g in range(0, len(picks), group):
        x = raw.get_data(picks=picks[g:g + group])
        x = mne.filter.notch_filter(x, sf, NOTCH_HZ, verbose=False)
        x = mne.filter.filter_data(x, sf, L_FREQ, H_FREQ, verbose=False)
        data[:, g:g + group, :] = x[:, idx].transpose(1, 0, 2)
        del x
    info = mne.pick_info(raw.info, picks)
    with info._unlock():
        info["highpass"], info["lowpass"] = L_FREQ, H_FREQ
    epochs = mne.EpochsArray(data, info, events=events, tmin=TMIN, event_id=EVENT_ID,
                             baseline=BASELINE, verbose=False)
    fpath = fpath or out_path(out_root, sub).fpath
    Path(fpath).parent.mkdir(parents=True, exist_ok=True)
    epochs.save(fpath, overwrite=True, verbose=False)
    return epochs


def finalize(root, out_root):
    rows = []
    for f in sorted(out_root.glob(f"sub-*/eeg/*_task-{TASK}_desc-cleaned_epo.fif")):
        ep = mne.read_epochs(f, preload=False, verbose=False)
        rows.append({"participant_id": f.name.split("_")[0], "n_channels": len(ep.ch_names),
                     "sfreq_Hz": ep.info["sfreq"], "n_overt": len(ep["Stimulus/Overt"]),
                     "n_covert": len(ep["Stimulus/Covert"]), "file": str(f.relative_to(out_root))})
    qc = pd.DataFrame(rows)
    qc.to_csv(out_root / "qc_summary.tsv", sep="\t", index=False)
    desc = {
        "Name": "Cleaned epochs (notch 60 Hz, band-pass 1-40 Hz, baseline-corrected, array channels only)",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {"Name": "MNE-Python", "Version": mne.__version__,
             "CodeURL": CODE_URL, "Description": "make_cleaned_epochs.py"},
            {"Name": "MNE-BIDS", "Version": mne_bids.__version__},
        ],
        "SourceDatasets": [{"URL": SOURCE_DATASET_URL}],
    }
    (out_root / "dataset_description.json").write_text(json.dumps(desc, indent=4))
    readme = f"""Cleaned epochs derivative
=========================

Generated from the raw BIDS dataset with make_cleaned_epochs.py ({CODE_URL}).

Processing steps (applied to the TCRE array channels of every participant):
1. Notch filter at {NOTCH_HZ:g} Hz.
2. Band-pass filter {L_FREQ:g}-{H_FREQ:g} Hz (MNE-Python {mne.__version__}, default zero-phase FIR).
3. Epochs from {TMIN:g} to {TMAX:g} s relative to each picture-onset marker (Stimulus/Overt, Stimulus/Covert).
4. Baseline correction using {BASELINE[0]:g} to {BASELINE[1]:g} s.
No channels or epochs were rejected. The original sampling rate is kept.

Channels: only the TCRE array channels are included. Odd-numbered channels are tripolar (TCRE Laplacian)
signals and even-numbered channels are the co-located emulated disc signals (channel names with the "h" suffix).
The EOG channels (two standard disc electrodes placed above and to the right of one eye; channels 41-42 of sub-07, sub-08, sub-09, sub-14, sub-15 and sub-16, and channels 63-64
of sub-11, sub-12 and sub-13) are not included here; they are available, labelled as EOG, in the raw data.
Electrodes per array (F4/P6/F3/P5): 5/5/5/5 (40 channels), 7/7/7/7 (56), 8/8/8/8 (64), and 8/7/8/8 (62 channels;
sub-11, sub-12, sub-13, whose P6 array had seven electrodes).

Files: sub-XX/eeg/sub-XX_task-{TASK}_desc-cleaned_epo.fif (MNE .fif format).
Load with:  epochs = mne.read_epochs("sub-01/eeg/sub-01_task-{TASK}_desc-cleaned_epo.fif")

Channel and trial counts per participant are listed in qc_summary.tsv.
"""
    (out_root / "README").write_text(readme)
    print(qc.to_string(index=False))


if __name__ == "__main__":
    root = Path(sys.argv[1]).expanduser()
    out_root = root / "derivatives" / PIPELINE
    args = sys.argv[2:]
    if args == ["--finalize"]:
        finalize(root, out_root)
    else:
        subs = args or sorted(p.name[4:] for p in root.glob("sub-*") if p.is_dir())
        for s in subs:
            ep = process_subject(root, out_root, s)
            print(f"sub-{s}: {len(ep.ch_names)} channels, {len(ep['Stimulus/Overt'])} overt, "
                  f"{len(ep['Stimulus/Covert'])} covert", flush=True)
