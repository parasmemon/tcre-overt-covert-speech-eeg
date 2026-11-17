"""
Generate derivatives/resampled-500hz from the raw BIDS dataset.

Continuous recordings of all channels (TCRE array channels and, where recorded, the two EOG
channels), resampled to a common rate of 500 Hz with MNE-Python's polyphase resampler
(includes an anti-aliasing low-pass filter). No other filtering, no re-referencing, no
rejection. The picture-onset events are kept as annotations (Stimulus/Overt, Stimulus/Covert).
Artifact removal with ICA or SSP can be applied to these files with artifact_ica_ssp.py.

Usage:
    python make_resampled.py /path/to/bids_root [01 02 ...]   # all or selected participants
    python make_resampled.py /path/to/bids_root --finalize     # write dataset_description, README, qc_summary.tsv
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
SFREQ_OUT = 500.0
PIPELINE = "resampled-500hz"
MAX_SAMPLES = 30_000_000
SOURCE_DATASET_URL = "https://openneuro.org/datasets/ds008895"
CODE_URL = "https://github.com/parasmemon/tcre-overt-covert-speech-eeg"


def out_path(out_root, sub):
    return BIDSPath(subject=sub, task=TASK, datatype="eeg", root=out_root, description="resampled500",
                    suffix="eeg", extension=".fif", check=False)


def process_subject(root, out_root, sub):
    raw = read_raw_bids(BIDSPath(subject=sub, task=TASK, datatype="eeg", root=root), verbose=False)
    sf = raw.info["sfreq"]
    down = sf / SFREQ_OUT
    assert abs(down - round(down)) < 1e-9, f"sub-{sub}: {sf} Hz is not a multiple of {SFREQ_OUT} Hz"
    down = int(round(down))
    n_out = int(np.ceil(raw.n_times / down))
    data = np.empty((len(raw.ch_names), n_out), dtype=np.float32)
    group = max(1, min(8, MAX_SAMPLES // raw.n_times))
    for g in range(0, len(raw.ch_names), group):
        x = raw.get_data(picks=np.arange(g, min(g + group, len(raw.ch_names))))
        y = mne.filter.resample(x, up=1, down=down, method="polyphase", verbose=False)
        data[g:g + group, :] = y[:, :n_out]
        del x, y
    info = raw.info.copy()
    with info._unlock():
        info["sfreq"] = SFREQ_OUT
        info["lowpass"] = min(info["lowpass"], SFREQ_OUT / 2)
    out = mne.io.RawArray(data, info, verbose=False)
    out.set_annotations(raw.annotations.copy())
    fpath = out_path(out_root, sub).fpath
    Path(fpath).parent.mkdir(parents=True, exist_ok=True)
    out.save(fpath, fmt="single", overwrite=True, verbose=False)
    return out, sf


def finalize(root, out_root):
    rows = []
    for f in sorted(out_root.glob(f"sub-*/eeg/*_task-{TASK}_desc-resampled500_eeg.fif")):
        raw = mne.io.read_raw_fif(f, preload=False, verbose=False)
        types = raw.get_channel_types()
        ev = raw.annotations.description
        rows.append({"participant_id": f.name.split("_")[0], "sfreq_Hz": raw.info["sfreq"],
                     "n_eeg": types.count("eeg"), "n_eog": types.count("eog"),
                     "duration_s": round(raw.times[-1], 1),
                     "n_overt": int(np.sum(ev == "Stimulus/Overt")),
                     "n_covert": int(np.sum(ev == "Stimulus/Covert")),
                     "file": str(f.relative_to(out_root))})
    qc = pd.DataFrame(rows)
    qc.to_csv(out_root / "qc_summary.tsv", sep="\t", index=False)
    desc = {
        "Name": "Continuous recordings resampled to 500 Hz (all channels, EOG retained)",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {"Name": "MNE-Python", "Version": mne.__version__,
             "CodeURL": CODE_URL, "Description": "code/make_resampled.py of the raw dataset"},
            {"Name": "MNE-BIDS", "Version": mne_bids.__version__},
        ],
        "SourceDatasets": [{"URL": SOURCE_DATASET_URL}],
    }
    (out_root / "dataset_description.json").write_text(json.dumps(desc, indent=4))
    readme = f"""Resampled continuous derivative (500 Hz)
========================================

Generated from the raw BIDS dataset with code/make_resampled.py ({CODE_URL}).

Processing: the continuous recording of every participant was resampled from its original rate
(1000 Hz for sub-01, 2500 Hz for all others) to a common rate of {SFREQ_OUT:g} Hz with the polyphase
resampler of MNE-Python {mne.__version__}, which applies an anti-aliasing low-pass filter. No other filtering,
re-referencing or rejection was applied. All channels are included: the TCRE array channels and, in the nine
participants who have them (sub-07, sub-08, sub-09, sub-11, sub-12, sub-13, sub-14, sub-15, sub-16), the two EOG
channels. The picture-onset events are stored as annotations (Stimulus/Overt, Stimulus/Covert).

Channels: odd-numbered array channels are TCRE (tripolar) signals and even-numbered array channels are the
co-located emulated disc signals. The channel names are placeholder labels and do not give electrode positions
(see the README of the raw dataset). The TCRE and disc signals differ in amplitude by more than an order of magnitude.

Artifact removal: code/artifact_ica_ssp.py applies ICA or SSP to these files, separately to the TCRE and the
disc signals. In participants with EOG channels, ocular components and projectors are identified from the EOG.

Files: sub-XX/eeg/sub-XX_task-{TASK}_desc-resampled500_eeg.fif (MNE .fif format, 32-bit float).
Load with:  raw = mne.io.read_raw_fif("sub-01/eeg/sub-01_task-{TASK}_desc-resampled500_eeg.fif")

Sampling rates, channel counts and event counts per participant are listed in qc_summary.tsv.
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
            out, sf = process_subject(root, out_root, s)
            types = out.get_channel_types()
            print(f"sub-{s}: {sf:g} -> {out.info['sfreq']:g} Hz, {types.count('eeg')} EEG, "
                  f"{types.count('eog')} EOG, {len(out.annotations)} events", flush=True)
