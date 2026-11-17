"""
Optional artifact removal (ICA or SSP) for derivatives/resampled-500hz.

The TCRE and disc signals differ in amplitude by more than an order of magnitude, so ICA and SSP are
applied to each signal type separately (TCRE = odd-numbered array channels, disc = even-numbered
array channels). In participants with EOG channels, ocular ICA components are identified
automatically from their correlation with the EOG (mne.preprocessing.ICA.find_bads_eog), and SSP
projectors are computed from EOG-detected blinks (mne.preprocessing.compute_proj_eog).
In participants without EOG, ICA is fitted and saved without automatic exclusion so that components
can be inspected and excluded manually; SSP requires EOG and is not available for them.

Usage:
    python artifact_ica_ssp.py /path/to/bids_root 07 --method ica  [--signal tcre|disc|both] [--out DIR]
    python artifact_ica_ssp.py /path/to/bids_root 07 --method ssp  [--signal tcre|disc|both] [--out DIR]

Outputs (default DIR: derivatives/artifact-corrected/sub-XX/eeg/):
    *_desc-<signal>ica_ica.fif     fitted ICA object (ICA only; mne.preprocessing.read_ica)
    *_desc-<signal>ssp_proj.fif    SSP projectors (SSP only; mne.read_proj)
    *_desc-<signal><method>_eeg.fif  corrected continuous data of that signal type (+ EOG channels)
"""
import argparse
from pathlib import Path

import mne
import numpy as np

TASK = "OvertCovert"
ICA_BAND = (1.0, 40.0)      # filter used only for fitting ICA
ICA_DECIM = 5               # 500 Hz -> 100 Hz for fitting
RANDOM_STATE = 97


def load(root, sub):
    f = (Path(root) / "derivatives" / "resampled-500hz" / f"sub-{sub}" / "eeg"
         / f"sub-{sub}_task-{TASK}_desc-resampled500_eeg.fif")
    return mne.io.read_raw_fif(f, preload=False, verbose=False)


def signal_picks(raw, signal):
    eeg = mne.pick_types(raw.info, eeg=True, eog=False)
    return eeg[0::2] if signal == "tcre" else eeg[1::2]       # odd / even array channels (1-based)


def run(root, sub, method, signal, out_dir):
    raw = load(root, sub)
    eog_idx = list(mne.pick_types(raw.info, eeg=False, eog=True))
    eog = [raw.ch_names[i] for i in eog_idx]
    if method == "ssp" and not eog:
        raise SystemExit(f"sub-{sub} has no EOG channels: SSP is not available, use --method ica.")
    picks = list(signal_picks(raw, signal)) + eog_idx
    data = raw.copy().pick(picks).load_data(verbose=False)
    stem = f"sub-{sub}_task-{TASK}_desc-{signal}{method}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if method == "ica":
        fit = data.copy().filter(*ICA_BAND, picks="eeg", verbose=False)
        ica = mne.preprocessing.ICA(n_components=0.99, method="infomax", fit_params=dict(extended=True),
                                    random_state=RANDOM_STATE, max_iter="auto", verbose=False)
        ica.fit(fit, picks="eeg", decim=ICA_DECIM, verbose=False)
        if eog:
            idx, scores = ica.find_bads_eog(fit, ch_name=eog, verbose=False)
            ica.exclude = [int(i) for i in idx]
        ica.save(out_dir / f"{stem}_ica.fif", overwrite=True, verbose=False)
        ica.apply(data, verbose=False)
        info = f"{ica.n_components_} components, excluded {ica.exclude}" + ("" if eog else " (no EOG: select manually)")
    else:
        projs, events = mne.preprocessing.compute_proj_eog(
            data, n_eeg=1, n_grad=0, n_mag=0, ch_name=eog, average=True,
            reject=None, no_proj=True, verbose=False)
        mne.write_proj(out_dir / f"{stem}_proj.fif", projs, overwrite=True, verbose=False)
        data.add_proj(projs).apply_proj(verbose=False)
        info = f"{len(projs)} projector(s) from {len(events)} blinks"
    data.save(out_dir / f"{stem}_eeg.fif", fmt="single", overwrite=True, verbose=False)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("subject", help="participant label without 'sub-', e.g. 07")
    ap.add_argument("--method", choices=["ica", "ssp"], required=True)
    ap.add_argument("--signal", choices=["tcre", "disc", "both"], default="both")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else Path(a.root) / "derivatives" / "artifact-corrected" / f"sub-{a.subject}" / "eeg"
    for sig in (["tcre", "disc"] if a.signal == "both" else [a.signal]):
        print(f"sub-{a.subject} {sig} {a.method}: {run(a.root, a.subject, a.method, sig, out)}", flush=True)
