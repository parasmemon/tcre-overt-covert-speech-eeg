"""
Shared channel definitions for the TCRE overt/covert speech dataset.

Channel convention (recording notes): electrode k is recorded on channels 2k-1 and 2k;
  odd channel  (2k-1) = tripolar (TCRE) signal
  even channel (2k)   = emulated disc signal (outer ring) of the same electrode
Scalp sites in channel order (right hemisphere first): F4, P6, F3, P5, each block = one array.

Electrodes per array (= per site, F4/P6/F3/P5), from the number of channels:
  40 channels -> 5/5/5/5   (sub-01; and sub-07, 08, 09, 14, 15, 16 in the cleaned epochs)
  42 channels -> 5/5/5/5   (raw data of sub-07, 08, 09, 14, 15, 16: channels 41-42 = EOG)
  56 channels -> 7/7/7/7   (sub-05, 06)
  64 channels -> 8/8/8/8   (sub-02, 03, 04, 10: all 64 channels are array channels)
Exception (recording log, confirmed by the signal correlations):
  sub-11, 12, 13 -> 8/7/8/8 (7 electrodes in P6): F4 ch1-16, P6 ch17-30, F3 ch31-46, P5 ch47-62;
                    raw channels 63-64 = EOG; the cleaned epochs have 62 channels.

The cleaned-epochs derivative contains array channels only. In the raw data, the EOG channels (two standard disc electrodes, above and to
the right of one eye) lie
after the array electrodes and are never selected by array_electrodes().
"""

import pickle
from pathlib import Path

import numpy as np

SITES = ["F4", "P6", "F3", "P5"]
ELECTRODES_PER_SITE = {40: 5, 42: 5, 56: 7, 64: 8}
LAYOUT = {s: (8, 7, 8, 8) for s in ["sub-11", "sub-12", "sub-13"]}   # per-site counts, F4/P6/F3/P5

NON_ARRAY = {}          # no array channel is excluded (see docstring)


def array_electrodes(sub, n_channels):
    """Array electrodes of one recording.

    Returns a list of (electrode_number, site, tripolar_channel_index, disc_channel_index),
    channel indices 0-based."""
    counts = LAYOUT.get(sub) or (ELECTRODES_PER_SITE[n_channels],) * len(SITES)
    sites = [site for site, n in zip(SITES, counts) for _ in range(n)]
    excluded = set(NON_ARRAY.get(sub, []))
    out = []
    for k, site in enumerate(sites, start=1):
        tri_ch, disc_ch = 2 * k - 1, 2 * k                    # 1-based
        if tri_ch in excluded or disc_ch in excluded:
            continue
        out.append((k, site, tri_ch - 1, disc_ch - 1))
    return out


def array_channels(sub, n_channels):
    """0-based channel indices (tripolar, disc) of the array electrodes, in electrode order."""
    el = array_electrodes(sub, n_channels)
    return np.array([e[2] for e in el]), np.array([e[3] for e in el])


def layout_label(sub, n_channels):
    el = array_electrodes(sub, n_channels)
    counts = {s: sum(e[1] == s for e in el) for s in SITES}
    excl = NON_ARRAY.get(sub, [])
    return ("/".join(str(counts[s]) for s in SITES) + " electrodes (F4/P6/F3/P5)"
            + (f"; excluded ch {','.join(map(str, excl))}" if excl else ""))


def cached(out_dir, tag, sub, compute):
    """Per-participant cache so a run can be stopped and resumed (results are recomputed
    only for participants without a cache file)."""
    f = Path(out_dir) / "cache" / f"{tag}_{sub}.pkl"
    if f.exists():
        with open(f, "rb") as fh:
            return pickle.load(fh)
    res = compute()
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(res, fh)
    tmp.replace(f)
    return res


def load_epochs(fpath, target_sfreq=None):
    """Read the cleaned epochs with low memory use (decimated while loading)."""
    import mne
    ep = mne.read_epochs(fpath, preload=False, verbose=False)
    if target_sfreq:
        decim = int(round(ep.info["sfreq"] / target_sfreq))
        if decim > 1:
            ep.decimate(decim, verbose=False)
    ep = ep[["Stimulus/Overt", "Stimulus/Covert"]]
    ep.load_data()
    return ep
