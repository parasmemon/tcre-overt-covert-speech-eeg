"""
Spatial selectivity of tripolar (TCRE) vs disc signals, measured by inter-electrode correlation.

Idea: electrodes in the same array (same scalp site) are only a few mm apart. If a signal is
spatially selective (local), neighbouring electrodes carry different signals -> LOW correlation.
If it is spatially smeared (volume conduction), neighbours carry the same signal -> HIGH correlation.

Uses the cleaned-epochs derivative (1-40 Hz, -0.5 to 2.5 s). Correlation is scale-free, so the
result does not depend on the unknown amplification of the tripolar output.

Channel convention (from the recording notes):
  odd channels (1, 3, 5, ...)  = tripolar signal of electrode k = (ch + 1) / 2
  even channels (2, 4, 6, ...) = disc signal (outer ring) of the same electrode
Site blocks (right hemisphere first): F4, P6, F3, P5; electrodes per site from the channel count
(40/42 -> 5, 56 -> 7, 64 -> 8). Array electrodes only: non-array channels (EOG) are
excluded as listed in channel_sets.py, so a site can have one electrode fewer.

For every participant, condition and modality:
  1. correlation matrix between electrodes, computed per epoch
  2. averaged over epochs with Fisher z
  3. all-pairs r    = mean over ALL electrode pairs (layout-free; computed for every participant)
     within-site r  = mean over electrode pairs in the same site
     between-site r = mean over electrode pairs in different sites
Also: tripolar-vs-disc correlation of the SAME electrode.

Nothing in the dataset is modified. Channels with zero variance are ignored in the analysis.
Per-participant results are cached in OUT_DIR/cache, so a stopped run resumes where it stopped.

Outputs (in OUT_DIR):
  spatial_selectivity_per_subject.tsv
  spatial_selectivity_summary.tsv   (group mean/SD + Wilcoxon signed-rank test, tripolar vs disc)

Usage:
    python spatial_selectivity.py /path/to/bids_root /path/to/output_dir
"""

import sys
from itertools import combinations
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from channel_sets import array_electrodes, cached, layout_label, load_epochs   # noqa: E402

TASK = "OvertCovert"
CONDITIONS = ["Overt", "Covert"]
TARGET_SFREQ = 250.0          # data are low-passed at 40 Hz, so 250 Hz loses nothing


def fisher_mean_corr(x):
    """x: epochs x signals x times -> Fisher-z averaged correlation matrix (signals x signals)."""
    z = []
    for ep in x:
        r = np.corrcoef(ep)
        z.append(np.arctanh(np.clip(r, -0.999999, 0.999999)))
    return np.tanh(np.nanmean(z, axis=0))


def pair_means(r, sites, valid):
    within, between = [], []
    for i, j in combinations(range(len(sites)), 2):
        if not (valid[i] and valid[j]):
            continue
        (within if sites[i] == sites[j] else between).append(r[i, j])
    return np.mean(within), np.mean(between), len(within), len(between)


def analyse_subject(fpath, sub):
    ep = load_epochs(fpath, TARGET_SFREQ)
    el = array_electrodes(sub, len(ep.ch_names))
    layout = layout_label(sub, len(ep.ch_names))
    sites = np.array([e[1] for e in el])
    tri_idx, disc_idx = [e[2] for e in el], [e[3] for e in el]

    rows = []
    for cond in CONDITIONS:
        x = ep[f"Stimulus/{cond}"].get_data()            # epochs x channels x times
        tri, disc = x[:, tri_idx, :], x[:, disc_idx, :]
        valid = (tri.std(axis=(0, 2)) > 0) & (disc.std(axis=(0, 2)) > 0)

        for name, sig in (("tripolar", tri), ("disc", disc)):
            r = fisher_mean_corr(sig)
            iu = [(i, j) for i, j in combinations(range(len(valid)), 2) if valid[i] and valid[j]]
            allp = float(np.mean([r[i, j] for i, j in iu]))
            w, b, nw, nb = pair_means(r, sites, valid)
            rows.append({"layout": layout, "condition": cond, "modality": name,
                         "all_pairs_r": round(allp, 3), "n_all_pairs": len(iu),
                         "within_site_r": round(w, 3), "between_site_r": round(b, 3),
                         "n_within_pairs": nw, "n_between_pairs": nb})

        # tripolar vs disc of the same electrode
        same = []
        for k in np.where(valid)[0]:
            z = [np.arctanh(np.clip(np.corrcoef(tri[e, k], disc[e, k])[0, 1], -0.999999, 0.999999))
                 for e in range(len(tri))]
            same.append(np.tanh(np.mean(z)))
        rows.append({"layout": layout, "condition": cond, "modality": "tripolar_vs_disc_same_electrode",
                     "same_electrode_r": round(float(np.mean(same)), 3), "n_electrodes": len(same)})
    return rows, f"{layout}; {int(valid.sum())}/{len(valid)} electrodes used"


def _compare(d, metric):
    t = d[d.modality == "tripolar"].set_index("participant_id")[metric].dropna()
    c = d[d.modality == "disc"].set_index("participant_id")[metric].dropna()
    t, c = t.align(c, join="inner")
    p = wilcoxon(t, c).pvalue if len(t) >= 5 and (t != c).any() else np.nan
    return {"n_subjects": len(t), "tripolar_mean": round(t.mean(), 3), "tripolar_sd": round(t.std(), 3),
            "disc_mean": round(c.mean(), 3), "disc_sd": round(c.std(), 3),
            "tripolar_lower_in": f"{int((t < c).sum())}/{len(t)}",
            "wilcoxon_p": round(p, 4) if p == p else np.nan}


def summarise(df):
    out = []
    sig = df[df.modality.isin(["tripolar", "disc"])]
    for cond in CONDITIONS:
        d = sig[sig.condition == cond]
        out.append({"condition": cond, "metric": "all_pairs_r", **_compare(d, "all_pairs_r")})
        for metric in ["within_site_r", "between_site_r"]:
            out.append({"condition": cond, "metric": metric, **_compare(d, metric)})
        s = df[(df.condition == cond) & (df.modality == "tripolar_vs_disc_same_electrode")]["same_electrode_r"]
        out.append({"condition": cond, "metric": "tripolar_vs_disc_same_electrode_r", "n_subjects": len(s),
                    "tripolar_mean": round(s.mean(), 3), "tripolar_sd": round(s.std(), 3)})
    return pd.DataFrame(out)


def main(root, out_dir):
    root, out_dir = Path(root).expanduser(), Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted((root / "derivatives" / "cleaned-epochs").glob(f"sub-*/eeg/*_task-{TASK}_desc-cleaned_epo.fif"))
    if not files:
        sys.exit("No cleaned-epochs files found. Run make_cleaned_epochs.py first.")

    rows = []
    for f in files:
        sub = f.name.split("_")[0]
        r, msg = cached(out_dir, "spatial", sub, lambda: analyse_subject(f, sub))
        print(f"{sub}: {msg}", flush=True)
        rows += [{"participant_id": sub, **x} for x in r]

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "spatial_selectivity_per_subject.tsv", sep="\t", index=False)
    summ = summarise(df)
    summ.to_csv(out_dir / "spatial_selectivity_summary.tsv", sep="\t", index=False)
    print("\n" + summ.to_string(index=False))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "spatial_selectivity_output")
