# TCRE overt and covert speech EEG: code for the data descriptor

Code and results for the technical validation of the dataset

> **An EEG Dataset of Overt and Covert Speech Recorded with Tripolar Concentric Ring Electrodes**
> Paras Qadir Memon, Shoaib Memon, Walter G. Besio, Charles W. Anderson (submitted to *Sensors*).

The dataset (BIDS format) is available on OpenNeuro: accession **ds008895**, version **1.1.0** (https://doi.org/10.18112/openneuro.ds008895.v1.1.0).
It contains EEG from 16 adults who named pictures aloud (overt) or silently (covert), recorded with
four arrays of tripolar concentric ring electrodes (TCREs) centred on F4, P6, F3 and P5. Every electrode
provides a tripolar (Laplacian) signal on the odd channels and an emulated disc signal on the even channels.

## Contents

```
code/      Python scripts (one per analysis or figure)
results/   Output of the scripts: per-participant tables (.tsv), group data (.npz) and figures (.png)
```

| Paper item | Script | Output in `results/` |
|---|---|---|
| Cleaned-epochs derivative (Section 2.5) | `make_cleaned_epochs.py`, `channel_sets.py` | written into the dataset (`derivatives/cleaned-epochs/`) |
| Figure 1, array positions | `make_fig1.py` | `fig1_array_positions.png` |
| Figure 2, paradigm; marker interval 6.08–6.13 s | `make_paradigm_figure.py` | `fig_task_paradigm.png`, `task_timing_per_subject.tsv` |
| Amplitude of raw recordings (245 vs 14 µV); peak-to-peak of epochs (200 µV example) | `qc_metrics.py` | `qc_recording.tsv`, `qc_epochs.tsv` |
| Table 4, event-related SNR | `snr_analysis.py` | `snr_per_subject.tsv`, `snr_stats.tsv` |
| Figure 3, event-related potentials | `erp_analysis.py` | `fig_erp_tripolar_disc.png`, `erp_*.tsv`, `erp_group.npz` |
| Figure 4 and Table 4, relative power and overt/covert gamma ratio | `psd_analysis.py` | `fig_psd_tripolar_disc.png`, `psd_*.tsv` |
| Figure 5, ERD/ERS; alpha and beta changes | `tfr_erd_ers.py` | `fig_tfr_erd_ers.png`, `tfr_band_erd_*.tsv`, `tfr_group_maps.npz` |
| Table 4, within-site correlation | `spatial_selectivity.py` | `spatial_selectivity_*.tsv` |
| Figure 6, spatial selectivity | `make_spatial_figure.py` | `fig_spatial_selectivity.png` |

`channel_sets.py` holds the channel layout of every participant and is used by all analysis scripts.

## Requirements

Python 3.10 or later and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

## How to run

1. Download the dataset from OpenNeuro, e.g. to `~/data/tcre_speech` (the folder that contains `dataset_description.json`).
2. Run the scripts from the `code/` folder, with the dataset folder and an output folder as arguments:

```
cd code
BIDS=~/data/tcre_speech
OUT=../results

python make_fig1.py $OUT
python make_paradigm_figure.py $BIDS $OUT
python qc_metrics.py $BIDS $OUT
python snr_analysis.py $BIDS $OUT
python erp_analysis.py $BIDS $OUT
python psd_analysis.py $BIDS $OUT
python tfr_erd_ers.py $BIDS $OUT
python spatial_selectivity.py $BIDS $OUT
python make_spatial_figure.py $BIDS $OUT auto
```

`make_spatial_figure.py` needs the output of `spatial_selectivity.py`. The other scripts are independent.
All analysis scripts except `qc_metrics.py` (raw amplitude) use the cleaned-epochs derivative that is part of the dataset.
To regenerate the derivative itself: `python make_cleaned_epochs.py $BIDS`, then `python make_cleaned_epochs.py $BIDS --finalize`.

Per-participant results are cached in `OUT/cache/`, so a stopped run continues where it stopped.
Delete the `cache/` folder to recompute everything. No script modifies the dataset, except
`make_cleaned_epochs.py`, which writes the derivative.

## Notes

- Tripolar and disc channels have different amplitude scales, so all comparisons use scale-free measures
  (ratios, relative power, percentage change, correlation).
- Statistics are two-sided Wilcoxon signed-rank tests across the 16 participants, not corrected for multiple comparisons.

## Licence

Code: MIT licence (see `LICENSE`). The dataset itself is released under CC0 on OpenNeuro.

## Citation

If you use this code or the dataset, please cite the data descriptor (reference added on publication) and the OpenNeuro dataset.
