# Derived experimental data

This directory contains machine-readable quantities generated from the externally archived experimental dataset at DOI **10.3929/ethz-b-000669567**.

The raw experimental archive is not redistributed in this repository.

Main files:

- `experimental_summary.json` — weighted longitudinal-relaxation fit summary.
- `experimental_liouvillian_reconstruction.csv` — bias-resolved reconstructed Liouvillian quantities.
- `experimental_uncertainties_mc.csv` — propagated uncertainty intervals from deterministic Monte Carlo sampling.
- `critical_bias_extrapolation.json` — local extrapolation of the critical-bias crossing.
- `cw_linewidth_check.csv` — continuous-wave linewidth consistency check.
- `fig2a_trace.csv` — committed trace used as a figure-reproduction fallback when the external raw archive is not mounted. Before release, regenerate it directly from the DOI-hosted source file with `code/export_fig2a_trace.py`.
- `analyze_log.txt` — analysis log from the released run.

Regenerate the data products with:

```bash
python code/analyze_experiment.py \
  --data-root /path/to/Data_Repository \
  --out data_processed \
  --figures _analysis_figures

python code/critical_bias_extrapolation.py

python code/cw_linewidth_check.py \
  --data-root /path/to/Data_Repository \
  --out data_processed/cw_linewidth_check.csv
```

Before creating the public release, regenerate the fallback trace directly from the raw archive:

```bash
python code/export_fig2a_trace.py --data-root /path/to/Data_Repository
```
