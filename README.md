# One-port Liouvillian sensing — reproducibility package

Reproducibility package for the manuscript:

**Aishwarya Vishwakarma, “One-port Liouvillian sensing: critical dynamics, measurement information, and collective-mode visibility.”**

The repository contains the numerical source code, derived tables, deterministic simulation outputs, figure-generation scripts, and manuscript source used in the study. The original experimental raw data are **not redistributed**.

## External experimental data

The experimental analysis uses the public dataset accompanying:

> S. Kovarik *et al.*, “Spin torque-driven electron paramagnetic resonance of a single spin in a pentacene molecule,” *Science* **384**, 1368–1373 (2024).

Dataset DOI: **10.3929/ethz-b-000669567**

Download and extract that archive separately. The directory passed to the scripts as `--data-root` must contain the `SM/` and `F4/` subdirectories.

## Repository layout

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── .zenodo.json
├── .gitignore
├── Main.tex
├── Main.pdf
├── Supplementary.tex
├── Supplementary.pdf
├── references.bib
├── main.bbl
├── supplement.bbl
├── environment.yml
├── requirements.txt
├── run_all.py
├── FIGURE_PROVENANCE.md
├── code/
│   ├── analyze_experiment.py
│   ├── critical_bias_extrapolation.py
│   ├── cw_linewidth_check.py
│   ├── export_fig2a_trace.py
│   ├── transport_floquet.py
│   ├── run_ep_scan.py
│   ├── run_convergence.py
│   ├── run_transport_sensitivity.py
│   ├── run_metrology_extended.py
│   ├── target_spin_gap.py
│   ├── run_target_gap.py
│   ├── multispin_chain.py
│   ├── run_one_port_response.py
│   ├── make_figures.py
│   └── superseded/
├── data_processed/
│   └── README.md
├── results/
│   └── README.md
└── figures/
```

## Python environment

Recommended environment:

```bash
conda env create -f environment.yml
conda activate esrstm-liouvillian
```

Alternatively:

```bash
python -m pip install -r requirements.txt
```

The tested Python target is Python 3.11. LaTeX compilation additionally requires a local TeX installation providing `pdflatex` and `bibtex`; these are not installed by the Python environment file.

## Prepare the committed Fig. 2a fallback from the raw archive

Before the public release, regenerate `data_processed/fig2a_trace.csv` directly from the DOI-hosted raw file:

```bash
python code/export_fig2a_trace.py --data-root /path/to/Data_Repository
```

This ensures the committed fallback trace has direct raw-data provenance rather than figure digitization.

## Reproduce the analysis

### Standard reproduction

This recomputes the experimental processing, critical-bias extrapolation, CW consistency check, one-port response tables, and figures while using the committed deterministic transport tables:

```bash
python run_all.py --data-root /path/to/Data_Repository
```

### Recompute all numerical scans

To regenerate the Floquet, convergence, transport-sensitivity, metrology, target-spin, and one-port outputs:

```bash
python run_all.py \
  --data-root /path/to/Data_Repository \
  --recompute-transport
```

### Recompute and compile the manuscripts

After ensuring the manuscript filename case in `run_all.py` matches `Main.tex` and `Supplementary.tex`:

```bash
python run_all.py \
  --data-root /path/to/Data_Repository \
  --recompute-transport \
  --compile-latex
```

## Output directories

- `data_processed/` contains quantities derived from the public experimental dataset.
- `results/` contains deterministic numerical outputs from the model calculations.
- `figures/` contains the rendered main-text and supplementary figures.

The machine-readable tables should be treated as the primary numerical outputs behind the plotted quantities.

## Raw-data policy

Do not commit `Data_Repository.zip` or an extracted copy of the external raw dataset to this repository. The `.gitignore` file includes guards against accidental inclusion of common raw-data directory names.

## Citation

GitHub will display the citation metadata from `CITATION.cff`. The archival DOI for this reproducibility package should be added here after the first Zenodo-backed GitHub release is archived.

## License

The source code is released under the MIT License; see `LICENSE`. Third-party experimental data retain the terms of their original archive and are not redistributed here. Manuscript text and figures are scholarly outputs and are not relicensed by the software license.
