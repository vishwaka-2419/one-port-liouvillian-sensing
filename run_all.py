#!/usr/bin/env python3
"""Reproduce the one-port Liouvillian sensing analysis package (v7).

Usage:
  python run_all.py --data-root /path/to/Data_Repository
  python run_all.py --data-root /path/to/Data_Repository --recompute-transport --compile-latex

The package contains deterministic transport result tables. Recompute the expensive
Floquet scans only when changing model parameters.
"""
from pathlib import Path
import argparse, subprocess, sys, shutil
ROOT=Path(__file__).resolve().parent

def run(cmd):
    print('+',' '.join(map(str,cmd)),flush=True)
    subprocess.run(list(map(str,cmd)),cwd=ROOT,check=True)

def compile_tex(stem):
    run(['pdflatex','-interaction=nonstopmode',f'{stem}.tex'])
    bib=shutil.which('bibtex')
    if not bib and Path('/usr/bin/bibtex.original').exists(): bib='/usr/bin/bibtex.original'
    if bib: run([bib,stem])
    run(['pdflatex','-interaction=nonstopmode',f'{stem}.tex'])
    run(['pdflatex','-interaction=nonstopmode',f'{stem}.tex'])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--recompute-transport',action='store_true'); ap.add_argument('--compile-latex',action='store_true'); a=ap.parse_args(); py=sys.executable
    run([py,'code/analyze_experiment.py','--data-root',a.data_root,'--out','data_processed','--figures','_analysis_figures'])
    run([py,'code/critical_bias_extrapolation.py'])
    run([py,'code/cw_linewidth_check.py','--data-root',a.data_root,'--out','data_processed/cw_linewidth_check.csv'])
    if a.recompute_transport:
        run([py,'code/run_ep_scan.py']); run([py,'code/run_convergence.py']); run([py,'code/run_transport_sensitivity.py']); run([py,'code/run_metrology_extended.py']); run([py,'code/run_target_gap.py'])
    # cheap (< 5 s): always recompute the one-port response tables
    run([py,'code/run_one_port_response.py'])
    # Single unified rendering module (supersedes make_quantum_figures.py,
    # make_feedback_figures.py and the rendering half of make_multispin_figures.py).
    run([py,'code/make_figures.py','--data-root',a.data_root])
    if a.compile_latex:
        compile_tex('main'); compile_tex('supplement')
if __name__=='__main__': main()
