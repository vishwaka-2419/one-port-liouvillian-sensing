#!/usr/bin/env python3
"""One-port response of the exchange-coupled chain: residues, participation, phase rigidity.

Writes
  results/one_port_slow_mode.csv     slow-mode gap, participation, |Res|, rigidity versus N
  results/one_port_rigidity_scan.csv slow-mode and minimum rigidity versus gamma_b/J
  results/one_port_spectrum.csv      |chi_11(delta)| for a short chain with its modal decomposition
  results/one_port_checks.json       machine-precision defects of the exact relations and the
                                     illustrative ESR-STM numbers quoted in the text
"""
from pathlib import Path
import sys, json, numpy as np, pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from multispin_chain import (one_port_modes, one_port_susceptibility, one_port_sum_rules,
                             slow_weight_asymptotic, boundary_coefficient)
ROOT=Path(__file__).resolve().parents[1]; RES=ROOT/'results'; RES.mkdir(exist_ok=True)
J=1.0

# 1. slow mode versus N for the benchmark and for the illustrative ESR-STM ratios
rows=[]
Ns=np.unique(np.round(np.logspace(np.log10(2),2,80)).astype(int))
for gs,gt,tag in [(.25,0.,'benchmark'),(.25,.01,'benchmark_gt0.01'),(.3,.0101,'esrstm_illustrative')]:
    for N in Ns:
        m=one_port_modes(int(N),J,gs,gt)
        rows.append(dict(case=tag,N=int(N),gamma_s=gs,gamma_t=gt,gap=-m['lam'][0].real,
                         participation=m['part'][0],residue_abs=abs(m['res'][0]),rigidity=m['rig'][0],
                         residue_over_participation=abs(m['res'][0])/m['part'][0],
                         asymptotic_participation=slow_weight_asymptotic(int(N),J,gs-gt)))
pd.DataFrame(rows).to_csv(RES/'one_port_slow_mode.csv',index=False)

# 2. rigidity versus boundary loss
rows=[]
gbs=np.logspace(-2,np.log10(4),49)
for N in [6,10,20,40]:
    for gb in gbs:
        m=one_port_modes(N,J,gb,0.)
        rows.append(dict(N=N,gamma_b_over_J=gb,slow_rigidity=m['rig'][0],min_rigidity=m['rig'].min(),
                         slow_residue_over_participation=abs(m['res'][0])/m['part'][0],
                         max_residue_over_participation=float(np.max(np.abs(m['res'])/m['part']))))
pd.DataFrame(rows).to_csv(RES/'one_port_rigidity_scan.csv',index=False)

# 3. spectrum of a short chain (N=6) with modal decomposition
N=6; gs=.25; gt=.01
m=one_port_modes(N,J,gs,gt)
delta=np.linspace(-1.4,1.4,1401)
chi=one_port_susceptibility(delta,N,J,gs,gt)
df=pd.DataFrame(dict(delta_over_J=delta,abs_chi=np.abs(chi),re_chi=chi.real,im_chi=chi.imag))
for k in range(N):
    term=m['res'][k]/(1j*delta-m['lam'][k])
    df[f'abs_mode{k+1}']=np.abs(term)
pd.DataFrame(dict(mode=np.arange(1,N+1),re_lam=m['lam'].real,im_lam=m['lam'].imag,
                  participation=m['part'],abs_residue=np.abs(m['res']),rigidity=m['rig']))\
   .to_csv(RES/'one_port_spectrum_modes.csv',index=False)
df.to_csv(RES/'one_port_spectrum.csv',index=False)

# 4. exact-relation checks and the illustrative ESR-STM numbers
checks={}
for (N_,gs_,gt_) in [(8,.25,0.),(20,.25,.01),(40,.6,.001),(12,1.2,0.),(7,.3,.0101)]:
    c=one_port_sum_rules(N_,J,gs_,gt_); c['residue_sum']=[c['residue_sum'].real,c['residue_sum'].imag]
    checks[f'N{N_}_gs{gs_}_gt{gt_}']=c
gamma_s=2.2e8; ratio=0.3; Jrad=gamma_s/ratio; gamma_t=1/135e-9
mi=one_port_modes(7,1.0,ratio,gamma_t/Jrad)
checks['esrstm_N7']=dict(J_over_2pi_MHz=Jrad/2/np.pi/1e6,gamma_t_over_J=gamma_t/Jrad,
                         slow_gap_s=-mi['lam'][0].real*Jrad,slow_participation=mi['part'][0],
                         slow_residue_abs=abs(mi['res'][0]),slow_rigidity=mi['rig'][0],
                         slow_lifetime_ns=1e9/(-mi['lam'][0].real*Jrad),
                         boundary_share_of_slow_decay=(-mi['lam'][0].real-gamma_t/Jrad)/(-mi['lam'][0].real))
json.dump(checks,open(RES/'one_port_checks.json','w'),indent=1)
print(json.dumps(checks['esrstm_N7'],indent=1)); print('slow-mode rigidity N=7 benchmark:',one_port_modes(7,1.,.25,0.)['rig'][0])
