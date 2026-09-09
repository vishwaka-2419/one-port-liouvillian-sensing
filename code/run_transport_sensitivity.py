#!/usr/bin/env python3
"""Tip-polarisation / tip-angle robustness scan of the coherent-drive EP (v7: bracketed root)."""
from pathlib import Path
import sys,numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from transport_floquet import *
TWOPI=2*np.pi; rows=[]
base=Params(f_drive_Hz=12.94e9,V_dc_V=.8,gamma_L=1.5e8,gamma_R=1e9,T1_intrinsic_s=138.23e-9,
            Tphi_intrinsic_s=10e-9,steps_per_period=12,a_D=.6)
for pL in [.2,.4,.6]:
    for th in [15,25,40]:
        p=replace(base,p_L=pL,theta=np.deg2rad(th))
        f,d=locate_ep(p,6.5,9.5)
        pp=replace(p,Omega_F=TWOPI*f*1e6); g=dissipative_gap(pp); I=cycle_averaged_current(pp)*1e12
        rows.append((pL,th,f,d['sep']/1e6,d['sep_norm'],d['overlap'],max(d['rig1'],d['rig2']),g/1e6,I)); print(rows[-1],flush=True)
pd.DataFrame(rows,columns=['pL','theta_deg','fEP_MHz','sep_Mrad_s','sep_norm','overlap','phase_rigidity','gap_Mrad_s','current_pA'])\
  .to_csv(Path(__file__).resolve().parents[1]/'results/transport_sensitivity.csv',index=False)
