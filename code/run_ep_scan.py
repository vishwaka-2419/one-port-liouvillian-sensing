#!/usr/bin/env python3
"""Dissipative-modulation scan of the coherent-drive EP (revised v7: bracketed root)."""
from pathlib import Path
import sys, numpy as np, pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from transport_floquet import *
TWOPI=2*np.pi
rows=[]
base=Params(f_drive_Hz=12.94e9,V_dc_V=.8,p_L=.4,theta=np.deg2rad(25),gamma_L=1.5e8,gamma_R=1e9,
            T1_intrinsic_s=138.23e-9,Tphi_intrinsic_s=10e-9,steps_per_period=20)
for aD in np.linspace(0,1,6):
    p=replace(base,a_D=float(aD))
    f,d=locate_ep(p,7.3,8.3)
    pp=replace(p,Omega_F=TWOPI*f*1e6)
    g=dissipative_gap(pp); I=cycle_averaged_current(pp)*1e12
    rows.append((aD,f,d['sep']/1e6,d['sep_norm'],d['overlap'],max(d['rig1'],d['rig2']),d['cond'],
                 g/1e6,I,d['mu1'].real/1e6,d['mu2'].real/1e6))
    print(rows[-1],flush=True)
out=Path(__file__).resolve().parents[1]/'results'/'floquet_ep_scan.csv'
pd.DataFrame(rows,columns=['aD','fEP_MHz','sep_Mrad_s','sep_norm','overlap','phase_rigidity','eigvec_cond',
                           'gap_Mrad_s','current_pA','mu1_real_Mrad_s','mu2_real_Mrad_s']).to_csv(out,index=False)
