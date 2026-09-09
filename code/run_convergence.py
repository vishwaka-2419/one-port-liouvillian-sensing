#!/usr/bin/env python3
"""Floquet time-step convergence of the coherent-drive EP (revised v7).

At every discretisation N_t the EP is located as the bracketed root of the real
discriminant Re[(mu_1-mu_2)^2] (see transport_floquet.locate_ep) instead of a bounded
minimisation of |mu_1-mu_2|, whose tolerance dominated the residual separation in
earlier revisions.  Reported per N_t: f_EP, the normalised pair separation
|mu_1-mu_2|/(|mu_1|+|mu_2|), the right-eigenvector overlap, the phase rigidity of the
pair and the condition number of the eigenvector matrix.  A Richardson fit
f_EP(N_t) = f_inf + a N_t^-2 documents the second-order midpoint convergence.
"""
from pathlib import Path
import sys, numpy as np, pandas as pd, json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from transport_floquet import *
rows=[]
for n in [8,12,16,20,24,32,48,64]:
    p=Params(f_drive_Hz=12.94e9,V_dc_V=.8,p_L=.4,theta=np.deg2rad(25),gamma_L=1.5e8,gamma_R=1e9,
             T1_intrinsic_s=138.23e-9,Tphi_intrinsic_s=10e-9,steps_per_period=n,a_D=.6)
    f,d=locate_ep(p,7.5,8.2)
    rows.append(dict(steps_per_period=n,fEP_MHz=f,sep_Mrad_s=d['sep']/1e6,sep_norm=d['sep_norm'],
                     overlap=d['overlap'],phase_rigidity=max(d['rig1'],d['rig2']),eigvec_cond=d['cond'],
                     mu_real_Mrad_s=d['mu1'].real/1e6))
    print(rows[-1],flush=True)
df=pd.DataFrame(rows)
out=Path(__file__).resolve().parents[1]/'results'
df.to_csv(out/'floquet_convergence.csv',index=False)
# Richardson-type fit on the four finest grids
sub=df[df.steps_per_period>=20]
X=np.c_[np.ones(len(sub)),1/sub.steps_per_period.to_numpy()**2]
coef,*_=np.linalg.lstsq(X,sub.fEP_MHz.to_numpy(),rcond=None)
fit=dict(f_inf_MHz=float(coef[0]),a_MHz=float(coef[1]),
         residual_MHz=float(np.max(np.abs(X@coef-sub.fEP_MHz.to_numpy()))))
json.dump(fit,open(out/'floquet_convergence_fit.json','w'),indent=1); print(fit)
