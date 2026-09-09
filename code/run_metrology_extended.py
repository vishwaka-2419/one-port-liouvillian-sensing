from pathlib import Path
import sys, numpy as np, pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from transport_floquet import *
TWOPI=2*np.pi
root=Path(__file__).resolve().parents[1]
ep=pd.read_csv(root/'results/floquet_ep_scan.csv')
fep=float(np.interp(.6,ep.aD,ep.fEP_MHz))
base=Params(f_drive_Hz=12.94e9,V_dc_V=.8,p_L=.4,theta=np.deg2rad(25),gamma_L=1.5e8,gamma_R=1e9,T1_intrinsic_s=138.23e-9,Tphi_intrinsic_s=10e-9,steps_per_period=12,a_D=.6)
# Dense around EP, logarithmically extended to ten times the EP.
ratios=np.unique(np.concatenate([np.linspace(.60,2.0,17),np.geomspace(2.1,10.0,16)]))
rows=[]
for ratio in ratios:
    f=ratio*fep
    p=replace(base,Omega_F=TWOPI*f*1e6)
    sep,ov,*_=spin_pair_diagnostics(p)
    fi=gaussian_counting_fi_rate(p,'detuning_rad_s',step=TWOPI*1e5)
    qfi,occ=periodic_state_qfi(p,'detuning_rad_s',step=TWOPI*1e5)
    I=cycle_averaged_current(p)*1e12
    rows.append((ratio,f,sep/1e6,ov,fi,qfi,occ,I))
    print(f'{ratio:.4f} {f:.3f} FI={fi:.6e} QFI={qfi:.6e} I={I:.6f}',flush=True)
df=pd.DataFrame(rows,columns=['Omega_over_EP','f_MHz','pair_sep_Mrad_s','overlap','counting_FI_per_rad2s','state_QFI_per_rad2s','singly_occ','current_pA'])
df.to_csv(root/'results/metrology_scan.csv',index=False)
imax=int(np.nanargmax(df.counting_FI_per_rad2s.to_numpy()))
print('MAX',df.iloc[imax].to_dict())
