#!/usr/bin/env python3
from pathlib import Path
import argparse, re
import numpy as np, pandas as pd
from scipy.optimize import curve_fit

def lorentz_mix(x,c,As,Aa,x0,g):
    dx=x-x0
    den=dx*dx+g*g
    return c + As*g*g/den + Aa*g*dx/den

def fit_one(df,bias=-250,vrf=80):
    prefix=f'vdc{bias:.1f}mV,1.3696T,vrf{vrf}mV'
    xcol=f'f (fit) (GHz)@{prefix}'; ycol=f'dI (fit) (pA)@{prefix}'
    x=df[xcol].dropna().to_numpy(float); y=df.loc[df[xcol].notna(),ycol].to_numpy(float)
    # use fit curve supplied in repository; this estimates its linewidth only
    k=np.argmax(np.abs(y-np.nanmedian(y)))
    x0=x[k]; span=max(x.max()-x.min(),1e-3)
    p0=[np.median(y), y[k]-np.median(y), 0.0, x0, 0.05]
    bounds=([-np.inf,-np.inf,-np.inf,x.min(),1e-5],[np.inf,np.inf,np.inf,x.max(),span])
    popt,_=curve_fit(lorentz_mix,x,y,p0=p0,bounds=bounds,maxfev=100000)
    return 2*abs(popt[4])*1e3 # MHz

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    df=pd.read_csv(Path(a.data_root)/'SM/S17/S17-Sweeps.csv')
    rows=[]
    for vrf in (40,80):
        width=fit_one(df,-250,vrf)
        rows.append({'bias_mV':-250,'current_pA':50,'U_RF_mV':vrf,'cw_FWHM_MHz':width,'source':'SM/S17 repository fit curve'})
    # Effective transient linewidth implied by the Liouvillian inversion at -260 mV.
    rec=pd.read_csv(Path(a.out).parent/'experimental_liouvillian_reconstruction.csv')
    r=rec.loc[np.isclose(rec.bias_mV,-260)].iloc[0]
    rows.append({'bias_mV':-260,'current_pA':13,'U_RF_mV':240,'cw_FWHM_MHz':float(r.Gamma2_Mrad_s/np.pi),'source':'homogeneous-equivalent width Gamma2/pi from Rabi inversion'})
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out,index=False)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__': main()
