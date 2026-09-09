#!/usr/bin/env python3
from pathlib import Path
import argparse, json
import numpy as np, pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--processed',default='data_processed'); ap.add_argument('--out',default='data_processed/critical_bias_extrapolation.json'); a=ap.parse_args()
    root=Path(__file__).resolve().parents[1]; proc=root/a.processed
    d=pd.read_csv(proc/'experimental_uncertainties_mc.csv')
    d=d[d.bias_mV.isin([-260,-240,-220,-200])].sort_values('bias_mV')
    x=d.bias_mV.to_numpy(float); med=d.ratio_med.to_numpy(float); sig=(d.ratio_p84.to_numpy()-d.ratio_p16.to_numpy())/2
    rng=np.random.default_rng(20260909); y=rng.normal(med,sig,(100000,len(x)))
    xm=x.mean(); den=np.sum((x-xm)**2); slopes=np.sum((y-y.mean(1)[:,None])*(x-xm),axis=1)/den; intercepts=y.mean(1)-slopes*xm
    xc=(1-intercepts)/slopes; q=np.percentile(xc,[16,50,84])
    out={'bias_p16_mV':float(q[0]),'bias_median_mV':float(q[1]),'bias_p84_mV':float(q[2]),'method':'linear fit to -260,-240,-220,-200 mV with independent Gaussian approximation to propagated 68% intervals','note':'local extrapolation outside measured underdamped Rabi window'}
    (root/a.out).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
