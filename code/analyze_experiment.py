#!/usr/bin/env python3
"""Data-constrained reduced Liouvillian analysis for pentacene ESR-STM.

Reads the Kovarik et al. public repository.  No raw data are redistributed.
Outputs processed parameter tables and figures used by the manuscript.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

TWOPI=2*np.pi

def exp_model(t,yinf,A,T1): return yinf + A*np.exp(-t/T1)

def weighted_t1_fit(root: Path):
    f=root/'SM/S16/dI(Delay).csv'
    d=pd.read_csv(f)
    t=d['Delay (ns)'].to_numpy(float)
    y=d['dI (sig) (fA)'].to_numpy(float)
    s=d['dI_err (sig) (fA)'].to_numpy(float)
    p0=[y[-10:].mean(), y[0]-y[-10:].mean(), 140.]
    popt,pcov=curve_fit(exp_model,t,y,p0=p0,sigma=s,absolute_sigma=True,maxfev=100000)
    err=np.sqrt(np.diag(pcov)); res=(y-exp_model(t,*popt))/s
    chi2=float(np.sum(res**2)); dof=len(y)-len(popt)
    return d,popt,err,chi2,dof

def reported_rabi(root: Path):
    d=pd.read_csv(root/'F4/E-F/FittedParameters.csv')
    cols=['U_DC (mV)','T_2 (ns)','T_2 (error) (ns)','Omega/2pi (MHz)','Omega/2pi (error) (MHz)']
    d=d[cols].dropna().copy().sort_values('U_DC (mV)').reset_index(drop=True)
    return d

def derive_liouvillian(rabi: pd.DataFrame, T1_ns: float):
    d=rabi.copy()
    G1=1/(T1_ns*1e-9)
    TR=d['T_2 (ns)'].to_numpy()*1e-9
    wd=TWOPI*d['Omega/2pi (MHz)'].to_numpy()*1e6
    A=1/TR
    G2=2*A-G1
    D=np.abs(0.5*(G1-G2)) # = |G1-A|
    Om=np.sqrt(wd**2+D**2)
    lam_re=-A
    ratio=Om/np.maximum(D,1e-300)
    Tcrit=1/(G1+Om)
    out=pd.DataFrame({
        'bias_mV':d['U_DC (mV)'],
        'T2Rabi_ns':d['T_2 (ns)'],
        'T2Rabi_err_ns':d['T_2 (error) (ns)'],
        'f_damped_MHz':d['Omega/2pi (MHz)'],
        'f_damped_err_MHz':d['Omega/2pi (error) (MHz)'],
        'Gamma1_Mrad_s':np.full(len(d),G1/1e6),
        'Gamma2_Mrad_s':G2/1e6,
        'decay_A_Mrad_s':A/1e6,
        'EP_halfdifference_Mrad_s':D/1e6,
        'Omega_coherent_Mrad_s':Om/1e6,
        'f_coherent_MHz':Om/TWOPI/1e6,
        'EP_ratio_Omega_over_D':ratio,
        'Tcrit_ns':Tcrit*1e9,
        'lambda_pair_real_Mrad_s':lam_re/1e6,
        'lambda_pair_imag_Mrad_s':wd/1e6,
    })
    return out

def mc_uncertainties(rabi,T1,T1err,n=40000,seed=731):
    rng=np.random.default_rng(seed); rows=[]
    for _,r in rabi.iterrows():
        T1s=rng.normal(T1,T1err,n); TR=rng.normal(r['T_2 (ns)'],r['T_2 (error) (ns)'],n)
        f=rng.normal(r['Omega/2pi (MHz)'],r['Omega/2pi (error) (MHz)'],n)
        good=(T1s>0)&(TR>0)&(f>0); T1s=T1s[good]*1e-9; TR=TR[good]*1e-9; f=f[good]*1e6
        G1=1/T1s; A=1/TR; G2=2*A-G1; D=np.abs(0.5*(G1-G2)); wd=TWOPI*f; Om=np.sqrt(wd**2+D**2); ratio=Om/D; Tcrit=1/(G1+Om)
        def q(x): return np.percentile(x,[16,50,84])
        rr=q(ratio); oo=q(Om/TWOPI/1e6); tt=q(Tcrit*1e9); gg=q(G2/1e6)
        rows.append(dict(bias_mV=r['U_DC (mV)'],ratio_p16=rr[0],ratio_med=rr[1],ratio_p84=rr[2],
                         fcoh_p16_MHz=oo[0],fcoh_med_MHz=oo[1],fcoh_p84_MHz=oo[2],
                         Tcrit_p16_ns=tt[0],Tcrit_med_ns=tt[1],Tcrit_p84_ns=tt[2],
                         Gamma2_p16_Mrad_s=gg[0],Gamma2_med_Mrad_s=gg[1],Gamma2_p84_Mrad_s=gg[2],
                         p_underdamped=float(np.mean(ratio>1))))
    return pd.DataFrame(rows)

def fit_linear_overlay(t,y,TR_ns,f_MHz):
    e=np.exp(-t/TR_ns); w=TWOPI*f_MHz*1e-3 # rad/ns
    X=np.column_stack([np.ones_like(t),e*np.cos(w*t),e*np.sin(w*t)])
    beta=np.linalg.lstsq(X,y,rcond=None)[0]
    return X@beta,beta

def make_figures(root,outdir,t1fit,t1err,derived,mc):
    outdir.mkdir(parents=True,exist_ok=True)
    # Fig: direct experimental reconstruction
    fig,axs=plt.subplots(2,2,figsize=(10.5,7.5),constrained_layout=True)
    d,popt,err,chi2,dof=t1fit
    ax=axs[0,0]; t=d['Delay (ns)'].to_numpy(); y=d['dI (sig) (fA)']; s=d['dI_err (sig) (fA)'];
    ax.errorbar(t,y,yerr=s,fmt='o',ms=2.5,lw=.7,capsize=1.2,label='repository data')
    tt=np.linspace(t.min(),t.max(),800); ax.plot(tt,exp_model(tt,*popt),lw=1.8,label='weighted exponential fit')
    ax.set(xlabel='pump--probe delay (ns)',ylabel=r'$\Delta I$ (fA)',title='(a) Independent $T_1$ reconstruction')
    ax.text(.98,.95,fr'$T_1={popt[2]:.1f}\pm{err[2]:.1f}$ ns'+'\n'+fr'$\chi^2_\nu={chi2/dof:.2f}$',ha='right',va='top',transform=ax.transAxes,fontsize=9)
    ax.legend(fontsize=8)
    # raw traces with constrained overlays
    raw=pd.read_csv(root/'SM/S21/AllSpectra.csv')
    ax=axs[0,1]
    for k,V in enumerate([-60,-140,-220,-260]):
        t=raw[f'Pulse duration (ns)@{V} mV'].to_numpy(); y=raw[f'dI (fA)@{V} mV'].to_numpy()
        row=derived.loc[np.isclose(derived.bias_mV,V)].iloc[0]
        yh,_=fit_linear_overlay(t,y,row.T2Rabi_ns,row.f_damped_MHz)
        shift=45*k
        ax.plot(t,y+shift,'.',ms=2,label=f'{V} mV data')
        ax.plot(t,yh+shift,lw=1.2)
    ax.set(xlabel='pulse duration (ns)',ylabel=r'$\Delta I$ + offset (fA)',title='(b) Raw Rabi traces and repository-constrained modes')
    ax.legend(ncol=2,fontsize=7)
    # reconstructed rates and EP ratio
    ax=axs[1,0]
    x=derived.bias_mV.to_numpy(); ax.errorbar(x,derived.f_damped_MHz,yerr=derived.f_damped_err_MHz,fmt='o',ms=4,label=r'observed $\omega_d/2\pi$')
    ax.plot(x,derived.f_coherent_MHz,'s-',ms=4,label=r'reconstructed $\Omega_F/2\pi$')
    ax.plot(x,derived.EP_halfdifference_Mrad_s/TWOPI,'^-',ms=4,label=r'EP threshold $D/2\pi$')
    ax.set(xlabel=r'$U_{DC}$ (mV)',ylabel='frequency scale (MHz)',title='(c) Liouvillian reconstruction from measured transients')
    ax.legend(fontsize=7.5)
    # ratio with MC interval, log scale perhaps
    ax=axs[1,1]; mm=mc.sort_values('bias_mV')
    ax.fill_between(mm.bias_mV,mm.ratio_p16,mm.ratio_p84,alpha=.22,label='68% propagated interval')
    ax.plot(mm.bias_mV,mm.ratio_med,'o-',ms=4,label=r'$\Omega_F/D$')
    ax.axhline(1,ls='--',lw=1,label='EP')
    ax.set(xlabel=r'$U_{DC}$ (mV)',ylabel='distance ratio',title='(d) All measured Rabi points remain underdamped')
    ax.set_ylim(bottom=.8); ax.legend(fontsize=7.5)
    fig.savefig(outdir/'fig_data_reconstruction.pdf',bbox_inches='tight'); fig.savefig(outdir/'fig_data_reconstruction.png',dpi=220,bbox_inches='tight'); plt.close(fig)

    # Critical design prescription
    fig,ax=plt.subplots(figsize=(7.2,4.7),constrained_layout=True)
    x=derived.bias_mV.to_numpy()
    ax.errorbar(x,derived.T2Rabi_ns,yerr=derived.T2Rabi_err_ns,fmt='o-',label=r'measured $T_{2,\mathrm{Rabi}}$')
    ax.plot(x,derived.Tcrit_ns,'s--',label=r'critical envelope $T_{\rm crit}=1/(\Gamma_1+\Omega_F)$')
    ax.fill_between(x,mc.Tcrit_p16_ns,mc.Tcrit_p84_ns,alpha=.2)
    ax.set(xlabel=r'$U_{DC}$ (mV)',ylabel='time (ns)',title='Experimentally testable route to critical damping')
    ax.legend(); fig.savefig(outdir/'fig_ep_prescription.pdf',bbox_inches='tight'); fig.savefig(outdir/'fig_ep_prescription.png',dpi=220,bbox_inches='tight'); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--out',required=True); ap.add_argument('--figures',required=True); args=ap.parse_args()
    root=Path(args.data_root); out=Path(args.out); figs=Path(args.figures); out.mkdir(parents=True,exist_ok=True)
    d,p,e,chi2,dof=weighted_t1_fit(root); T1=p[2]; T1err=e[2]
    r=reported_rabi(root); der=derive_liouvillian(r,T1); mc=mc_uncertainties(r,T1,T1err)
    der.to_csv(out/'experimental_liouvillian_reconstruction.csv',index=False); mc.to_csv(out/'experimental_uncertainties_mc.csv',index=False)
    summary={'T1_ns':float(T1),'T1_err_ns':float(T1err),'T1_chi2':float(chi2),'T1_dof':int(dof),'T1_reduced_chi2':float(chi2/dof),
             'minimum_EP_ratio_median':float(mc.ratio_med.min()),'minimum_EP_ratio_bias_mV':float(mc.loc[mc.ratio_med.idxmin(),'bias_mV']),
             'all_p_underdamped_min':float(mc.p_underdamped.min())}
    (out/'experimental_summary.json').write_text(json.dumps(summary,indent=2))
    make_figures(root,figs,(d,p,e,chi2,dof),T1err,der,mc)
    print(json.dumps(summary,indent=2)); print(der.to_string(index=False))
if __name__=='__main__': main()
