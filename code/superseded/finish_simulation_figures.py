from pathlib import Path
import sys,json
import numpy as np,pandas as pd,matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parent))
from transport_floquet import *
from target_spin_gap import gap as target_gap, weak_coupling_fit
TWOPI=2*np.pi
root=Path(__file__).resolve().parents[1]; out=root/'figures'; res=root/'results'
def save(fig,name):
 fig.savefig(out/(name+'.pdf'),bbox_inches='tight'); fig.savefig(out/(name+'.png'),dpi=220,bbox_inches='tight'); plt.close(fig)
# floquet figure from prior scan
df=pd.read_csv(res/'floquet_ep_scan.csv'); cv=pd.read_csv(res/'floquet_convergence.csv')
fig,axs=plt.subplots(2,2,figsize=(10.5,7.2),constrained_layout=True)
axs[0,0].plot(df.aD,df.fEP_MHz,'o-'); axs[0,0].set(xlabel=r'DLT modulation $a_D$',ylabel=r'$\Omega_{F,EP}/2\pi$ (MHz)',title='(a) DLT shifts the Floquet EP')
axs[0,1].plot(df.aD,df.gap_Mrad_s,'s-'); axs[0,1].set(xlabel=r'$a_D$',ylabel=r'$\Delta_{\cal L}$ ($10^6$ s$^{-1}$)',title='(b) DLT controls the relaxation gap')
axs[1,0].plot(df.aD,df.overlap,'^-'); axs[1,0].set_ylim(.99997,1.000002); axs[1,0].set(xlabel=r'$a_D$',ylabel='right-mode overlap',title='(c) Eigenvector coalescence')
axs[1,1].plot(cv.steps_per_period,cv.fEP_MHz,'o-'); axs[1,1].axhline(cv.fEP_MHz.iloc[-1],ls=':',lw=1); axs[1,1].set(xlabel='Floquet slices per period',ylabel=r'$\Omega_{F,EP}/2\pi$ (MHz)',title='(d) Time-discretization convergence')
save(fig,'fig5_floquet_transport')
# metrology
base=Params(f_drive_Hz=12.94e9,V_dc_V=.8,p_L=.4,theta=np.deg2rad(25),gamma_L=1.5e8,gamma_R=1e9,T1_intrinsic_s=138.23e-9,Tphi_intrinsic_s=10e-9,steps_per_period=12,a_D=.6)
fep=float(np.interp(.6,df.aD,df.fEP_MHz)); fs=np.linspace(4.8,15.,17); rows=[]
for f in fs:
 p=replace(base,Omega_F=TWOPI*f*1e6); sep,ov,*_=spin_pair_diagnostics(p)
 fi=gaussian_counting_fi_rate(p,'detuning_rad_s',step=TWOPI*1e5); qfi,occ=periodic_state_qfi(p,'detuning_rad_s',step=TWOPI*1e5); I=cycle_averaged_current(p)*1e12
 rows.append((f,sep/1e6,ov,fi,qfi,occ,I)); print('met',rows[-1],flush=True)
md=pd.DataFrame(rows,columns=['f_MHz','pair_sep_Mrad_s','overlap','counting_FI_per_rad2s','state_QFI_per_rad2s','singly_occ','current_pA']); md.to_csv(res/'metrology_scan.csv',index=False)
def norm(x):
 a=np.asarray(x,float); return a/np.nanmax(a)
fig,axs=plt.subplots(1,2,figsize=(10.4,4.2),constrained_layout=True)
axs[0].plot(md.f_MHz,norm(1/np.maximum(md.pair_sep_Mrad_s,1e-8)),'o-',label='spectral coalescence index')
axs[0].plot(md.f_MHz,norm(md.counting_FI_per_rad2s),'s-',label='counting FI rate')
axs[0].plot(md.f_MHz,norm(md.state_QFI_per_rad2s),'^-',label='periodic-state QFI')
axs[0].axvline(fep,ls='--',label='Floquet EP'); axs[0].set(xlabel=r'$\Omega_F/2\pi$ (MHz)',ylabel='normalized quantity',title='(a) Spectral amplification vs information'); axs[0].legend(fontsize=7.2)
axs[1].plot(md.f_MHz,md.current_pA,'o-'); axs[1].axvline(fep,ls='--'); axs[1].set(xlabel=r'$\Omega_F/2\pi$ (MHz)',ylabel='cycle-averaged current (pA)',title='(b) Transport observable is regular at the EP')
save(fig,'fig6_counting_metrology')
# target gap
fs=np.concatenate([[0],np.logspace(-5,1,110)]); Js=TWOPI*fs*1e6; gaps=np.array([target_gap(J_rad_s=x) for x in Js]); g0,c=weak_coupling_fit(Js,gaps)
pd.DataFrame({'J_over_2pi_MHz':fs,'gap_s-1':gaps}).to_csv(res/'target_gap_scan.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(10.4,4.2),constrained_layout=True)
axs[0].loglog(fs[1:],gaps[1:],'o',ms=2.8,label='two-spin Liouvillian'); pred=g0+c*Js**2; axs[0].loglog(fs[1:35],pred[1:35],'--',label=r'weak coupling $\Delta_0+CJ^2$'); axs[0].set(xlabel=r'$J/2\pi$ (MHz)',ylabel=r'$\Delta_{\cal L}$ (s$^{-1}$)',title='(a) Hybridization opens the slow-mode gap'); axs[0].legend(fontsize=8)
axs[1].plot(fs[:50],gaps[:50]); axs[1].axhline(gaps[0],ls='--',label='intrinsic target floor'); axs[1].set(xlabel=r'$J/2\pi$ (MHz)',ylabel=r'$\Delta_{\cal L}$ (s$^{-1}$)',title='(b) Isolated target limit gives gap closing'); axs[1].legend(fontsize=8)
save(fig,'fig7_target_gap'); (res/'target_gap_fit.json').write_text(json.dumps({'gap_at_J0_s-1':float(gaps[0]),'gap0_fit_s-1':g0,'quadratic_coefficient_s':c},indent=2))
