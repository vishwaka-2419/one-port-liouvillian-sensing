#!/usr/bin/env python3
from pathlib import Path
import json, argparse
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, RegularPolygon, Arc
from matplotlib.ticker import AutoMinorLocator

HERE = Path(__file__).resolve().parents[1]
OUT = HERE/'figures'
PKG = HERE
DATA = PKG/'data_processed'
RES = PKG/'results'
RAW = None

NAVY='#18364F'; TEAL='#0C8F8B'; CYAN='#55B8C7'; ORANGE='#DD7438'; RED='#B23C49'; PURPLE='#7151A4'
GREEN='#458A61'; GOLD='#C49936'; DARK='#22262A'; GREY='#67737D'; LIGHT='#EDF2F5'; MID='#AAB6BE'; PALE='#F7F8F9'

mpl.rcParams.update({
    'font.family':'serif',
    'font.serif':['Latin Modern Roman','CMU Serif','Computer Modern Roman','DejaVu Serif'],
    'mathtext.fontset':'stix',
    'font.size':8.2,
    'axes.labelsize':8.3,
    'legend.fontsize':7.6,
    'xtick.labelsize':7.3,
    'ytick.labelsize':7.3,
    'axes.linewidth':0.85,
    'xtick.direction':'in','ytick.direction':'in',
    'xtick.major.size':4.0,'ytick.major.size':4.0,
    'xtick.minor.size':2.3,'ytick.minor.size':2.3,
    'xtick.major.width':0.75,'ytick.major.width':0.75,
    'xtick.minor.width':0.6,'ytick.minor.width':0.6,
    'pdf.fonttype':42,'ps.fonttype':42,
    'savefig.facecolor':'white'
})

def style(ax, xminor=True, yminor=True):
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.85); s.set_color('#30363B')
    ax.tick_params(which='both', direction='in', top=True, right=True)
    if xminor: ax.xaxis.set_minor_locator(AutoMinorLocator())
    if yminor: ax.yaxis.set_minor_locator(AutoMinorLocator())
    return ax

def panel(ax, letter, x=-0.13, y=1.04):
    ax.text(x,y,letter,transform=ax.transAxes,ha='left',va='bottom',fontsize=9.6,fontweight='bold')

def save(fig,name):
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',pad_inches=0.025)
    fig.savefig(OUT/f'{name}.png',dpi=350,bbox_inches='tight',pad_inches=0.025)
    plt.close(fig)

# ---------------- Figure 1 ----------------
def fig1():
    fig,ax=plt.subplots(figsize=(7.25,3.1))
    ax.set_xlim(0,10); ax.set_ylim(0,4.2); ax.axis('off')
    xs=[1.05,3.45,5.85,8.35]; y=2.72
    cols=[NAVY,TEAL,PURPLE,ORANGE]
    labels=[
        ('ESR-STM','molecular spin\nRF current readout'),
        ('Torque separation','coherent FLT\ndissipative DLT'),
        ('Floquet spectrum','EP coalescence\nrelaxation gap'),
        ('Measured information','counting statistics\nFisher information'),
    ]
    ax.plot([.55,8.85],[y,y],color='#D8E0E5',lw=4.2,solid_capstyle='round',zorder=0)
    for i in range(3):
        ax.add_patch(FancyArrowPatch((xs[i]+.33,y),(xs[i+1]-.34,y),arrowstyle='-|>',mutation_scale=12,
                                     lw=1.15,color='#8B979F',zorder=2))
    for i,(x,c) in enumerate(zip(xs,cols)):
        ax.add_patch(Circle((x,y),.44,facecolor='white',edgecolor=c,lw=1.8,zorder=3))
        if i==0:
            ax.text(x,y+.02,'STM',ha='center',va='center',fontsize=8.2,color=NAVY)
            ax.add_patch(FancyArrowPatch((x,y+.22),(x,y+.08),arrowstyle='-|>',mutation_scale=8,lw=.9,color=TEAL))
        elif i==1:
            ax.add_patch(FancyArrowPatch((x-.20,y-.08),(x+.20,y+.14),connectionstyle='arc3,rad=.32',arrowstyle='-|>',mutation_scale=10,lw=1.2,color=TEAL))
            ax.add_patch(FancyArrowPatch((x+.20,y-.08),(x-.20,y+.14),connectionstyle='arc3,rad=-.32',arrowstyle='-|>',mutation_scale=10,lw=1.2,color=ORANGE))
            ax.text(x,y-.01,r'$H$ / $\mathcal{D}$',ha='center',va='center',fontsize=7.0)
        elif i==2:
            t=np.linspace(0,2*np.pi,80)
            ax.plot(x+.18*np.cos(t),y+.10*np.sin(t),color=PURPLE,lw=.9)
            ax.scatter([x-.13,x+.13],[y,y],s=16,c=[PURPLE,ORANGE],zorder=4)
            ax.add_patch(FancyArrowPatch((x-.09,y),(x+.09,y),arrowstyle='->',mutation_scale=8,lw=.8,color=GREY))
        else:
            ax.text(x,y+.04,r'$I,\;\mathcal{F}$',ha='center',va='center',fontsize=8.2,color=ORANGE)
        title,sub=labels[i]
        ax.text(x,y-.64,title,ha='center',va='top',fontsize=7.5,fontweight='bold',color=c)
        ax.text(x,y-.90,sub,ha='center',va='top',fontsize=7.0,linespacing=1.22,color=DARK)
    tx,ty=8.35,3.65
    ax.add_patch(Circle((tx,ty),.18,fc='white',ec=GREEN,lw=1.4))
    ax.arrow(tx,ty-.02,0,.13,width=.008,head_width=.07,head_length=.07,color=GREEN,length_includes_head=True)
    ax.add_patch(FancyArrowPatch((tx,ty-.20),(8.35,3.20),arrowstyle='-|>',mutation_scale=9,lw=1.0,color=GREEN))
    ax.text(tx+.30,ty,r'weak target $J$',ha='left',va='center',fontsize=7.0,color=GREEN)
    ry=.47
    ax.text(.58,.92,'experiment-constrained route',fontsize=7.4,fontweight='bold',color=NAVY,ha='left')
    rx=np.linspace(.85,9.15,5)
    rlab=['public transients',r'$T_1,\;T_{2,\mathrm{Rabi}},\;\omega_d$',r'$\Omega_F/D$',r'$T_{\rm crit}$','target-spin gap']
    ax.plot([rx[0],rx[-1]],[ry,ry],color='#B5C0C7',lw=1.05,zorder=0)
    for i,(xx,ll) in enumerate(zip(rx,rlab)):
        ax.scatter([xx],[ry],s=34,c=[TEAL if i<4 else GREEN],edgecolor='white',lw=.8,zorder=3)
        ax.text(xx,ry-.18,ll,ha='center',va='top',fontsize=6.75,color=DARK)
    save(fig,'fig1_story')

# ---------------- Figure 2 ----------------
def fig2():
    d=pd.read_csv(DATA/'experimental_liouvillian_reconstruction.csv').sort_values('bias_mV')
    mc=pd.read_csv(DATA/'experimental_uncertainties_mc.csv').sort_values('bias_mV')
    summary=json.load(open(DATA/'experimental_summary.json'))
    fig,axs=plt.subplots(2,2,figsize=(7.25,5.0),constrained_layout=True)
    # a
    ax=style(axs[0,0]); panel(ax,'a')
    raw=pd.read_csv(RAW/'SM/S16/dI(Delay).csv')
    t=raw['Delay (ns)'].to_numpy(); yy=raw['dI (sig) (fA)'].to_numpy(); ss=raw['dI_err (sig) (fA)'].to_numpy(); T=summary['T1_ns']
    X=np.c_[np.ones_like(t),np.exp(-t/T)]; b=np.linalg.lstsq(X/ss[:,None],yy/ss,rcond=None)[0]; tt=np.linspace(t.min(),t.max(),500)
    ax.errorbar(t,yy,yerr=ss,fmt='o',ms=2.6,mfc='white',mec=NAVY,ecolor='#AAB6BE',elinewidth=.65,capsize=1.1,label='data')
    ax.plot(tt,b[0]+b[1]*np.exp(-tt/T),color=RED,lw=1.55,label='fit')
    ax.set(xlabel=r'$\tau$ (ns)',ylabel=r'$\Delta I$ (fA)')
    ax.legend(frameon=False,loc='best',handlelength=1.7,fontsize=8.0)
    ax.text(.96,.94,fr'$T_1={summary["T1_ns"]:.1f}\pm{summary["T1_err_ns"]:.1f}$ ns'+'\n'+fr'$\chi^2_\nu={summary["T1_reduced_chi2"]:.2f}$',transform=ax.transAxes,ha='right',va='top',fontsize=7.1)
    # b
    ax=style(axs[0,1]); panel(ax,'b'); x=d.bias_mV.to_numpy()
    ax.errorbar(x,d.f_damped_MHz,yerr=d.f_damped_err_MHz,fmt='o',ms=3.9,mfc='white',mec=NAVY,ecolor='#AAB6BE',label=r'$\omega_d/2\pi$')
    ax.plot(x,d.f_coherent_MHz,'o-',color=TEAL,lw=1.5,ms=3.2,label=r'$\Omega_F/2\pi$')
    ax.plot(x,d.EP_halfdifference_Mrad_s/(2*np.pi),'--',color=RED,lw=1.4,label=r'$D/2\pi$')
    ax.set(xlabel=r'$U_{\rm DC}$ (mV)',ylabel=r'$f$ (MHz)')
    ax.legend(frameon=False,loc='upper left',handlelength=1.8,fontsize=8.0)
    # c
    ax=style(axs[1,0]); panel(ax,'c')
    ax.fill_between(mc.bias_mV,mc.ratio_p16,mc.ratio_p84,color=TEAL,alpha=.13,lw=0)
    ax.plot(mc.bias_mV,mc.ratio_med,'o-',color=TEAL,lw=1.55,ms=3.8)
    ax.axhline(1,color=RED,ls='--',lw=1.15)
    imin=np.argmin(mc.ratio_med.to_numpy()); bx=float(mc.bias_mV.iloc[imin]); by=float(mc.ratio_med.iloc[imin])
    ax.scatter([bx],[by],s=38,color=ORANGE,edgecolor='white',lw=.6,zorder=4)
    ax.annotate(fr'closest: ${by:.2f}$',xy=(bx,by),xytext=(bx+78,by+1.75),ha='left',va='center',fontsize=7.6,
                arrowprops=dict(arrowstyle='->',lw=.75,color=GREY,shrinkA=4,shrinkB=6))
    ax.set(xlabel=r'$U_{\rm DC}$ (mV)',ylabel=r'$\Omega_F/D$'); ax.set_ylim(.78,max(mc.ratio_p84)*1.07)
    # d
    ax=style(axs[1,1]); panel(ax,'d')
    ax.errorbar(x,d.T2Rabi_ns,yerr=d.T2Rabi_err_ns,fmt='o-',color=NAVY,mfc='white',mec=NAVY,lw=1.35,ms=3.6,label=r'$T_{2,\rm Rabi}$')
    ax.fill_between(mc.bias_mV,mc.Tcrit_p16_ns,mc.Tcrit_p84_ns,color=ORANGE,alpha=.13,lw=0)
    ax.plot(mc.bias_mV,mc.Tcrit_med_ns,'s--',color=ORANGE,lw=1.4,ms=3.2,label=r'$T_{\rm crit}$')
    ax.set(xlabel=r'$U_{\rm DC}$ (mV)',ylabel=r'$T$ (ns)')
    ax.legend(frameon=False,loc='upper left',handlelength=1.8,fontsize=8.0)
    save(fig,'fig2_experiment')

# ---------------- Figure 3 ----------------
def fig3():
    G1,G2=1.0,3.5; D=abs(G1-G2)/2; A=(G1+G2)/2
    Om=np.linspace(0,2.2*D,700); root=np.sqrt((D**2-Om**2)+0j); lp=-A+root; lm=-A-root
    fig,axs=plt.subplots(2,2,figsize=(7.25,5.0),constrained_layout=True)
    ax=style(axs[0,0]); panel(ax,'a')
    ax.plot(Om/D,lp.real/D,color=TEAL,lw=1.55,label=r'$\mathrm{Re}\,\lambda_+/D$')
    ax.plot(Om/D,lm.real/D,color=NAVY,lw=1.55,label=r'$\mathrm{Re}\,\lambda_-/D$')
    ax.plot(Om/D,np.abs(lp.imag)/D,color=ORANGE,lw=1.55,label=r'$|\mathrm{Im}\,\lambda_\pm|/D$')
    ax.axvline(1,color=RED,ls='--',lw=1.1); ax.text(1.035,-.40,'EP',color=RED,fontsize=7.6)
    ax.set(xlabel=r'$\Omega_F/D$',ylabel=r'$\lambda/D$')
    ax.legend(frameon=False,ncol=2,loc='lower right',columnspacing=.9,handlelength=1.5,fontsize=8.2)
    ax=style(axs[0,1],xminor=False,yminor=False); panel(ax,'b')
    eps=np.logspace(-5,-.15,180); split=2*np.sqrt(2*D**2*eps+eps**2*D**2)
    ax.loglog(eps,split/D,color=NAVY,lw=1.55,label='exact')
    ax.loglog(eps,2*np.sqrt(2)*np.sqrt(eps),'--',color=ORANGE,lw=1.3,label=r'$\propto\sqrt{\epsilon}$')
    ax.tick_params(which='both',direction='in',top=True,right=True)
    ax.set(xlabel=r'$\epsilon$',ylabel=r'$|\Delta\lambda|/D$')
    ax.legend(frameon=False,loc='upper left',fontsize=8.2)
    ax=style(axs[1,0]); panel(ax,'c')
    t=np.linspace(0,4/A,400); yj=(1+A*t)*np.exp(-A*t); yu=np.exp(-A*t)*np.cos(1.2*A*t); yo=.5*np.exp(-.55*A*t)+.5*np.exp(-1.55*A*t)
    ax.plot(t*A,yu,color=TEAL,lw=1.4,label='underdamped'); ax.plot(t*A,yj,color=RED,lw=1.6,label='EP'); ax.plot(t*A,yo,color=NAVY,lw=1.4,label='overdamped')
    ax.set(xlabel=r'$At$',ylabel=r'$z(t)/z(0)$'); ax.legend(frameon=False,loc='upper right',fontsize=7.6)
    ax=style(axs[1,1]); panel(ax,'d')
    ax.plot(lp.real/D,lp.imag/D,color=TEAL,lw=1.45); ax.plot(lm.real/D,lm.imag/D,color=NAVY,lw=1.45)
    idx=np.argmin(abs(Om-D)); ax.scatter([lp.real[idx]/D],[lp.imag[idx]/D],s=42,color=RED,edgecolor='white',zorder=4)
    ax.annotate('coalescence',xy=(lp.real[idx]/D,lp.imag[idx]/D),xytext=(-3.18,.98),fontsize=8.0,
                arrowprops=dict(arrowstyle='->',lw=.75,color=GREY),ha='left',va='center')
    ax.axhline(0,color='#D6DDE2',lw=.8); ax.set(xlabel=r'$\mathrm{Re}\,\lambda/D$',ylabel=r'$\mathrm{Im}\,\lambda/D$')
    ax.set_aspect('equal',adjustable='datalim')
    save(fig,'fig3_ep_anatomy')

# ---------------- Figure 4 ----------------
def fig4():
    ep=pd.read_csv(RES/'floquet_ep_scan.csv'); conv=pd.read_csv(RES/'floquet_convergence.csv'); sens=pd.read_csv(RES/'transport_sensitivity.csv')
    fig,axs=plt.subplots(2,2,figsize=(7.25,5.0),constrained_layout=True)
    ax=style(axs[0,0]); panel(ax,'a'); ax.plot(ep.aD,ep.fEP_MHz,'o-',color=TEAL,lw=1.55,ms=3.7)
    ax.set(xlabel=r'$a_D$',ylabel=r'$f_{\rm EP}$ (MHz)')
    ax=style(axs[0,1]); panel(ax,'b'); l1=ax.plot(ep.aD,ep.gap_Mrad_s,'s-',color=ORANGE,lw=1.5,ms=3.5,label=r'$\Delta_{\mathcal{L}}$')
    ax2=ax.twinx(); l2=ax2.plot(ep.aD,ep.current_pA,'o--',color=NAVY,lw=1.25,ms=3.3,label=r'$I$')
    for s in ax2.spines.values(): s.set_visible(True); s.set_linewidth(.85); s.set_color('#30363B')
    ax2.tick_params(which='both',direction='in',top=True,right=True,colors=NAVY); ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set(xlabel=r'$a_D$',ylabel=r'$\Delta_{\mathcal{L}}$ ($10^6$ s$^{-1}$)'); ax2.set_ylabel(r'$I$ (pA)',color=NAVY)
    lines=l1+l2; ax.legend(lines,[l.get_label() for l in lines],frameon=False,loc='upper left',fontsize=8.2)
    ax=style(axs[1,0]); panel(ax,'c'); ax.plot(conv.steps_per_period,conv.fEP_MHz,'o-',color=PURPLE,lw=1.55,ms=3.6)
    ref=float(ep.loc[np.isclose(ep.aD,.6),'fEP_MHz'].iloc[0]); ax.axhline(ref,color='#AFB9C0',ls='--',lw=.9)
    ax.set(xlabel=r'$N_t$',ylabel=r'$f_{\rm EP}$ (MHz)')
    axin=ax.inset_axes([.56,.58,.37,.30])
    axin.semilogy(conv.steps_per_period,np.maximum(1-conv.overlap,1e-16),'o-',color=RED,ms=2.6,lw=1.0)
    for s in axin.spines.values(): s.set_visible(True); s.set_linewidth(.65); s.set_color('#555E64')
    axin.tick_params(which='both',direction='in',top=True,right=True,labelsize=5.5,length=2)
    axin.set_xlabel(r'$N_t$',fontsize=5.8,labelpad=0); axin.set_ylabel(r'$1-|\langle R_1|R_2\rangle|$',fontsize=5.8,labelpad=1)
    axin.patch.set_facecolor('white'); axin.patch.set_alpha(.96)
    ax=style(axs[1,1]); panel(ax,'d')
    sc=ax.scatter(sens.theta_deg,sens.fEP_MHz,c=sens.pL,s=34,cmap='viridis',edgecolor='white',linewidth=.4)
    for _,grp in sens.groupby('pL'): ax.plot(grp.theta_deg,grp.fEP_MHz,color='#C6CDD2',lw=.6,zorder=0)
    ax.set(xlabel=r'$\theta$ (deg)',ylabel=r'$f_{\rm EP}$ (MHz)')
    cax=ax.inset_axes([.90,.15,.035,.72])
    cb=fig.colorbar(sc,cax=cax)
    cb.set_label(r'$p_L$',fontsize=7.2,labelpad=3)
    cb.ax.tick_params(labelsize=6.5,direction='in',length=2.2,pad=1.5)
    cb.outline.set_linewidth(.65)
    save(fig,'fig4_floquet_transport')

# ---------------- Figure 5 ----------------
def fig5():
    m=pd.read_csv(RES/'metrology_scan.csv'); ep=pd.read_csv(RES/'floquet_ep_scan.csv'); fep=float(ep.loc[np.isclose(ep.aD,.6),'fEP_MHz'].iloc[0])
    fig,axs=plt.subplots(1,3,figsize=(7.25,2.55),constrained_layout=True)
    ax=style(axs[0]); panel(ax,'a',x=-.18)
    l1=ax.plot(m.f_MHz,m.pair_sep_Mrad_s,color=NAVY,lw=1.5,label=r'$|\Delta\mu|$')
    ax2=ax.twinx(); l2=ax2.plot(m.f_MHz,m.overlap,color=RED,lw=1.35,label=r'$\mathcal{O}_R$')
    for s in ax2.spines.values(): s.set_visible(True); s.set_linewidth(.85); s.set_color('#30363B')
    ax2.tick_params(which='both',direction='in',top=True,right=True,colors=RED); ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax.axvline(fep,color=ORANGE,ls='--',lw=1.05)
    ax.set(xlabel=r'$\Omega_F/2\pi$ (MHz)',ylabel=r'$|\Delta\mu|$ ($10^6$ s$^{-1}$)'); ax2.set_ylabel(r'$\mathcal{O}_R$',color=RED)
    ax.legend(l1+l2,[x.get_label() for x in l1+l2],frameon=False,loc='lower left',handlelength=1.3,fontsize=8.2)
    ax=style(axs[1]); panel(ax,'b',x=-.18)
    y=m.counting_FI_per_rad2s.to_numpy(); q=m.state_QFI_per_rad2s.to_numpy();
    ax.plot(m.f_MHz,y/y.max(),color=TEAL,lw=1.55,label=r'$\dot{\mathcal{F}}_{\rm count}$')
    ax.plot(m.f_MHz,q/q.max(),'--',color=PURPLE,lw=1.35,label=r'$\mathcal{F}_Q$')
    ax.axvline(fep,color=ORANGE,ls='--',lw=1.05,label='EP'); ax.set(xlabel=r'$\Omega_F/2\pi$ (MHz)',ylabel=r'$\mathcal{F}/\mathcal{F}_{\max}$'); ax.legend(frameon=False,loc='lower right',handlelength=1.4,fontsize=8.2)
    ax=style(axs[2]); panel(ax,'c',x=-.18)
    l1=ax.plot(m.f_MHz,m.current_pA,color=NAVY,lw=1.5,label=r'$I$'); ax2=ax.twinx(); l2=ax2.plot(m.f_MHz,m.singly_occ,color=GREEN,lw=1.35,label=r'$P_1$')
    for s in ax2.spines.values(): s.set_visible(True); s.set_linewidth(.85); s.set_color('#30363B')
    ax2.tick_params(which='both',direction='in',top=True,right=True,colors=GREEN); ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax.axvline(fep,color=ORANGE,ls='--',lw=1.05); ax.set(xlabel=r'$\Omega_F/2\pi$ (MHz)',ylabel=r'$I$ (pA)'); ax2.set_ylabel(r'$P_1$',color=GREEN)
    ax.legend(l1+l2,[x.get_label() for x in l1+l2],frameon=False,loc='center left',bbox_to_anchor=(0.02,0.72),handlelength=1.3,fontsize=8.2)
    save(fig,'fig5_metrology')

# ---------------- Figure 6 ----------------
def fig6():
    d=pd.read_csv(RES/'target_gap_scan.csv'); fit=json.load(open(RES/'target_gap_fit.json')); J=d.J_over_2pi_MHz.to_numpy(); gap=d['gap_s-1'].to_numpy()
    fig,axs=plt.subplots(1,3,figsize=(7.25,2.62),constrained_layout=True)
    ax=axs[0]; ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off'); panel(ax,'a',x=-.12)
    ax.add_patch(Polygon([[.04,.13],[.96,.13],[.90,.18],[.10,.18]],closed=True,fc='#EFF2F4',ec='#93A0A8',lw=1.0))
    ax.text(.52,.06,'MgO/Ag',ha='center',fontsize=6.8,color=GREY)
    # flat-lying pentacene as fused aromatic units on surface
    centers=np.linspace(.28,.56,5)
    for cx in centers:
        ax.add_patch(RegularPolygon((cx,.29),6,radius=.050,orientation=np.pi/6,facecolor='white',edgecolor=TEAL,lw=1.0))
    ax.text(.42,.20,'pentacene sensor',ha='center',fontsize=6.8,color=TEAL)
    # beautiful STM tip with shaft + apex
    ax.add_patch(Polygon([[.18,.93],[.52,.93],[.47,.80],[.44,.66],[.41,.58],[.29,.58],[.26,.66],[.23,.80]],closed=True,fc=LIGHT,ec=NAVY,lw=1.15))
    ax.add_patch(Polygon([[.34,.58],[.36,.52],[.35,.46],[.33,.52]],closed=True,fc='white',ec=NAVY,lw=.9))
    ax.arrow(.345,.46,0,-.08,width=.004,head_width=.03,head_length=.03,color=NAVY,length_includes_head=True)
    ax.text(.12,.82,'spin-polarized tip',ha='left',va='center',fontsize=6.9,color=NAVY)
    ax.add_patch(Circle((.42,.38),.030,fc='white',ec=TEAL,lw=1.1)); ax.arrow(.42,.38,0,.10,width=.004,head_width=.028,head_length=.028,color=TEAL,length_includes_head=True)
    ax.add_patch(Circle((.80,.37),.055,fc='#F2F7F2',ec=GREEN,lw=1.3)); ax.arrow(.80,.37,.022,.11,width=.004,head_width=.032,head_length=.032,color=GREEN,length_includes_head=True)
    ax.text(.80,.20,'target spin',ha='center',fontsize=6.8,color=GREEN)
    ax.add_patch(Arc((.61,.38),.32,.22,theta1=12,theta2=168,color=PURPLE,lw=1.1,ls='--'))
    ax.text(.62,.54,r'$J$',ha='center',fontsize=8,color=PURPLE)
    ax=style(axs[1],xminor=False,yminor=False); panel(ax,'b',x=-.18)
    pred=fit['gap0_fit_s-1']+fit['quadratic_coefficient_s']*(2*np.pi*J*1e6)**2
    ax.loglog(J[1:],gap[1:],color=TEAL,lw=1.55,label=r'$\Delta_{\mathcal{L}}$'); ax.loglog(J[1:],pred[1:],'--',color=ORANGE,lw=1.3,label=r'$\Delta_0+CJ^2$'); ax.axhline(fit['gap0_fit_s-1'],color=NAVY,ls=':',lw=1.05,label=r'$\Delta_0$')
    ax.tick_params(which='both',direction='in',top=True,right=True); ax.set(xlabel=r'$J/2\pi$ (MHz)',ylabel=r'$\Delta_{\mathcal{L}}$ (s$^{-1}$)'); ax.legend(frameon=False,loc='lower right',handlelength=1.4,fontsize=7.8)
    ax=style(axs[2],xminor=False,yminor=False); panel(ax,'c',x=-.18)
    rel=(gap-fit['gap0_fit_s-1'])/fit['gap0_fit_s-1']; ax.loglog(J[1:],np.maximum(rel[1:],1e-12),color=PURPLE,lw=1.55); ax.axhline(1,color='#ADB7BE',ls='--',lw=.9)
    ax.tick_params(which='both',direction='in',top=True,right=True); ax.set(xlabel=r'$J/2\pi$ (MHz)',ylabel=r'$(\Delta-\Delta_0)/\Delta_0$')
    save(fig,'fig6_target_gap')

# ---------------- S1: combined comparison ----------------
T1_NS=138.23; G1=1/T1_NS

def kov_model(t,c,A,T,f):
    return c+A*np.exp(-t/T)*np.cos(2*np.pi*f*1e-3*t)

def liouv_model(t,c,A,G2,Om_MHz):
    Om=2*np.pi*Om_MHz*1e-3; D=.5*(G1-G2); wd2=Om*Om-D*D
    wd=np.sqrt(np.maximum(wd2,1e-12)); aa=.5*(G1+G2); q=(G2-G1)/(2*wd)
    return c+A*np.exp(-aa*t)*(np.cos(wd*t)+q*np.sin(wd*t))

def fit_stats():
    raw=pd.read_csv(RAW/'SM/S21/AllSpectra.csv'); fp=pd.read_csv(RAW/'F4/E-F/FittedParameters.csv')
    biases=sorted([int(c.split('@')[1].split(' mV')[0]) for c in raw.columns if c.startswith('Pulse duration')])
    rows=[]; curves={}
    for V in biases:
        t=raw[f'Pulse duration (ns)@{V} mV'].to_numpy(float); y=raw[f'dI (fA)@{V} mV'].to_numpy(float)
        r=fp.loc[np.isclose(fp['U_DC (mV)'],V)]
        if r.empty: continue
        r=r.iloc[0]; T0=float(r['T_2 (ns)']); f0=float(r['Omega/2pi (MHz)'])
        p0=[np.mean(y[-10:]),y[0]-np.mean(y[-10:]),T0,f0]
        pk,_=curve_fit(kov_model,t,y,p0=p0,bounds=([-1500,-3000,1,.1],[1500,3000,500,500]),maxfev=200000)
        G20=max(1e-5,2/T0-G1); Om0=np.sqrt((2*np.pi*f0*1e-3)**2+(.5*(G1-G20))**2)/(2*np.pi*1e-3)
        pl0=[np.mean(y[-10:]),y[0]-np.mean(y[-10:]),G20,Om0]
        pl,_=curve_fit(liouv_model,t,y,p0=pl0,bounds=([-1500,-3000,1e-5,.1],[1500,3000,2,500]),maxfev=200000)
        yhK=kov_model(t,*pk); yhL=liouv_model(t,*pl); n=len(y); k=4
        def stat(pred):
            rss=float(np.sum((y-pred)**2)); rm=np.sqrt(rss/n); aic=n*np.log(rss/n)+2*k; aicc=aic+2*k*(k+1)/(n-k-1); bic=n*np.log(rss/n)+k*np.log(n); return rm,aicc,bic
        sk=stat(yhK); sl=stat(yhL)
        rows.append(dict(bias_mV=V,rmse_published=sk[0],rmse_liouv=sl[0],aicc_published=sk[1],aicc_liouv=sl[1],delta_aicc=sl[1]-sk[1],bic_published=sk[2],bic_liouv=sl[2]))
        tt=np.linspace(t.min(),t.max(),700); curves[V]=(t,y,kov_model(tt,*pk),liouv_model(tt,*pl),tt)
    return pd.DataFrame(rows),curves

def figS1():
    stats,curves=fit_stats(); stats.to_csv(HERE/'fit_comparison_S1.csv',index=False)
    chosen=[-60,-100,-140,-180,-220,-260]
    fig,axs=plt.subplots(2,1,figsize=(7.25,5.7),constrained_layout=True,gridspec_kw={'height_ratios':[2.2,1.0]})
    ax=axs[0]; style(ax); panel(ax,'a',x=-.06)
    offsets=np.arange(len(chosen))[::-1]*380.0
    for off,V in zip(offsets,chosen):
        t,y,kfit,lfit,tt=curves[V]
        y0=y-np.mean(y[-12:])
        scale=1.0
        ax.plot(t,y0*scale+off,'o',ms=1.9,mfc='#CBD2D7',mec='#89959E',mew=.35)
        ax.plot(tt,kfit-np.mean(kfit[-20:])+off,'--',color=NAVY,lw=1.15)
        ax.plot(tt,lfit-np.mean(lfit[-20:])+off,color=ORANGE,lw=1.25)
        ax.text(tt.max()*0.98,off+20,fr'$U_{{\rm DC}}={V}$ mV',ha='right',va='bottom',fontsize=6.9,color=DARK)
    ax.plot([],[],'o',ms=3.0,mfc='#CBD2D7',mec='#89959E',mew=.35,label='data')
    ax.plot([],[],'--',color=NAVY,lw=1.2,label='published fit')
    ax.plot([],[],color=ORANGE,lw=1.3,label='Liouvillian fit')
    ax.set(xlabel=r'$t_p$ (ns)',ylabel=r'offset $\Delta I$ (fA)')
    ax.legend(frameon=False,loc='upper right',fontsize=7.8,handlelength=1.7)
    ax=axs[1]; style(ax); panel(ax,'b',x=-.06)
    ax.axhline(0,color='#8F999F',lw=.85)
    pos=stats.delta_aicc<0
    ax.plot(stats.bias_mV,stats.delta_aicc,'o-',color=PURPLE,lw=1.2,ms=3.2)
    ax.scatter(stats.loc[pos,'bias_mV'],stats.loc[pos,'delta_aicc'],s=28,color=ORANGE,zorder=3,label='Liouvillian preferred')
    ax.scatter(stats.loc[~pos,'bias_mV'],stats.loc[~pos,'delta_aicc'],s=28,facecolor='white',edgecolor=NAVY,lw=.8,zorder=3,label='published fit preferred')
    ax.set(xlabel=r'$U_{\rm DC}$ (mV)',ylabel=r'$\Delta\mathrm{AIC}_c$')
    ax.legend(frameon=False,ncol=2,loc='upper left',handletextpad=.4,columnspacing=1.1,fontsize=7.8)
    save(fig,'figS1_rabi_overlays')

# ---------------- S2 ----------------
def figS2():
    conv=pd.read_csv(RES/'floquet_convergence.csv'); ep=pd.read_csv(RES/'floquet_ep_scan.csv')
    fig,axs=plt.subplots(1,3,figsize=(7.25,2.55),constrained_layout=True)
    ax=style(axs[0]); panel(ax,'a',x=-.18); ax.plot(conv.steps_per_period,conv.sep_Mrad_s,'o-',color=NAVY,lw=1.45,ms=3.4); ax.set(xlabel=r'$N_t$',ylabel=r'$|\Delta\mu|$ ($10^6$ s$^{-1}$)')
    ax=style(axs[1],xminor=False,yminor=False); panel(ax,'b',x=-.18); ax.semilogy(conv.steps_per_period,np.maximum(1-conv.overlap,1e-16),'o-',color=RED,lw=1.45,ms=3.4); ax.tick_params(which='both',direction='in',top=True,right=True); ax.set(xlabel=r'$N_t$',ylabel=r'$1-|\langle R_1|R_2\rangle|$')
    ax=style(axs[2]); panel(ax,'c',x=-.18); ax.plot(ep.aD,ep.mu1_real_Mrad_s,'o-',color=TEAL,lw=1.4,ms=3.3,label=r'$\mathrm{Re}\,\mu_1$'); ax.plot(ep.aD,ep.mu2_real_Mrad_s,'s--',color=ORANGE,lw=1.25,ms=3.1,label=r'$\mathrm{Re}\,\mu_2$'); ax.set(xlabel=r'$a_D$',ylabel=r'$\mathrm{Re}\,\mu$ ($10^6$ s$^{-1}$)'); ax.legend(frameon=False,loc='best',handlelength=1.4,fontsize=7.8)
    save(fig,'figS2_numerics')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); a=ap.parse_args()
    RAW=Path(a.data_root)
    OUT.mkdir(parents=True,exist_ok=True)
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); figS1(); figS2()
    print('Wrote',OUT)
