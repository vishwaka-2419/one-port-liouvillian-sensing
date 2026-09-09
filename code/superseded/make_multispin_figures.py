#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.ticker import AutoMinorLocator
from multispin_chain import (slow_gap_and_weight,slow_gap_asymptotic,slow_weight_asymptotic,
    boundary_coefficient,propagate_single_excitation,propagation_length_readout,
    intrinsic_floor_length,mode_resolvability_length,validation_scan)

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'figures'; RES=ROOT/'results'; FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
NAVY='#18364F'; TEAL='#0C8F8B'; ORANGE='#DD7438'; RED='#B23C49'; PURPLE='#7151A4'; GREEN='#458A61'; GREY='#67737D'; MID='#AAB6BE'; DARK='#22262A'
mpl.rcParams.update({'font.family':'serif','font.serif':['Latin Modern Roman','CMU Serif','DejaVu Serif'],'mathtext.fontset':'stix',
                     'font.size':8.2,'axes.labelsize':8.4,'legend.fontsize':7.6,'xtick.labelsize':7.3,'ytick.labelsize':7.3,
                     'axes.linewidth':.85,'xtick.direction':'in','ytick.direction':'in','pdf.fonttype':42,'ps.fonttype':42})

def style(ax,minor=True):
    for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(.85); s.set_color('#30363B')
    ax.tick_params(which='both',direction='in',top=True,right=True)
    if minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())

def panel(ax,l,x=-.13,y=1.04): ax.text(x,y,l,transform=ax.transAxes,fontweight='bold',fontsize=9.5,va='bottom')
def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',pad_inches=.025)
    fig.savefig(FIG/(name+'.png'),dpi=350,bbox_inches='tight',pad_inches=.025)
    plt.close(fig)

def figure7():
    J=1.; gs=.25
    Ns=np.unique(np.round(np.logspace(np.log10(2),2,80)).astype(int))
    fig,axs=plt.subplots(2,2,figsize=(7.25,5.0),constrained_layout=True)
    # a schematic
    ax=axs[0,0]; ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off'); panel(ax,'a',x=-.10)
    xs=np.linspace(.18,.86,7); y=.50
    ax.plot(xs,y*np.ones_like(xs),color=PURPLE,lw=1.5,zorder=0)
    for k,x in enumerate(xs):
        col=ORANGE if k==0 else TEAL
        ax.add_patch(Circle((x,y),.045,fc='white',ec=col,lw=1.4))
        ax.arrow(x,y,0,.09,width=.004,head_width=.025,head_length=.025,color=col,length_includes_head=True)
    ax.add_patch(FancyArrowPatch((xs[0]-.03,.84),(xs[0]-.03,.60),arrowstyle='-|>',mutation_scale=12,lw=1.2,color=ORANGE))
    ax.text(xs[0]-.12,.88,r'$\Omega_F$',color=ORANGE,fontsize=8.5,fontweight='bold')
    ax.add_patch(FancyArrowPatch((xs[0]-.07,.25),(xs[0]-.07,.43),arrowstyle='<|-|>',mutation_scale=8,lw=1.0,color=NAVY))
    ax.text(xs[0]-.13,.18,r'$I$',color=ORANGE,fontsize=8.6,fontweight='bold')
    ax.text(.53,.57,r'$J$',color=PURPLE,fontsize=8.0)
    ax.text(.56,.30,r'$\gamma_t$',color=TEAL,fontsize=8.5,fontweight='bold')
    ax.text(xs[0]+.01,.12,r'$\gamma_s$',color=NAVY,fontsize=8.5,fontweight='bold')
    # b gap scaling
    ax=axs[0,1]; style(ax,minor=False); panel(ax,'b')
    rows=[]
    for gt,c in [(0,NAVY),(.001,TEAL),(.01,ORANGE),(.03,RED)]:
        vals=[]
        for N in Ns:
            gap,w,_=slow_gap_and_weight(N,J,gs,gt); vals.append(gap); rows.append((N,gt,gap,w))
        ax.loglog(Ns,vals,color=c,lw=1.45,label=fr'$\gamma_t/J={gt:g}$')
        if gt>0: ax.axhline(gt,color=c,ls=':',lw=.8,alpha=.65)
    nref=np.array([15,100]); cref=(slow_gap_asymptotic(30,J,gs,0))*31**3
    ax.loglog(nref,cref/(nref+1)**3,'--',color=GREY,lw=1.15,label=r'$N^{-3}$')
    ax.set(xlabel='chain length $N$',ylabel=r'$\Delta_{\rm slow}/J$'); ax.legend(frameon=False,loc='lower left',fontsize=7.4)
    # c exact identity + asymptotic weight
    ax=axs[1,0]; style(ax,minor=False); panel(ax,'c')
    gt=0; weights=[]; excess=[]
    for N in Ns:
        gap,w,_=slow_gap_and_weight(N,J,gs,gt); weights.append(w); excess.append((gap-gt)/(gs-gt))
    ax.loglog(Ns,weights,color=TEAL,lw=1.45,label=r'$w_1=|R_1(1)|^2$')
    ax.loglog(Ns,[slow_weight_asymptotic(N,J,gs-gt) for N in Ns],'--',color=ORANGE,lw=1.2,label=r'asymptotic $N^{-3}$')
    ax.set(xlabel='chain length $N$',ylabel='sensor weight $w_1$'); ax.legend(frameon=False,loc='lower left')
    ax.text(.98,.93,r'$\Delta_{\rm slow}-\gamma_t=(\gamma_s-\gamma_t)w_1$',transform=ax.transAxes,ha='right',va='top',fontsize=8.0,color=NAVY)
    # d Zeno turnover
    ax=axs[1,1]; style(ax,minor=False); panel(ax,'d')
    gbs=np.logspace(-2,np.log10(4),160); law=np.array([boundary_coefficient(J,g) for g in gbs])
    ax.semilogx(gbs,law,'--',color=ORANGE,lw=1.35,label='large-$N$ law')
    for N,mark,col in [(20,'o',TEAL),(40,'s',PURPLE),(80,'^',NAVY)]:
        ys=[]
        for gb in np.logspace(-2,np.log10(4),11):
            gap,_,_=slow_gap_and_weight(N,J,gb,0); ys.append((N+1)**3*gap)
        ax.semilogx(np.logspace(-2,np.log10(4),11),ys,mark+'-',ms=3.2,lw=1.0,color=col,label=fr'$N={N}$')
    ax.axvline(.5,color=GREY,ls=':',lw=1.0)
    ax.text(.08,.92,r'maximum at $\gamma_b=J/2$',transform=ax.transAxes,ha='left',va='top',fontsize=7.4)
    ax.set(xlabel=r'boundary excess decay $\gamma_b/J$',ylabel=r'$(N+1)^3\Delta_{\rm slow}/J$')
    ax.legend(frameon=False,ncol=2,loc='lower right',fontsize=7.0,columnspacing=0.9,handlelength=1.5)
    pd.DataFrame(rows,columns=['N','gamma_t_over_J','gap_over_J','sensor_weight']).to_csv(RES/'multispin_scaling.csv',index=False)
    save(fig,'fig7_multispin_scaling')

def figureS3():
    J=1.; gs=.25; N=24; times=np.linspace(0,35,180)
    fig,axs=plt.subplots(1,3,figsize=(7.25,2.6),constrained_layout=True)
    for k,(gt,label) in enumerate([(0,r'$\gamma_t=0$'),(.03,r'$\gamma_t/J=0.03$')]):
        ax=axs[k]; panel(ax,chr(97+k),x=-.18)
        dat=propagate_single_excitation(N,times,J,gs,gt).T
        im=ax.imshow(dat,origin='lower',aspect='auto',extent=[times.min(),times.max(),1,N],cmap='magma',vmin=0,vmax=1)
        style(ax,minor=False); ax.set(xlabel=r'$Jt$',ylabel='site $j$'); ax.text(.94,.92,label,transform=ax.transAxes,ha='right',color='white',fontsize=7.4)
        cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.02); cb.set_label(r'$|c_j|^2$',fontsize=7.0); cb.ax.tick_params(labelsize=6.5,direction='in')
    ax=axs[2]; style(ax,minor=False); panel(ax,'c',x=-.18)
    ratios=np.logspace(1,5,160); gt=1/ratios; gs_arr=.25+gt
    xi=np.array([propagation_length_readout(J,g) for g in gt])
    nf=np.array([intrinsic_floor_length(J,s,t) for s,t in zip(gs_arr,gt)])
    nr=np.array([mode_resolvability_length(J,g) for g in gt])
    ax.loglog(ratios,xi,color=NAVY,lw=1.45,label=r'$\xi_{\rm read}$')
    ax.loglog(ratios,nf,color=ORANGE,lw=1.45,label=r'$N_{\rm floor}$')
    ax.loglog(ratios,nr,color=TEAL,lw=1.45,label=r'$N_{\rm res}$')
    ax.set(xlabel=r'$J/\gamma_t$',ylabel='sites'); ax.legend(frameon=False,loc='upper left')
    pd.DataFrame({'J_over_gamma_t':ratios,'xi_read':xi,'N_floor':nf,'N_res':nr}).to_csv(RES/'multispin_length_scales.csv',index=False)
    save(fig,'figS3_multispin_propagation')

def figureS4():
    vals=validation_scan()
    df=pd.DataFrame(vals); df.to_csv(RES/'multispin_validation.csv',index=False)
    fig,axs=plt.subplots(1,3,figsize=(7.25,2.6),constrained_layout=True)
    colors={2:NAVY,3:TEAL,4:ORANGE,5:PURPLE}
    for N,g in df.groupby('N'):
        ax=axs[0]; ax.loglog(g.Omega_over_J,g.linear_response_error,'o-',ms=3.0,lw=1.1,color=colors[N],label=fr'$N={N}$')
        ax=axs[1]; ax.loglog(g.Omega_over_J,g.excitation_density,'o-',ms=3.0,lw=1.1,color=colors[N],label=fr'$N={N}$')
        if N<=4:
            ax=axs[2]; ax.loglog(g.Omega_over_J,g.xxz_xy_response_difference,'o-',ms=3.0,lw=1.1,color=colors[N],label=fr'$N={N}$')
    for i,ax in enumerate(axs): style(ax,minor=False); panel(ax,chr(97+i),x=-.18); ax.set_xlabel(r'$\Omega_F/J$')
    axs[0].set_ylabel('linear-response error'); axs[0].axhline(.05,color=GREY,ls='--',lw=.9); axs[0].legend(frameon=False,ncol=2,loc='upper left')
    axs[1].set_ylabel(r'excitation density $n_{\rm ex}$')
    axs[2].set_ylabel('XXZ-XY response difference'); axs[2].axhline(.05,color=GREY,ls='--',lw=.9); axs[2].legend(frameon=False,loc='lower left')
    save(fig,'figS4_multispin_validation')

if __name__=='__main__':
    figure7(); figureS3(); figureS4(); print('wrote multispin figures/results')
