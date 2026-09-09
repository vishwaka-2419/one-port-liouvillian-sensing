#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, RegularPolygon, Arc
from matplotlib.ticker import AutoMinorLocator
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'figures'; RES=ROOT/'results'
NAVY='#18364F'; TEAL='#0C8F8B'; ORANGE='#DD7438'; RED='#B23C49'; PURPLE='#7151A4'; GREEN='#458A61'; GREY='#67737D'; LIGHT='#EDF2F5'; DARK='#22262A'
mpl.rcParams.update({'font.family':'serif','font.serif':['Latin Modern Roman','CMU Serif','Computer Modern Roman','DejaVu Serif'],'mathtext.fontset':'stix','font.size':8.2,'axes.labelsize':8.3,'legend.fontsize':7.7,'xtick.labelsize':7.3,'ytick.labelsize':7.3,'axes.linewidth':.85,'xtick.direction':'in','ytick.direction':'in','pdf.fonttype':42,'ps.fonttype':42})
def style(ax,minor=True):
    for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(.85); s.set_color('#30363B')
    ax.tick_params(which='both',direction='in',top=True,right=True)
    if minor: ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())
def panel(ax,l,x=-.16): ax.text(x,1.04,l,transform=ax.transAxes,fontweight='bold',fontsize=9.6,va='bottom')
def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.025); fig.savefig(OUT/(name+'.png'),dpi=350,bbox_inches='tight',pad_inches=.025); plt.close(fig)
def fig5():
    m=pd.read_csv(RES/'metrology_scan.csv'); ep=pd.read_csv(RES/'floquet_ep_scan.csv'); fep=float(np.interp(.6,ep.aD,ep.fEP_MHz))
    x=m.Omega_over_EP.to_numpy(); fi=m.counting_FI_per_rad2s.to_numpy(); imax=int(np.nanargmax(fi)); xopt=x[imax]
    fig,axs=plt.subplots(1,3,figsize=(7.25,2.55),constrained_layout=True)
    ax=axs[0]; style(ax); panel(ax,'a')
    l1=ax.plot(x,m.pair_sep_Mrad_s,color=NAVY,lw=1.5,label=r'$|\Delta\mu|$')
    ax2=ax.twinx(); l2=ax2.plot(x,m.overlap,color=RED,lw=1.35,label=r'$\mathcal{O}_R$')
    for s in ax2.spines.values(): s.set_visible(True); s.set_linewidth(.85); s.set_color('#30363B')
    ax2.tick_params(which='both',direction='in',top=True,right=True,colors=RED); ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax.axvline(1,color=ORANGE,ls='--',lw=1.05)
    ax.set(xlabel=r'$\Omega_F/\Omega_{\rm EP}$',ylabel=r'$|\Delta\mu|$ ($10^6$ s$^{-1}$)'); ax2.set_ylabel(r'$\mathcal{O}_R$',color=RED)
    ax.legend(l1+l2,[z.get_label() for z in l1+l2],frameon=False,loc='upper left',bbox_to_anchor=(.03,.76),handlelength=1.35,fontsize=7.8)
    ax=axs[1]; style(ax); panel(ax,'b')
    ax.plot(x,fi/fi[imax],color=TEAL,lw=1.6,label=r'$\dot{\mathcal{F}}_{\rm count}$')
    ax.axvline(1,color=ORANGE,ls='--',lw=1.05,label='EP'); ax.axvline(xopt,color=PURPLE,ls=':',lw=1.15,label='information optimum')
    ax.scatter([xopt],[1],s=28,color=PURPLE,edgecolor='white',lw=.5,zorder=4)
    ax.annotate(fr'$\Omega_F\simeq {xopt:.2f}\,\Omega_{{\rm EP}}$',xy=(xopt,1),xytext=(4.15,.84),fontsize=7.3,arrowprops=dict(arrowstyle='->',lw=.7,color=GREY),ha='left')
    ax.set(xlabel=r'$\Omega_F/\Omega_{\rm EP}$',ylabel=r'$\dot{\mathcal{F}}_{\rm count}/\dot{\mathcal{F}}_{\max}$',ylim=(0.1,1.06)); ax.legend(frameon=False,loc='lower right',fontsize=7.5,handlelength=1.4)
    ax=axs[2]; style(ax); panel(ax,'c')
    l1=ax.plot(x,m.current_pA,color=NAVY,lw=1.5,label=r'$I$'); ax2=ax.twinx(); l2=ax2.plot(x,m.singly_occ,color=GREEN,lw=1.35,label=r'$P_1$')
    for s in ax2.spines.values(): s.set_visible(True); s.set_linewidth(.85); s.set_color('#30363B')
    ax2.tick_params(which='both',direction='in',top=True,right=True,colors=GREEN); ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax.axvline(1,color=ORANGE,ls='--',lw=1.05); ax.axvline(xopt,color=PURPLE,ls=':',lw=1.0)
    ax.set(xlabel=r'$\Omega_F/\Omega_{\rm EP}$',ylabel=r'$I$ (pA)'); ax2.set_ylabel(r'$P_1$',color=GREEN)
    ax.legend(l1+l2,[z.get_label() for z in l1+l2],frameon=False,loc='upper left',bbox_to_anchor=(.03,.78),handlelength=1.35,fontsize=7.8)
    save(fig,'fig5_metrology')
def fig6():
    d=pd.read_csv(RES/'target_gap_scan.csv'); fit=json.load(open(RES/'target_gap_fit.json')); J=d.J_over_2pi_MHz.to_numpy(); gap=d['gap_s-1'].to_numpy()
    fig,axs=plt.subplots(1,2,figsize=(7.25,2.75),constrained_layout=True,width_ratios=[.9,1.45])
    ax=axs[0]; ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off'); panel(ax,'a',x=-.10)
    # Retain the restrained schematic style of the approved v3 figure.
    ax.add_patch(Polygon([[.16,.94],[.55,.94],[.47,.57],[.38,.57]],closed=True,fc=LIGHT,ec=NAVY,lw=1.15))
    ax.plot([.425,.425],[.57,.45],color=NAVY,lw=1.0); ax.add_patch(Polygon([[.405,.49],[.445,.49],[.425,.42]],closed=True,fc='white',ec=NAVY,lw=.9))
    ax.arrow(.425,.42,0,-.10,width=.004,head_width=.032,head_length=.035,color=NAVY,length_includes_head=True)
    ax.text(.10,.82,'spin-polarized tip',fontsize=6.9,color=NAVY,ha='left')
    for cx in np.linspace(.30,.58,5): ax.add_patch(RegularPolygon((cx,.28),6,radius=.050,orientation=np.pi/6,facecolor='white',edgecolor=TEAL,lw=1.0))
    ax.plot([.08,.88],[.12,.12],color='#95A1A8',lw=1.0); ax.add_patch(Polygon([[.08,.12],[.88,.12],[.82,.18],[.14,.18]],closed=True,fc='#EFF2F4',ec='#95A1A8',lw=1.0))
    ax.text(.44,.19,'pentacene sensor',ha='center',fontsize=6.8,color=TEAL); ax.text(.48,.035,'MgO/Ag',ha='center',fontsize=6.8,color=GREY)
    ax.add_patch(Circle((.44,.39),.031,fc='white',ec=TEAL,lw=1.1)); ax.arrow(.44,.39,0,.11,width=.004,head_width=.028,head_length=.03,color=TEAL,length_includes_head=True)
    ax.add_patch(Circle((.78,.38),.055,fc='#F2F7F2',ec=GREEN,lw=1.3)); ax.arrow(.78,.38,.022,.12,width=.004,head_width=.032,head_length=.032,color=GREEN,length_includes_head=True); ax.text(.78,.20,'target spin',ha='center',fontsize=6.8,color=GREEN)
    ax.add_patch(Arc((.61,.40),.30,.22,theta1=12,theta2=168,color=PURPLE,lw=1.1,ls='--')); ax.text(.61,.55,r'$J$',ha='center',fontsize=8,color=PURPLE)
    ax=axs[1]; style(ax,minor=False); panel(ax,'b',x=-.12)
    pred=fit['gap0_fit_s-1']+fit['quadratic_coefficient_s']*(2*np.pi*J*1e6)**2
    ax.loglog(J[1:],gap[1:],color=TEAL,lw=1.6,label=r'$\Delta_{\mathcal{L}}$')
    # Only show perturbative law over its fitted weak-coupling range.
    weak=(J>0)&(J<=0.03)
    ax.loglog(J[weak],pred[weak],'--',color=ORANGE,lw=1.35,label=r'$\Delta_0+CJ^2$')
    ax.axhline(fit['gap0_fit_s-1'],color=NAVY,ls=':',lw=1.05,label=r'$\Delta_0$')
    ax.set(xlabel=r'$J/2\pi$ (MHz)',ylabel=r'$\Delta_{\mathcal{L}}$ (s$^{-1}$)'); ax.legend(frameon=False,loc='lower right',fontsize=7.8,handlelength=1.5)
    ins=ax.inset_axes([.11,.56,.36,.34]); rel=(gap-fit['gap0_fit_s-1'])/fit['gap0_fit_s-1']; ins.loglog(J[1:],np.maximum(rel[1:],1e-12),color=PURPLE,lw=1.1); ins.axhline(1,color='#ADB7BE',ls='--',lw=.75)
    for s in ins.spines.values(): s.set_visible(True); s.set_linewidth(.6); s.set_color('#555E64')
    ins.tick_params(which='both',direction='in',top=True,right=True,labelsize=5.5,length=2); ins.set_xlabel(r'$J/2\pi$',fontsize=5.7,labelpad=0); ins.set_ylabel(r'$(\Delta-\Delta_0)/\Delta_0$',fontsize=5.7,labelpad=1)
    save(fig,'fig6_target_gap')
if __name__=='__main__': fig5(); fig6()
