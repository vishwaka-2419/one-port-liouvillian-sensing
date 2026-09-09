#!/usr/bin/env python3
"""Two-spin extension for dissipative-gap sensing.

This is not a reproduction of the coupled-spin calculations in Ruckert et al. 2026.
Their work establishes a microscopic driven-dissipative spin-torque framework for coupled
spins. Here the different question is the low-lying Liouvillian spectrum: how a long-lived
weakly coupled target creates a closing slow mode and how coupling J opens its decay gap.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import eigvals

I2=np.eye(2,dtype=complex)
sx=np.array([[0,1],[1,0]],complex); sy=np.array([[0,-1j],[1j,0]],complex); sz=np.array([[1,0],[0,-1]],complex)
sm=np.array([[0,0],[1,0]],complex)
def K(a,b): return np.kron(a,b)

def dsuper(C):
    d=C.shape[0]; I=np.eye(d,dtype=complex); Q=C.conj().T@C
    return np.kron(C.conj(),C)-0.5*np.kron(I,Q)-0.5*np.kron(Q.T,I)
def csuper(H):
    d=H.shape[0]; I=np.eye(d,dtype=complex)
    return -1j*(np.kron(I,H)-np.kron(H.T,I))

def liouvillian(J_rad_s, delta_target=0.0, Omega_sensor=0.0,
                Gamma1_sensor=1/140e-9, Gamma_phi_sensor=1/40e-9,
                Gamma1_target=1/5e-3, Gamma_phi_target=0.0):
    H=0.5*Omega_sensor*K(sx,I2)+0.5*delta_target*K(I2,sz)
    H+=0.25*J_rad_s*(K(sx,sx)+K(sy,sy))
    L=csuper(H)
    for rate,C in [(Gamma1_sensor,K(sm,I2)),(Gamma_phi_sensor,K(sz,I2)/np.sqrt(2)),
                   (Gamma1_target,K(I2,sm)),(Gamma_phi_target,K(I2,sz)/np.sqrt(2))]:
        if rate>0: L+=dsuper(np.sqrt(rate)*C)
    return L

def spectrum(**kw): return eigvals(liouvillian(**kw))
def gap(**kw):
    z=spectrum(**kw); r=np.array([-x.real for x in z if abs(x)>1e-6 and x.real<1e-7])
    return float(np.min(r)) if len(r) else np.nan

def weak_coupling_fit(Js_rad_s,gaps):
    x=np.asarray(Js_rad_s)**2; y=np.asarray(gaps)
    n=max(4,min(10,len(x)//3)); A=np.column_stack([np.ones(n),x[:n]])
    c=np.linalg.lstsq(A,y[:n],rcond=None)[0]
    return float(c[0]),float(c[1])
