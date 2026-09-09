#!/usr/bin/env python3
"""One-port exchange-coupled spin network: linear generator, exact one-port response, validation.

Conventions (hbar = 1):
H = sum_j delta_j sigma_z(j)/2
  + J/4 sum_j [sigma_x(j)sigma_x(j+1)+sigma_y(j)sigma_y(j+1)]
  + Omega/2 sigma_x(1).

Only spin 1 is directly driven/read out.  Its transverse coherence rate is gamma_s;
spins 2..N have rate gamma_t.  In the polarized weak-drive sector the coherences
c_j=<sigma^-_j> obey exactly to linear order
    dc/dt = A c - i Omega/2 e_1,
where A_jj = -(gamma_j + i delta_j), A_{j,j+1}=A_{j+1,j}=-i J/2.

The file also provides a full Hilbert-space Lindblad solver for N<=5 used to
validate the linearized description and to test an XXZ interaction away from the
single-excitation limit.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import linalg
from scipy.sparse import csc_matrix, csr_matrix, eye, kron
from scipy.sparse.linalg import spsolve

SX=np.array([[0,1],[1,0]],complex)
SY=np.array([[0,-1j],[1j,0]],complex)
SZ=np.array([[1,0],[0,-1]],complex)
SM=np.array([[0,0],[1,0]],complex)  # basis |up>,|down>; lowering |up> -> |down>
ID2=np.eye(2,dtype=complex)


def linear_generator(N:int,J:float=1.0,gamma_s:float=.25,gamma_t:float=0.0,
                     detunings=None):
    if detunings is None: detunings=np.zeros(N)
    detunings=np.asarray(detunings,float)
    if len(detunings)!=N: raise ValueError('detunings must have length N')
    A=np.zeros((N,N),complex)
    gam=np.full(N,gamma_t,float); gam[0]=gamma_s
    A[np.diag_indices(N)]=-(gam+1j*detunings)
    for j in range(N-1):
        A[j,j+1]=A[j+1,j]=-1j*J/2
    return A


def steady_linear_response(N:int,Omega:float,J:float=1.0,gamma_s:float=.25,
                           gamma_t:float=0.0,detunings=None):
    A=linear_generator(N,J,gamma_s,gamma_t,detunings)
    b=np.zeros(N,complex); b[0]=-1j*Omega/2
    return np.linalg.solve(A,-b)


def slow_mode(N:int,J:float=1.0,gamma_s:float=.25,gamma_t:float=0.0):
    A=linear_generator(N,J,gamma_s,gamma_t)
    vals,vecs=np.linalg.eig(A)
    idx=np.argmin(-vals.real)
    v=vecs[:,idx]; v=v/np.linalg.norm(v)
    return vals[idx],v


def slow_gap_and_weight(N:int,J:float=1.0,gamma_s:float=.25,gamma_t:float=0.0):
    lam,v=slow_mode(N,J,gamma_s,gamma_t)
    return -lam.real, abs(v[0])**2, lam


def boundary_coefficient(J:float,gamma_b:float):
    """Large-N coefficient C_b in Delta-gamma_t = C_b/(N+1)^3."""
    return 2*np.pi**2*gamma_b/(1+(2*gamma_b/J)**2)


def slow_weight_asymptotic(N:int,J:float,gamma_b:float):
    return 2*np.pi**2/(1+(2*gamma_b/J)**2)/(N+1)**3


def slow_gap_asymptotic(N:int,J:float,gamma_s:float,gamma_t:float):
    gamma_b=gamma_s-gamma_t
    return gamma_t+boundary_coefficient(J,gamma_b)/(N+1)**3


def propagation_length_readout(J:float,gamma_t:float):
    """Round-trip intensity/readout length at band center, in lattice sites."""
    if gamma_t<=0: return np.inf
    return 1/(2*np.arcsinh(gamma_t/J))


def intrinsic_floor_length(J:float,gamma_s:float,gamma_t:float):
    """N where boundary-induced slow-mode decay equals intrinsic target decay."""
    if gamma_t<=0: return np.inf
    cb=boundary_coefficient(J,gamma_s-gamma_t)
    return max(0.0,(cb/gamma_t)**(1/3)-1)


def mode_resolvability_length(J:float,gamma_t:float):
    """Near-band-edge estimate from adjacent-mode spacing ~ 3*pi^2 J/[2(N+1)^2]=gamma_t."""
    if gamma_t<=0: return np.inf
    return max(0.0,np.pi*np.sqrt(3*J/(2*gamma_t))-1)


def propagate_single_excitation(N:int,times,J:float=1.0,gamma_s:float=.25,gamma_t:float=0.0):
    """Non-Hermitian one-magnon propagation from site 1; returns normalized intensities."""
    A=linear_generator(N,J,gamma_s,gamma_t)
    c0=np.zeros(N,complex); c0[0]=1.0
    out=np.empty((len(times),N))
    for k,t in enumerate(times):
        c=linalg.expm(A*t)@c0
        out[k]=abs(c)**2
    m=out.max()
    if m>0: out/=m
    return out

# ---------- One-port response: residues, participation, phase rigidity ----------
def one_port_modes(N:int,J:float=1.0,gamma_s:float=.25,gamma_t:float=0.0,detunings=None):
    """Modal data of the complex-symmetric one-port generator A_N (A_N^T = A_N).

    For every right eigenvector R_m (normalised, ||R_m||=1) the function returns
      lam   : eigenvalue lambda_m,
      part  : boundary participation |R_m(1)|^2,
      rig   : phase rigidity r_m = |R_m^T R_m|  (1 for a normal mode, 0 at an EP),
      res   : residue of the one-port susceptibility chi_11 at lambda_m,
              Res_m = R_m(1)^2 / (R_m^T R_m)   (complex-symmetric structure: L_m = R_m^*).
    Exact relations checked in the accompanying script:
      -Re lam_m - gamma_t = gamma_b |R_m(1)|^2 = gamma_b r_m |Res_m|,
      sum_m Res_m = 1,   sum_m |R_m(1)|^2 = 1.
    Modes are returned slowest first.
    """
    A=linear_generator(N,J,gamma_s,gamma_t,detunings)
    lam,R=np.linalg.eig(A)
    R=R/np.linalg.norm(R,axis=0)
    rtr=np.einsum('im,im->m',R,R)
    part=np.abs(R[0,:])**2
    rig=np.abs(rtr)
    res=R[0,:]**2/rtr
    order=np.argsort(-lam.real)
    return dict(lam=lam[order],part=part[order],rig=rig[order],res=res[order],R=R[:,order],A=A)


def one_port_susceptibility(omega,N:int,J:float=1.0,gamma_s:float=.25,gamma_t:float=0.0,detunings=None):
    """chi_11(omega) = e_1^T (i omega - A_N)^{-1} e_1, evaluated directly (no modal sum)."""
    A=linear_generator(N,J,gamma_s,gamma_t,detunings)
    omega=np.atleast_1d(np.asarray(omega,float))
    e1=np.zeros(N,complex); e1[0]=1
    return np.array([np.linalg.solve(1j*w*np.eye(N)-A,e1)[0] for w in omega])


def one_port_sum_rules(N:int,J:float=1.0,gamma_s:float=.25,gamma_t:float=0.0,detunings=None):
    """Return the machine-precision defects of the exact one-port relations."""
    m=one_port_modes(N,J,gamma_s,gamma_t,detunings)
    gb=gamma_s-gamma_t
    lhs=-m['lam'].real-gamma_t
    return dict(identity=float(np.max(np.abs(lhs-gb*m['part']))),
                residue_form=float(np.max(np.abs(m['part']-m['rig']*np.abs(m['res'])))),
                residue_sum=complex(m['res'].sum()),
                participation_sum=float(m['part'].sum()),
                min_rigidity=float(m['rig'].min()),
                slow_rigidity=float(m['rig'][0]))


# ---------- Full Lindblad validation ----------
def _op_on_site(op,site,N):
    out=np.array([[1]],complex)
    for j in range(N): out=np.kron(out,op if j==site else ID2)
    return out


def build_hamiltonian(N,J=1.0,Omega=0.0,delta=0.0,Delta_z=0.0):
    d=2**N
    H=np.zeros((d,d),complex)
    for j in range(N):
        H += delta/2*_op_on_site(SZ,j,N)
    H += Omega/2*_op_on_site(SX,0,N)
    for j in range(N-1):
        H += J/4*(_op_on_site(SX,j,N)@_op_on_site(SX,j+1,N)+
                  _op_on_site(SY,j,N)@_op_on_site(SY,j+1,N))
        if Delta_z:
            H += Delta_z/4*(_op_on_site(SZ,j,N)@_op_on_site(SZ,j+1,N))
    return H


def liouvillian_sparse(H,jumps):
    d=H.shape[0]; I=eye(d,format='csc',dtype=complex); Hs=csc_matrix(H)
    # vec convention column stacking: vec(A rho B)=(B^T \otimes A)vec(rho)
    L=-1j*(kron(I,Hs)-kron(Hs.T,I))
    for C in jumps:
        C=csc_matrix(C); CdC=(C.getH()@C).tocsc()
        L += kron(C.conjugate(),C)-0.5*kron(I,CdC)-0.5*kron(CdC.T,I)
    return L.tocsc()


def steady_state_sparse(L,d):
    M=L.tolil(copy=True)
    b=np.zeros(d*d,complex)
    # replace one row by trace constraint
    M[0,:]=0
    diag_indices=[j+j*d for j in range(d)]
    M[0,diag_indices]=1
    b[0]=1
    rho_vec=spsolve(M.tocsc(),b)
    rho=rho_vec.reshape((d,d),order='F')
    rho=(rho+rho.conj().T)/2
    rho/=np.trace(rho)
    return rho


def full_steady_state(N,Omega,J=1.0,gamma_s=.25,gamma_t=.01,Delta_z=0.0):
    """Amplitude damping rates are 2*gamma_j so transverse coherence decays at gamma_j."""
    H=build_hamiltonian(N,J,Omega,Delta_z=Delta_z)
    jumps=[]
    for j in range(N):
        gam=gamma_s if j==0 else gamma_t
        jumps.append(np.sqrt(2*gam)*_op_on_site(SM,j,N))
    L=liouvillian_sparse(H,jumps)
    return steady_state_sparse(L,2**N)


def full_observables(N,rho):
    coh=np.array([np.trace(rho@_op_on_site(SM,j,N)) for j in range(N)])
    # excitation number n=(1+z)/2 in basis |up>,|down>
    nex=np.mean([(1+np.trace(rho@_op_on_site(SZ,j,N)).real)/2 for j in range(N)])
    return coh,nex


def validation_scan(Ns=(2,3,4,5),Omegas=None,J=1.0,gamma_s=.25,gamma_t=.01,Delta_z=.5):
    if Omegas is None: Omegas=np.logspace(-3.2,-0.35,10)*J
    rows=[]
    for N in Ns:
        for Om in Omegas:
            lin=steady_linear_response(N,Om,J,gamma_s,gamma_t)
            rho=full_steady_state(N,Om,J,gamma_s,gamma_t,Delta_z=0)
            coh,nex=full_observables(N,rho)
            # compare sensor response because it is the measured port; avoid vanishing remote response zeros
            denom=max(abs(lin[0]),1e-14)
            err=abs(coh[0]-lin[0])/denom
            row=dict(N=N,Omega_over_J=Om/J,linear_response_error=err,excitation_density=nex)
            if N<=4:
                rho_z=full_steady_state(N,Om,J,gamma_s,gamma_t,Delta_z=Delta_z)
                coh_z,_=full_observables(N,rho_z)
                row['xxz_xy_response_difference']=abs(coh_z[0]-coh[0])/max(abs(coh[0]),1e-14)
            else:
                row['xxz_xy_response_difference']=np.nan
            rows.append(row)
    return rows

if __name__=='__main__':
    for N in [10,20,40,80]:
        g,w,_=slow_gap_and_weight(N)
        print(N,g,w)
