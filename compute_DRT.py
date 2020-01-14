from math import pi
from math import log
import numpy as np

def A_re(freq):
    
    omega = 2.*pi*freq
    tau = 1./freq
    N_freqs = freq.size
    
    out_A_re = np.zeros((N_freqs, N_freqs))
    
    for p in range(0, N_freqs):
        for q in range(0, N_freqs):
            if q == 0:
                out_A_re[p, q] = -0.5/(1+(omega[p]*tau[q])**2)*log(tau[q+1]/tau[q])
            elif q == N_freqs-1:
                out_A_re[p, q] = -0.5/(1+(omega[p]*tau[q])**2)*log(tau[q]/tau[q-1])
            else:
                out_A_re[p, q] = -0.5/(1+(omega[p]*tau[q])**2)*log(tau[q+1]/tau[q-1])
                
    return out_A_re

def A_im(freq):
    
    omega = 2.*pi*freq
    tau = 1./freq
    N_freqs = freq.size
    
    out_A_im = np.zeros((N_freqs, N_freqs))
    
    for p in range(0, N_freqs):
        for q in range(0, N_freqs):
            if q == 0:
                out_A_im[p, q] = 0.5*(omega[p]*tau[q])/(1+(omega[p]*tau[q])**2)*log(tau[q+1]/tau[q])
            elif q == N_freqs-1:
                out_A_im[p, q] = 0.5*(omega[p]*tau[q])/(1+(omega[p]*tau[q])**2)*log(tau[q]/tau[q-1])
            else:
                out_A_im[p, q] = 0.5*(omega[p]*tau[q])/(1+(omega[p]*tau[q])**2)*log(tau[q+1]/tau[q-1])
                
    return out_A_im

def L(freq):
    
    tau = 1./freq
    N_freqs = freq.size
    
    out_L = np.zeros((N_freqs-2, N_freqs))
    
    for p in range(0, N_freqs-2):

        delta_loc = log(tau[p+1]/tau[p])
        
        if p==0 or p == N_freqs-3:
            out_L[p,p] = 2./(delta_loc**2)
            out_L[p,p+1] = -4./(delta_loc**2)
            out_L[p,p+2] = 2./(delta_loc**2)
        else:
            out_L[p,p] = 1./(delta_loc**2)
            out_L[p,p+1] = -2./(delta_loc**2)
            out_L[p,p+2] = 1./(delta_loc**2)

    return out_L


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)