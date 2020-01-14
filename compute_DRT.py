from math import pi
from math import log
import numpy as np

# This script defines the matrices used for calculating the impedance from DRT
# The detail equations and derivations can be found in this paper "Saccoccio, M., Wan, T. H., Chen, C., & Ciucci, F. (2014). Optimal regularization in distribution of relaxation times applied to electrochemical impedance spectroscopy: ridge and lasso regression methods-a theoretical and experimental study. Electrochimica Acta, 147, 470-482." [doi.org/10.1016/j.electacta.2014.09.058]
# Please feel free to use it and cite the following publications 
# 1. Saccoccio, M., Wan, T. H., Chen, C., & Ciucci, F. (2014). Optimal regularization in distribution of relaxation times applied to electrochemical impedance spectroscopy: ridge and lasso regression methods-a theoretical and experimental study. Electrochimica Acta, 147, 470-482. [doi.org/10.1016/j.electacta.2014.09.058](https://doi.org/10.1016/j.electacta.2014.09.058)
# 2. Liu, J., & Ciucci, F. (2020). The Deep-Prior Distribution of Relaxation Times. Journal of The Electrochemical Society, 167(2), 026506.[10.1149/1945-7111/ab631a](https://iopscience.iop.org/article/10.1149/1945-7111/ab631a/meta)
# For contact, please email: francesco.ciucci@ust.hk

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