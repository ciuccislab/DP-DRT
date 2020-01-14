# imports
import numpy as np
import os
import matplotlib.pyplot as plt
import random as rnd
import math
from math import sin, cos, pi
import torch
import torch.nn.functional as F
import compute_DRT


# check the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
    
    
# create random noisy DRTs
rng = rnd.seed(271932)
rng_np = np.random.seed(216532)

torch.manual_seed(216532)

# define frequency range
N_freqs = 81
freq_vec = np.logspace(-4., 4., num=N_freqs, endpoint=True)
tau_vec  = 1./freq_vec

# define frequency range
N_freqs = 81
freq_vec = np.logspace(-4., 4., num=N_freqs, endpoint=True)
tau_vec  = 1./freq_vec

sigma_n_exp = 0.1

# define variables for exact
R_inf_1 = 10
R_ct_1 = 50
phi_1 = 0.8
tau_1 = 0.1

T_1 = tau_1**phi_1/R_ct_1
Z_exact_1 = R_inf_1 + 1./(1./R_ct_1+T_1*(1j*2.*pi*freq_vec)**phi_1)

# define variables for exact
R_inf_2 = 10
R_ct_2 = 50
phi_2 = 0.8
tau_2 = 1E1

T_2 = tau_2**phi_2/R_ct_2
Z_exact_2 = 1./(1./R_ct_2+T_2*(1j*2.*pi*freq_vec)**phi_2)

Z = Z_exact_1  + Z_exact_2
Z_exp = Z + (sigma_n_exp**2)*np.random.normal(0,1,N_freqs) + 1j*(sigma_n_exp**2)*np.random.normal(0,1,N_freqs)

# this part of the code is pretty bad
Z_exp_re = np.zeros(N_freqs)
Z_exp_im = np.zeros(N_freqs)
Z_exp_re = np.real(Z_exp)
Z_exp_im = np.imag(Z_exp)

# Create tensors to hold inputs
# as this is an inverse problem the input (x) is Z
Z_exp_re_torch = torch.from_numpy(Z_exp_re.T).type(torch.FloatTensor).reshape(1,N_freqs)
Z_exp_im_torch = torch.from_numpy(Z_exp_im.T).type(torch.FloatTensor).reshape(1,N_freqs)

#compute gamma exact
gamma_fct_1 = (R_ct_1)/(2.*pi)*sin((1.-phi_1)*pi) / \
    (np.cosh(phi_1*np.log(tau_vec/tau_1))-cos((1.-phi_1)*pi))
gamma_fct_2 = (R_ct_2)/(2.*pi)*sin((1.-phi_2)*pi) / \
    (np.cosh(phi_2*np.log(tau_vec/tau_2))-cos((1.-phi_2)*pi))
gamma_exact = gamma_fct_1 + gamma_fct_2
gamma_exact_torch = torch.from_numpy(gamma_exact).type(torch.FloatTensor)

# define the matrices neeeded in the NN
A_re = compute_DRT.A_re(freq_vec)
A_im = compute_DRT.A_im(freq_vec)
L = compute_DRT.L(freq_vec)

A_re_torch = torch.from_numpy(A_re.T).type(torch.FloatTensor)
A_im_torch = torch.from_numpy(A_im.T).type(torch.FloatTensor)
L_torch = torch.from_numpy(L.T).type(torch.FloatTensor)

# size of the arbitrary zeta input
# which will be declared as
# zeta = torch.randn(N, N_zeta)

N_zeta = 1
# define the neural network
# N is batch size
# D_in is input dimension;
# H is hidden dimension
# D_out is output dimension.
N = 1
D_in = N_zeta # this is the dimension of zeta
H = max(N_freqs,10*N_zeta)
D_out = N_freqs+1


# Define model
class vanilla_model(torch.nn.Module):
    def __init__(self):
        super(vanilla_model, self).__init__()
        self.fct_1 = torch.nn.Linear(D_in, H)
        self.fct_2 = torch.nn.Linear(H, H) 
        self.fct_3 = torch.nn.Linear(H, H)
        self.fct_4 = torch.nn.Linear(H, D_out) 

        torch.nn.init.xavier_uniform_(self.fct_1.weight)
        torch.nn.init.xavier_uniform_(self.fct_2.weight)
        torch.nn.init.xavier_uniform_(self.fct_3.weight)
        torch.nn.init.xavier_uniform_(self.fct_4.weight)

        torch.nn.init.zeros_(self.fct_1.weight)
        torch.nn.init.zeros_(self.fct_2.weight)
        torch.nn.init.zeros_(self.fct_3.weight)
        torch.nn.init.zeros_(self.fct_4.weight)
        
    def forward(self, zeta):

        h = F.elu(self.fct_1(zeta))
        h = F.elu(self.fct_2(h))
        h = F.elu(self.fct_3(h))
        gamma_pred = F.softplus(self.fct_4(h), beta = 5)        
        
        return gamma_pred 
    
model = vanilla_model()

def loss_fn(output, Z_exp_re_torch, Z_exp_im_torch, A_re_torch, A_im_torch, L_torch):
    
    MSE_re = torch.sum((output[:, -1]+torch.mm(output[:, 0:-1], A_re_torch)-Z_exp_re_torch)**2)
    MSE_im = torch.sum((torch.mm(output[:, 0:-1], A_im_torch)-Z_exp_im_torch)**2)

    MSE = MSE_re + MSE_im
    return MSE

# optimize the neural network
loss_vec = np.array([])
distance_vec = np.array([])
lambda_vec = np.array([])

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which tensors it should update.


zeta = torch.randn(N, N_zeta)

learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12000, gamma=0.1)
max_iters = 100001
gamma_NN_store = torch.zeros((max_iters, N_freqs))
R_inf_NN_store = torch.zeros((max_iters, 1))

for t in range(max_iters):
    # Forward pass: compute predicted y by passing x to the model.
    gamma = model(zeta)
    
    # Compute the loss
    loss = loss_fn(gamma, Z_exp_re_torch, Z_exp_im_torch, A_re_torch, A_im_torch, L_torch)
    # save it
    loss_vec = np.append(loss_vec, loss.item())
    
    # store gamma
    gamma_NN = gamma[:, 0:-1].detach().reshape(-1) 
    gamma_NN_store[t, :] = gamma_NN
    
    # store gamma
    R_inf_NN_store[t,:] = gamma[:, -1].detach().reshape(-1)

    # Compute the distance
    distance = math.sqrt(torch.sum((gamma_NN-gamma_exact_torch)**2).item()) 
    # save it
    distance_vec = np.append(distance_vec, distance)

    # and print it
    if not t%100:
        print('iter=', t, '; loss=', loss.item(), '; distance=', distance)

    # Before starting the optimizer we can note that the learning rate
    # can be modified on the go, see
    # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    # It would be nice to implement this option in the future.
    
    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()
    
    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()
    # scheduler.step()
    
index_opt = np.argmin(distance_vec)    
index_early_stop = np.flatnonzero(np.abs(np.diff(loss_vec))<1E-8)

gamma_DIP_torch_opt = gamma_NN_store[index_opt, :]
R_inf_DIP_torch_opt = R_inf_NN_store[index_opt, :]

gamma_DIP_opt = gamma_DIP_torch_opt.detach().numpy()
R_DIP_opt = R_inf_DIP_torch_opt.detach().numpy()

if len(index_early_stop):
    gamma_DIP_torch_early_stop = gamma_NN_store[index_early_stop[0], :]
    gamma_DIP = gamma_DIP_torch_early_stop.detach().numpy()
    R_DIP = R_inf_NN_store[index_early_stop[0], :]
    R_DIP = R_DIP.detach().numpy()
else:
    gamma_DIP = gamma_DIP_opt
    R_DIP = R_DIP_opt
    
Z_DIP = R_DIP + np.matmul(A_re, gamma_DIP) + 1j*np.matmul(A_im, gamma_DIP)

plt.plot(np.real(Z_exp), -np.imag(Z_exp), "o", markersize=10, color="black", label="synth exp")
plt.plot(np.real(Z_DIP), -np.imag(Z_DIP), linewidth=4, color="red", label="DP-DRT")
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.annotate(r'$10^{-2}$', xy=(np.real(Z_exp[20]), -np.imag(Z_exp[20])), 
             xytext=(np.real(Z_exp[20])-1, 8-np.imag(Z_exp[20])), 
             arrowprops=dict(arrowstyle="-",connectionstyle="arc"))
plt.annotate(r'$10^{-1}$', xy=(np.real(Z_exp[30]), -np.imag(Z_exp[30])), 
             xytext=(np.real(Z_exp[30])-1, 8-np.imag(Z_exp[30])), 
             arrowprops=dict(arrowstyle="-",connectionstyle="arc"))
plt.annotate(r'$1$', xy=(np.real(Z_exp[40]), -np.imag(Z_exp[40])), 
             xytext=(np.real(Z_exp[40])+1, 9-np.imag(Z_exp[40])), 
             arrowprops=dict(arrowstyle="-",connectionstyle="arc"))
plt.annotate(r'$10$', xy=(np.real(Z_exp[50]), -np.imag(Z_exp[50])), 
             xytext=(np.real(Z_exp[50])-1, 11-np.imag(Z_exp[50])), 
             arrowprops=dict(arrowstyle="-",connectionstyle="arc"))
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.legend(frameon=False, fontsize = 15)
plt.xlim(10, 120)
plt.ylim(0, 55)
plt.xticks(range(0, 130, 10))
plt.yticks(range(0, 60, 10))
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel(r'$Z_{\rm re}/\Omega$', fontsize = 20)
plt.ylabel(r'$-Z_{\rm im}/\Omega$', fontsize = 20)
plt.savefig('figs/Fig_3_a.eps', dpi=300, bbox_inches='tight')
plt.savefig('figs/Fig_3_a.svg', dpi=300, bbox_inches='tight') 
fig = plt.gcf()
size = fig.get_size_inches()
plt.show()


plt.semilogx(tau_vec, gamma_exact, linewidth=4, color="black", label="exact")
#plt.semilogx(tau_vec, gamma_DIP_opt, linestyle='None', marker='o', color="blue", label="optimal")
plt.semilogx(tau_vec, gamma_DIP, linewidth=4, color="red", label="DP-DRT")
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.axis([1E-4,1E4,-0.4,33])
plt.legend(frameon=False, fontsize = 15)
plt.xlabel(r'$\tau/{\rm s}$', fontsize = 20)
plt.ylabel(r'$\gamma/\Omega$', fontsize = 20)
fig = plt.gcf()
fig.set_size_inches(5, 4)
plt.savefig('figs/Fig_3_b.eps', dpi=300, bbox_inches='tight')
plt.savefig('figs/Fig_3_b.svg', dpi=300, bbox_inches='tight')
plt.show()

plt.semilogy(loss_vec, linewidth=4, color="black")
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.xlabel(r'iter', fontsize=20)
plt.ylabel(r'loss', fontsize=20)
plt.axis([0,1E5,1E-2,1E6])
fig=plt.gcf()
fig.set_size_inches(5, 4)
plt.savefig('figs/Fig_3_abc.eps', dpi=300, bbox_inches='tight')
plt.savefig('figs/Fig_3_abc.svg', dpi=300, bbox_inches='tight')
plt.show()

plt.semilogy(distance_vec, linewidth=4, color="black")
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.xlabel(r'iter', fontsize=20)
plt.ylabel(r'error', fontsize=20)
plt.axis([0,1E5,1E0,1E2])
fig=plt.gcf()
fig.set_size_inches(5, 4)
plt.savefig('figs/Fig_3_abd.eps', dpi=300, bbox_inches='tight')
plt.savefig('figs/Fig_3_abd.svg', dpi=300, bbox_inches='tight')
plt.show()

print('total number parameters = ', compute_DRT.count_parameters(model))
print('distance_early_stop = ', distance_vec[index_early_stop[0]])
print('distance_opt= ', distance_vec[index_opt])

PATH = os.getcwd()
torch.save(model.state_dict(), 'model_wo_lambda.pth')
