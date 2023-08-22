"""
@author: rmojgani
Model discovery
Run the KS model without u_xxxx, 
Find the u_xxxx using RVM
"""
#%%
import sys
# insert at 1, 0 is the script path (or '' in REPL)
import numpy as np
#%%
# $$
# u_t + uu_x + u_{xx} + u_{xxxx} = 0
# $$
#%%
if_local= True
if_run  = True
if_save = False
if_load = False
if_SP   = False
# 'estimatedterms', 'fixedterms', 'none'
if_keep_only_last = False
dmRBoolean = 0
D = 4; P = 4 # D = 4; P = 4 or D = 4; P = 0 
#%%
#%%
import numpy as np
from canonicalPDEs import save_sol, load_sol
from canonicalPDEs import ETRNK4intKS as intKS 
L = 100
N = 256#*2
dx = L/N
x = np.linspace(-L/2, L/2, N, endpoint=False)
dx = x[1] - x[0]

kappa = 2 * np.pi*np.fft.fftfreq(N,d=dx)
lambdas = [1.0,1.0,1.0]
LL = 1/(L/100)
u0 = -np.cos(x*2*np.pi/100*LL)*(1+np.sin(-x*2*np.pi/100*LL))

dt = 0.01/10
Nt = 30000*10-int(100.0/dt)
t = np.arange(0,Nt*dt,dt)#dt=0.1, N = 256
dt = t[1]-t[0]
Nt_spinoff = int(100.0/dt)

print('spin up',Nt_spinoff)
BETA = 10000#*1000 #beta = 100*100
NDT = 5

X, T = np.meshgrid(x, t)
#------------------- 
# Perfect model run
#------------------- 
print('/ Simulation start ... ')
print('dt',dt)
u_truth_long = intKS(u0,t,kappa,N,lambdas)
save_sol(u_truth_long, 'KS_256')

