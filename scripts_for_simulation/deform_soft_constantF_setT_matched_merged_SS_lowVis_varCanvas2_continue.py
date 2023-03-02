from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import odespy
import os
import re
from scipy.spatial import distance_matrix
from scipy.sparse import coo_matrix
import cython
from cython.parallel import prange
from datetime import datetime
import time
from tqdm import tqdm
import h5py
import pylab
import math
from scipy.integrate import solve_ivp
from scipy.spatial import Delaunay
import sys
cwd = os.getcwd()
sys.path.insert(1, cwd)
from rs_helper import rock_string_helpers_realistic as rs
from rs_helper import rock_string_analysis_helper_new as rsh
from rs_helper import rock_string_soft_helpers_SSmethod_ERbreak_fullc as rsssb
from rs_helper import rock_string_soft_helpers_SSmethod_ERbreak_fullc_canvasVar as rsssbc

PI = np.pi

env_var = os.environ
hardstarter_name = env_var['HARD']

items_file = hardstarter_name.split('/')
test_str = items_file[-1]
items = test_str.split('_')
id_Nc = items.index('Nc')
id_Nf = items.index('Nf')
id_mLf = [s for s in items if 'mLf' in s]
id_seed = [s for s in items if 'seed' in s]
Nc = int(items[id_Nc+1])
Nf = int(items[id_Nf+1])
mLf = float(id_mLf[0][3:])
seed = int(id_seed[0].split('.')[0][4:])

ERbreak = bool(env_var['ERbreak'])
print('ER break: %s'%ERbreak)
ER_break_threshold = 1.015

folder_name = env_var['FOLDER']

try:
    Nv = int(env_var['Nv'])
except:
    Nv = 8
try:
    random_init = bool(env_var['RANDOMINIT'])
except:
    random_init = True
try:
    Lx = float(env_var['LXHALF'])
    Ly = float(env_var['LYHALF'])
except:
    Lx = 50.0
    Ly = 5.0

canvas_initial_xy = np.array([[-Lx, -Ly], [Lx, -Ly], [Lx, Ly], [-Lx, Ly]])
CF_sys = rsssbc.Soft_and_Filament(Nc = Nc, Nf = Nf, Nv = Nv, 
    K1 = 0.91E-3, K2_pos = 1.8, K2_neg = 1.8, K3 = 0.25, K4 = 5E-9, 
    Re_R = 0.4, Stk = 0.24, Ca = 0.005, viscous_ratio = 5,
    canvas_initial_xy = canvas_initial_xy, random_init = random_init, mLf = mLf,
    periodic_bc = False, full_repulsion = True, seed = seed, kr_bf = 0)

list_of_files = np.sort(os.listdir(folder_name))

def extract_number(f):
    s = re.findall("\d+$",f)
    return (int(s[0]) if s else -1,f)

last_file = max(list_of_files,key=extract_number)
CF_sys.load_data(folder_name + last_file)
CF_sys.change_r_expanded(CF_sys.R[:, -1].flatten())

# obtain force and run time
F = CF_sys.boundary_forces_external
Lx0 = CF_sys.Lx0
Ly0 = CF_sys.Ly0
T_contr = CF_sys.T_contr
eta_cyto_sim = CF_sys.eta
eta_water_sim = eta_cyto_sim/5
shrinkage_min = 0.05
dT = T_contr/6250
ddt = 0.01

UE_ave = 1
KE_ave = 1
if ERbreak:
    note = '_mLf'+str(mLf)+'_soft_ERbreak_lowVis_constantF_VarC_'+str(round(F,2))+'_seed'+str(seed)+'_18Aug'
else:
    note = '_mLf'+str(mLf)+'_soft_ERintact_lowVis_constantF_VarC_'+str(round(F,2))+'_seed'+str(seed)+'_18Aug'

Lx_new = np.max(CF_sys.canvas.xy[:,0])

# shrinkage
shrinkage = Lx_new/Lx
t0 = CF_sys.Time[-1]
if ERbreak:
    ER_intact = CF_sys.ER_intact

while (shrinkage > shrinkage_min) and (t0 < T_contr):
    rsh.tic()
    CF_sys.simulate(Tf = dT, t0 = t0, method = 'bdf', save = True, Npts = int(dT/ddt), note = note, use_odespy = True, 
            path = '/scratch/users/jrc612', atol = 1E-7, rtol = 1E-6, subsampling = 20, order = 5)
    rsh.toc()
    t0 = CF_sys.Time[-1]
    CF_sys.change_r_expanded(CF_sys.R[:, -1].flatten())
    hLx = np.max(CF_sys.canvas.xy[:,0])
    shrinkage = hLx/Lx
    hLy_corrected = Lx*Ly/hLx
    CF_sys.change_canvas(np.array([[-hLx, -hLy_corrected], 
                    [hLx, -hLy_corrected], 
                    [hLx, hLy_corrected], 
                    [-hLx, hLy_corrected]]))

    if ERbreak:
        # evaluate ER break
        CF_sys.change_r_expanded(CF_sys.R[:, -1].flatten())
        CF_sys.get_separation_vectors()
        ER_intact *= (CF_sys.dr < ER_break_threshold*CF_sys.dLf)
        CF_sys.change_ER_intact(ER_intact)
    print('shrinkage = %s' %shrinkage)
    print(CF_sys.canvas.xy)

os.system('say "your program has finished"')
