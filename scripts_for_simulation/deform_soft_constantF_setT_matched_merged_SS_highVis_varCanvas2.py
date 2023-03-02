from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import odespy
import os
import cython
from cython.parallel import prange
from scipy.spatial import distance_matrix
from scipy.sparse import coo_matrix
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
ER_break_threshold = 1.015

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

# load hard particle system
canvas_initial_xy = np.array([[-Lx, -Ly], [Lx, -Ly], [Lx, Ly], [-Lx, Ly]])
CF_sysH = rs.Colloid_and_Filament(Nc = Nc, Nf = Nf, 
    canvas_initial_xy = canvas_initial_xy, random_init = random_init, mLf = mLf,
    periodic_bc = False, full_repulsion = True, seed = seed)
CF_sysH.load_data(file = hardstarter_name)
CF_sysH.change_r_expanded(CF_sysH.R[:, -1].flatten())

# initialize soft particle system
corrected_theta = (CF_sysH.theta[CF_sysH.connection_table] + CF_sysH.connection_theta)%(2*PI)
#     Re_R = 0.4, Stk = 0.24, Ca = 0.0039,
CF_sys0 = rsssb.Soft_and_Filament(Nc = Nc, Nf = Nf, Nv = Nv, 
    K1 = 0.91E-3, K2_pos = 1.8, K2_neg = 1.8, K3 = 0.25, K4 = 5E-9,
    Re_R = 0.04, Stk = 0.0024, Ca = 0.05,
    canvas_initial_xy = canvas_initial_xy, random_init = random_init, mLf = mLf,
    periodic_bc = False, full_repulsion = True, seed = seed, kr_bf = 0)

CF_sys0.initialize_colloid_location_from_starter(radius = CF_sysH.radius/CF_sysH.Rc, 
    r_matrix = CF_sysH.r_matrix)
CF_sys0.initialize_filament_location_from_starter(delaunay = CF_sysH.delaunay, connection_table = CF_sysH.connection_table, 
                                                 connection_theta = corrected_theta, f_x_array = CF_sysH.f_x_array, 
                                                 f_y_array = CF_sysH.f_y_array)

CF_sys = rsssbc.Soft_and_Filament(Nc = Nc, Nf = Nf, Nv = Nv, 
    K1 = 0.91E-3, K2_pos = 1.8, K2_neg = 1.8, K3 = 0.25, K4 = 5E-9,
    Re_R = 0.04, Stk = 0.0024, Ca = 0.05, viscous_ratio = 50,
    canvas_initial_xy = canvas_initial_xy, random_init = random_init, mLf = mLf,
    periodic_bc = False, full_repulsion = True, seed = seed, kr_bf = 0)

CF_sys.initialize_colloid_location_from_starter(radius = CF_sysH.radius/CF_sysH.Rc, 
    r_matrix = CF_sysH.r_matrix)
CF_sys.initialize_filament_location_from_starter(delaunay = CF_sysH.delaunay, connection_table = CF_sysH.connection_table, 
                                                 connection_theta = corrected_theta, f_x_array = CF_sysH.f_x_array, 
                                                 f_y_array = CF_sysH.f_y_array)

# obtain force and run time
F = CF_sys.boundary_forces_external
Lx0 = CF_sys0.Lx0
Ly0 = CF_sys0.Ly0
T_contr = CF_sys0.T_contr
shrinkage_min = 0.05
dT = T_contr/625
ddt = 0.1

UE_ave = 1
KE_ave = 1

if ERbreak:
    note = '_mLf'+str(mLf)+'_soft_ERbreak_highVis_constantF_VarC_'+str(round(F,2))+'_seed'+str(seed)+'_18Aug'
else:
    note = '_mLf'+str(mLf)+'_soft_ERintact_highVis_constantF_VarC_'+str(round(F,2))+'_seed'+str(seed)+'_18Aug'

print('volume fraction = %s' %CF_sys.vol_frac)
counter = 0
counterMax = 100
percent_changes = 1
ER_too_long = 1

while (counter < counterMax) and (ER_too_long > 0 or percent_changes > 1E-3):
    try:
        t0 = CF_sys.Time[-1]
        CF_sys0.change_r_expanded(CF_sys.R[:, -1].flatten())
    except:
        t0 = 0
    rsh.tic()
    CF_sys0.simulate(Tf = dT, t0 = t0, method = 'bdf', save = True, Npts = int(dT/ddt), note = note+'init', use_odespy = True, 
        path = '/scratch/users/jrc612', atol = 1E-7, rtol = 1E-6, subsampling = 20, order = 5)
    rsh.toc()

    U_tot_arr = []
    KE_arr = []
    for i, t in enumerate(CF_sys0.Time):
        CF_sys0.change_r_expanded(CF_sys0.R[:, i].flatten())
        result = CF_sys0.compute_potential_energy(CF_sys0.R[:, i])
        U_tot_arr.append(result['U_colloid_average'])
        KE_arr.append(CF_sys0._KE(CF_sys0.R[:, i], CF_sys0.Time[i])['KE_velocity_ave'])
    percent_changes = np.abs(UE_ave - np.min(U_tot_arr))/UE_ave

    CF_sys0.get_separation_vectors()
    ER_too_long = np.sum(CF_sys0.dr > ER_break_threshold*CF_sys0.dLf)

    UE_ave = np.min(U_tot_arr)
    KE_ave = np.min(KE_arr)
    print('time = %s' %t0)
    counter += 1
    print('counter = %s' %counter)

# shrinkage
shrinkage = 1
t0 = 0
if ERbreak:
    ER_intact = CF_sys.ER_intact

CF_sys.change_r_expanded(np.hstack([CF_sys0.R[:, -1].flatten(), Lx, Ly]))
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
