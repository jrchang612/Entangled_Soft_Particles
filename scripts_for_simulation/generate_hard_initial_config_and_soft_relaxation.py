from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import odespy
import os
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
from rs_helper import rock_string_soft_helpers_realistic_ERbreak as rssb
from rs_helper import rock_string_analysis_helper_new as rsh

PI = np.pi

env_var = os.environ
Nc = int(env_var['Nc'])
Nf = int(env_var['Nf'])
seed = int(env_var['SEED'])
try:
    random_init = bool(env_var['RANDOMINIT'])
except:
    random_init = True
try:
    mLf = float(env_var['mLf'])
except:
    mLf = 6.4
try:
    Lx = float(env_var['LXHALF'])
    Ly = float(env_var['LYHALF'])
except:
    Lx = 50
    Ly = 5

UE_ave = 1
KE_ave = 1
note = '_mLf'+str(mLf)+'_FQ_Config_seed'+str(seed)
canvas_initial_xy = np.array([[-Lx, -Ly], [Lx, -Ly], [Lx, Ly], [-Lx, Ly]])

if Nf == 0:
    CF_sysH = rs.Colloid_and_Filament(Nc = Nc, Nf = Nf, kr_bf = 1,
    	canvas_initial_xy = canvas_initial_xy, random_init = random_init, mLf = mLf,
        periodic_bc = False, full_repulsion = True, seed = seed)

    print('volume fraction = %s' %CF_sysH.vol_frac)
    counter = 0
    # fully relax in initialization phase
    while ((KE_ave > 1E-10) or (UE_ave > 1E-8)) and ((counter < 50) or (percent_changes > 0.01)):
        try:
            t0 = CF_sysH.Time[-1]
            CF_sysH.change_r_expanded(CF_sysH.R[:, -1].flatten())
        except Exception as e:
            print(e)
            t0 = 0
        rsh.tic()
        CF_sysH.simulate(Tf = 2000, t0 = t0, method = 'bdf', save = True, Npts = 40, note = note, use_odespy = True, 
                path = '/scratch/users/jrc612')
        rsh.toc()

        U_tot_arr = []
        KE_arr = []
        for i, t in enumerate(CF_sysH.Time):
            CF_sysH.change_r_expanded(CF_sysH.R[:, i].flatten())
            result = CF_sysH.compute_potential_energy(CF_sysH.R[:, i])
            U_tot_arr.append(result['U_colloid_average'])
            KE_arr.append(CF_sysH._KE(CF_sysH.R[:, i], CF_sysH.Time[i])['KE_colloid_ave'])
        percent_changes = np.abs(UE_ave - np.min(U_tot_arr))/UE_ave + np.abs(KE_ave - np.min(KE_arr))/KE_ave

        UE_ave = np.min(U_tot_arr)
        KE_ave = np.min(KE_arr)
        print('time = %s' %t0)
        counter += 1
        print('counter = %s' %counter)
else:
    # pre-run with Nf = 0
    CF_sysH0 = rs.Colloid_and_Filament(Nc = Nc, Nf = 0, kr_bf = 1,
        canvas_initial_xy = canvas_initial_xy, random_init = random_init, mLf = mLf,
        periodic_bc = False, full_repulsion = True, seed = seed)
    CF_sysH0.simulate(Tf = 2000, t0 = 0, method = 'bdf', save = True, Npts = 40, note = note, use_odespy = True, 
                path = '/scratch/users/jrc612')
    CF_sysH0.change_r_expanded(CF_sysH0.R[:, -1].flatten())

    # initialize the actual system with prerun system to make it faster
    CF_sysH = rs.Colloid_and_Filament(Nc = Nc, Nf = Nf, kr_bf = 1,
        canvas_initial_xy = canvas_initial_xy, random_init = random_init, mLf = mLf,
        periodic_bc = False, full_repulsion = True, seed = seed)
    CF_sysH.r0 = CF_sysH0.r
    CF_sysH.radius = CF_sysH0.radius
    CF_sysH.six_pi_eta_r = 6*PI*CF_sysH.eta*CF_sysH.radius
    CF_sysH.eight_pi_eta_r_cube = 8*PI*CF_sysH.eta*CF_sysH.radius**3
    CF_sysH.effective_diameter = CF_sysH.radius + CF_sysH.radius.reshape((CF_sysH.Nc,1))
    CF_sysH.r = CF_sysH.r0
    CF_sysH.r_matrix = CF_sysH.reshape_to_matrix(CF_sysH.r)
    CF_sysH.r_expanded_0[0:CF_sysH.Nc*CF_sysH.dim] = CF_sysH.r0
    CF_sysH.r_expanded[0:CF_sysH.Nc*CF_sysH.dim] = CF_sysH.r
    tri = Delaunay(CF_sysH.r_matrix)
    CF_sysH.delaunay = tri
    connection_list = []
    for simplex in tri.simplices:
        connection_list.append(np.array([simplex[2], simplex[0]]))
        connection_list.append(np.array([simplex[0], simplex[1]]))
        connection_list.append(np.array([simplex[1], simplex[2]]))        
    new_connection_list = np.unique(np.sort(connection_list), axis = 0)
    dist_rp = np.sqrt(np.sum(
        (tri.points[new_connection_list[:,0], :] - tri.points[new_connection_list[:,1], :])**2, axis = 1))
    threshold = CF_sysH.Rc*CF_sysH.bidisperse*4
    [drop_list] = np.where(dist_rp > threshold)
    drop_list = np.flip(drop_list)
    new_connection_list = list(new_connection_list)
    for drop in drop_list:
        new_connection_list.pop(drop)
    new_connection_list = np.array(new_connection_list)
    CF_sysH.Nf_Delaunay = len(new_connection_list)
    list_id_selected = np.random.choice(np.arange(len(new_connection_list)), CF_sysH.Nf, replace = False)
    CF_sysH.list_id_selected = list_id_selected
    CF_sysH.connection_list = new_connection_list
    CF_sysH.connection_table[:, 0:2] = CF_sysH.connection_list[list_id_selected, :]
    CF_sysH.connection_theta[:, 0] = 2*PI*np.random.rand(CF_sysH.Nf)
    CF_sysH.connection_theta[:, 1] = (CF_sysH.connection_theta[:, 0] + PI)%(2*PI)
    CF_sysH.f_x_array = np.linspace(CF_sysH.r_matrix[CF_sysH.connection_table[:,0], 0] 
                                     + CF_sysH.radius[CF_sysH.connection_table[:,0]]*np.cos(CF_sysH.connection_theta[:,0]), 
                                     CF_sysH.r_matrix[CF_sysH.connection_table[:,1], 0] 
                                     + CF_sysH.radius[CF_sysH.connection_table[:,1]]*np.cos(CF_sysH.connection_theta[:,1]), 
                                     CF_sysH.Np).T
    CF_sysH.f_y_array = np.linspace(CF_sysH.r_matrix[CF_sysH.connection_table[:,0], 1] 
                                     + CF_sysH.radius[CF_sysH.connection_table[:,0]]*np.sin(CF_sysH.connection_theta[:,0]), 
                                     CF_sysH.r_matrix[CF_sysH.connection_table[:,1], 1] 
                                     + CF_sysH.radius[CF_sysH.connection_table[:,1]]*np.sin(CF_sysH.connection_theta[:,1]), 
                                     CF_sysH.Np).T
    CF_sysH.get_separation_vectors()
    CF_sysH.r_expanded_0[CF_sysH.Nc*(CF_sysH.dim+1):(CF_sysH.Nc*(CF_sysH.dim+1) + CF_sysH.Nf*CF_sysH.Np)] = CF_sysH.f_x_array.flatten()
    CF_sysH.r_expanded_0[(CF_sysH.Nc*(CF_sysH.dim+1) + CF_sysH.Nf*CF_sysH.Np):(CF_sysH.Nc*(CF_sysH.dim+1) + 2*CF_sysH.Nf*CF_sysH.Np)] = CF_sysH.f_y_array.flatten()
    CF_sysH.r_expanded[CF_sysH.Nc*(CF_sysH.dim+1):(CF_sysH.Nc*(CF_sysH.dim+1) + CF_sysH.Nf*CF_sysH.Np)] = CF_sysH.r_expanded_0[
            CF_sysH.Nc*(CF_sysH.dim+1):(CF_sysH.Nc*(CF_sysH.dim+1) + CF_sysH.Nf*CF_sysH.Np)]
    CF_sysH.r_expanded[(CF_sysH.Nc*(CF_sysH.dim+1) + CF_sysH.Nf*CF_sysH.Np):(CF_sysH.Nc*(CF_sysH.dim+1) + 2*CF_sysH.Nf*CF_sysH.Np)] = CF_sysH.r_expanded_0[
            (CF_sysH.Nc*(CF_sysH.dim+1) + CF_sysH.Nf*CF_sysH.Np):(CF_sysH.Nc*(CF_sysH.dim+1) + 2*CF_sysH.Nf*CF_sysH.Np)]
    CF_sysH.r_vac0 = CF_sysH.radius[CF_sysH.connection_table[:, 0]].reshape((CF_sysH.Nf,1))
    CF_sysH.r_vac1 = CF_sysH.radius[CF_sysH.connection_table[:, 1]].reshape((CF_sysH.Nf,1))
    CF_sysH.filament_frac_initial = CF_sysH.Nf/CF_sysH.Nf_Delaunay

    # simulation of the actual system
    print('volume fraction = %s' %CF_sysH.vol_frac)
    counter = 0
    # fully relax in initialization phase
    while ((KE_ave > 1E-10) or (UE_ave > 1E-8)) and ((counter < 50) or (percent_changes > 0.01)):
        try:
            t0 = CF_sysH.Time[-1]
            CF_sysH.change_r_expanded(CF_sysH.R[:, -1].flatten())
        except Exception as e:
            print(e)
            t0 = 0
        rsh.tic()
        CF_sysH.simulate(Tf = 2000, t0 = t0, method = 'bdf', save = True, Npts = 40, note = note, use_odespy = True, 
                path = '/scratch/users/jrc612')
        rsh.toc()

        U_tot_arr = []
        KE_arr = []
        for i, t in enumerate(CF_sysH.Time):
            CF_sysH.change_r_expanded(CF_sysH.R[:, i].flatten())
            result = CF_sysH.compute_potential_energy(CF_sysH.R[:, i])
            U_tot_arr.append(result['U_colloid_average'])
            KE_arr.append(CF_sysH._KE(CF_sysH.R[:, i], CF_sysH.Time[i])['KE_colloid_ave'])
        percent_changes = np.abs(UE_ave - np.min(U_tot_arr))/UE_ave + np.abs(KE_ave - np.min(KE_arr))/KE_ave

        UE_ave = np.min(U_tot_arr)
        KE_ave = np.min(KE_arr)
        print('time = %s' %t0)
        counter += 1
        print('counter = %s' %counter)

CF_sysH.change_r_expanded(CF_sysH.R[:, -1].flatten())

# initialize soft particle system
corrected_theta = (CF_sysH.theta[CF_sysH.connection_table] + CF_sysH.connection_theta)%(2*PI)
CF_sys = rssb.Soft_and_Filament(Nc = Nc, Nf = Nf, Nv = Nv,
    canvas_initial_xy = canvas_initial_xy, random_init = random_init, mLf = mLf,
    periodic_bc = False, full_repulsion = True, seed = seed, kr_bf = 1)
CF_sys.initialize_colloid_location_from_starter(radius = CF_sysH.radius, r_matrix = CF_sysH.r_matrix)
CF_sys.initialize_filament_location_from_starter(delaunay = CF_sysH.delaunay, connection_table = CF_sysH.connection_table, 
                                                 connection_theta = corrected_theta, f_x_array = CF_sysH.f_x_array, 
                                                 f_y_array = CF_sysH.f_y_array)

# obtain force and run time
F = CF_sys.boundary_forces_external
Lx0 = CF_sys.Lx0
Ly0 = CF_sys.Ly0
T_contr = CF_sys.T_contr
eta_cyto_sim = CF_sys.eta
eta_water_sim = eta_cyto_sim/50
shrinkage_min = 0.05
dT = T_contr/5000

UE_ave = 1
KE_ave = 1
note = '_mLf'+str(mLf)+'_deform_debug_'+str(round(F,2))+'_seed'+str(seed)+'_corrected'

print('volume fraction = %s' %CF_sys.vol_frac)
counter = 0
percent_changes = 1
# fully relax in initialization phase
#while (percent_changes > 0.001):
while counter < 500:
    try:
        t0 = CF_sys.Time[-1]
        CF_sys.change_r_expanded(CF_sys.R[:, -1].flatten())
    except:
        t0 = 0
    rsh.tic()
    CF_sys.simulate(Tf = dT*50, t0 = t0, method = 'bdf', save = True, Npts = 4, note = note, use_odespy = True, 
        path = '/scratch/users/jrc612')
    rsh.toc()

    U_tot_arr = []
    KE_arr = []
    for i, t in enumerate(CF_sys.Time):
        CF_sys.change_r_expanded(CF_sys.R[:, i].flatten())
        result = CF_sys.compute_potential_energy(CF_sys.R[:, i])
        U_tot_arr.append(result['U_colloid_average'])
        KE_arr.append(CF_sys._KE(CF_sys.R[:, i], CF_sys.Time[i])['KE_velocity_ave'])
    percent_changes = np.abs(UE_ave - np.min(U_tot_arr))/UE_ave + np.abs(KE_ave - np.min(KE_arr))/KE_ave

    UE_ave = np.min(U_tot_arr)
    KE_ave = np.min(KE_arr)
    print('time = %s' %t0)
    counter += 1
    print('counter = %s' %counter)

os.system('say "your program has finished"')
