from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import time
from tqdm import tqdm
import h5py
import pylab
import math
from scipy.integrate import solve_ivp
from scipy.spatial import Delaunay
import sys
from matplotlib import animation, rc
from IPython.display import HTML
from rs_helper import rock_string_soft_helpers as rss
from moviepy.editor import VideoFileClip, concatenate_videoclips
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.sparse import coo_matrix
from scipy import optimize
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from shutil import copyfile

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

def analyze_deformation_folder_result(folder_name0, copy_number0 = 0, save_file_name = None, *folder_names):
    """
    analyze the result output by deformation with hard BC
    """
    # load metadata and create empty class object
    metadata_file = pd.read_csv(folder_name0 + 'metadata.csv')

    CF_sys = rss.Soft_and_Filament(dim = int(metadata_file['Dimensions']),
                                  Nc = int(metadata_file['N colloids']),
                                  Nv = int(metadata_file['N vertices']),
                                  Np = int(metadata_file['N particles per filament']),
                                  Nf = int(metadata_file['N filaments']), 
                                  Rc = float(metadata_file['baseline radius']), 
                                  bidisperse = float(metadata_file['bidisperse']), 
                                  Rep = float(metadata_file['repulsive number']), 
                                  kr = float(metadata_file['repulsive constant']), 
                                  kr_b = float(metadata_file['repulsive constant from wall']), 
                                  Contr = float(metadata_file['contractility']),
                                  Inext = float(metadata_file['inextensibility']),
                                  mLf = float(metadata_file['normalized length of strings']),
                                  K1 = float(metadata_file['K1']),
                                  K2_pos = float(metadata_file['K2_pos']),
                                  K2_neg = float(metadata_file['K2_neg']),
                                  K3 = float(metadata_file['K3']),
                                  K4 = float(metadata_file['K4']),
                                  periodic_bc = metadata_file['periodic boundary'][0],
                                  full_repulsion = metadata_file['full repulsion'][0],
                                  seed = int(metadata_file['Seed']),
                                  random_init = True)

    # initialize empty array
    Ul_arr = []
    Ur_arr = []
    Urb_arr = []
    U_rep_arr = []
    Ua_arr = []
    Ub_arr = []
    Ulc_arr = []
    Ugamma_arr = []
    U_tot_arr = []

    #KE_arr = []
    KE_velocity_arr = []
    KE_neighbor_arr = []
    time_arr = []
    shrinkage_arr = []
    z_colloid_arr = []
    z_wall_arr = []
    z_filament_arr = []
    z_filament_filtered_arr = []
    z_total_arr = []
    z_total_filtered_arr = []
    z_total_below4_arr = []
    z_total_nowall_below4_arr = []
    Fx_arr = []
    Fy_arr = []
    preserved_fraction_arr = []
    x_arr = []
    y_arr = []
    vx_arr = []
    vy_arr = []
    # correlation length
    l_corr_arr = []
    x_max_arr = []
    l_corr_cv_arr = []
    l_corr_R2_arr = []
    t_last = 0
    # characterize run
    for i, folder_name in enumerate([folder_name0, *folder_names]):
        run = True
        if i == 0:
            copy_number = copy_number0
        else:
            copy_number = 0

        while run == True:
            try:
                # load data
                CF_sys.load_data(file = folder_name + 'SimResults_{0:03d}.hdf5'.format(copy_number))

                if (i == 0) and (copy_number == copy_number0):
                    CF_sys.change_r_expanded(CF_sys.R[:, 0].flatten())
                    rp_original = CF_sys.compute_ridge_point()
                    length_original = max(CF_sys.canvas.xy[:,0]) - min(CF_sys.canvas.xy[:,0])

                # add to array
                time_arr.append(CF_sys.Time)
                length = max(CF_sys.canvas.xy[:,0]) - min(CF_sys.canvas.xy[:,0])
                shrinkage = (length_original - length)/length_original
                for j, t in enumerate(CF_sys.Time):
                    CF_sys.change_r_expanded(CF_sys.R[:, j].flatten())
                    rp = CF_sys.compute_ridge_point()
                    result = CF_sys.compute_potential_energy(CF_sys.R[:, j])
                    x_arr.append(CF_sys.R[0:CF_sys.Nc, j])
                    y_arr.append(CF_sys.R[CF_sys.Nc:CF_sys.Nc*2, j])
                    try:
                        vx_arr.append((x_arr[-1] - x_arr[-2])/(t - t_last))
                        vy_arr.append((y_arr[-1] - y_arr[-2])/(t - t_last))
                    except:
                        vx_arr.append(np.zeros_like(CF_sys.R[0:CF_sys.Nc, 0]))
                        vy_arr.append(np.zeros_like(CF_sys.R[0:CF_sys.Nc, 0]))

                    Ul_arr.append(result['Ul'])
                    Ur_arr.append(result['Ur'])
                    Urb_arr.append(result['Urb'])
                    U_rep_arr.append(result['U_rep_colloid'])
                    Ua_arr.append(result['Ua'])
                    Ub_arr.append(result['Ub'])
                    Ulc_arr.append(result['Ulc'])
                    Ugamma_arr.append(result['Ugamma'])
                    U_tot_arr.append(result['U_colloid_average'])
                    res = CF_sys._KE(CF_sys.R[:, j], CF_sys.Time[j])
                    dr = CF_sys.drEdt[0:CF_sys.Nc*CF_sys.dim].reshape([CF_sys.dim, CF_sys.Nc])
                    dr_pair = dr[:, rp]
                    KE_neighbor = np.sum((dr_pair[0, :, :] - dr_pair[1, :, :])**2)
                    KE_neighbor_arr.append(KE_neighbor)
                    #KE_arr.append(res['KE_colloid_ave'])
                    KE_velocity_arr.append(res['KE_velocity_ave'])

                    params, cv, rSquared = velocity_correlation_distance(CF_sys, j)
                    l_corr_arr.append(params[0])
                    x_max_arr.append(params[1])
                    l_corr_cv_arr.append(cv)
                    l_corr_R2_arr.append(rSquared)
                    shrinkage_arr.append(shrinkage)
                    preserved_fraction_arr.append(compare_ridge_points(rp_original, rp))

                    _, _, _, z_wall = CF_sys.canvas.find_wall_potential_for_plot(
                        CF_sys.r_matrix[:, 0:1], CF_sys.r_matrix[:, 1:2], CF_sys.radius.reshape((-1,1)))
                    z_wall = z_wall.flatten()
                    z_colloid_arr.append(np.mean(CF_sys.z_colloid))
                    z_wall_arr.append(np.mean(z_wall))
                    z_filament_arr.append(np.mean(CF_sys.z_filaments))
                    z_filament_filtered_arr.append(np.mean(CF_sys.z_filaments_filtered))
                    z_total_arr.append(np.mean(CF_sys.z_colloid + CF_sys.z_filaments + z_wall))
                    z_total_filtered_arr.append(np.mean(CF_sys.z_colloid + CF_sys.z_filaments_filtered + z_wall))
                    z_total_below4_arr.append(np.sum(CF_sys.z_colloid+ z_wall + CF_sys.z_filaments <4)/CF_sys.Nc)
                    z_total_nowall_below4_arr.append(np.sum(CF_sys.z_colloid + CF_sys.z_filaments <4)/CF_sys.Nc)
                    Fx, Fy = CF_sys.canvas.find_force_on_walls(CF_sys.r[0:CF_sys.Nc].reshape((CF_sys.Nc,1)), 
                                                    CF_sys.r[CF_sys.Nc:].reshape((CF_sys.Nc,1)), 
                                                    CF_sys.radius.reshape((CF_sys.Nc,1)))
                    Fx_arr.append(np.sum(np.abs(Fx)))
                    Fy_arr.append(np.sum(np.abs(Fy)))
                    t_last = t
                # copy number +1
                copy_number += 1
            except Exception as e:
                run = False
                print(e)

    vx_arr = np.vstack(vx_arr)
    vy_arr = np.vstack(vy_arr)
    KE_kinematics_sum = np.sum(vx_arr**2 + vy_arr**2, axis = 1)
    target_loc = np.where(np.isnan(KE_kinematics_sum))[0] - 1
    target_loc = target_loc[target_loc >= 0]
    KE_selected = KE_kinematics_sum[target_loc]
    ts = np.hstack(time_arr)
    time_selected = ts[target_loc]
    # save the result
    result = {'ts': ts, 
              'Ul_arr': Ul_arr, 
              'Ur_arr': Ur_arr, 
              'Urb_arr': Urb_arr, 
              'U_rep_arr': U_rep_arr, 
              'U_tot_arr': U_tot_arr, 
              #'KE_arr': KE_arr, 
              'KE_velocity_arr': KE_velocity_arr,
              'KE_neighbor_arr': KE_neighbor_arr,
              'shrinkage_arr': shrinkage_arr,
              'preserved_fraction_arr': preserved_fraction_arr,
              'z_colloid_arr': z_colloid_arr,
              'z_wall_arr': z_wall_arr,
              'z_filament_arr': z_filament_arr,
              'z_filament_filtered_arr': z_filament_filtered_arr,
              'z_total_arr': z_total_arr,
              'z_total_filtered_arr': z_total_filtered_arr,
              'z_total_below4_arr': z_total_below4_arr,
              'z_total_nowall_below4_arr': z_total_nowall_below4_arr,
              'Fx_arr': Fx_arr,
              'Fy_arr': Fy_arr, 
              'filament_fraction': CF_sys.filament_frac_initial, 
              'volume_fraction': CF_sys.vol_frac,
              'x_arr': x_arr, 
              'y_arr': y_arr, 
              'vx_arr': vx_arr, 
              'vy_arr': vy_arr,
              'KE_kinematics': KE_selected,
              'l_corr_arr': l_corr_arr,
              'x_max_arr': x_max_arr,
              'l_corr_cv_arr': l_corr_cv_arr,
              'l_corr_R2_arr': l_corr_R2_arr,
              'time_KE_kinematics': time_selected}
    if save_file_name is None:
        pass
    else:
        file_name = save_file_name + '.hdf5'
        with h5py.File(file_name, "w") as f:
            dset = f.create_group("summarized data")
            dset.create_dataset('Time', data = result['ts'])
            dset.create_dataset('Filament contractile energy', data = result['Ul_arr'])
            dset.create_dataset('Filament repulsive energy', data = result['Ur_arr'])
            dset.create_dataset('Boundary repulsive energy', data = result['Urb_arr'])
            dset.create_dataset('Colloid repulsive energy', data = result['U_rep_arr'])
            dset.create_dataset('Averaged repulsive energy from colloid', data = result['U_tot_arr'])
            #dset.create_dataset('Averaged kinetic energy from colloid', data = result['KE_arr'])
            dset.create_dataset('Averaged kinetic energy from the velocity of colloid', data = result['KE_velocity_arr'])
            dset.create_dataset('Total relative kinetic energy between neighbor', data = result['KE_neighbor_arr'])
            dset.create_dataset('Shrinage', data = result['shrinkage_arr'])
            dset.create_dataset('Preserved connection', data = result['preserved_fraction_arr'])
            dset.create_dataset('Z colloid', data = result['z_colloid_arr'])
            dset.create_dataset('Z wall', data = result['z_wall_arr'])
            dset.create_dataset('Z filaments', data = result['z_filament_arr'])
            dset.create_dataset('Z filaments from others', data = result['z_filament_filtered_arr'])
            dset.create_dataset('Z total', data = result['z_total_arr'])
            dset.create_dataset('Z total exclude connecting filaments', data = result['z_total_filtered_arr'])
            dset.create_dataset('fraction of Z total below 4', data = result['z_total_below4_arr'])
            dset.create_dataset('fraction of Z total below 4 (exclude wall)', data = result['z_total_nowall_below4_arr'])
            dset.create_dataset('Force on the wall in x', data = result['Fx_arr'])
            dset.create_dataset('Force on the wall in y', data = result['Fy_arr'])
            dset.create_dataset('Colloid velocity in x', data = result['vx_arr'])
            dset.create_dataset('Colloid velocity in y', data = result['vy_arr'])
            try:
                dset.create_dataset('Colloid kinetic energy based on kinematics', data = result['KE_kinematics'])
                dset.create_dataset('Velocity correlation length', data = result['l_corr_arr'])
                dset.create_dataset('Maximum distance of correlation length', data = result['x_max_arr'])
                dset.create_dataset('Velocity correlation covariance', data = result['l_corr_cv_arr'])
                dset.create_dataset('Velocity correlation fitting R2', data = result['l_corr_R2_arr'])
                dset.create_dataset('Time for selected kinematics', data = result['time_KE_kinematics'])
            except Exception as e:
                print(e)
            dset.attrs['filament fraction'] = CF_sys.filament_frac_initial
            dset.attrs['volume fraction'] = CF_sys.vol_frac
            dset.attrs['N colloids'] = CF_sys.Nc
            dset.attrs['N filaments'] = CF_sys.Nf

    return result, CF_sys

def velocity_correlation_distance(CF_sys, j):
    """
    Compute the correlation distance of particle velocity
    """
    def biExp(x, t, xmax):
        return np.exp(-t * x)-np.exp(-t*(xmax-x))*(x<=xmax) - 1*(x>xmax)
    CF_sys.change_r_expanded(CF_sys.R[:, j])
    CF_sys.rhs_cython(CF_sys.R[:, j], CF_sys.Time[j])
    v_particle = CF_sys.drEdt[0:2*CF_sys.Nc].reshape((2, -1)) + 1E-20 # to avoid NaN
    v_particle_mag = np.sqrt(np.sum(v_particle**2, axis = 0))
    velocity_corr_matrix = (v_particle[0,:]*v_particle[0,:].reshape((-1,1)) + 
        v_particle[1,:]*v_particle[1,:].reshape((-1,1)))/(v_particle_mag*v_particle_mag.reshape((-1,1)))
    v_corr_flatten = np.triu(velocity_corr_matrix, k = 1)[np.nonzero(np.triu(velocity_corr_matrix, k = 1))]
    dist_flatten = np.triu(CF_sys.colloid_dist_arr, k = 1)[np.nonzero(np.triu(CF_sys.colloid_dist_arr, k = 1))]

    p0 = (0.16, 30)
    params, cv = optimize.curve_fit(biExp, dist_flatten, v_corr_flatten, p0)

    squaredDiffs = np.square(v_corr_flatten - biExp(dist_flatten, *params))
    squaredDiffsFromMean = np.square(v_corr_flatten - np.mean(v_corr_flatten))
    rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
    #print(f"R² = {rSquared}")
    return params, cv, rSquared

def plot_folder_result(folder_name0, copy_number0 = 0, save_file_name = None, *folder_names):
    # load metadata and create empty class object
    metadata_file = pd.read_csv(folder_name0 + 'metadata.csv')
    try:
        mLf = float(metadata_file['normalized length of strings'])
    except:
        mLf = 3.3

    CF_sys = rs.Colloid_and_Filament(dim = int(metadata_file['Dimensions']),
                                    Nc = int(metadata_file['N colloids']),
                                    Np = int(metadata_file['N particles per filament']),
                                    Nf = int(metadata_file['N filaments']), 
                                    Rc = float(metadata_file['baseline radius']), 
                                    bidisperse = float(metadata_file['bidisperse']), 
                                    Rep = float(metadata_file['repulsive number']), 
                                    kr = float(metadata_file['repulsive constant']), 
                                    kr_b = float(metadata_file['repulsive constant from wall']), 
                                    Contr = float(metadata_file['contractility']),
                                    Inext = float(metadata_file['inextensibility']),
                                    mLf = mLf,
                                    random_init = False)

    # initialize empty array
    Ul_arr = []
    Ur_arr = []
    Urb_arr = []
    U_rep_arr = []
    U_tot_arr = []
    KE_arr = []
    KE_velocity_arr = []
    KE_neighbor_arr = []
    time_arr = []
    vol_frac_arr = []

    # characterize run
    for i, folder_name in enumerate([folder_name0, *folder_names]):
        run = True
        if i == 0:
            copy_number = copy_number0
        else:
            copy_number = 0

        while run == True:
            try:
                # load data
                try:
                    CF_sys.load_data(file = folder_name + 'SimResults_{0:02d}.hdf5'.format(copy_number))
                except:
                    CF_sys.load_data(file = folder_name + 'SimResults_{0:03d}.hdf5'.format(copy_number))

                # add to array
                time_arr.append(CF_sys.Time)
                for j, t in enumerate(CF_sys.Time):
                    CF_sys.change_r_expanded(CF_sys.R[:, j].flatten())
                    rp = CF_sys.compute_ridge_point()
                    result = CF_sys.compute_potential_energy(CF_sys.R[:, j])
                    Ul_arr.append(result['Ul'])
                    Ur_arr.append(result['Ur'])
                    Urb_arr.append(result['Urb'])
                    U_rep_arr.append(result['U_rep_colloid'])
                    U_tot_arr.append(result['U_colloid_average'])
                    res = CF_sys._KE(CF_sys.R[:, j], CF_sys.Time[j])
                    dr = CF_sys.drEdt[0:CF_sys.Nc*CF_sys.dim].reshape([CF_sys.dim, CF_sys.Nc])
                    dr_pair = dr[:, rp]
                    KE_neighbor = np.sum((dr_pair[0, :, :] - dr_pair[1, :, :])**2)
                    KE_neighbor_arr.append(KE_neighbor)
                    KE_arr.append(res['KE_colloid_ave'])
                    KE_velocity_arr.append(res['KE_velocity_ave'])
                    vol_frac_arr.append(CF_sys.vol_frac)

                # copy number +1
                copy_number += 1
            except:
                run = False

    # plot
    ts = np.hstack(time_arr)
    plt.figure(dpi = 200)
    plt.plot(ts, Ul_arr, label = 'Ul')
    plt.plot(ts, Ur_arr, label = 'Ur')
    plt.plot(ts, Urb_arr, label = 'Urb')
    plt.plot(ts, U_rep_arr, label = 'U_rep_colloid')
    plt.plot(ts, U_tot_arr, label = 'U_total')
    plt.plot(ts, KE_arr, label = 'KE')
    plt.xlabel('time (sec)')
    plt.ylabel('potential energy')
    plt.yscale('log')
    plt.legend()
    
    result = {'ts': ts, 
              'Ul_arr': Ul_arr, 
              'Ur_arr': Ur_arr, 
              'Urb_arr': Urb_arr, 
              'U_rep_arr': U_rep_arr, 
              'U_tot_arr': U_tot_arr, 
              'KE_arr': KE_arr, 
              'KE_velocity_arr': KE_velocity_arr,
              'KE_neighbor_arr': KE_neighbor_arr,
              'vol_frac_arr': vol_frac_arr}
    if save_file_name is None:
        pass
    else:
        file_name = save_file_name + '.hdf5'
        with h5py.File(file_name, "w") as f:
            dset = f.create_group("summarized data")
            dset.create_dataset('Time', data = result['ts'])
            dset.create_dataset('Filament contractile energy', data = result['Ul_arr'])
            dset.create_dataset('Filament repulsive energy', data = result['Ur_arr'])
            dset.create_dataset('Boundary repulsive energy', data = result['Urb_arr'])
            dset.create_dataset('Colloid repulsive energy', data = result['U_rep_arr'])
            dset.create_dataset('Averaged repulsive energy from colloid', data = result['U_tot_arr'])
            dset.create_dataset('Averaged kinetic energy from colloid', data = result['KE_arr'])
            dset.create_dataset('Averaged kinetic energy from the velocity of colloid', data = result['KE_velocity_arr'])
            dset.create_dataset('Total relative kinetic energy between neighbor', data = result['KE_neighbor_arr'])
            dset.create_dataset('Volume fraction', data = result['vol_frac_arr'])

    return result, CF_sys

def plot_folder_result_deformation(folder_name0, copy_number0 = 0, save_file_name = None, *folder_names):
    # load metadata and create empty class object
    metadata_file = pd.read_csv(folder_name0 + 'metadata.csv')
    try:
        mLf = float(metadata_file['normalized length of strings'])
    except:
        mLf = 3.3

    CF_sys = rs.Colloid_and_Filament(dim = int(metadata_file['Dimensions']),
                                    Nc = int(metadata_file['N colloids']),
                                    Np = int(metadata_file['N particles per filament']),
                                    Nf = int(metadata_file['N filaments']), 
                                    Rc = float(metadata_file['baseline radius']), 
                                    bidisperse = float(metadata_file['bidisperse']), 
                                    Rep = float(metadata_file['repulsive number']), 
                                    kr = float(metadata_file['repulsive constant']), 
                                    kr_b = float(metadata_file['repulsive constant from wall']), 
                                    Contr = float(metadata_file['contractility']),
                                    Inext = float(metadata_file['inextensibility']),
                                    mLf = mLf,
                                    random_init = False)

    # initialize empty array
    Ul_arr = []
    Ur_arr = []
    Urb_arr = []
    U_rep_arr = []
    U_tot_arr = []
    KE_arr = []
    KE_velocity_arr = []
    KE_neighbor_arr = []
    time_arr = []
    vol_frac_arr = []
    shrinkage_arr = []
    x_arr = []
    y_arr = []
    vx_arr = []
    vy_arr = []

    # characterize run
    for i, folder_name in enumerate([folder_name0, *folder_names]):
        run = True
        if i == 0:
            copy_number = copy_number0
        else:
            copy_number = 0
        t_last = 0
        while run == True:
            try:
                # load data
                try:
                    CF_sys.load_data(file = folder_name + 'SimResults_{0:02d}.hdf5'.format(copy_number))
                except:
                    CF_sys.load_data(file = folder_name + 'SimResults_{0:03d}.hdf5'.format(copy_number))

                # add to array
                time_arr.append(CF_sys.Time)
                for j, t in enumerate(CF_sys.Time):
                    CF_sys.change_r_expanded(CF_sys.R[:, j].flatten())
                    rp = CF_sys.compute_ridge_point()
                    result = CF_sys.compute_potential_energy(CF_sys.R[:, j])
                    x_arr.append(CF_sys.R[0:CF_sys.Nc, j])
                    y_arr.append(CF_sys.R[CF_sys.Nc:CF_sys.Nc*2, j])
                    try:
                        vx_arr.append((x_arr[-1] - x_arr[-2])/(t - t_last))
                        vy_arr.append((y_arr[-1] - y_arr[-2])/(t - t_last))
                    except:
                        vx_arr.append(np.zeros_like(CF_sys.R[0:CF_sys.Nc, 0]))
                        vy_arr.append(np.zeros_like(CF_sys.R[0:CF_sys.Nc, 0]))

                    half_box_size_x = CF_sys.half_box_size_x
                    shrinkage_arr.append(half_box_size_x/15)
                    Ul_arr.append(result['Ul'])
                    Ur_arr.append(result['Ur'])
                    Urb_arr.append(result['Urb'])
                    U_rep_arr.append(result['U_rep_colloid'])
                    U_tot_arr.append(result['U_colloid_average'])
                    res = CF_sys._KE(CF_sys.R[:, j], CF_sys.Time[j])
                    dr = CF_sys.drEdt[0:CF_sys.Nc*CF_sys.dim].reshape([CF_sys.dim, CF_sys.Nc])
                    dr_pair = dr[:, rp]
                    KE_neighbor = np.sum((dr_pair[0, :, :] - dr_pair[1, :, :])**2)
                    KE_neighbor_arr.append(KE_neighbor)
                    KE_arr.append(res['KE_colloid_ave'])
                    KE_velocity_arr.append(res['KE_velocity_ave'])
                    vol_frac_arr.append(CF_sys.vol_frac)
                    t_last = t

                # copy number +1
                copy_number += 1
            except:
                run = False

    # plot
    ts = np.hstack(time_arr)
    plt.figure(dpi = 200)
    plt.plot(ts, Ul_arr, label = 'Ul')
    plt.plot(ts, Ur_arr, label = 'Ur')
    plt.plot(ts, Urb_arr, label = 'Urb')
    plt.plot(ts, U_rep_arr, label = 'U_rep_colloid')
    plt.plot(ts, U_tot_arr, label = 'U_total')
    plt.plot(ts, KE_arr, label = 'KE')
    plt.xlabel('time (sec)')
    plt.ylabel('potential energy')
    plt.yscale('log')
    plt.legend()
    
    result = {'ts': ts, 
              'Ul_arr': Ul_arr, 
              'Ur_arr': Ur_arr, 
              'Urb_arr': Urb_arr, 
              'U_rep_arr': U_rep_arr, 
              'U_tot_arr': U_tot_arr, 
              'KE_arr': KE_arr, 
              'KE_velocity_arr': KE_velocity_arr,
              'KE_neighbor_arr': KE_neighbor_arr,
              'vol_frac_arr': vol_frac_arr, 
              'shrinkage_arr': shrinkage_arr, 
              'x_arr': x_arr, 
              'y_arr': y_arr, 
              'vx_arr': vx_arr, 
              'vy_arr': vy_arr}
    if save_file_name is None:
        pass
    else:
        file_name = save_file_name + '.hdf5'
        with h5py.File(file_name, "w") as f:
            dset = f.create_group("summarized data")
            dset.create_dataset('Time', data = result['ts'])
            dset.create_dataset('Filament contractile energy', data = result['Ul_arr'])
            dset.create_dataset('Filament repulsive energy', data = result['Ur_arr'])
            dset.create_dataset('Boundary repulsive energy', data = result['Urb_arr'])
            dset.create_dataset('Colloid repulsive energy', data = result['U_rep_arr'])
            dset.create_dataset('Averaged repulsive energy from colloid', data = result['U_tot_arr'])
            dset.create_dataset('Averaged kinetic energy from colloid', data = result['KE_arr'])
            dset.create_dataset('Volume fraction', data = result['vol_frac_arr'])
            dset.create_dataset('Shrinkage', data = result['shrinkage_arr'])
            dset.attrs['N colloids'] = CF_sys.Nc
            dset.attrs['N filaments'] = CF_sys.Nf

    return result, CF_sys

def plot_single_snapshot(folder_name, copy_number = 0, frame = 0):
    metadata_file = pd.read_csv(folder_name + 'metadata.csv')
    try:
        mLf = float(metadata_file['normalized length of strings'])
    except:
        mLf = 3.3

    CF_sys = rs.Colloid_and_Filament(dim = int(metadata_file['Dimensions']),
                                    Nc = int(metadata_file['N colloids']),
                                    Np = int(metadata_file['N particles per filament']),
                                    Nf = int(metadata_file['N filaments']), 
                                    Rc = float(metadata_file['baseline radius']), 
                                    bidisperse = float(metadata_file['bidisperse']), 
                                    Rep = float(metadata_file['repulsive number']), 
                                    kr = float(metadata_file['repulsive constant']), 
                                    kr_b = float(metadata_file['repulsive constant from wall']), 
                                    Contr = float(metadata_file['contractility']),
                                    Inext = float(metadata_file['inextensibility']),
                                    mLf = mLf,
                                    random_init = False)
    try:
        CF_sys.load_data(file = folder_name + 'SimResults_{0:02d}.hdf5'.format(copy_number))
    except:
        CF_sys.load_data(file = folder_name + 'SimResults_{0:03d}.hdf5'.format(copy_number))
    CF_sys.change_r_expanded(CF_sys.R[:, frame].flatten())
    return CF_sys

def read_and_plot_folder_hdf5(file_name, plot = False, plot_title = None):
    """
    read the pre-processed hdf5 file, and plot the result
    """
    result = {'ts': None, 
              'Ul_arr': None, 
              'Ur_arr': None, 
              'Urb_arr': None, 
              'U_rep_arr': None, 
              'U_tot_arr': None, 
              'KE_arr': None, 
              'vol_frac_arr': None}

    with h5py.File(file_name, "r") as f:
        if('summarized data' in f.keys()):
            dset = f['summarized data']
            result['ts'] = dset["Time"][:]
            result['Ul_arr'] = dset['Filament contractile energy'][:]
            result['Ur_arr'] = dset['Filament repulsive energy'][:]
            result['Urb_arr'] = dset["Boundary repulsive energy"][:]
            result['U_rep_arr'] = dset["Colloid repulsive energy"][:]
            result['U_tot_arr'] = dset["Averaged repulsive energy from colloid"][:]
            result['KE_arr'] = dset["Averaged kinetic energy from colloid"][:]
            result['vol_frac_arr'] = dset["Volume fraction"][:]

    if plot == True:
        fig, ax1 = plt.subplots(dpi = 200)
        ax2 = ax1.twinx()

        ax1.plot(result['ts'], result['Ul_arr'], label = 'Ul')
        ax1.plot(result['ts'], result['Ur_arr'], label = 'Ur')
        ax1.plot(result['ts'], result['Urb_arr'], label = 'Urb')
        ax1.plot(result['ts'], result['U_rep_arr'], label = 'U_rep_colloid')
        ax1.plot(result['ts'], result['U_tot_arr'], label = 'U_total')
        ax1.plot(result['ts'], result['KE_arr'], label = 'KE')
        ax1.set_xlabel('time (sec)')
        ax1.set_ylabel('energy')
        ax1.set_yscale('log')
        ax1.legend()
        ax2.plot(result['ts'], result['vol_frac_arr'], label = 'volume fraction', color = 'black')
        ax2.set_ylabel('volume fraction')
        #plt.legend(bbox_to_anchor = (0.64,0.95))
        plt.title(plot_title)

    return result

def animate_folder_result(folder_name, copy_number = 0, ending_number = 100000, fps = 20, 
    plot_energy = False, colormap = cm.viridis):
    # load metadata and create empty class object
    metadata_file = pd.read_csv(folder_name + 'metadata.csv')

    CF_sys = rss.Soft_and_Filament(dim = int(metadata_file['Dimensions']),
                                  Nc = int(metadata_file['N colloids']),
                                  Nv = int(metadata_file['N vertices']),
                                  Np = int(metadata_file['N particles per filament']),
                                  Nf = int(metadata_file['N filaments']), 
                                  Rc = float(metadata_file['baseline radius']), 
                                  bidisperse = float(metadata_file['bidisperse']), 
                                  Rep = float(metadata_file['repulsive number']), 
                                  kr = float(metadata_file['repulsive constant']), 
                                  kr_b = float(metadata_file['repulsive constant from wall']), 
                                  Contr = float(metadata_file['contractility']),
                                  Inext = float(metadata_file['inextensibility']),
                                  mLf = float(metadata_file['normalized length of strings']),
                                  K1 = float(metadata_file['K1']),
                                  K2_pos = float(metadata_file['K2_pos']),
                                  K2_neg = float(metadata_file['K2_neg']),
                                  K3 = float(metadata_file['K3']),
                                  K4 = float(metadata_file['K4']),
                                  periodic_bc = metadata_file['periodic boundary'][0],
                                  full_repulsion = metadata_file['full repulsion'][0],
                                  seed = int(metadata_file['Seed']),
                                  random_init = True)

    Nf = int(metadata_file['N filaments'])
    Nc = int(metadata_file['N colloids'])
    Nv = int(metadata_file['N vertices'])
    # characterize run
    run = True
    
    while run == True:
        try:
            # load data
            if copy_number <= ending_number:
                CF_sys.load_data(file = folder_name + 'SimResults_{0:03d}.hdf5'.format(copy_number))
                
                if CF_sys.periodic_bc == True:
                    try:
                        half_box_size_x = CF_sys.half_box_size_x
                        half_box_size_y = CF_sys.half_box_size_y
                    except:
                        half_box_size_x = CF_sys.half_box_size
                        half_box_size_y = CF_sys.half_box_size

                    t_steps = len(CF_sys.Time)
                    # initialize plot
                    CF_sys.canvas.plot_canvas()
                    plt.xlim([np.min(CF_sys.canvas.xy[:,0]), np.max(CF_sys.canvas.xy[:,0])])
                    plt.ylim([np.min(CF_sys.canvas.xy[:,1]), np.max(CF_sys.canvas.xy[:,1])])
                    fig = plt.gcf()
                    ax = plt.gca()
                    ax.set_aspect('equal')
                    #[line] = ax.plot([], [], 'black', lw = 1, solid_capstyle = 'round', zorder = 2)
                    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
                    time_text.set_text('')

                    i = 0
                    CF_sys.change_r_expanded(CF_sys.R[:, i].flatten())
                    CF_sys.realign_filaments()

                    circle_list = []
                    for i, (x, y) in enumerate(CF_sys.r.reshape((CF_sys.Nc, CF_sys.dim), order = 'F')):
                        for ix in range(-1, 2):
                            for iy in range(-1, 2):
                                circle = plt.Circle((x + ix*2*half_box_size_x, y + iy*2*half_box_size_y), 
                                                    CF_sys.radius[i], color='red')
                                ax.add_artist(circle)
                                circle_list.append(circle)

                    if Nf == 0:
                        pass
                    else:
                        line_list = []
                        for i in range(CF_sys.Nf):
                            for ix in range(-1, 2):
                                for iy in range(-1, 2):                        
                                    line, = ax.plot([], [], color = 'C'+str(i))
                                    line.set_data(CF_sys.f_x_array[i, :] + ix*2*half_box_size_x, 
                                                  CF_sys.f_y_array[i, :] + iy*2*half_box_size_y)
                                    line_list.append(line)

                    if plot_energy:
                        # energy between colloids
                        self.get_distance_bet_colloid()
                        coo = coo_matrix(np.triu((self.effective_diameter - self.colloid_dist_arr))**2
                            *self.interaction_true_table)
                        normalize = mcolors.Normalize(vmin=0, vmax=np.max(coo.data))
                        ax = plt.gca()
                        for i, data in enumerate(coo.data):
                            row = coo.row[i]
                            col = coo.col[i]
                            normed = normalize(data)
                            color = colormap(normed)
                            for ix in range(-1, 2):
                                for iy in range(-1, 2):
                                    ax.plot(
                                        [self.r_matrix[row, 0] + ix*2*self.half_box_size_x, 
                                        self.r_matrix[row, 0] + ix*2*self.half_box_size_x - self.colloid_delta_x_arr[row, col]], 
                                        [self.r_matrix[row, 1] + iy*2*self.half_box_size_y, 
                                        self.r_matrix[row, 1] + iy*2*self.half_box_size_y - self.colloid_delta_y_arr[row, col]], 
                                        color=color, alpha = normed)

                    def _init2():
                        time_text.set_text('')
                        return circle_list[0],

                    def _animate2(i):
                        CF_sys.change_r_expanded(CF_sys.R[:, i].flatten())
                        CF_sys.realign_filaments()
                        for j, (x, y) in enumerate(CF_sys.r.reshape((CF_sys.Nc, CF_sys.dim), order = 'F')):
                            m = 0
                            for ix in range(-1, 2):
                                for iy in range(-1, 2):
                                    circle_list[9*j+m].center = (x + ix*2*half_box_size_x, y + iy*2*half_box_size_y)
                                    m += 1

                        if Nf == 0:
                            pass
                        else:
                            for j in range(CF_sys.Nf):
                                m = 0
                                for ix in range(-1, 2):
                                    for iy in range(-1, 2):
                                        line_list[9*j+m].set_data(CF_sys.f_x_array[j, :] + ix*2*half_box_size_x, 
                                                                  CF_sys.f_y_array[j, :] + iy*2*half_box_size_y)
                                        m += 1

                        t = CF_sys.Time[i]
                        time_text.set_text('time = %s' % t)
                        return circle_list[0],

                    # animation
                    anim = animation.FuncAnimation(fig, _animate2, init_func=_init2,
                                                   frames=t_steps, interval = 1, blit=True)
                    HTML(anim.to_html5_video())
                    anim.save(folder_name+'Anim_{0:03d}.mp4'.format(copy_number), fps = fps)
                    copy_number += 1
                  
                else:
                    t_steps = len(CF_sys.Time)
                    # initialize plot
                    CF_sys.canvas.plot_canvas()
                    fig = plt.gcf()
                    ax = plt.gca()
                    ax.set_aspect('equal')
                    #[line] = ax.plot([], [], 'black', lw = 1, solid_capstyle = 'round', zorder = 2)
                    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
                    time_text.set_text('')

                    i = 0
                    CF_sys.change_r_expanded(CF_sys.R[:, i].flatten())

                    circle_list = []
                    patch_list = []
                    for m in range(Nc):
                        circle, = ax.plot([], [], color = 'black', linewidth = 1)
                        circle.set_data(CF_sys.r_matrix[0, m, range(-1, Nv)], CF_sys.r_matrix[1, m, range(-1,Nv)])
                        patch, = ax.fill([], [], color = 'silver')
                        patch.set_xy(
                            np.vstack([CF_sys.r_matrix[0, m, range(-1, Nv)], CF_sys.r_matrix[1, m, range(-1,Nv)]]).T)
                        circle_list.append(circle)
                        patch_list.append(patch)

                    if Nf == 0:
                        pass
                    else:
                        line_list = []
                        for i in range(CF_sys.Nf):
                            line, = ax.plot([], [], color = 'green')
                            line.set_data(CF_sys.f_x_array[i, :], CF_sys.f_y_array[i, :])
                            line_list.append(line)


                    def _init():
                        time_text.set_text('')
                        return circle_list[0],

                    def _animate(i):
                        CF_sys.change_r_expanded(CF_sys.R[:, i].flatten())

                        for j in range(Nc):
                            circle_list[j].set_data(CF_sys.r_matrix[0, j, range(-1,Nv)], CF_sys.r_matrix[1, j, range(-1,Nv)])
                            patch_list[j].set_xy(
                                np.vstack([CF_sys.r_matrix[0, j, range(-1,Nv)], CF_sys.r_matrix[1, j, range(-1,Nv)]]).T)

                        if Nf == 0:
                            pass
                        else:
                            for j in range(CF_sys.Nf):
                                line_list[j].set_data(CF_sys.f_x_array[j, :], CF_sys.f_y_array[j, :])

                        t = CF_sys.Time[i]
                        time_text.set_text('time = %s' % t)
                        return circle_list[0],

                    # animation
                    anim = animation.FuncAnimation(fig, _animate, init_func=_init,
                                                   frames=t_steps, interval = 1, blit=True)
                    HTML(anim.to_html5_video())
                    anim.save(folder_name+'Anim_{0:03d}.mp4'.format(copy_number), fps = fps)
                    copy_number += 1
            else:
                run = False
        except Exception as e: 
            print(e)
            run = False

def stitch_video(folder_name, copy_number = 0, ending_number = 100000):
    final_clip = None
    run = True
    while (run == True) and (copy_number <= ending_number):

        try:
            clip_1 = VideoFileClip(folder_name+'Anim_{0:03d}.mp4'.format(copy_number))
            if final_clip is None:
                final_clip = clip_1
            else:
                final_clip = concatenate_videoclips([final_clip, clip_1])
            copy_number += 1
            print(copy_number)
        except:
            run = False

    final_clip.set_fps(40)
    final_clip.write_videofile(folder_name + "Anim_final.mp4", audio = False)

def compare_ridge_points(rp1, rp2):
    """
    compute the ratio of preserved connection between neighbor vacuoles
    """
    n_ridges_1 = len(rp1)
    preserved = 0
    for rp in rp1:
        if rp in rp2:
            preserved += 1
        else:
            pass
    percentage = preserved/n_ridges_1
    return percentage

def merge_folder_result(output_folder, *input_folders):
    if os.path.isdir(output_folder):
        pass
    else:
        os.mkdir(output_folder)
    cumulative_count = 0
    for i, folder_name in enumerate([*input_folders]):
        copy_number = 0
        run = True
        copyfile(folder_name+'metadata.csv', output_folder+'metadata.csv')
        while run:
            try:
                old_name = folder_name+'SimResults_{0:03d}.hdf5'.format(copy_number)
                new_name = output_folder+'SimResults_{0:03d}.hdf5'.format(cumulative_count)
                copyfile(old_name, new_name)
                try:
                    old_name = folder_name+'Anim_{0:03d}.mp4'.format(copy_number)
                    new_name = output_folder+'Anim_{0:03d}.mp4'.format(cumulative_count)
                    copyfile(old_name, new_name)
                except:
                    pass
                copy_number += 1
                cumulative_count += 1
            except:
                run = False

def plot_single_snapshot(folder_name, copy_number = 0, frame = 0):
    metadata_file = pd.read_csv(folder_name + 'metadata.csv')

    CF_sys = rss.Soft_and_Filament(dim = int(metadata_file['Dimensions']),
                                  Nc = int(metadata_file['N colloids']),
                                  Nv = int(metadata_file['N vertices']),
                                  Np = int(metadata_file['N particles per filament']),
                                  Nf = int(metadata_file['N filaments']), 
                                  Rc = float(metadata_file['baseline radius']), 
                                  bidisperse = float(metadata_file['bidisperse']), 
                                  Rep = float(metadata_file['repulsive number']), 
                                  kr = float(metadata_file['repulsive constant']), 
                                  kr_b = float(metadata_file['repulsive constant from wall']), 
                                  Contr = float(metadata_file['contractility']),
                                  Inext = float(metadata_file['inextensibility']),
                                  mLf = float(metadata_file['normalized length of strings']),
                                  K1 = float(metadata_file['K1']),
                                  K2_pos = float(metadata_file['K2_pos']),
                                  K2_neg = float(metadata_file['K2_neg']),
                                  K3 = float(metadata_file['K3']),
                                  K4 = float(metadata_file['K4']),
                                  periodic_bc = metadata_file['periodic boundary'][0],
                                  full_repulsion = metadata_file['full repulsion'][0],
                                  seed = int(metadata_file['Seed']),
                                  random_init = True)
    try:
        CF_sys.load_data(file = folder_name + 'SimResults_{0:02d}.hdf5'.format(copy_number))
    except:
        CF_sys.load_data(file = folder_name + 'SimResults_{0:03d}.hdf5'.format(copy_number))
    CF_sys.change_r_expanded(CF_sys.R[:, frame].flatten())
    return CF_sys

def analyze_raw_data_folder(folder_name, output_folder_name = 'output_summarized/', 
    prefix = 'deform', replace = False, Ran = False, constantF = False):
    list_file = sorted(os.listdir(folder_name))
    for test_str in list_file:
        try:
            items = test_str.split('_')
            id_Nc = items.index('Nc')
            id_Nf = items.index('Nf')
            id_volfrac = items.index('volfrac')
            id_filfrac = items.index('filfrac')
            id_mLf = [s for s in items if 'mLf' in s]
            id_long = [s for s in items if 'long' in s]
            id_seed = -1
            output_file_name = prefix + '_Nf' + items[id_Nf+1] + '_volfrac' + items[id_volfrac+1] + '_filfrac' + items[id_filfrac+1]
            try:
                output_file_name += '_' + id_mLf[0]
            except:
                pass
            try:
                output_file_name += '_' + id_long[0]
            except:
                pass
            output_file_name += '_' + items[id_seed]

            if constantF:
                output_file_name = 'constantF/' + output_file_name
            else:
                pass
            
            if Ran == True:
                final_output_dest = output_folder_name + '/' + prefix + '/' + 'NcRan' + '/' + output_file_name
                if constantF:
                    complete_input_folder = prefix + '_data/'+ 'NcRan_constantF' + '/' + test_str + '/'
                else:
                    complete_input_folder = prefix + '_data/'+ 'NcRan' + '/' + test_str + '/'
            else:
                final_output_dest = output_folder_name + '/' + prefix + '/' + 'Nc' + items[id_Nc+1] + '/' + output_file_name
                complete_input_folder = prefix + '_data/'+ 'Nc' + items[id_Nc+1] + '/' + test_str + '/'
                
            if replace == False:
                if os.path.exists(final_output_dest+'.hdf5'):
                    pass
                else:
                    if prefix == 'deform':
                        run = True
                        copy_number = 0
                        shrinkage_list = []
                        while run:
                            try:
                                if prefix == 'deform':
                                    periodic_bc = False
                                else:
                                    periodic_bc = True
                                try:
                                    mLf = float(id_mLf[0][3:])
                                except:
                                    mLf = 3.3
                                canvas_initial_xy = np.array([[-15, -15], [15, -15], [15, 15], [-15, 15]])
                                CF_sys = rs.Colloid_and_Filament(Nc = int(items[id_Nc+1]), Nf = int(items[id_Nf+1]), 
                                    canvas_initial_xy = canvas_initial_xy, random_init = False, mLf = mLf,
                                    periodic_bc = periodic_bc, full_repulsion = True, seed = 0)
                                CF_sys.load_data(file = complete_input_folder + 'SimResults_{0:03d}.hdf5'.format(copy_number))
                                half_box_size_x = CF_sys.half_box_size_x
                                shrinkage_list.append(half_box_size_x/15)
                                if shrinkage_list[-1] < 0.999:
                                    run = False
                                    copy_number -= 1
                                else:
                                    copy_number += 1
                            except:
                                run = False
                        start = copy_number
                    else:
                        start = 0
                    _, _ = analyze_deformation_folder_result(complete_input_folder, start, final_output_dest)
            else:
                if prefix == 'deform':
                    run = True
                    copy_number = 0
                    shrinkage_list = []
                    while run:
                        try:
                            if prefix == 'deform':
                                periodic_bc = False
                            else:
                                periodic_bc = True
                            try:
                                mLf = float(id_mLf[0][3:])
                            except:
                                mLf = 3.3
                            canvas_initial_xy = np.array([[-15, -15], [15, -15], [15, 15], [-15, 15]])
                            CF_sys = rs.Colloid_and_Filament(Nc = int(items[id_Nc+1]), Nf = int(items[id_Nf+1]), 
                                    canvas_initial_xy = canvas_initial_xy, random_init = False, mLf = mLf,
                                    periodic_bc = periodic_bc, full_repulsion = True, seed = 0)
                            CF_sys.load_data(file = complete_input_folder + 'SimResults_{0:03d}.hdf5'.format(copy_number))
                            half_box_size_x = CF_sys.half_box_size_x
                            shrinkage_list.append(half_box_size_x/15)
                            if shrinkage_list[-1] < 0.999:
                                run = False
                                copy_number -= 1
                            else:
                                copy_number += 1
                        except:
                            run = False
                    start = copy_number
                else:
                    start = 0
                _, _ = analyze_deformation_folder_result(complete_input_folder, start, final_output_dest)
        except Exception as e:
            print(e)

def plot_time_series_result(result, key, ylabel, ncol = 2, selection_list = None, fontsize = 12):
    plt.figure(dpi = 150)
    if selection_list:
        for i in selection_list:
            t_correct = result['t_correct_all'][i]
            plt.plot(t_correct, result[key][i], label = r'$\phi_f = $ %s'%round(result['fil_frac_all'][i],3))
    else:
        for i in np.argsort(result['fil_frac_all']):
            t_correct = result['t_correct_all'][i]
            plt.plot(t_correct, result[key][i], label = r'$\phi_f = $ %s'%round(result['fil_frac_all'][i],3))
    plt.xlabel('time', fontsize = fontsize)
    plt.ylabel(ylabel, fontsize = fontsize)
    plt.legend(ncol=ncol, bbox_to_anchor=(1.05, 1), loc='upper left')

def plot_scatter_result(result, key1, key2, xlabel, ylabel, new_fig = True):
    if new_fig:
        plt.figure(dpi = 150)
    plt.scatter(result[key1], result[key2])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
