cimport cython
import numpy as np
cimport numpy as np
np.import_array()
from libc.math cimport sqrt, atan2
from libc.math cimport round as roundc
from cython.parallel import prange
cdef double PI = 3.14159265359

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.c_api_binop_methods(True)

cdef class vacuole_interactions:
	cdef long Nv
	cdef double kr
	cdef long [:] axis_1_plus_1
	cdef long Nc
	cdef long Nf
	cdef long Np
	cpdef interact_with_vacuole(self, double[:, :] xy_x, double[:, :] xy_y, double[:, :] xy2_x, double[:, :] xy2_y, double[:, :] xs, double[:, :] ys, double DELTA, int n_pair)
	cpdef add_string_interact_with_vacuole(self, double[:, :] xy_x, double[:, :] xy_y, double[:, :] xy2_x, double[:, :] xy2_y, double[:] xs, double[:] ys, double DELTA, int n_pair, int[:] c_all, int[:] f_collide, int[:] p_collide, long[:] axis_1_plus_1)
	cpdef add_string_interact_with_vacuole_complete(self, double[:, :] xy_x, double[:, :] xy_y, double[:, :] xy2_x, double[:, :] xy2_y, double[:] xs, double[:] ys, double DELTA, int n_pair, int[:] c_all, int[:] f_collide, int[:] p_collide, long[:] axis_1_plus_1)
	cpdef add_vacuole_interact_with_vacuole(self, double[:, :] xy_x, double[:, :] xy_y, double[:, :] xy2_x, double[:, :] xy2_y, double[:] xs, double[:] ys, double DELTA, int n_pair, int[:] c_all, int[:] c_collide, int[:] v_collide, long[:] axis_1_plus_1)
	cpdef generate_c_c_interaction(self, double[:,:,:] r_matrix, double[:,:] xy_vacuole_min_mi_delt, double[:,:] xy_vacuole_max_pl_delt)
	cpdef generate_c_c_intxn_add_vac_force(self, double[:,:,:] r_matrix, double[:,:] xy_vacuole_min_mi_delt, double[:,:] xy_vacuole_max_pl_delt, double[:] rx, double[:] ry, long[:] axis_1_plus_1, double DELTA)
	cpdef generate_fp_c_interaction(self, double[:,:] f_x_array, double[:,:] f_y_array, double[:,:] xy_vacuole_min_mi_delt, double[:,:] xy_vacuole_max_pl_delt, long[:,:] connection_table)
	cpdef generate_fp_c_intxn_add_string_force(self, double[:,:,:] r_matrix, double[:,:] f_x_array, double[:,:] f_y_array, double[:,:] xy_vacuole_min_mi_delt, double[:,:] xy_vacuole_max_pl_delt, long[:,:] connection_table, double DELTA, long[:] axis_1_plus_1)

cdef class boundary_interactions:
	cdef long n_corners
	cpdef find_within_canvas_and_wall_force(self, double[:] xs, double[:] ys, double delta, long n_site, double[:,:] xy, double[:,:] xy2, double[:] a, double[:] b, double[:] c, double[:] M, int[:] center_sign_to_border)

cdef class vacuole_computations:
	cdef long Nv
	cdef long Nc
	cdef double gamma
	cdef double kl_c_pos_Nv
	cdef double kl_c_neg_Nv
	cpdef find_Fb_Flc_Fgamma(self, double[:,:] lx_arr, double[:,:] ly_arr, double[:,:] l_arr, double[:] l0_arr, double[:,:] l_m_l0_arr, double[:,:] lx_div_l, double[:,:] ly_div_l, double[:] vacuole_bending_prefactor, long[:] axis_1_plus_1, long[:] axis_1_minus_1, long[:] axis_1_plus_2, int[:,:] neg)

cdef class filament_computations:
	cdef long Nf
	cdef long Np
	cdef double kl
	cdef double filament_bending_prefactor
	cpdef find_Fl_Fbf(self, double[:,:] dx, double[:,:] dy, long[:,:] ER_intact, double[:,:] Fl_expanded)
