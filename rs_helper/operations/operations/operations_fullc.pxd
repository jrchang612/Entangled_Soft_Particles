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
@cython.initializedcheck(False)
cdef class full_rhs_cython:
	cdef int Nc
	cdef int Nv
	cdef int Nf
	cdef int Np
	cdef int n_corners 
	cdef double kr
	cdef double kr_b
	cdef double ka
	cdef double gamma
	cdef double kl_c_pos_Nv
	cdef double kl_c_neg_Nv
	cdef double kl
	cdef double filament_bending_prefactor
	cdef double kr_bf
	cdef double six_pi_eta_r
	cdef double delt
	cdef double dLf
	cdef double eta
	cpdef compute_Fa(self, double[:,:,:] r_matrix, long[:] axis_1_minus_1, long[:] axis_1_plus_1, double[:] area0_arr)
	cpdef compute_segment_length_Fb_Flc_Fgamma(self, double[:,:,:] r_matrix, double[:] l0_arr, double[:] vacuole_bending_prefactor, long[:] axis_1_plus_1, long[:] axis_1_minus_1, long[:] axis_1_plus_2)
	cpdef get_separation_vectors_Fl_Fbf(self, double[:,:] f_x_array, double[:,:] f_y_array, long[:,:] ER_intact)
	cpdef generate_c_c_intxn_add_vac_force(self, double[:, :] Fvx, double[:, :] Fvy, double[:,:,:] r_matrix, double[:,:] xy_vacuole_min_mi_delt, double[:,:] xy_vacuole_max_pl_delt, double[:] rx, double[:] ry, long[:] axis_1_plus_1, double DELTA)
	cpdef generate_fp_c_intxn_add_string_force(self, double[:,:] Fvx, double[:,:] Fvy, double[:,:,:] r_matrix, double[:,:] f_x_array, double[:,:] f_y_array, double[:,:] xy_vacuole_min_mi_delt, double[:,:] xy_vacuole_max_pl_delt, long[:,:] connection_table, double DELTA, long[:] axis_1_plus_1)
	cpdef find_within_canvas_and_wall_force(self, double[:] xs, double[:] ys, double delta, long n_site, double[:,:] xy, double[:,:] xy2, double[:] a, double[:] b, double[:] c, double[:] M, int[:] center_sign_to_border)
	cpdef (double, double) minmax(self, double[:] arr)
	cpdef rhs_cython(self, double[:,:,:] r_matrix, double[:] rx, double[:] ry, double[:,:] f_x_array, double[:,:] f_y_array
        , double[:] f_x_array_flat, double[:] f_y_array_flat
        , long[:,:] connection_table, long[:,:] connection_vertex, long[:] axis_1_plus_1, long[:] axis_1_minus_1, long[:] axis_1_plus_2
        , double[:] area0_arr, double[:] l0_arr, double[:,:] xy, double[:,:] xy2, double[:] a, double[:] b, double[:] c, double[:] M, 
        int[:] center_sign_to_border, long[:,:] ER_intact, double[:] vacuole_bending_prefactor)
