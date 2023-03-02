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
    def __init__(self, long Nc, long Nv, long Nf, long Np, long n_corners, double kr, double kr_b, double ka, double gamma, double kl_c_pos_Nv, double kl_c_neg_Nv, double kl, double filament_bending_prefactor, double kr_bf, double six_pi_eta_r, double delt, double dLf, double eta):
        self.Nc = Nc
        self.Nv = Nv
        self.Nf = Nf
        self.Np = Np
        self.n_corners = n_corners
        self.kr = kr
        self.kr_b = kr_b
        self.ka = ka
        self.gamma = gamma
        self.kl_c_pos_Nv = kl_c_pos_Nv
        self.kl_c_neg_Nv = kl_c_neg_Nv
        self.kl = kl
        self.filament_bending_prefactor = filament_bending_prefactor
        self.kr_bf = kr_bf
        self.six_pi_eta_r = six_pi_eta_r
        self.delt = delt
        self.dLf = dLf
        self.eta = eta

    cpdef compute_Fa(self, double[:,:,:] r_matrix, long[:] axis_1_minus_1, long[:] axis_1_plus_1, double[:] area0_arr):
        cdef double[:,:] Fax = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef double[:,:] Fay = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef int ii, vv
        cdef double area_m_area0
        for ii in range(self.Nc):
            area_m_area0 = 0.0
            for vv in range(self.Nv):
                area_m_area0 += r_matrix[0, ii, vv]*r_matrix[1, ii, axis_1_minus_1[vv]]
                area_m_area0 -= r_matrix[0, ii, axis_1_minus_1[vv]]*r_matrix[1, ii, vv] ###
            area_m_area0 *= 0.5
            area_m_area0 -= area0_arr[ii]
            for vv in range(self.Nv):
                Fax[ii,vv] = (-r_matrix[1, ii, axis_1_minus_1[vv]] + r_matrix[1, ii, axis_1_plus_1[vv]])*area_m_area0*self.ka/2 ####
                Fay[ii,vv] = (-r_matrix[0, ii, axis_1_plus_1[vv]] + r_matrix[0, ii, axis_1_minus_1[vv]])*area_m_area0*self.ka/2
        return Fax, Fay

    cpdef compute_segment_length_Fb_Flc_Fgamma(self, double[:,:,:] r_matrix, double[:] l0_arr, double[:] vacuole_bending_prefactor, long[:] axis_1_plus_1, long[:] axis_1_minus_1, long[:] axis_1_plus_2):
        cdef double[:,:] lx_arr = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef double[:,:] ly_arr = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef double[:,:] l_arr = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef double[:,:] lx_div_l = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef double[:,:] ly_div_l = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef double[:,:] l_m_l0_arr = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef double[:,:] Fbx = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef double[:,:] Fby = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef double[:,:] Flcx = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef double[:,:] Flcy = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef double[:,:] Fgx = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef double[:,:] Fgy = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef int[:,:] neg = np.zeros((self.Nc, self.Nv), dtype = np.int32)
        cdef double lx, ly, l
        cdef int ii, cc, vv

        for cc in range(self.Nc):
            for vv in range(self.Nv):
                lx = r_matrix[0, cc, axis_1_minus_1[vv]] - r_matrix[0, cc, vv]
                ly = r_matrix[1, cc, axis_1_minus_1[vv]] - r_matrix[1, cc, vv]
                lx_arr[cc,vv] = lx
                ly_arr[cc,vv] = ly
                l = sqrt(lx**2 + ly**2)
                lx_div_l[cc,vv] = lx/l
                ly_div_l[cc,vv] = ly/l
                l_arr[cc,vv] = l
                l_m_l0_arr[cc,vv] = l - l0_arr[cc]
                if l_m_l0_arr[cc,vv] < 0:
                    neg[cc,vv] = 1

        for ii in range(self.Nc):
            for vv in range(self.Nv):
                Fbx[ii,vv] = vacuole_bending_prefactor[ii]*(3*lx_arr[ii,vv] - 3*lx_arr[ii, axis_1_plus_1[vv]]
                    + lx_arr[ii, axis_1_plus_2[vv]] - lx_arr[ii, axis_1_minus_1[vv]])
                Fby[ii,vv] = vacuole_bending_prefactor[ii]*(3*ly_arr[ii,vv] - 3*ly_arr[ii, axis_1_plus_1[vv]] 
                    + ly_arr[ii, axis_1_plus_2[vv]] - ly_arr[ii, axis_1_minus_1[vv]])
                Fgx[ii,vv] = self.gamma*(lx_div_l[ii,vv] - lx_div_l[ii, axis_1_plus_1[vv]])
                Fgy[ii,vv] = self.gamma*(ly_div_l[ii,vv] - ly_div_l[ii, axis_1_plus_1[vv]])
                if neg[ii,vv] > 0:
                    Flcx[ii,vv] += self.kl_c_neg_Nv*l_m_l0_arr[ii,vv]*lx_div_l[ii,vv]
                    Flcy[ii,vv] += self.kl_c_neg_Nv*l_m_l0_arr[ii,vv]*ly_div_l[ii,vv]
                else:
                    Flcx[ii,vv] += self.kl_c_pos_Nv*l_m_l0_arr[ii,vv]*lx_div_l[ii,vv]
                    Flcy[ii,vv] += self.kl_c_pos_Nv*l_m_l0_arr[ii,vv]*ly_div_l[ii,vv]

                if neg[ii,axis_1_plus_1[vv]] > 0:
                    Flcx[ii,vv] -= self.kl_c_neg_Nv*l_m_l0_arr[ii, axis_1_plus_1[vv]]*lx_div_l[ii, axis_1_plus_1[vv]]
                    Flcy[ii,vv] -= self.kl_c_neg_Nv*l_m_l0_arr[ii, axis_1_plus_1[vv]]*ly_div_l[ii, axis_1_plus_1[vv]]
                else:
                    Flcx[ii,vv] -= self.kl_c_pos_Nv*l_m_l0_arr[ii, axis_1_plus_1[vv]]*lx_div_l[ii, axis_1_plus_1[vv]]
                    Flcy[ii,vv] -= self.kl_c_pos_Nv*l_m_l0_arr[ii, axis_1_plus_1[vv]]*ly_div_l[ii, axis_1_plus_1[vv]]
        return Fbx, Fby, Flcx, Flcy, Fgx, Fgy

    cpdef get_separation_vectors_Fl_Fbf(self, double[:,:] f_x_array, double[:,:] f_y_array, long[:,:] ER_intact):
        cdef int Np_m1 = self.Np - 1
        cdef int ff, pp
        cdef double FX, FY
        cdef double[:,:] Flx = np.zeros((self.Nf, self.Np), dtype = np.double)
        cdef double[:,:] Fly = np.zeros((self.Nf, self.Np), dtype = np.double)
        cdef double[:,:] Fbfx = np.zeros((self.Nf, self.Np), dtype = np.double)
        cdef double[:,:] Fbfy = np.zeros((self.Nf, self.Np), dtype = np.double)
        cdef double[:,:] dx = np.zeros((self.Nf, Np_m1), dtype = np.double)
        cdef double[:,:] dy = np.zeros((self.Nf, Np_m1), dtype = np.double)
        cdef double[:,:] dr = np.zeros((self.Nf, Np_m1), dtype = np.double)
        cdef double[:,:] Fl_expanded = np.zeros((self.Nf, Np_m1), dtype = np.double)

        for ff in range(self.Nf):
            for pp in range(self.Np-1):
                dx[ff,pp] = f_x_array[ff, pp+1] - f_x_array[ff, pp]
                dy[ff,pp] = f_y_array[ff, pp+1] - f_y_array[ff, pp]
                dr[ff,pp] = sqrt(dx[ff,pp]**2 + dy[ff,pp]**2)
                Fl_expanded[ff,pp] = self.kl*Np_m1*(dr[ff,pp] - self.dLf)/dr[ff,pp]

        for ff in range(self.Nf):
            for pp in range(self.Np-2):
                if ER_intact[ff, pp] != 0:
                    FX = Fl_expanded[ff, pp]*dx[ff, pp]
                    FY = Fl_expanded[ff, pp]*dy[ff, pp]
                    Flx[ff, pp] += FX
                    Fly[ff, pp] += FY
                    Flx[ff, pp+1] -= FX
                    Fly[ff, pp+1] -= FY
                    if ER_intact[ff, pp+1] != 0:
                        FX = self.filament_bending_prefactor*(dx[ff,pp] - dx[ff,pp+1])
                        FY = self.filament_bending_prefactor*(dy[ff,pp] - dy[ff,pp+1])
                        Fbfx[ff, pp] += FX
                        Fbfy[ff, pp] += FY
                        Fbfx[ff, pp+1] -= 2*FX
                        Fbfy[ff, pp+1] -= 2*FY
                        Fbfx[ff, pp+2] += FX
                        Fbfy[ff, pp+2] += FY

            pp = self.Np-2
            if ER_intact[ff, pp] != 0:
                FX = Fl_expanded[ff, pp]*dx[ff, pp]
                FY = Fl_expanded[ff, pp]*dy[ff, pp]
                Flx[ff, pp] += FX
                Fly[ff, pp] += FY
                Flx[ff, pp+1] -= FX
                Fly[ff, pp+1] -= FY

        return Flx, Fly, Fbfx, Fbfy

    cpdef generate_c_c_intxn_add_vac_force(self, double[:, :] Fvx, double[:, :] Fvy, double[:,:,:] r_matrix, double[:,:] xy_vacuole_min_mi_delt, double[:,:] xy_vacuole_max_pl_delt, double[:] rx, double[:] ry, long[:] axis_1_plus_1, double DELTA):
        cdef int c1, v, c2, ii, jj, sign, kk 
        cdef double DX, DY, DR, DELTA_eff_M_DR, a, b, c, M, N, dist_perp, min_x_vac, max_x_vac, min_y_vac, max_y_vac, xp, yp
        cdef double fx, fy, f0x, f0y, f1x, f1y
        cdef double xy_x, xy_y, xy2_x, xy2_y, xs, ys

        for c1 in range(self.Nc):
            for c2 in range(self.Nc):
                if c1 == c2:
                    pass
                else:
                    for v in range(self.Nv):
                        xs = r_matrix[0, c1, v]
                        ys = r_matrix[1, c1, v]
                        if xs >= xy_vacuole_min_mi_delt[c2, 0]:
                            if xs <= xy_vacuole_max_pl_delt[c2, 0]:
                                if ys >= xy_vacuole_min_mi_delt[c2, 1]:
                                    if ys <= xy_vacuole_max_pl_delt[c2, 1]:
                                        for ii in range(self.Nv):
                                            kk = axis_1_plus_1[ii]
                                            xy_x = r_matrix[0, c2, ii]
                                            xy_y = r_matrix[1, c2, ii]
                                            xy2_x = r_matrix[0, c2, kk]
                                            xy2_y = r_matrix[1, c2, kk]

                                            DX = (xs - xy_x)
                                            DY = (ys - xy_y)
                                            DR = sqrt(DX**2 + DY**2)
                                            DELTA_eff_M_DR = DELTA - DR
                                            if DELTA_eff_M_DR > 0:
                                                fx = self.kr*(DX/(DR+1E-10)*(DELTA_eff_M_DR))
                                                fy = self.kr*(DY/(DR+1E-10)*(DELTA_eff_M_DR))
                                                Fvx[c2, ii] -= fx
                                                Fvy[c2, ii] -= fy
                                                Fvx[c1, v] += fx
                                                Fvy[c1, v] += fy

                                            a = xy2_y - xy_y
                                            b = - (xy2_x - xy_x)
                                            c = xy_y*(xy2_x - xy_x) - xy_x*(xy2_y - xy_y)
                                            M = a**2 + b**2
                                            N = b*xs - a*ys

                                            dist_perp = (a*xs + b*ys + c)/sqrt(M)
                                            if dist_perp >= 0:
                                                sign = 1
                                            else:
                                                sign = -1

                                            dist_perp = abs(dist_perp)

                                            min_x_vac = min(xy_x, xy2_x)
                                            max_x_vac = max(xy_x, xy2_x)
                                            min_y_vac = min(xy_y, xy2_y)
                                            max_y_vac = max(xy_y, xy2_y)

                                            xp = (b*N - a*c)/M
                                            yp = (a*(-N) - b*c)/M
                                            DELTA_eff_M_DR = DELTA - dist_perp
                                            if DELTA_eff_M_DR > 0:
                                                if xp >= min_x_vac:
                                                    if xp <= max_x_vac:
                                                        if yp >= min_y_vac:
                                                            if yp <= max_y_vac:
                                                                TF = True
                                                            else:
                                                                TF = False
                                                        else:
                                                            TF = False
                                                    else:
                                                        TF = False
                                                else:
                                                    TF = False
                                            else:
                                                TF = False
                                                
                                            if TF:
                                                fx = self.kr*((xs - xp)/(dist_perp+1E-10)*(DELTA_eff_M_DR))
                                                fy = self.kr*((ys - yp)/(dist_perp+1E-10)*(DELTA_eff_M_DR))
                                                f0x = 2*self.kr*DELTA_eff_M_DR*((-b)*dist_perp/M + (ys - xy2_y)*sign/sqrt(M))
                                                f0y = 2*self.kr*DELTA_eff_M_DR*((xy2_x - xs)*sign/sqrt(M) + a*dist_perp/M)
                                                f1x = 2*self.kr*DELTA_eff_M_DR*(b*dist_perp/M + (xy_y - ys)*sign/sqrt(M))
                                                f1y = 2*self.kr*DELTA_eff_M_DR*((-xy_x + xs)*sign/sqrt(M) - a*dist_perp/M)
                                                Fvx[c1, v] += fx
                                                Fvy[c1, v] += fy
                                                Fvx[c2,ii] += f0x
                                                Fvy[c2,ii] += f0y
                                                Fvx[c2,kk] += f1x
                                                Fvy[c2,kk] += f1y
        return Fvx, Fvy

    cpdef generate_fp_c_intxn_add_string_force(self, double[:,:] Fvx, double[:,:] Fvy, double[:,:,:] r_matrix, double[:,:] f_x_array, double[:,:] f_y_array, double[:,:] xy_vacuole_min_mi_delt, double[:,:] xy_vacuole_max_pl_delt, long[:,:] connection_table, double DELTA, long[:] axis_1_plus_1):
        cdef int f, p, cc, ii, sign, kk
        cdef double xs, ys, xy_x, xy_y, xy2_x, xy2_y
        cdef double[:,:] Frx = np.zeros((self.Nf, self.Np), dtype = np.double)
        cdef double[:,:] Fry = np.zeros((self.Nf, self.Np), dtype = np.double)
        cdef double DX, DY, DR, DELTA_eff_M_DR, a, b, c, M, N, dist_perp, min_x_vac, max_x_vac, min_y_vac, max_y_vac, xp, yp
        cdef double fx, fy, f0x, f0y, f1x, f1y
        for f in range(self.Nf):
            for cc in range(self.Nc):
                if cc == connection_table[f, 0]:
                    for p in range(1, self.Np):
                        if f_x_array[f,p] >= xy_vacuole_min_mi_delt[cc, 0]:
                            if f_x_array[f,p] <= xy_vacuole_max_pl_delt[cc, 0]:
                                if f_y_array[f,p] >= xy_vacuole_min_mi_delt[cc, 1]:
                                    if f_y_array[f,p] <= xy_vacuole_max_pl_delt[cc, 1]:
                                        xs = f_x_array[f,p]
                                        ys = f_y_array[f,p]
                                        for ii in range(self.Nv):
                                            kk = axis_1_plus_1[ii]
                                            xy_x = r_matrix[0, cc, ii]
                                            xy_y = r_matrix[1, cc, ii]
                                            xy2_x = r_matrix[0, cc, kk]
                                            xy2_y = r_matrix[1, cc, kk]

                                            DX = (xs - xy_x)
                                            DY = (ys - xy_y)
                                            DR = sqrt(DX**2 + DY**2)
                                            DELTA_eff_M_DR = DELTA - DR
                                            if DELTA_eff_M_DR > 0:
                                                fx = self.kr*(DX/(DR+1E-10)*(DELTA_eff_M_DR))
                                                fy = self.kr*(DY/(DR+1E-10)*(DELTA_eff_M_DR))
                                                Fvx[cc, ii] -= fx
                                                Fvy[cc, ii] -= fy
                                                Frx[f,p] += fx
                                                Fry[f,p] += fy

                                            a = xy2_y - xy_y
                                            b = - (xy2_x - xy_x)
                                            c = xy_y*(xy2_x - xy_x) - xy_x*(xy2_y - xy_y)
                                            M = a**2 + b**2
                                            N = b*xs - a*ys

                                            dist_perp = (a*xs + b*ys + c)/sqrt(M)
                                            if dist_perp >= 0:
                                                sign = 1
                                            else:
                                                sign = -1

                                            dist_perp = abs(dist_perp)
                                            min_x_vac = min(xy_x, xy2_x)
                                            max_x_vac = max(xy_x, xy2_x)
                                            min_y_vac = min(xy_y, xy2_y)
                                            max_y_vac = max(xy_y, xy2_y)

                                            xp = (b*N - a*c)/M
                                            yp = (a*(-N) - b*c)/M
                                            DELTA_eff_M_DR = DELTA - dist_perp
                                            if DELTA_eff_M_DR > 0:
                                                if xp >= min_x_vac:
                                                    if xp <= max_x_vac:
                                                        if yp >= min_y_vac:
                                                            if yp <= max_y_vac:
                                                                TF = True
                                                            else:
                                                                TF = False
                                                        else:
                                                            TF = False
                                                    else:
                                                        TF = False
                                                else:
                                                    TF = False
                                            else:
                                                TF = False
                                                
                                            if TF:
                                                fx = self.kr*((xs - xp)/(dist_perp+1E-10)*(DELTA_eff_M_DR))
                                                fy = self.kr*((ys - yp)/(dist_perp+1E-10)*(DELTA_eff_M_DR))
                                                f0x = 2*self.kr*DELTA_eff_M_DR*((-b)*dist_perp/M + (ys - xy2_y)*sign/sqrt(M))
                                                f0y = 2*self.kr*DELTA_eff_M_DR*((xy2_x - xs)*sign/sqrt(M) + a*dist_perp/M)
                                                f1x = 2*self.kr*DELTA_eff_M_DR*(b*dist_perp/M + (xy_y - ys)*sign/sqrt(M))
                                                f1y = 2*self.kr*DELTA_eff_M_DR*((-xy_x + xs)*sign/sqrt(M) - a*dist_perp/M)
                                                Frx[f,p] += fx
                                                Fry[f,p] += fy
                                                Fvx[cc,ii] += f0x
                                                Fvy[cc,ii] += f0y
                                                Fvx[cc,kk] += f1x
                                                Fvy[cc,kk] += f1y

                elif cc == connection_table[f, 1]:
                    for p in range(self.Np-1):
                        if f_x_array[f,p] >= xy_vacuole_min_mi_delt[cc, 0]:
                            if f_x_array[f,p] <= xy_vacuole_max_pl_delt[cc, 0]:
                                if f_y_array[f,p] >= xy_vacuole_min_mi_delt[cc, 1]:
                                    if f_y_array[f,p] <= xy_vacuole_max_pl_delt[cc, 1]:
                                        xs = f_x_array[f,p]
                                        ys = f_y_array[f,p]
                                        for ii in range(self.Nv):
                                            kk = axis_1_plus_1[ii]
                                            xy_x = r_matrix[0, cc, ii]
                                            xy_y = r_matrix[1, cc, ii]
                                            xy2_x = r_matrix[0, cc, kk]
                                            xy2_y = r_matrix[1, cc, kk]

                                            DX = (xs - xy_x)
                                            DY = (ys - xy_y)
                                            DR = sqrt(DX**2 + DY**2)
                                            DELTA_eff_M_DR = DELTA - DR
                                            if DELTA_eff_M_DR > 0:
                                                fx = self.kr*(DX/(DR+1E-10)*(DELTA_eff_M_DR))
                                                fy = self.kr*(DY/(DR+1E-10)*(DELTA_eff_M_DR))
                                                Fvx[cc, ii] -= fx
                                                Fvy[cc, ii] -= fy
                                                Frx[f,p] += fx
                                                Fry[f,p] += fy

                                            a = xy2_y - xy_y
                                            b = - (xy2_x - xy_x)
                                            c = xy_y*(xy2_x - xy_x) - xy_x*(xy2_y - xy_y)
                                            M = a**2 + b**2
                                            N = b*xs - a*ys

                                            dist_perp = (a*xs + b*ys + c)/sqrt(M)
                                            if dist_perp >= 0:
                                                sign = 1
                                            else:
                                                sign = -1

                                            dist_perp = abs(dist_perp)
                                            min_x_vac = min(xy_x, xy2_x)
                                            max_x_vac = max(xy_x, xy2_x)
                                            min_y_vac = min(xy_y, xy2_y)
                                            max_y_vac = max(xy_y, xy2_y)

                                            xp = (b*N - a*c)/M
                                            yp = (a*(-N) - b*c)/M
                                            DELTA_eff_M_DR = DELTA - dist_perp
                                            if DELTA_eff_M_DR > 0:
                                                if xp >= min_x_vac:
                                                    if xp <= max_x_vac:
                                                        if yp >= min_y_vac:
                                                            if yp <= max_y_vac:
                                                                TF = True
                                                            else:
                                                                TF = False
                                                        else:
                                                            TF = False
                                                    else:
                                                        TF = False
                                                else:
                                                    TF = False
                                            else:
                                                TF = False
                                                
                                            if TF:
                                                fx = self.kr*((xs - xp)/(dist_perp+1E-10)*(DELTA_eff_M_DR))
                                                fy = self.kr*((ys - yp)/(dist_perp+1E-10)*(DELTA_eff_M_DR))
                                                f0x = 2*self.kr*DELTA_eff_M_DR*((-b)*dist_perp/M + (ys - xy2_y)*sign/sqrt(M))
                                                f0y = 2*self.kr*DELTA_eff_M_DR*((xy2_x - xs)*sign/sqrt(M) + a*dist_perp/M)
                                                f1x = 2*self.kr*DELTA_eff_M_DR*(b*dist_perp/M + (xy_y - ys)*sign/sqrt(M))
                                                f1y = 2*self.kr*DELTA_eff_M_DR*((-xy_x + xs)*sign/sqrt(M) - a*dist_perp/M)
                                                Frx[f,p] += fx
                                                Fry[f,p] += fy
                                                Fvx[cc,ii] += f0x
                                                Fvy[cc,ii] += f0y
                                                Fvx[cc,kk] += f1x
                                                Fvy[cc,kk] += f1y
                else:
                    for p in range(self.Np):
                        if f_x_array[f,p] >= xy_vacuole_min_mi_delt[cc, 0]:
                            if f_x_array[f,p] <= xy_vacuole_max_pl_delt[cc, 0]:
                                if f_y_array[f,p] >= xy_vacuole_min_mi_delt[cc, 1]:
                                    if f_y_array[f,p] <= xy_vacuole_max_pl_delt[cc, 1]:
                                        xs = f_x_array[f,p]
                                        ys = f_y_array[f,p]
                                        for ii in range(self.Nv):
                                            kk = axis_1_plus_1[ii]
                                            xy_x = r_matrix[0, cc, ii]
                                            xy_y = r_matrix[1, cc, ii]
                                            xy2_x = r_matrix[0, cc, kk]
                                            xy2_y = r_matrix[1, cc, kk]

                                            DX = (xs - xy_x)
                                            DY = (ys - xy_y)
                                            DR = sqrt(DX**2 + DY**2)
                                            DELTA_eff_M_DR = DELTA - DR
                                            if DELTA_eff_M_DR > 0:
                                                fx = self.kr*(DX/(DR+1E-10)*(DELTA_eff_M_DR))
                                                fy = self.kr*(DY/(DR+1E-10)*(DELTA_eff_M_DR))
                                                Fvx[cc, ii] -= fx
                                                Fvy[cc, ii] -= fy
                                                Frx[f,p] += fx
                                                Fry[f,p] += fy

                                            a = xy2_y - xy_y
                                            b = - (xy2_x - xy_x)
                                            c = xy_y*(xy2_x - xy_x) - xy_x*(xy2_y - xy_y)
                                            M = a**2 + b**2
                                            N = b*xs - a*ys

                                            dist_perp = (a*xs + b*ys + c)/sqrt(M)
                                            if dist_perp >= 0:
                                                sign = 1
                                            else:
                                                sign = -1

                                            dist_perp = abs(dist_perp)
                                            min_x_vac = min(xy_x, xy2_x)
                                            max_x_vac = max(xy_x, xy2_x)
                                            min_y_vac = min(xy_y, xy2_y)
                                            max_y_vac = max(xy_y, xy2_y)

                                            xp = (b*N - a*c)/M
                                            yp = (a*(-N) - b*c)/M
                                            DELTA_eff_M_DR = DELTA - dist_perp
                                            if DELTA_eff_M_DR > 0:
                                                if xp >= min_x_vac:
                                                    if xp <= max_x_vac:
                                                        if yp >= min_y_vac:
                                                            if yp <= max_y_vac:
                                                                TF = True
                                                            else:
                                                                TF = False
                                                        else:
                                                            TF = False
                                                    else:
                                                        TF = False
                                                else:
                                                    TF = False
                                            else:
                                                TF = False
                                                
                                            if TF:
                                                fx = self.kr*((xs - xp)/(dist_perp+1E-10)*(DELTA_eff_M_DR))
                                                fy = self.kr*((ys - yp)/(dist_perp+1E-10)*(DELTA_eff_M_DR))
                                                f0x = 2*self.kr*DELTA_eff_M_DR*((-b)*dist_perp/M + (ys - xy2_y)*sign/sqrt(M))
                                                f0y = 2*self.kr*DELTA_eff_M_DR*((xy2_x - xs)*sign/sqrt(M) + a*dist_perp/M)
                                                f1x = 2*self.kr*DELTA_eff_M_DR*(b*dist_perp/M + (xy_y - ys)*sign/sqrt(M))
                                                f1y = 2*self.kr*DELTA_eff_M_DR*((-xy_x + xs)*sign/sqrt(M) - a*dist_perp/M)
                                                Frx[f,p] += fx
                                                Fry[f,p] += fy
                                                Fvx[cc,ii] += f0x
                                                Fvy[cc,ii] += f0y
                                                Fvx[cc,kk] += f1x
                                                Fvy[cc,kk] += f1y
        return Frx, Fry, Fvx, Fvy

    cpdef find_within_canvas_and_wall_force(self, double[:] xs, double[:] ys, double delta, long n_site, double[:,:] xy, double[:,:] xy2, double[:] a, double[:] b, double[:] c, double[:] M, int[:] center_sign_to_border):
        cdef double v1x, v1y, v2x, v2y, ang, N, xp, yp, dist_perp, delta_m_dist
        cdef int ii, cc, sign
        cdef double[:] f_x = np.zeros(n_site, dtype = np.double)
        cdef double[:] f_y = np.zeros(n_site, dtype = np.double)
        for ii in range(n_site):
            ang = 0
            for cc in range(self.n_corners):
                v1x = xy[cc,0] - xs[ii]
                v1y = xy[cc,1] - ys[ii]
                v2x = xy2[cc,0] - xs[ii]
                v2y = xy2[cc,1] - ys[ii]
                ang += atan2(-v2x*v1y + v2y*v1x, v2x*v1x + v2y*v1y)
            if roundc((abs(ang) - 2*PI)*10**9) == 0:
                pass
                # within canvas
                for cc in range(self.n_corners):
                    N = b[cc]*xs[ii] - a[cc]*ys[ii]
                    xp = (b[cc]*N - a[cc]*c[cc])/M[cc]
                    yp = (-a[cc]*N - b[cc]*c[cc])/M[cc]
                    dist_perp = abs(a[cc]*xs[ii] + b[cc]*ys[ii] + c[cc])/sqrt(M[cc])
                    delta_m_dist = delta - dist_perp
                    if delta_m_dist > 0:
                        f_x[ii] += (xs[ii] - xp)/(dist_perp+1E-10)*delta_m_dist
                        f_y[ii] += (ys[ii] - yp)/(dist_perp+1E-10)*delta_m_dist
            else:
                # not within canvas
                for cc in range(self.n_corners):
                    N = b[cc]*xs[ii] - a[cc]*ys[ii]
                    xp = (b[cc]*N - a[cc]*c[cc])/M[cc]
                    yp = (-a[cc]*N - b[cc]*c[cc])/M[cc]
                    dist_perp = (a[cc]*xs[ii] + b[cc]*ys[ii] + c[cc])/sqrt(M[cc])
                    if dist_perp > 0:
                        sign = 1
                    else:
                        sign = 0
                    if sign != center_sign_to_border[cc]:
                        dist_perp = abs(dist_perp)
                        f_x[ii] -= (xs[ii] - xp)/(dist_perp+1E-10)*(dist_perp+delta)
                        f_y[ii] -= (ys[ii] - yp)/(dist_perp+1E-10)*(dist_perp+delta)
        return f_x, f_y

    cpdef (double, double) minmax(self, double[:] arr):
        cdef double mini = 1000.
        cdef double maxi = -1000.
        cdef int i
        for i in range(arr.shape[0]):
            if arr[i] < mini:
                mini = arr[i]
            if arr[i] > maxi:
                maxi = arr[i]
        return mini, maxi

    cpdef rhs_cython(self, double[:,:,:] r_matrix, double[:] rx, double[:] ry, double[:,:] f_x_array, double[:,:] f_y_array
        , double[:] f_x_array_flat, double[:] f_y_array_flat
        , long[:,:] connection_table, long[:,:] connection_vertex, long[:] axis_1_plus_1, long[:] axis_1_minus_1, long[:] axis_1_plus_2
        , double[:] area0_arr, double[:] l0_arr, double[:,:] xy, double[:,:] xy2, double[:] a, double[:] b, double[:] c, double[:] M, 
        int[:] center_sign_to_border, long[:,:] ER_intact, double[:] vacuole_bending_prefactor):

        cdef double[:,:] Fvx = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef double[:,:] Fvy = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef double[:,:] Fax 
        cdef double[:,:] Fay
        cdef double[:,:] Fbx
        cdef double[:,:] Fby
        cdef double[:,:] Flcx 
        cdef double[:,:] Flcy 
        cdef double[:,:] Fgx
        cdef double[:,:] Fgy #= np.zeros((self.Nc, self.Nv), dtype = double)
        cdef double[:] fx
        cdef double[:] fy
        cdef double[:] drEdt = np.zeros((2*self.Nc*self.Nv+2*self.Nf*self.Np), dtype = np.double)

        cdef double[:,:] xy_vacuole_min_m_delt = np.zeros((self.Nc, 2), dtype = np.double)
        cdef double[:,:] xy_vacuole_max_p_delt = np.zeros((self.Nc, 2), dtype = np.double)
        cdef int cc, ff, vv, pp

        cdef double[:,:] Flx = np.zeros((self.Nf, self.Np), dtype = np.double)
        cdef double[:,:] Fly = np.zeros((self.Nf, self.Np), dtype = np.double)
        cdef double[:,:] Frx = np.zeros((self.Nf, self.Np), dtype = np.double)
        cdef double[:,:] Fry = np.zeros((self.Nf, self.Np), dtype = np.double)
        cdef double[:,:] Fbfx = np.zeros((self.Nf, self.Np), dtype = np.double)
        cdef double[:,:] Fbfy = np.zeros((self.Nf, self.Np), dtype = np.double)
        cdef double[:] ffx = np.zeros((self.Nf*self.Np), dtype = np.double)
        cdef double[:] ffy = np.zeros((self.Nf*self.Np), dtype = np.double)
            
        # Forces on vacuoles
        Fax, Fay = self.compute_Fa(r_matrix, axis_1_minus_1, axis_1_plus_1, area0_arr)
        Fbx, Fby, Flcx, Flcy, Fgx, Fgy = self.compute_segment_length_Fb_Flc_Fgamma(r_matrix, 
            l0_arr, vacuole_bending_prefactor, axis_1_plus_1, axis_1_minus_1, axis_1_plus_2)

        # Forces on vacuoles among vacuoles
        for cc in range(self.Nc):
            xy_vacuole_min_m_delt[cc, 0], xy_vacuole_max_p_delt[cc, 0] = self.minmax(r_matrix[0, cc, 0:self.Nv])
            xy_vacuole_min_m_delt[cc, 1], xy_vacuole_max_p_delt[cc, 1] = self.minmax(r_matrix[1, cc, 0:self.Nv])
            xy_vacuole_min_m_delt[cc, 0] -= self.delt
            xy_vacuole_min_m_delt[cc, 1] -= self.delt
            xy_vacuole_max_p_delt[cc, 0] += self.delt
            xy_vacuole_max_p_delt[cc, 1] += self.delt

        Fvx, Fvy = self.generate_c_c_intxn_add_vac_force(Fvx, Fvy, r_matrix, xy_vacuole_min_m_delt,
                                           xy_vacuole_max_p_delt, rx, ry, axis_1_plus_1, self.delt)

        fx, fy = self.find_within_canvas_and_wall_force(rx, ry, self.delt/2, self.Nc*self.Nv, xy, xy2, 
                a, b, c, M, center_sign_to_border)

        if self.Nf > 0:
            # Forces on filaments
            Flx, Fly, Fbfx, Fbfy = self.get_separation_vectors_Fl_Fbf(f_x_array, f_y_array, ER_intact)
            Frx, Fry, Fvx, Fvy = self.generate_fp_c_intxn_add_string_force(Fvx, Fvy, 
                    r_matrix, f_x_array, f_y_array, xy_vacuole_min_m_delt, xy_vacuole_max_p_delt, 
                    connection_table, self.delt/2, axis_1_plus_1)

        if self.Nf > 0:
            # Forces on vacuoles from the end of the filaments
            for ff in range(self.Nf):
                Fvx[connection_table[ff, 0], connection_vertex[ff, 0]] += Flx[ff, 0] + Fbfx[ff, 0] + Frx[ff, 0]
                Fvx[connection_table[ff, 1], connection_vertex[ff, 1]] += Flx[ff, self.Np-1] + Fbfx[ff, self.Np-1] + Frx[ff, self.Np-1]
                Fvy[connection_table[ff, 0], connection_vertex[ff, 0]] += Fly[ff, 0] + Fbfy[ff, 0] + Fry[ff, 0]
                Fvy[connection_table[ff, 1], connection_vertex[ff, 1]] += Fly[ff, self.Np-1] + Fbfy[ff, self.Np-1] + Fry[ff, self.Np-1]
        
        for cc in range(self.Nc):
            for vv in range(self.Nv):
                drEdt[self.Nv*cc+vv] = (Fax[cc,vv] + Fbx[cc,vv] + Flcx[cc,vv] + Fgx[cc,vv] + Fvx[cc,vv] + self.kr_b*self.kr*fx[cc*self.Nv+vv])/self.six_pi_eta_r
                drEdt[self.Nc*self.Nv + self.Nv*cc+vv] = (Fay[cc,vv] + Fby[cc,vv] + Flcy[cc,vv] + Fgy[cc,vv] + Fvy[cc,vv] + self.kr_b*self.kr*fy[cc*self.Nv+vv])/self.six_pi_eta_r

        # Forces on ER from the boundaries
        if (self.kr_bf > 0) and (self.Nf > 0):
            ffx, ffy = self.find_within_canvas_and_wall_force(
                    f_x_array_flat, f_y_array_flat, 0.0, self.Nf*self.Np, 
                    xy, xy2, a, b, c, M, center_sign_to_border)

        if self.Nf > 0:
            if self.kr_bf > 0:
                for ff in range(self.Nf):
                    pp = 0
                    drEdt[2*self.Nc*self.Nv+self.Np*ff+pp] = drEdt[connection_table[ff, 0]*self.Nv+connection_vertex[ff, 0]]
                    drEdt[2*self.Nc*self.Nv+self.Np*self.Nf+self.Np*ff+pp] = drEdt[self.Nc*self.Nv+connection_table[ff, 0]*self.Nv+connection_vertex[ff, 0]]

                    for pp in range(1, self.Np-1):
                        drEdt[2*self.Nc*self.Nv+self.Np*ff+pp] = (Flx[ff,pp] + Frx[ff,pp] + Fbfx[ff,pp] + self.kr_bf*self.kr*ffx[ff*self.Np+pp])/(self.eta*self.dLf)
                        drEdt[2*self.Nc*self.Nv+self.Np*self.Nf+self.Np*ff+pp] = (Fly[ff,pp] + Fry[ff,pp] + Fbfy[ff,pp] + self.kr_bf*self.kr*ffy[ff*self.Np+pp])/(self.eta*self.dLf)
                        
                    pp = self.Np-1
                    drEdt[2*self.Nc*self.Nv+self.Np*ff+pp] = drEdt[connection_table[ff, 1]*self.Nv+connection_vertex[ff, 1]]
                    drEdt[2*self.Nc*self.Nv+self.Np*self.Nf+self.Np*ff+pp] = drEdt[self.Nc*self.Nv+connection_table[ff, 1]*self.Nv+connection_vertex[ff, 1]]
            else:
                for ff in range(self.Nf):
                    pp = 0
                    drEdt[2*self.Nc*self.Nv+self.Np*ff+pp] = drEdt[connection_table[ff, 0]*self.Nv+connection_vertex[ff, 0]]
                    drEdt[2*self.Nc*self.Nv+self.Np*self.Nf+self.Np*ff+pp] = drEdt[self.Nc*self.Nv+connection_table[ff, 0]*self.Nv+connection_vertex[ff, 0]]

                    for pp in range(1, self.Np-1):
                        drEdt[2*self.Nc*self.Nv+self.Np*ff+pp] = (Flx[ff,pp] + Frx[ff,pp] + Fbfx[ff,pp])/(self.eta*self.dLf)
                        drEdt[2*self.Nc*self.Nv+self.Np*self.Nf+self.Np*ff+pp] = (Fly[ff,pp] + Fry[ff,pp] + Fbfy[ff,pp])/(self.eta*self.dLf)
                        
                    pp = self.Np-1
                    drEdt[2*self.Nc*self.Nv+self.Np*ff+pp] = drEdt[connection_table[ff, 1]*self.Nv+connection_vertex[ff, 1]]
                    drEdt[2*self.Nc*self.Nv+self.Np*self.Nf+self.Np*ff+pp] = drEdt[self.Nc*self.Nv+connection_table[ff, 1]*self.Nv+connection_vertex[ff, 1]]

        return Fvx, Fvy, Fax, Fay, Fbx, Fby, Flcx, Flcy, Fgx, Fgy, Flx, Fly, Frx, Fry, Fbfx, Fbfy, drEdt
