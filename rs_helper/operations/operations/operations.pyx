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
    def __init__(self, long Nv, double kr, long[:] axis_1_plus_1, long Nc, long Nf, long Np): 
        self.Nv = Nv
        self.kr = kr
        self.axis_1_plus_1 = axis_1_plus_1
        self.Nc = Nc
        self.Nf = Nf
        self.Np = Np

    cpdef interact_with_vacuole(self, double[:, :] xy_x, double[:, :] xy_y, double[:, :] xy2_x, double[:, :] xy2_y, double[:, :] xs, double[:, :] ys, double DELTA, int n_pair):
        """
        xy_x, xy_y, xy2_x, xy2_y: shape: (Nv, # of potential interactions)
        xs, ys: shape: (1, # of potential interaction)
        This function is only valid if the particle never enters the vacuole.
        """
        cdef int ii, jj, sign, kk
        cdef double[:,:] Fvx_int = np.zeros((self.Nv, n_pair), dtype = np.double)
        cdef double[:,:] Fvy_int = np.zeros((self.Nv, n_pair), dtype = np.double)
        cdef double[:,:] Frx = np.zeros((1, n_pair), dtype = np.double)
        cdef double[:,:] Fry = np.zeros((1, n_pair), dtype = np.double)
        cdef double DX, DY, DR, DELTA_eff_M_DR, a, b, c, M, N, dist_perp, min_x_vac, max_x_vac, min_y_vac, max_y_vac, xp, yp
        cdef double fx, fy, f0x, f0y, f1x, f1y
        
        for ii in range(self.Nv):
            for jj in range(n_pair):
                DX = (xs[0,jj] - xy_x[ii,jj])
                DY = (ys[0,jj] - xy_y[ii,jj])
                DR = sqrt(DX**2 + DY**2)
                DELTA_eff_M_DR = DELTA - DR
                if DELTA_eff_M_DR > 0:
                    fx = self.kr*(DX/(DR+1E-10)*(DELTA_eff_M_DR))
                    fy = self.kr*(DY/(DR+1E-10)*(DELTA_eff_M_DR))
                    Fvx_int[ii, jj] -= fx
                    Fvy_int[ii, jj] -= fy
                    Frx[0,jj] += fx
                    Fry[0,jj] += fy

                a = xy2_y[ii,jj] - xy_y[ii,jj]
                b = - (xy2_x[ii,jj] - xy_x[ii,jj])
                c = xy_y[ii,jj]*(xy2_x[ii,jj] - xy_x[ii,jj]) - xy_x[ii,jj]*(xy2_y[ii,jj] - xy_y[ii,jj])
                M = a**2 + b**2
                N = b*xs[0,jj] - a*ys[0,jj]

                dist_perp = (a*xs[0,jj] + b*ys[0,jj] + c)/sqrt(M)
                if dist_perp >= 0:
                    sign = 1
                else:
                    sign = -1

                dist_perp = abs(dist_perp)

                min_x_vac = min(xy_x[ii,jj], xy2_x[ii,jj])
                max_x_vac = max(xy_x[ii,jj], xy2_x[ii,jj])
                min_y_vac = min(xy_y[ii,jj], xy2_y[ii,jj])
                max_y_vac = max(xy_y[ii,jj], xy2_y[ii,jj])

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
                    fx = self.kr*((xs[0,jj] - xp)/(dist_perp+1E-10)*(DELTA_eff_M_DR))
                    fy = self.kr*((ys[0,jj] - yp)/(dist_perp+1E-10)*(DELTA_eff_M_DR))
                    f0x = 2*self.kr*DELTA_eff_M_DR*((-b)*dist_perp/M + (ys[0,jj] - xy2_y[ii,jj])*sign/sqrt(M))
                    f0y = 2*self.kr*DELTA_eff_M_DR*((xy2_x[ii,jj] - xs[0,jj])*sign/sqrt(M) + a*dist_perp/M)
                    f1x = 2*self.kr*DELTA_eff_M_DR*(b*dist_perp/M + (xy_y[ii,jj] - ys[0,jj])*sign/sqrt(M))
                    f1y = 2*self.kr*DELTA_eff_M_DR*((-xy_x[ii,jj] + xs[0,jj])*sign/sqrt(M) - a*dist_perp/M)
                    Frx[0,jj] += fx
                    Fry[0,jj] += fy
                    Fvx_int[ii,jj] += f0x
                    Fvy_int[ii,jj] += f0y
                    kk = self.axis_1_plus_1[ii]
                    Fvx_int[kk, jj] += f1x
                    Fvy_int[kk, jj] += f1y
        
        return np.array(Frx), np.array(Fry), np.array(Fvx_int), np.array(Fvy_int)

    cpdef add_string_interact_with_vacuole(self, double[:, :] xy_x, double[:, :] xy_y, double[:, :] xy2_x, double[:, :] xy2_y, double[:] xs, double[:] ys, double DELTA, int n_pair, int[:] c_all, int[:] f_collide, int[:] p_collide, long[:] axis_1_plus_1):
        """
        xy_x, xy_y, xy2_x, xy2_y: shape: (Nv, # of potential interactions)
        xs, ys: shape: (1, # of potential interaction)
        This function is only valid if the particle never enters the vacuole.
        """
        cdef int ii, jj, sign, kk, cc, ff, pp
        cdef double[:,:] Fvx_int = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef double[:,:] Fvy_int = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef double[:,:] Frx = np.zeros((self.Nf, self.Np), dtype = np.double)
        cdef double[:,:] Fry = np.zeros((self.Nf, self.Np), dtype = np.double)
        cdef double DX, DY, DR, DELTA_eff_M_DR, a, b, c, M, N, dist_perp, min_x_vac, max_x_vac, min_y_vac, max_y_vac, xp, yp
        cdef double fx, fy, f0x, f0y, f1x, f1y
        
        for ii in range(self.Nv):
            for jj in range(n_pair):
                DX = (xs[jj] - xy_x[ii,jj])
                DY = (ys[jj] - xy_y[ii,jj])
                DR = sqrt(DX**2 + DY**2)
                DELTA_eff_M_DR = DELTA - DR
                if DELTA_eff_M_DR > 0:
                    fx = self.kr*(DX/(DR+1E-10)*(DELTA_eff_M_DR))
                    fy = self.kr*(DY/(DR+1E-10)*(DELTA_eff_M_DR))
                    cc = c_all[jj]
                    ff = f_collide[jj]
                    pp = p_collide[jj]
                    Fvx_int[cc, ii] -= fx
                    Fvy_int[cc, ii] -= fy
                    Frx[ff,pp] += fx
                    Fry[ff,pp] += fy

                a = xy2_y[ii,jj] - xy_y[ii,jj]
                b = - (xy2_x[ii,jj] - xy_x[ii,jj])
                c = xy_y[ii,jj]*(xy2_x[ii,jj] - xy_x[ii,jj]) - xy_x[ii,jj]*(xy2_y[ii,jj] - xy_y[ii,jj])
                M = a**2 + b**2
                N = b*xs[jj] - a*ys[jj]

                dist_perp = (a*xs[jj] + b*ys[jj] + c)/sqrt(M)
                if dist_perp >= 0:
                    sign = 1
                else:
                    sign = -1

                dist_perp = abs(dist_perp)

                min_x_vac = min(xy_x[ii,jj], xy2_x[ii,jj])
                max_x_vac = max(xy_x[ii,jj], xy2_x[ii,jj])
                min_y_vac = min(xy_y[ii,jj], xy2_y[ii,jj])
                max_y_vac = max(xy_y[ii,jj], xy2_y[ii,jj])

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
                    cc = c_all[jj]
                    ff = f_collide[jj]
                    pp = p_collide[jj]
                    fx = self.kr*((xs[jj] - xp)/(dist_perp+1E-10)*(DELTA_eff_M_DR))
                    fy = self.kr*((ys[jj] - yp)/(dist_perp+1E-10)*(DELTA_eff_M_DR))
                    f0x = 2*self.kr*DELTA_eff_M_DR*((-b)*dist_perp/M + (ys[jj] - xy2_y[ii,jj])*sign/sqrt(M))
                    f0y = 2*self.kr*DELTA_eff_M_DR*((xy2_x[ii,jj] - xs[jj])*sign/sqrt(M) + a*dist_perp/M)
                    f1x = 2*self.kr*DELTA_eff_M_DR*(b*dist_perp/M + (xy_y[ii,jj] - ys[jj])*sign/sqrt(M))
                    f1y = 2*self.kr*DELTA_eff_M_DR*((-xy_x[ii,jj] + xs[jj])*sign/sqrt(M) - a*dist_perp/M)
                    Frx[ff,pp] += fx
                    Fry[ff,pp] += fy
                    Fvx_int[cc,ii] += f0x
                    Fvy_int[cc,ii] += f0y
                    kk = axis_1_plus_1[ii]
                    Fvx_int[cc,kk] += f1x
                    Fvy_int[cc,kk] += f1y
        
        return np.array(Frx), np.array(Fry), np.array(Fvx_int), np.array(Fvy_int)

    cpdef add_string_interact_with_vacuole_complete(self, double[:, :] xy_x, double[:, :] xy_y, double[:, :] xy2_x, double[:, :] xy2_y, double[:] xs, double[:] ys, double DELTA, int n_pair, int[:] c_all, int[:] f_collide, int[:] p_collide, long[:] axis_1_plus_1):
        """
        xy_x, xy_y, xy2_x, xy2_y: shape: (Nv, # of potential interactions)
        xs, ys: shape: (1, # of potential interaction)
        This function is only valid if the particle never enters the vacuole.
        """
        cdef int ii, jj, sign, kk, cc, ff, pp, i_min
        cdef double[:,:] Fvx_int = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef double[:,:] Fvy_int = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef double[:,:] Frx = np.zeros((self.Nf, self.Np), dtype = np.double)
        cdef double[:,:] Fry = np.zeros((self.Nf, self.Np), dtype = np.double)
        cdef double DX, DY, DR, DELTA_eff_M_DR, a, b, c, M, N, dist_perp, min_x_vac, max_x_vac, min_y_vac, max_y_vac, xp, yp
        #cdef double a_new, b_new, c_new, M_new, N_new, dist_perp_new, xp_new, yp_new
        #cdef int sign_new
        cdef double fx, fy, f0x, f0y, f1x, f1y#, ang, v1x, v1y, v2x, v2y
        #cdef int[:] point_is_within_canvas = np.zeros(n_pair, dtype = np.int32)
        
        for jj in range(n_pair):
            for ii in range(self.Nv):
                DX = (xs[jj] - xy_x[ii,jj])
                DY = (ys[jj] - xy_y[ii,jj])
                DR = sqrt(DX**2 + DY**2)
                DELTA_eff_M_DR = DELTA - DR
                if DELTA_eff_M_DR > 0:
                    fx = self.kr*(DX/(DR+1E-10)*(DELTA_eff_M_DR))
                    fy = self.kr*(DY/(DR+1E-10)*(DELTA_eff_M_DR))
                    cc = c_all[jj]
                    ff = f_collide[jj]
                    pp = p_collide[jj]
                    Fvx_int[cc, ii] -= fx
                    Fvy_int[cc, ii] -= fy
                    Frx[ff,pp] += fx
                    Fry[ff,pp] += fy

                a = xy2_y[ii,jj] - xy_y[ii,jj]
                b = - (xy2_x[ii,jj] - xy_x[ii,jj])
                c = xy_y[ii,jj]*(xy2_x[ii,jj] - xy_x[ii,jj]) - xy_x[ii,jj]*(xy2_y[ii,jj] - xy_y[ii,jj])
                M = a**2 + b**2
                N = b*xs[jj] - a*ys[jj]

                dist_perp = (a*xs[jj] + b*ys[jj] + c)/sqrt(M)
                if dist_perp >= 0:
                    sign = 1
                else:
                    sign = -1

                dist_perp = abs(dist_perp)

                min_x_vac = min(xy_x[ii,jj], xy2_x[ii,jj])
                max_x_vac = max(xy_x[ii,jj], xy2_x[ii,jj])
                min_y_vac = min(xy_y[ii,jj], xy2_y[ii,jj])
                max_y_vac = max(xy_y[ii,jj], xy2_y[ii,jj])

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
                    cc = c_all[jj]
                    ff = f_collide[jj]
                    pp = p_collide[jj]
                    fx = self.kr*((xs[jj] - xp)/(dist_perp+1E-10)*(DELTA_eff_M_DR))
                    fy = self.kr*((ys[jj] - yp)/(dist_perp+1E-10)*(DELTA_eff_M_DR))
                    f0x = 2*self.kr*DELTA_eff_M_DR*((-b)*dist_perp/M + (ys[jj] - xy2_y[ii,jj])*sign/sqrt(M))
                    f0y = 2*self.kr*DELTA_eff_M_DR*((xy2_x[ii,jj] - xs[jj])*sign/sqrt(M) + a*dist_perp/M)
                    f1x = 2*self.kr*DELTA_eff_M_DR*(b*dist_perp/M + (xy_y[ii,jj] - ys[jj])*sign/sqrt(M))
                    f1y = 2*self.kr*DELTA_eff_M_DR*((-xy_x[ii,jj] + xs[jj])*sign/sqrt(M) - a*dist_perp/M)
                    Frx[ff,pp] += fx
                    Fry[ff,pp] += fy
                    Fvx_int[cc,ii] += f0x
                    Fvy_int[cc,ii] += f0y
                    kk = axis_1_plus_1[ii]
                    Fvx_int[cc,kk] += f1x
                    Fvy_int[cc,kk] += f1y
        return np.array(Frx), np.array(Fry), np.array(Fvx_int), np.array(Fvy_int)

        """for ii in range(n_pair):
            ang = 0
            for cc in range(self.Nv):
                v1x = xy_x[cc,ii] - xs[ii]
                v1y = xy_y[cc,ii] - ys[ii]
                v2x = xy2_x[cc,ii] - xs[ii]
                v2y = xy2_y[cc,ii] - ys[ii]
                ang += atan2(-v2x*v1y + v2y*v1x, v2x*v1x + v2y*v1y)
            if roundc((abs(ang) - 2*PI)*10**9) == 0:
                point_is_within_canvas[ii] = 1

        for jj in range(n_pair):
            if point_is_within_canvas[jj] == 1:
                for ii in range(self.Nv):
                    DX = (xs[jj] - xy_x[ii,jj])
                    DY = (ys[jj] - xy_y[ii,jj])
                    DR = sqrt(DX**2 + DY**2)
                    DELTA_eff_M_DR = DELTA - DR
                    if DELTA_eff_M_DR > 0:
                        fx = self.kr*(DX/(DR+1E-10)*(DELTA_eff_M_DR))
                        fy = self.kr*(DY/(DR+1E-10)*(DELTA_eff_M_DR))
                        cc = c_all[jj]
                        ff = f_collide[jj]
                        pp = p_collide[jj]
                        Fvx_int[cc, ii] += fx #
                        Fvy_int[cc, ii] += fy #
                        Frx[ff,pp] -= fx #
                        Fry[ff,pp] -= fy #

                    dist_perp = 100.0
                    a_new = xy2_y[ii,jj] - xy_y[ii,jj]
                    b_new = - (xy2_x[ii,jj] - xy_x[ii,jj])
                    c_new = xy_y[ii,jj]*(xy2_x[ii,jj] - xy_x[ii,jj]) - xy_x[ii,jj]*(xy2_y[ii,jj] - xy_y[ii,jj])
                    M_new = a_new**2 + b_new**2
                    N_new = b_new*xs[jj] - a_new*ys[jj]
                    dist_perp_new = (a_new*xs[jj] + b_new*ys[jj] + c_new)/sqrt(M_new)
                    xp_new = (b_new*N_new - a_new*c_new)/M_new
                    yp_new = (a_new*(-N_new) - b_new*c_new)/M_new

                    min_x_vac = min(xy_x[ii,jj], xy2_x[ii,jj])
                    max_x_vac = max(xy_x[ii,jj], xy2_x[ii,jj])
                    min_y_vac = min(xy_y[ii,jj], xy2_y[ii,jj])
                    max_y_vac = max(xy_y[ii,jj], xy2_y[ii,jj])
                    
                    if xp_new >= min_x_vac:
                        if xp_new <= max_x_vac:
                            if yp_new >= min_y_vac:
                                if yp_new <= max_y_vac:
                                    if dist_perp_new < dist_perp:
                                        a = a_new
                                        b = b_new
                                        c = c_new
                                        M = M_new
                                        N = N_new
                                        sign = sign_new
                                        dist_perp = dist_perp_new
                                        xp = xp_new
                                        yp = yp_new
                                        i_min = ii
                DELTA_eff_M_DR = DELTA + dist_perp #
                cc = c_all[jj]
                ff = f_collide[jj]
                pp = p_collide[jj]
                fx = self.kr*((xs[jj] - xp)/(dist_perp+1E-10)*(DELTA_eff_M_DR))
                fy = self.kr*((ys[jj] - yp)/(dist_perp+1E-10)*(DELTA_eff_M_DR))
                #f0x = 2*self.kr*DELTA_eff_M_DR*((-b)*dist_perp/M + (ys[jj] - xy2_y[i_min,jj])*sign/sqrt(M))
                #f0y = 2*self.kr*DELTA_eff_M_DR*((xy2_x[i_min,jj] - xs[jj])*sign/sqrt(M) + a*dist_perp/M)
                #f1x = 2*self.kr*DELTA_eff_M_DR*(b*dist_perp/M + (xy_y[i_min,jj] - ys[jj])*sign/sqrt(M))
                #f1y = 2*self.kr*DELTA_eff_M_DR*((-xy_x[i_min,jj] + xs[jj])*sign/sqrt(M) - a*dist_perp/M)
                Frx[ff,pp] -= fx #
                Fry[ff,pp] -= fy #
                #Fvx_int[cc,i_min] -= f0x #
                #Fvy_int[cc,i_min] -= f0y #
                #kk = axis_1_plus_1[i_min] #
                #Fvx_int[cc,kk] -= f1x #
                #Fvy_int[cc,kk] -= f1y #
            else:
                for ii in range(self.Nv):
                    DX = (xs[jj] - xy_x[ii,jj])
                    DY = (ys[jj] - xy_y[ii,jj])
                    DR = sqrt(DX**2 + DY**2)
                    DELTA_eff_M_DR = DELTA - DR
                    if DELTA_eff_M_DR > 0:
                        fx = self.kr*(DX/(DR+1E-10)*(DELTA_eff_M_DR))
                        fy = self.kr*(DY/(DR+1E-10)*(DELTA_eff_M_DR))
                        cc = c_all[jj]
                        ff = f_collide[jj]
                        pp = p_collide[jj]
                        Fvx_int[cc, ii] -= fx
                        Fvy_int[cc, ii] -= fy
                        Frx[ff,pp] += fx
                        Fry[ff,pp] += fy

                    a = xy2_y[ii,jj] - xy_y[ii,jj]
                    b = - (xy2_x[ii,jj] - xy_x[ii,jj])
                    c = xy_y[ii,jj]*(xy2_x[ii,jj] - xy_x[ii,jj]) - xy_x[ii,jj]*(xy2_y[ii,jj] - xy_y[ii,jj])
                    M = a**2 + b**2
                    N = b*xs[jj] - a*ys[jj]

                    dist_perp = (a*xs[jj] + b*ys[jj] + c)/sqrt(M)
                    if dist_perp >= 0:
                        sign = 1
                    else:
                        sign = -1

                    dist_perp = abs(dist_perp)

                    min_x_vac = min(xy_x[ii,jj], xy2_x[ii,jj])
                    max_x_vac = max(xy_x[ii,jj], xy2_x[ii,jj])
                    min_y_vac = min(xy_y[ii,jj], xy2_y[ii,jj])
                    max_y_vac = max(xy_y[ii,jj], xy2_y[ii,jj])

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
                        cc = c_all[jj]
                        ff = f_collide[jj]
                        pp = p_collide[jj]
                        fx = self.kr*((xs[jj] - xp)/(dist_perp+1E-10)*(DELTA_eff_M_DR))
                        fy = self.kr*((ys[jj] - yp)/(dist_perp+1E-10)*(DELTA_eff_M_DR))
                        f0x = 2*self.kr*DELTA_eff_M_DR*((-b)*dist_perp/M + (ys[jj] - xy2_y[ii,jj])*sign/sqrt(M))
                        f0y = 2*self.kr*DELTA_eff_M_DR*((xy2_x[ii,jj] - xs[jj])*sign/sqrt(M) + a*dist_perp/M)
                        f1x = 2*self.kr*DELTA_eff_M_DR*(b*dist_perp/M + (xy_y[ii,jj] - ys[jj])*sign/sqrt(M))
                        f1y = 2*self.kr*DELTA_eff_M_DR*((-xy_x[ii,jj] + xs[jj])*sign/sqrt(M) - a*dist_perp/M)
                        Frx[ff,pp] += fx
                        Fry[ff,pp] += fy
                        Fvx_int[cc,ii] += f0x
                        Fvy_int[cc,ii] += f0y
                        kk = axis_1_plus_1[ii]
                        Fvx_int[cc,kk] += f1x
                        Fvy_int[cc,kk] += f1y
        
        return np.array(Frx), np.array(Fry), np.array(Fvx_int), np.array(Fvy_int)"""

    cpdef add_vacuole_interact_with_vacuole(self, double[:, :] xy_x, double[:, :] xy_y, double[:, :] xy2_x, double[:, :] xy2_y, double[:] xs, double[:] ys, double DELTA, int n_pair, int[:] c_all, int[:] c_collide, int[:] v_collide, long[:] axis_1_plus_1):
        """
        xy_x, xy_y, xy2_x, xy2_y: shape: (Nv, # of potential interactions)
        xs, ys: shape: (# of potential interaction)
        This function is only valid if the particle never enters the vacuole.
        """
        cdef int ii, jj, sign, kk, cc, c2c2, v2v2
        cdef double[:,:] Fvx_int = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef double[:,:] Fvy_int = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef double DX, DY, DR, DELTA_eff_M_DR, a, b, c, M, N, dist_perp, min_x_vac, max_x_vac, min_y_vac, max_y_vac, xp, yp
        cdef double fx, fy, f0x, f0y, f1x, f1y
        
        for ii in range(self.Nv):
            for jj in range(n_pair):
                DX = (xs[jj] - xy_x[ii,jj])
                DY = (ys[jj] - xy_y[ii,jj])
                DR = sqrt(DX**2 + DY**2)
                DELTA_eff_M_DR = DELTA - DR
                if DELTA_eff_M_DR > 0:
                    fx = self.kr*(DX/(DR+1E-10)*(DELTA_eff_M_DR))
                    fy = self.kr*(DY/(DR+1E-10)*(DELTA_eff_M_DR))
                    cc = c_all[jj]
                    c2c2 = c_collide[jj]
                    v2v2 = v_collide[jj]
                    Fvx_int[cc, ii] -= fx
                    Fvy_int[cc, ii] -= fy
                    Fvx_int[c2c2, v2v2] += fx
                    Fvy_int[c2c2, v2v2] += fy

                a = xy2_y[ii,jj] - xy_y[ii,jj]
                b = - (xy2_x[ii,jj] - xy_x[ii,jj])
                c = xy_y[ii,jj]*(xy2_x[ii,jj] - xy_x[ii,jj]) - xy_x[ii,jj]*(xy2_y[ii,jj] - xy_y[ii,jj])
                M = a**2 + b**2
                N = b*xs[jj] - a*ys[jj]

                dist_perp = (a*xs[jj] + b*ys[jj] + c)/sqrt(M)
                if dist_perp >= 0:
                    sign = 1
                else:
                    sign = -1

                dist_perp = abs(dist_perp)

                min_x_vac = min(xy_x[ii,jj], xy2_x[ii,jj])
                max_x_vac = max(xy_x[ii,jj], xy2_x[ii,jj])
                min_y_vac = min(xy_y[ii,jj], xy2_y[ii,jj])
                max_y_vac = max(xy_y[ii,jj], xy2_y[ii,jj])

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
                    cc = c_all[jj]
                    c2c2 = c_collide[jj]
                    v2v2 = v_collide[jj]
                    fx = self.kr*((xs[jj] - xp)/(dist_perp+1E-10)*(DELTA_eff_M_DR))
                    fy = self.kr*((ys[jj] - yp)/(dist_perp+1E-10)*(DELTA_eff_M_DR))
                    f0x = 2*self.kr*DELTA_eff_M_DR*((-b)*dist_perp/M + (ys[jj] - xy2_y[ii,jj])*sign/sqrt(M))
                    f0y = 2*self.kr*DELTA_eff_M_DR*((xy2_x[ii,jj] - xs[jj])*sign/sqrt(M) + a*dist_perp/M)
                    f1x = 2*self.kr*DELTA_eff_M_DR*(b*dist_perp/M + (xy_y[ii,jj] - ys[jj])*sign/sqrt(M))
                    f1y = 2*self.kr*DELTA_eff_M_DR*((-xy_x[ii,jj] + xs[jj])*sign/sqrt(M) - a*dist_perp/M)
                    Fvx_int[c2c2, v2v2] += fx
                    Fvy_int[c2c2, v2v2] += fy
                    Fvx_int[cc,ii] += f0x
                    Fvy_int[cc,ii] += f0y
                    kk = axis_1_plus_1[ii]
                    Fvx_int[cc,kk] += f1x
                    Fvy_int[cc,kk] += f1y
        
        return np.array(Fvx_int), np.array(Fvy_int)

    cpdef generate_c_c_interaction(self, double[:,:,:] r_matrix, double[:,:] xy_vacuole_min_mi_delt, double[:,:] xy_vacuole_max_pl_delt):
        cdef int n_pair = 0
        cdef int size = self.Nc*self.Nc*self.Nv
        cdef int c1, v, c2
        cdef int[:] c_collide = np.zeros(size, dtype = np.int32)
        cdef int[:] v_collide = np.zeros(size, dtype = np.int32)
        cdef int[:] c_all = np.zeros(size, dtype = np.int32)

        for c1 in range(self.Nc):
            for c2 in range(self.Nc):
                if c1 == c2:
                    pass
                else:
                    for v in range(self.Nv):
                        if r_matrix[0, c1, v] >= xy_vacuole_min_mi_delt[c2, 0]:
                            if r_matrix[0, c1, v] <= xy_vacuole_max_pl_delt[c2, 0]:
                                if r_matrix[1, c1, v] >= xy_vacuole_min_mi_delt[c2, 1]:
                                    if r_matrix[1, c1, v] <= xy_vacuole_max_pl_delt[c2, 1]:
                                        c_collide[n_pair] = c1
                                        v_collide[n_pair] = v
                                        c_all[n_pair] = c2
                                        n_pair += 1
        return np.array(c_collide[0:n_pair]), np.array(v_collide[0:n_pair]), np.array(c_all[0:n_pair]), n_pair

    cpdef generate_c_c_intxn_add_vac_force(self, double[:,:,:] r_matrix, double[:,:] xy_vacuole_min_mi_delt, double[:,:] xy_vacuole_max_pl_delt, double[:] rx, double[:] ry, long[:] axis_1_plus_1, double DELTA):
        cdef int c1, v, c2, ii, jj, sign, kk 
        cdef double[:,:] Fvx_int = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef double[:,:] Fvy_int = np.zeros((self.Nc, self.Nv), dtype = np.double)
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
                                                Fvx_int[c2, ii] -= fx
                                                Fvy_int[c2, ii] -= fy
                                                Fvx_int[c1, v] += fx
                                                Fvy_int[c1, v] += fy

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
                                                Fvx_int[c1, v] += fx
                                                Fvy_int[c1, v] += fy
                                                Fvx_int[c2,ii] += f0x
                                                Fvy_int[c2,ii] += f0y
                                                Fvx_int[c2,kk] += f1x
                                                Fvy_int[c2,kk] += f1y
        return np.array(Fvx_int), np.array(Fvy_int)

    cpdef generate_fp_c_interaction(self, double[:,:] f_x_array, double[:,:] f_y_array, double[:,:] xy_vacuole_min_mi_delt, double[:,:] xy_vacuole_max_pl_delt, long[:,:] connection_table):
        cdef int n_pair = 0
        cdef int size = self.Nf*self.Np*self.Nc
        cdef int f, p, c
        cdef int[:] f_collide = np.zeros(size, dtype = np.int32)
        cdef int[:] p_collide = np.zeros(size, dtype = np.int32)
        cdef int[:] c_all = np.zeros(size, dtype = np.int32)
        for f in range(self.Nf):
            for c in range(self.Nc):
                if c == connection_table[f, 0]:
                    for p in range(1, self.Np):
                        if f_x_array[f,p] >= xy_vacuole_min_mi_delt[c, 0]:
                            if f_x_array[f,p] <= xy_vacuole_max_pl_delt[c, 0]:
                                if f_y_array[f,p] >= xy_vacuole_min_mi_delt[c, 1]:
                                    if f_y_array[f,p] <= xy_vacuole_max_pl_delt[c, 1]:
                                        f_collide[n_pair] = f
                                        p_collide[n_pair] = p
                                        c_all[n_pair] = c
                                        n_pair += 1
                elif c == connection_table[f, 1]:
                    for p in range(self.Np-1):
                        if f_x_array[f,p] >= xy_vacuole_min_mi_delt[c, 0]:
                            if f_x_array[f,p] <= xy_vacuole_max_pl_delt[c, 0]:
                                if f_y_array[f,p] >= xy_vacuole_min_mi_delt[c, 1]:
                                    if f_y_array[f,p] <= xy_vacuole_max_pl_delt[c, 1]:
                                        f_collide[n_pair] = f
                                        p_collide[n_pair] = p
                                        c_all[n_pair] = c
                                        n_pair += 1
                else:
                    for p in range(self.Np):
                        if f_x_array[f,p] >= xy_vacuole_min_mi_delt[c, 0]:
                            if f_x_array[f,p] <= xy_vacuole_max_pl_delt[c, 0]:
                                if f_y_array[f,p] >= xy_vacuole_min_mi_delt[c, 1]:
                                    if f_y_array[f,p] <= xy_vacuole_max_pl_delt[c, 1]:
                                        f_collide[n_pair] = f
                                        p_collide[n_pair] = p
                                        c_all[n_pair] = c
                                        n_pair += 1
        return np.array(f_collide[0:n_pair]), np.array(p_collide[0:n_pair]), np.array(c_all[0:n_pair]), n_pair

    cpdef generate_fp_c_intxn_add_string_force(self, double[:,:,:] r_matrix, double[:,:] f_x_array, double[:,:] f_y_array, double[:,:] xy_vacuole_min_mi_delt, double[:,:] xy_vacuole_max_pl_delt, long[:,:] connection_table, double DELTA, long[:] axis_1_plus_1):
        #cdef int n_pair = 0
        #cdef int size = self.Nf*self.Np*self.Nc
        cdef int f, p, cc, ii, sign, kk
        cdef double xs, ys, xy_x, xy_y, xy2_x, xy2_y
        #cdef int[:] f_collide = np.zeros(size, dtype = np.int32)
        #cdef int[:] p_collide = np.zeros(size, dtype = np.int32)
        #cdef int[:] c_all = np.zeros(size, dtype = np.int32)
        cdef double[:,:] Fvx_int = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef double[:,:] Fvy_int = np.zeros((self.Nc, self.Nv), dtype = np.double)
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
                                                Fvx_int[cc, ii] -= fx
                                                Fvy_int[cc, ii] -= fy
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
                                                Fvx_int[cc,ii] += f0x
                                                Fvy_int[cc,ii] += f0y
                                                Fvx_int[cc,kk] += f1x
                                                Fvy_int[cc,kk] += f1y

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
                                                Fvx_int[cc, ii] -= fx
                                                Fvy_int[cc, ii] -= fy
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
                                                Fvx_int[cc,ii] += f0x
                                                Fvy_int[cc,ii] += f0y
                                                Fvx_int[cc,kk] += f1x
                                                Fvy_int[cc,kk] += f1y
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
                                                Fvx_int[cc, ii] -= fx
                                                Fvy_int[cc, ii] -= fy
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
                                                Fvx_int[cc,ii] += f0x
                                                Fvy_int[cc,ii] += f0y
                                                Fvx_int[cc,kk] += f1x
                                                Fvy_int[cc,kk] += f1y
        return np.array(Frx), np.array(Fry), np.array(Fvx_int), np.array(Fvy_int)

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.c_api_binop_methods(True)

cdef class boundary_interactions:
    def __init__(self, long n_corners):
        self.n_corners = n_corners

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
        return np.array(f_x), np.array(f_y)

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.c_api_binop_methods(True)

cdef class vacuole_computations:
    def __init__(self, long Nv, long Nc, double gamma, double kl_c_pos_Nv, double kl_c_neg_Nv): 
        self.Nv = Nv
        self.Nc = Nc
        self.gamma = gamma
        self.kl_c_pos_Nv = kl_c_pos_Nv
        self.kl_c_neg_Nv = kl_c_neg_Nv
    cpdef find_Fb_Flc_Fgamma(self, double[:,:] lx_arr, double[:,:] ly_arr, double[:,:] l_arr, double[:] l0_arr, double[:,:] l_m_l0_arr, double[:,:] lx_div_l, double[:,:] ly_div_l, double[:] vacuole_bending_prefactor, long[:] axis_1_plus_1, long[:] axis_1_minus_1, long[:] axis_1_plus_2, int[:,:] neg):
        cdef double[:,:] Fbx = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef double[:,:] Fby = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef double[:,:] Flcx = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef double[:,:] Flcy = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef double[:,:] Fgx = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef double[:,:] Fgy = np.zeros((self.Nc, self.Nv), dtype = np.double)
        cdef int ii, vv
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
        return np.array(Fbx), np.array(Fby), np.array(Flcx), np.array(Flcy), np.array(Fgx), np.array(Fgy)

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.c_api_binop_methods(True)

cdef class filament_computations:
    def __init__(self, long Nf, long Np, double kl, double filament_bending_prefactor): 
        self.Nf = Nf
        self.Np = Np
        self.kl = kl
        self.filament_bending_prefactor = filament_bending_prefactor

    cpdef find_Fl_Fbf(self, double[:,:] dx, double[:,:] dy, long[:,:] ER_intact, double[:,:] Fl_expanded):
        cdef int Np_m1 = self.Np - 1
        cdef int ff, pp
        cdef double FX, FY
        cdef double[:,:] Flx = np.zeros((self.Nf, self.Np), dtype = np.double)
        cdef double[:,:] Fly = np.zeros((self.Nf, self.Np), dtype = np.double)
        cdef double[:,:] Fbfx = np.zeros((self.Nf, self.Np), dtype = np.double)
        cdef double[:,:] Fbfy = np.zeros((self.Nf, self.Np), dtype = np.double)

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

        return np.array(Flx), np.array(Fly), np.array(Fbfx), np.array(Fbfy)
