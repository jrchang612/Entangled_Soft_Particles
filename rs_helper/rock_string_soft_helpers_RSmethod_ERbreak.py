
from __future__ import division
import cython
from cython.parallel import prange
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import odespy
import os
from datetime import datetime
import time
from tqdm import tqdm
import h5py
import pylab
import math
from scipy.integrate import solve_ivp
from scipy.spatial import Delaunay, Voronoi, distance
from scipy.sparse import coo_matrix
from sklearn.metrics import pairwise_distances
import matplotlib.colors as mcolors
import matplotlib.cm as cm
PI = 3.141592653589793
#@cython.cdivision(True)
#@cython.nonecheck(False)
#@cython.wraparound(False) 
#@cython.boundscheck(False)

def angle_xy(v1x, v1y, v2x, v2y):
    """
    same function as angle, but accepting separate vx & vy, to avoid hstack and reshape
    v1x, v1y, v2x, v2y must be of the same shape
    """
    #cosine = (v1x*v2x + v1y*v2y)/np.sqrt((v1x**2+v1y**2)*(v2x**2 + v2y**2))
    #return np.arccos(np.maximum(np.minimum(cosine, 1), -1))
    return np.arctan2(-v2x*v1y + v2y*v1x, v2x*v1x + v2y*v1y)

def is_on_segment(x, y, edge):
    """
    Determine whether a point is on a segment by checking if Ax+By-C == 0 and falls between the two
    corners which define the edge.
    ## This is vectorized version of the function that can determine whether a series of points (x, y)
    are on a certain edge.
    """
    [[x1, y1],[x2, y2]] = edge
    # convert to ax + by = c
    a = (y2 - y1); 
    b = - (x2 - x1); 
    c = x1*(y2 - y1) - y1*(x2 - x1)
    test = (a*x + b*y - c)
    if (a**2 + b**2) == 0:
        result = (x == x1) * (y == y1)

    else:
        x = (x*(10**9) + 0.5).astype(int)/(10.**9)
        x1 = int(x1*(10**9) + 0.5)/(10.**9)
        x2 = int(x2*(10**9) + 0.5)/(10.**9)
        y = (y*(10**9) + 0.5).astype(int)/(10.**9)
        y1 = int(y1*(10**9) + 0.5)/(10.**9)
        y2 = int(y2*(10**9) + 0.5)/(10.**9)
        
        result = (((test*(10**9))/(10.**9)).astype(int) == 0)*(
            (x >= min(x1, x2))*(x <= max(x1, x2))*(y >= min(y1, y2))*(y <= max(y1, y2)))

    return result

def point_is_within_canvas(xy, xs, ys):
    """
    Check if a point is within canvas by computing the sum of the angle formed by adjacent corners
    and the point. If a point is within canvas, the sum of the angle should be 2*pi.
    ## This is vectorized version
    """
    xy2 = np.roll(xy, shift = 1, axis = 0)
    v1x = xy[:,0:1].T - xs
    v1y = xy[:,1:2].T - ys
    v2x = xy2[:,0:1].T - xs
    v2y = xy2[:,1:2].T - ys

    ang = angle_xy(v1x, v1y, v2x, v2y)
    sum_angle = np.abs(np.sum(ang, axis = 1, keepdims = True))
    result = (((sum_angle-(2*PI))*(10**9)).astype(int)/(10.**9) == 0).reshape((-1, 1))
    #result = result_angle + result_corner
    return result

def find_min_dist_to_vac_border(xy, xs, ys):
    """
    find the minimum distance of the site to the borders of the vacuole. (used for repulsive calculation)
    """
    xy2 = np.roll(xy, shift = 1, axis = 0)
    a = xy2[:, 1:2] - xy[:, 1:2]
    b = - (xy2[:, 0:1] - xy[:, 0:1])
    c = xy[:, 1:2]*(-b) - xy[:, 0:1]*a
    dist_perp = np.abs(a*xs.T + b*ys.T + c)/np.sqrt(a**2 + b**2)
    xp = (b*(b*xs.T - a*ys.T) - a*c)/(a**2 + b**2)
    yp = (a*(- b*xs.T + a*ys.T) - b*c)/(a**2 + b**2)
    fx = (xp - xs.T)
    fy = (yp - ys.T)
    min_dist_loc = np.argmin(dist_perp, axis = 0)
    n_site = len(xs)
    f_x = fx[min_dist_loc, range(n_site)].reshape((n_site, 1))
    f_y = fy[min_dist_loc, range(n_site)].reshape((n_site, 1))
    return f_x, f_y, min_dist_loc

class PolygonClass(object):
    def __init__(self, xy, disable_reorder = False):

        def reorder_points(corners):
            """
            This function reorders the corners of a polygon in a counterclockwise manner.
            The input should be a numpy array of nx2.
            """
            ordered_points = corners
            com = ordered_points.mean(axis = 0) # find center of mass
            ordered_points = ordered_points[np.argsort(np.arctan2((ordered_points - com)[:, 1], 
                (ordered_points - com)[:, 0]))]
            return ordered_points
        if disable_reorder:
            self.xy = xy
            self.n_corners = len(xy)
            self.xy2 = self.xy[np.arange(-1,self.n_corners-1), :]
        else:
            self.xy = reorder_points(xy)
            self.n_corners = len(xy)
            self.xy2 = self.xy[np.arange(-1,self.n_corners-1), :]
        self.edges = []
        for i in range(self.n_corners):
            self.edges.append(self.xy[[i-1, i], :])
        self.area = self.polygon_area(self.xy)

        self.center = np.mean(self.xy, axis = 0)
        self.a = self.xy2[:, 1:2] - self.xy[:, 1:2]
        self.b = - (self.xy2[:, 0:1] - self.xy[:, 0:1])
        self.c = (self.xy[:, 1:2]*(-self.b) - self.xy[:, 0:1]*self.a)
        self.M = self.a**2 + self.b**2
        self.center_sign_to_border = (self.a*self.center[0] + self.b*self.center[1] + self.c) > 0

    def polygon_area(self, corners):
        """
        Calculate polygon area using shoelace formula.
        Please make sure that the corners are reordered before calling polygon_area function!
        """
        n = len(corners)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += corners[i][0] * corners[j][1]
            area -= corners[j][0] * corners[i][1]
        area = abs(area) / 2.0
        return area

    def random_points_in_canvas(self, n_points):
        """
        This function assigns n random points within the canvas. The algorithm is as followed:
        (1) divide the canvas into multiple triangels
        (2) weight each triangle according to the area
        (3) randomly placed points within the triangles
        Please make sure that the points have been reordered properly!
        """
        vec_all = self.xy[1:, :] - self.xy[0, :]
        n_triangle = self.n_corners - 2
        area_triangle = np.zeros(n_triangle, dtype = np.double)

        for i in range(n_triangle):
            area_triangle[i] = self.polygon_area(np.vstack([[0, 0], vec_all[i], vec_all[i+1]]))
        rand_scale = np.sum(np.tril(area_triangle), axis = 1)/sum(area_triangle)
        rand_num = np.hstack([np.random.rand(n_points, 3), np.zeros([n_points, 1])])

        sites = np.zeros([n_points, 2], dtype = np.double)
        for i in range(n_triangle):
            mask = (rand_num[:, 0] <= rand_scale[i]) * (rand_num[:, 3] == 0)
            vec1_tile_xy = np.tile(vec_all[i, :], (sum(mask), 1))
            vec2_tile_xy = np.tile(vec_all[i+1, :], (sum(mask), 1))
            rand_num_masked = rand_num[mask, 1].reshape((-1, 1))
            rand_len_masked = np.sqrt(rand_num[mask, 2].reshape((-1, 1)))
            sites[mask, :] = (rand_num_masked*vec1_tile_xy*rand_len_masked 
                + (1 - rand_num_masked)*vec2_tile_xy*rand_len_masked)
            rand_num[mask, 3] = 1

        sites += np.tile(self.xy[0, :], (n_points, 1))
        return sites
    
    def point_is_within_canvas(self, xs, ys):
        """
        Check if a point is within canvas by computing the sum of the angle formed by adjacent corners
        and the point. If a point is within canvas, the sum of the angle should be 2*pi.
        ## This is vectorized version
        """
        result_corner = np.sum((xs == self.xy[:, 0])*(ys == self.xy[:, 1]), axis = 1, keepdims = True)
        n_site = len(xs)

        v1x = self.xy[:,0:1].T - xs
        v1y = self.xy[:,1:2].T - ys
        v2x = self.xy2[:,0:1].T - xs
        v2y = self.xy2[:,1:2].T - ys

        ang = angle_xy(v1x, v1y, v2x, v2y)
        sum_angle = np.abs(np.einsum('ij->i', ang).reshape((n_site, 1)))
        result_angle = (((sum_angle - (2*PI))*(10**9)).astype(int)/(10.**9) == 0).reshape((n_site, 1))
        result = result_angle + result_corner
        return result

    def find_min_dist_to_vac_border(self, xs, ys):
        """
        find the minimum distance of the site to the borders of the vacuole. (used for repulsive calculation)
        """
        #a = self.xy2[:, 1:2] - self.xy[:, 1:2]
        #b = - (self.xy2[:, 0:1] - self.xy[:, 0:1])
        #c = (self.xy[:, 1:2]*(-b) - self.xy[:, 0:1]*a)
        dist_perp = np.abs(self.a*xs.T + self.b*ys.T + self.c)/np.sqrt(self.M)
        xp = (self.b*(self.b*xs.T - self.a*ys.T) - self.a*self.c)/self.M
        yp = (self.a*(- self.b*xs.T + self.a*ys.T) - self.b*self.c)/self.M
        fx = (xp - xs.T)
        fy = (yp - ys.T)
        min_dist_loc = np.argmin(dist_perp, axis = 0)
        n_site = len(xs)
        f_x = fx[min_dist_loc, range(n_site)].reshape((n_site, 1))
        f_y = fy[min_dist_loc, range(n_site)].reshape((n_site, 1))
        return f_x, f_y, min_dist_loc
    
    def find_wall_force_vectorized_heuristic(self, xs, ys, delta):
        """
        xs & ys: both in shape ((-1,1))
        only compute the points that are outside canvas, only valid if delta = 0
        """
        n_site = len(xs)
        within_canvas = self.point_is_within_canvas(xs, ys)
        (loc_outside_canvas,_) = np.where(within_canvas == 0)
        N = self.b*xs.T - self.a*ys.T
        xp = (self.b*N - self.a*self.c)/self.M
        yp = (self.a*(-N) - self.b*self.c)/self.M
        dist_perp = (self.a*xs.T + self.b*ys.T + self.c)/np.sqrt(self.M)
        sign = dist_perp > 0
        mismatched_sign = (sign != self.center_sign_to_border)
        dist_perp = np.abs(dist_perp)
        delta_m_dist = delta - dist_perp
        TF = (delta_m_dist > 0)
        
        Fx_all = TF*(xs.T - xp)/(dist_perp+1E-10)*delta_m_dist*within_canvas.T
        Fy_all = TF*(ys.T - yp)/(dist_perp+1E-10)*delta_m_dist*within_canvas.T
        f_x = np.sum(Fx_all, axis = 0, keepdims = True).T
        f_y = np.sum(Fy_all, axis = 0, keepdims = True).T        
        
        xs2 = xs[loc_outside_canvas]
        ys2 = ys[loc_outside_canvas]
        xp2 = xp[:, loc_outside_canvas]
        yp2 = yp[:, loc_outside_canvas]
        delta2 = delta[:,loc_outside_canvas]
        mismatched_sign2 = mismatched_sign[:, loc_outside_canvas]
        dist_perp2 = dist_perp[:, loc_outside_canvas]
        Fx_all = - ((xs2.T - xp2)/(dist_perp2+1E-10)*(dist_perp2+delta2)).T
        Fy_all = - ((ys2.T - yp2)/(dist_perp2+1E-10)*(dist_perp2+delta2)).T
        f_x[loc_outside_canvas] = np.sum(Fx_all*mismatched_sign2.T, axis = 1, keepdims = True)
        f_y[loc_outside_canvas] = np.sum(Fy_all*mismatched_sign2.T, axis = 1, keepdims = True)     

        return f_x, f_y

    def find_force_on_walls_heuristic(self, xs, ys, delta):
        """
        xs & ys: both in shape ((-1,1))
        only compute the points that are outside canvas, only valid if delta = 0
        """
        n_site = len(xs)
        within_canvas = self.point_is_within_canvas(xs, ys)
        (loc_outside_canvas,_) = np.where(within_canvas == 0)

        N = self.b*xs.T - self.a*ys.T
        xp = (self.b*N - self.a*self.c)/self.M
        yp = (self.a*(-N) - self.b*self.c)/self.M
        dist_perp = (self.a*xs.T + self.b*ys.T + self.c)/np.sqrt(self.M)
        sign = dist_perp > 0
        mismatched_sign = (sign != self.center_sign_to_border)
        dist_perp = np.abs(dist_perp)
        delta_m_dist = delta - dist_perp
        TF = (delta_m_dist > 0)

        Fx_all = TF*(xs.T - xp)/(dist_perp+1E-10)*delta_m_dist*within_canvas.T
        Fy_all = TF*(ys.T - yp)/(dist_perp+1E-10)*delta_m_dist*within_canvas.T
        Fx_wall1 = np.sum(Fx_all, axis = 1, keepdims = True)
        Fy_wall1 = np.sum(Fy_all, axis = 1, keepdims = True)

        xs2 = xs[loc_outside_canvas]
        ys2 = ys[loc_outside_canvas]
        xp2 = xp[:, loc_outside_canvas]
        yp2 = yp[:, loc_outside_canvas]
        delta2 = delta[:,loc_outside_canvas]
        mismatched_sign2 = mismatched_sign[:, loc_outside_canvas]
        dist_perp2 = dist_perp[:, loc_outside_canvas]
        Fx_all = - ((xs2.T - xp2)/(dist_perp2+1E-10)*(dist_perp2+delta2)).T
        Fy_all = - ((ys2.T - yp2)/(dist_perp2+1E-10)*(dist_perp2+delta2)).T
        Fx_wall2 = np.sum(Fx_all*mismatched_sign2.T, axis = 0, keepdims = True)
        Fy_wall2 = np.sum(Fy_all*mismatched_sign2.T, axis = 0, keepdims = True)

        Fx_sum = -(Fx_wall1.T + Fx_wall2)
        Fy_sum = -(Fy_wall1.T + Fy_wall2)

        return Fx_sum, Fy_sum

    def find_wall_potential_vectorized_heuristic(self, xs, ys, delta):
        """
        xs & ys: both in shape ((-1,1))
        only compute the points that are outside canvas, only valid if delta = 0
        """
        n_site = len(xs)
        within_canvas = self.point_is_within_canvas(xs, ys)
        (loc_outside_canvas,_) = np.where(within_canvas == 0)

        N = self.b*xs.T - self.a*ys.T
        xp = (self.b*N - self.a*self.c)/self.M
        yp = (self.a*(-N) - self.b*self.c)/self.M
        dist_perp = (self.a*xs.T + self.b*ys.T + self.c)/np.sqrt(self.M)
        sign = dist_perp > 0
        mismatched_sign = (sign != self.center_sign_to_border)
        dist_perp = np.abs(dist_perp)
        delta_m_dist = delta - dist_perp
        TF = (delta_m_dist > 0)

        U1 = 1/2*np.sum(TF*delta_m_dist**2*within_canvas.T)

        xs2 = xs[loc_outside_canvas]
        ys2 = ys[loc_outside_canvas]
        n_site2 = len(xs2)
        xp2 = xp[:, loc_outside_canvas]
        yp2 = yp[:, loc_outside_canvas]
        delta2 = delta[:,loc_outside_canvas]
        mismatched_sign2 = mismatched_sign[:, loc_outside_canvas]
        dist_perp2 = dist_perp[:, loc_outside_canvas]

        U2 = 1/2*np.sum(mismatched_sign2*(dist_perp2+delta2)**2)

        U_final = U1 + U2
        return U_final, within_canvas
    
    def plot_canvas(self, lw = 2):
        fig = plt.figure(figsize = (5, 5), dpi = 200)
        ax = fig.add_axes([0, 0, 1, 1])

        # plot canvas
        for edge_canvas in self.edges:
            ax.plot(edge_canvas[:, 0], edge_canvas[:, 1], 
                'black', lw = lw, solid_capstyle = 'round', zorder = 2)

    def update_xy(self, xy):
        self.xy = xy
        for i in range(self.n_corners):
            self.edges[i] = (self.xy[[i-1, i], :])
        self.area = self.polygon_area(self.xy)


class Soft_and_Filament(object):
    """
    This is the main class of colloid(c) and filament(f) (Np particle per filament) simulations
    """
    def __init__(self, dim = 2, Nc = 10, Nv = 20, Nf = 0, Np = 10, Rc = 1, bidisperse = 1.4, mLf = 6.4, v_char = 0.01,
                 Ca = 0.039, St = 1, Re_R = 0.04, Stk = 2.4E-3, 
                 K1 = 1.2E-3, K2_pos = 17.6, K2_neg = 17.6, K3 = 0.39, K4 = 5.1E-8, kr_b = 1, kr_bf = 0, rho = 1000,
                 seed = 0, random_init = True, periodic_bc = False, full_repulsion = True, 
                 Aspect_ratio = 10, Length_radius_ratio = 100, canvas_initial_xy = None):
        """
        Nc = number of colloid particles
        Nv = number of discrete segments/vertices per colloid
        Nf = number of filaments
        Np = number of discrete segments per filament
        Rc = radius of colloid particles
        bidisperse = bidispersity of the system, the probability is set exactly to 50%/50%
        mLf = relative length of filaments compared to the radius of the larger colloid particles
        v_char = characteristic velocity of particles

        Ca = capillary number = viscosity * characteristic velocity/kl_c
        St = Strouhal number = magnitude of contraction/(time scale of contraction * characteristic velocity)
        Re_R = Reynolds number, based on particle radius
        Stk = Stokes number = particle density * boundary force / (54*pi*viscosity^2)
        K1 = kl_c/(ka*R^2), the smaller the number, the more imcompressible the vacuole is
        K2 = kr/kl_c, positive indicates l_{m,l} > l0, negative indicates l_{m,l} < l0
        K3 = gamma/(kl_c*R)
        K4 = kb/(kl_c*R^4)
        rho = mass density, same for surrounding & particles
        Aspect_ratio = length of the cell/diameter of the cell
        Length_radius_ratio = length of the cell/radius of vacuole

        delt = diameter of the vertex
        l0 = equilibrium length of each vertex composing the colloid particles
        kr = repulsive energy between to nearby colloid
        kr_b = relative repulsive strength from the boundary, compared to the repulsive energy among colloid
        kl = spring constant of filaments
        kl_c = spring constant of the segment of colloid
        ka = compressibility constant of soft colloid
        gamma = tension of soft colloid
        kb = bending coefficient of soft colloid

        The code is inspired from the paper "Jamming of Deformable Polygons" (PRL 2018).
        """
        # Main parameters
        self.dim = dim
        self.Nc = Nc
        self.Nv = Nv
        self.Nf = Nf
        self.Np = Np
        self.Rc = Rc
        self.bidisperse = bidisperse
        self.mLf = mLf
        self.v_char = v_char
        self.Ca = Ca
        self.St = St
        self.Re_R = Re_R
        self.Stk = Stk
        self.K1 = K1
        self.K2_pos = K2_pos
        self.K3 = K3
        self.K4 = K4
        self.K2_neg = K2_neg
        self.kr_b = kr_b # stronger/weaker repulsion from boundary, should equal 1
        self.kr_bf = kr_bf
        self.rho = rho
        self.Aspect_ratio = Aspect_ratio
        self.Length_radius_ratio = Length_radius_ratio
        
        self.delt = 2*PI*Rc/Nv
        self.l0 = self.delt

        # matching dimensionless numbers
        self.eta = self.rho*self.v_char*self.Rc/self.Re_R # matching the particle Reynolds number
        # matching Stokes number, and covert it to 2D
        self.boundary_forces_external = self.Stk*54*PI*self.eta**2/self.rho
        self.Lx0 = self.Rc*self.Length_radius_ratio # matching length radius ratio
        self.Ly0 = self.Lx0/self.Aspect_ratio # matching aspect ratio
        self.T_contr = self.Lx0/(self.v_char*self.St) # matching Strouhal number
        self.kl_c_neg = self.eta*self.v_char/self.Ca # matching capillary number
        self.kr = self.K2_neg*self.kl_c_neg # matching relative strength between repulsion and membrane stretching
        self.kl_c_pos = self.kr/self.K2_pos # = 0 for vacuoles
        self.ka = self.kl_c_neg/(self.K1*self.Rc**2) # matching relative strength between bulk modulus & expansion modulus
        self.gamma = self.K3*self.kl_c_neg*self.Rc # matching relative strength between interfacial tension & expansion modulus
        self.kb = self.K4*self.kl_c_neg*self.Rc**4 # matching relative strength between bending and expansion modulus
        
        self.Lf = self.Rc*self.bidisperse*self.mLf # length of each filament
        self.dLf = self.Lf/(self.Np - 1) # length of each filament segment
        self.kl = self.kl_c_neg # matching membrane properties
        self.kbf = self.kb # bending stiffness of RER, matching membrane properties

        self.seed = seed
        self.periodic_bc = periodic_bc
        self.full_repulsion = full_repulsion

        # Set up canvas
        if canvas_initial_xy is None:
            canvas_initial_xy = np.array([[-self.Lx0/2, -self.Ly0/2], 
                [self.Lx0/2, -self.Ly0/2], [self.Lx0/2, self.Ly0/2], [-self.Lx0/2, self.Ly0/2]])
            self.canvas_initial = PolygonClass(canvas_initial_xy)
            self.canvas = self.canvas_initial
            self.xmin = -self.Lx0/2
            self.xmax = self.Lx0/2
            self.ymin = -self.Ly0/2
            self.ymax = self.Ly0/2
            self.half_box_size_x = self.Lx0/2
            self.half_box_size_y = self.Ly0/2
            self.aspect_ratio_disabled = False
        else:
            # this is just for backward compatibility. just to debug the code. You are not supposed to run simulation based on this
            self.canvas_initial = PolygonClass(canvas_initial_xy)
            self.canvas = self.canvas_initial
            self.xmin = np.min(self.canvas.xy[:,0])
            self.xmax = np.max(self.canvas.xy[:,0])
            self.ymin = np.min(self.canvas.xy[:,1])
            self.ymax = np.max(self.canvas.xy[:,1])
            self.half_box_size_x = (self.xmax - self.xmin)/2
            self.half_box_size_y = (self.ymax - self.ymin)/2
            self.aspect_ratio_disabled = True

        # Initialize arrays for storing particle positions, activity strengths etc.
        self.allocate_arrays()
        self.filament_bending_prefactor = self.kbf/((self.Np-1)*self.dLf**4)
        
        # Other parameters
        self.cpu_time = 0
        
        # Initialize the colloids
        if random_init:
            self.random_init = True
            self.initialize_colloid_location()
        else:
            self.random_init = False
            #self.initialize_colloid_location_nonrandom()
        vol_frac_initial = np.sum(PI*self.radius**2)/self.canvas_initial.area
        self.vol_frac_initial = vol_frac_initial
        self.vol_frac = self.vol_frac_initial
        self.vol_frac_actual = (1+self.bidisperse)/2*self.Rc*self.Nc*(
            self.Nv/2*np.sin(2*PI/self.Nv) + PI**3/(2*self.Nv**2)*(self.Nv+2))/self.canvas_initial.area

        # Initialize the filament
        self.initialize_filament_location()
        self.filament_frac_initial = self.Nf/self.Nf_Delaunay
    
    def print_help(self):
        """
        print help information of the class
        """
        print('The followings are potential parameters for initialization:')
        print('dim = dimension of the system, default to 2. currently 3D simulation is not supported.')
        print('Nc = number of colloid particles, default to 10')
        print('Nv = number of discrete segments/vertices per colloid, default to 20')
        print('Nf = number of filaments, default to 0')
        print('Np = number of discrete segments per filament, default to 10')
        print('Rc = radius of colloid particles, default to 1')
        print('bidisperse = bidispersity of the system, the probability is set exactly to 50%/50%, default to 1.4')
        print('Rep = repulsive number = kr/(eta*v), default to 1')
        print('kr = repulsive energy between to nearby colloid, default to 1')
        print('kr_b = relative repulsive strength from the boundary, compared to the repulsive energy among colloid, default to 1')
        print('mLf = relative length of filaments compared to the radius of the larger colloid particles, default to 3.3')
        print('K1 = kl_c/(ka*l0**2), the smaller the number, the more imcompressible the vacuole is, default to 3.5E-4')
        print('K2_pos and K2_neg, K2 = kr/kl_c, positive indicates l_{m,l} > l0, negative indicates l_{m,l} < l0, default K2_pos = np.inf, K2_neg = 1')
        print('K3 = gamma/(kl_c*l0), indicate the surface tension of the soft colloid, default to 3.19')
        print('K4 = kb/(kl_c*l0**4), indicate the bending modulus of the surface, default to 4.4E-5')
        print('periodic_bc, default to False')
        print('full_repulsion, whether the strings are repulsed by all colloid, default to False')
        print('seed, the seed for random process, default to 0')
        print('canvas_initial_xy, the initialized canvas, default to np.array([[-15, -15], [15, -15], [15, 15], [-15, 15]])')
        print('random_init, whether colloid particles are initialized randomly, default to True')
    
    def change_canvas(self, canvas_xy):
        """
        change the boundary to new canvas.
        """
        self.canvas = PolygonClass(canvas_xy)
        self.vol_frac = np.sum(PI*self.radius**2)/self.canvas.area
        self.xmin = np.min(self.canvas.xy[:,0])
        self.xmax = np.max(self.canvas.xy[:,0])
        self.ymin = np.min(self.canvas.xy[:,1])
        self.ymax = np.max(self.canvas.xy[:,1])
        self.half_box_size_x = (self.xmax - self.xmin)/2
        self.half_box_size_y = (self.ymax - self.ymin)/2
    
    def change_ER_intact(self, ER_intact_new):
        if self.ER_intact.shape == ER_intact_new.shape:
            self.ER_intact = ER_intact_new
        else:
            print('the shape does not match')

    def allocate_arrays(self):
        # Initiate positions, orientations, forces etc of the particles
        self.r = np.zeros(self.Nc*self.Nv*self.dim, dtype = np.double)
        self.r_expanded = np.zeros(self.Nc*self.Nv*self.dim + self.dim*self.Nf*self.Np, 
            dtype = np.double)
        self.r_center_matrix = np.zeros((self.Nc, self.dim), dtype = np.double)
        self.x_particle_all = np.zeros((self.Nv*self.Nc, 1))
        self.y_particle_all = np.zeros((self.Nv*self.Nc, 1))
        self.r_matrix = np.zeros((self.dim, self.Nc, self.Nv), dtype = np.double)
        self.area_arr = np.zeros((self.Nc, 1), dtype = np.double)
        self.area0_arr = np.zeros((self.Nc, 1), dtype = np.double)
        self.lx_arr = np.zeros((self.Nc, self.Nv), dtype = np.double)
        self.ly_arr = np.zeros((self.Nc, self.Nv), dtype = np.double)
        self.l_arr = np.zeros((self.Nc, self.Nv), dtype = np.double)
        #self.perimeters_arr = np.zeros(self.Nc, dtype = np.double)
        
        self.r0 = np.zeros(self.Nc*self.Nv*self.dim, dtype = np.double)
        self.r_expanded_0 = np.zeros(self.Nc*self.Nv*self.dim+self.Nc+self.dim*self.Nf*self.Np, 
            dtype = np.double)
        
        self.radius = np.zeros(self.Nc, dtype = np.double)
        self.effective_diameter = np.zeros((self.Nc, self.Nc), dtype = np.double)
        self.delta_all = np.zeros(self.Nc*self.Nv, dtype = np.double)
        self.delta_string = np.zeros((1, self.Nf*self.Np), dtype = np.double)
        self.effective_diam_delta_simple = np.zeros((self.Nc, self.Nc), dtype = np.double)
        self.effective_diam_delta = np.zeros((self.Nc*self.Nv, self.Nc*self.Nv), dtype = np.double)
        self.interaction_true_table = np.zeros((self.Nc, self.Nc), dtype = bool)
        self.colloid_dist_arr = np.zeros((self.Nc, self.Nc), dtype = np.double)
        self.colloid_delta_r_arr = np.zeros((self.Nc, self.Nc, self.dim), dtype = np.double)
        self.colloid_delta_x_arr = np.zeros((self.Nc, self.Nc), dtype = np.double)
        self.colloid_delta_y_arr = np.zeros((self.Nc, self.Nc), dtype = np.double)
        self.rough_repulsive_force = np.zeros((self.Nc, self.Nc), dtype = np.double)
        self.xy_vacuole_min = np.zeros((self.Nc, self.dim), dtype = np.double)
        self.xy_vacuole_max = np.zeros((self.Nc, self.dim), dtype = np.double)
        self.X_pair = np.zeros((self.Nc, self.Nc), dtype = bool)
        self.Y_pair = np.zeros((self.Nc, self.Nc), dtype = bool)
        
        # filaments
        self.ER_intact = np.ones((self.Nf, self.Np-1), dtype = int)
        self.dx = np.zeros((self.Nf, self.Np-1), dtype = np.double)
        self.dy = np.zeros((self.Nf, self.Np-1), dtype = np.double)
        self.dr = np.zeros((self.Nf, self.Np-1), dtype = np.double)
        self.dx_hat = np.zeros((self.Nf, self.Np-1), dtype = np.double)
        self.dy_hat = np.zeros((self.Nf, self.Np-1), dtype = np.double)
        self.f_x_array = np.zeros((self.Nf, self.Np), dtype = np.double)
        self.f_y_array = np.zeros((self.Nf, self.Np), dtype = np.double)
        self.connection_table = np.zeros((self.Nf, 2), dtype = int) # i, j 
        self.connection_vertex = np.zeros((self.Nf, 2), dtype = int) # vertex_i, vertex_j
        self.Flx = np.zeros((self.Nf, self.Np), dtype = np.double)
        self.Fly = np.zeros((self.Nf, self.Np), dtype = np.double)
        self.Fbfx = np.zeros((self.Nf, self.Np), dtype = np.double)
        self.Fbfy = np.zeros((self.Nf, self.Np), dtype = np.double)
        self.Frx = np.zeros((self.Nf, self.Np), dtype = np.double)
        self.Fry = np.zeros((self.Nf, self.Np), dtype = np.double)
        self.Frx_expanded = np.zeros((self.Nf, self.Nc, self.Np-2), dtype = np.double)
        self.Fry_expanded = np.zeros((self.Nf, self.Nc, self.Np-2), dtype = np.double)
        # soft particles
        self.Fax = np.zeros((self.Nc, self.Nv), dtype = np.double)
        self.Fay = np.zeros((self.Nc, self.Nv), dtype = np.double)
        self.Fbx = np.zeros((self.Nc, self.Nv), dtype = np.double)
        self.Fby = np.zeros((self.Nc, self.Nv), dtype = np.double)
        self.Flcx = np.zeros((self.Nc, self.Nv), dtype = np.double)
        self.Flcy = np.zeros((self.Nc, self.Nv), dtype = np.double)
        self.Fgx = np.zeros((self.Nc, self.Nv), dtype = np.double)
        self.Fgy = np.zeros((self.Nc, self.Nv), dtype = np.double)
        self.Fv = np.zeros((self.Nc, self.Nv, self.dim), dtype = np.double) # repulsive force on vacuoles vertex

        # filaments to connectinng vacuoles
        self.dx_to_vac0 = np.zeros((self.Nf, self.Np-2), dtype = np.double)
        self.dx_to_vac1 = np.zeros((self.Nf, self.Np-2), dtype = np.double)
        self.dy_to_vac0 = np.zeros((self.Nf, self.Np-2), dtype = np.double)
        self.dy_to_vac1 = np.zeros((self.Nf, self.Np-2), dtype = np.double)
        self.dr_to_vac0 = np.zeros((self.Nf, self.Np-2), dtype = np.double)
        self.dr_to_vac1 = np.zeros((self.Nf, self.Np-2), dtype = np.double)
        ## for full repulsion cases
        self.dx_to_vac = np.zeros((self.Nf, self.Nc, self.Np-2), dtype = np.double)
        self.dy_to_vac = np.zeros((self.Nf, self.Nc, self.Np-2), dtype = np.double)
        self.dr_to_vac = np.zeros((self.Nf, self.Nc, self.Np-2), dtype = np.double)

        # Velocity of all the particles
        self.drdt = np.zeros(self.Nc*self.Nv*self.dim, dtype = np.double)
        self.dfil_x_dt = np.zeros(self.Nf*self.Np, dtype = np.double)
        self.dfil_y_dt = np.zeros(self.Nf*self.Np, dtype = np.double)
        self.drEdt = np.zeros(self.Nc*self.Nv*self.dim+self.Nf*self.Np*self.dim, dtype = np.double)
    
    def initialize_colloid_location(self):
        np.random.seed(self.seed)
        theta_initialized = np.arange(0, 2*PI, 2*PI/self.Nv)
        self.radius = np.array([self.Rc]*int(self.Nc/2)+[self.Rc*self.bidisperse]*int(self.Nc/2))
        np.random.shuffle(self.radius)
        self.effective_diameter = self.radius + self.radius.reshape((self.Nc,1))
        self.r_center_matrix = self.canvas.random_points_in_canvas(self.Nc)

        for i in range(self.Nc):
            self.x_particle_all[i*self.Nv:(i+1)*self.Nv,0] = (
                self.r_center_matrix[i, 0] + self.radius[i]*np.cos(theta_initialized))
            self.y_particle_all[i*self.Nv:(i+1)*self.Nv,0] = (
                self.r_center_matrix[i, 1] + self.radius[i]*np.sin(theta_initialized))
        
        self.r0[0:self.Nc*self.Nv] = self.x_particle_all.flatten()
        self.r0[self.Nc*self.Nv:2*self.Nc*self.Nv] = self.y_particle_all.flatten()
        self.r = np.copy(self.r0)
        
        self.six_pi_eta_r = self.eta*self.radius.reshape((-1,1))*(PI/self.Nv)
        self.axis_1_minus_1 = np.arange(1,self.Nv+1)%self.Nv
        self.axis_1_plus_1 = np.hstack([self.Nv-1, np.arange(0, self.Nv-1)])
        self.axis_1_plus_2 = np.hstack([self.Nv-2, self.Nv-1, np.arange(0, self.Nv-2)])
        #self.eight_pi_eta_r_cube = 8*PI*self.eta*self.radius**3

        self.r_matrix = self.reshape_to_matrix(self.r)
        self.r_expanded_0[0:self.Nc*self.Nv*self.dim] = self.r0
        self.r_expanded[0:self.Nc*self.Nv*self.dim] = self.r
        self.delta_all = (np.tile(self.radius*2*PI/self.Nv, (self.Nv, 1)).T).flatten()
        self.delta_wall = (self.delta_all/2).reshape((1, self.Nc*self.Nv))
        self.effective_diam_delta = self.delta_all + self.delta_all.reshape((self.Nc*self.Nv,1))
        self.effective_diam_delta_simple = self.radius*PI/self.Nv + self.radius.reshape((self.Nc,1))*PI/self.Nv
        self._compute_a()
        self._compute_segment_length()
        self.area0_arr = np.copy(self.area_arr)
        #self.l0_arr = (2*PI*self.radius/self.Nv).reshape((-1,1))
        self.l0_arr = (self.radius*2*np.sin(PI/self.Nv)).reshape((-1,1))
        self.vacuole_bending_prefactor = self.kb/(self.Nv*self.l0_arr**4)
        #self.create_vacuole_polygons()
    
    def initialize_colloid_location_from_starter(self, radius, r_matrix):
        theta_initialized = np.arange(0, 2*PI, 2*PI/self.Nv)
        self.radius = np.copy(radius)
        self.effective_diameter = self.radius + self.radius.reshape((self.Nc,1))
        self.r_center_matrix = np.copy(r_matrix)

        for i in range(self.Nc):
            self.x_particle_all[i*self.Nv:(i+1)*self.Nv,0] = (
                self.r_center_matrix[i, 0] + self.radius[i]*np.cos(theta_initialized))
            self.y_particle_all[i*self.Nv:(i+1)*self.Nv,0] = (
                self.r_center_matrix[i, 1] + self.radius[i]*np.sin(theta_initialized))
        
        self.r0[0:self.Nc*self.Nv] = self.x_particle_all.flatten()
        self.r0[self.Nc*self.Nv:2*self.Nc*self.Nv] = self.y_particle_all.flatten()
        self.r = np.copy(self.r0)
        
        self.six_pi_eta_r = self.eta*self.radius.reshape((-1,1))*(PI/self.Nv)
        self.axis_1_minus_1 = np.arange(1,self.Nv+1)%self.Nv
        self.axis_1_plus_1 = np.hstack([self.Nv-1, np.arange(0, self.Nv-1)])
        self.axis_1_plus_2 = np.hstack([self.Nv-2, self.Nv-1, np.arange(0, self.Nv-2)])
        #self.eight_pi_eta_r_cube = 8*PI*self.eta*self.radius**3

        self.r_matrix = self.reshape_to_matrix(self.r)
        self.r_expanded_0[0:self.Nc*self.Nv*self.dim] = self.r0
        self.r_expanded[0:self.Nc*self.Nv*self.dim] = self.r
        self.delta_all = (np.tile(self.radius*2*PI/self.Nv, (self.Nv, 1)).T).flatten()
        self.delta_wall = (self.delta_all/2).reshape((1, self.Nc*self.Nv))
        self.effective_diam_delta = self.delta_all + self.delta_all.reshape((self.Nc*self.Nv,1))
        self.effective_diam_delta_simple = self.radius*PI/self.Nv + self.radius.reshape((self.Nc,1))*PI/self.Nv
        self._compute_a()
        self._compute_segment_length()
        self.area0_arr = (1/2*self.Nv*np.sin(2*PI/self.Nv)*self.radius**2).reshape((self.Nc,1))
        #self.l0_arr = (2*PI*self.radius/self.Nv).reshape((-1,1))
        self.l0_arr = (self.radius*2*np.sin(PI/self.Nv)).reshape((-1,1))
        self.vacuole_bending_prefactor = self.kb/(self.Nv*self.l0_arr**4)
    
    def initialize_filament_location(self):
        tri = Delaunay(self.r_center_matrix)
        self.delaunay = tri
        connection_list = []
        for simplex in tri.simplices:
            connection_list.append(np.array([simplex[2], simplex[0]]))
            connection_list.append(np.array([simplex[0], simplex[1]]))
            connection_list.append(np.array([simplex[1], simplex[2]]))        
        new_connection_list = np.unique(np.sort(connection_list), axis = 0)
        self.Nf_Delaunay = len(new_connection_list)

        list_id_selected = np.random.choice(np.arange(len(new_connection_list)), self.Nf, replace = False)
        self.list_id_selected = list_id_selected
        self.connection_list = new_connection_list
        # connection table: (Nf, 2) = [id_vac1, id_vac2]
        # connection vertex: (Nf, 2) = [id_v_attach1, id_v_attach2]
        self.connection_table[:, 0:2] = self.connection_list[list_id_selected, :]
        self.connection_vertex[:, 0] = np.random.choice(range(self.Nv), size=(self.Nf,), replace=True)
        self.connection_vertex[:, 1] = (self.connection_vertex[:, 0] + int(self.Nv/2))%self.Nv
        #self.connection_vertex = self.connection_vertex.astype(int)
        self.f_x_array = np.linspace(self.r_matrix[0, self.connection_table[:, 0], self.connection_vertex[:, 0]], 
                                    self.r_matrix[0, self.connection_table[:, 1], self.connection_vertex[:, 1]], 
                                    self.Np).T
        self.f_y_array = np.linspace(self.r_matrix[1, self.connection_table[:, 0], self.connection_vertex[:, 0]], 
                                    self.r_matrix[1, self.connection_table[:, 1], self.connection_vertex[:, 1]], 
                                    self.Np).T
        self.get_separation_vectors()
        self.r_expanded_0[self.Nc*self.Nv*self.dim:(self.Nc*self.Nv*self.dim + self.Nf*self.Np)] = self.f_x_array.flatten()
        self.r_expanded_0[(self.Nc*self.Nv*self.dim + self.Nf*self.Np):(self.Nc*self.Nv*self.dim + 2*self.Nf*self.Np)] = self.f_y_array.flatten()
        self.r_expanded[self.Nc*self.Nv*self.dim:(self.Nc*self.Nv*self.dim + 2*self.Nf*self.Np)] = self.r_expanded_0[
            self.Nc*self.Nv*self.dim:(self.Nc*self.Nv*self.dim + 2*self.Nf*self.Np)]
        self.r_vac0 = self.radius[self.connection_table[:, 0]].reshape((self.Nf,1))
        self.r_vac1 = self.radius[self.connection_table[:, 1]].reshape((self.Nf,1))

        self.attach_F_list = []
        self.attach_P_list = []
        for j in range(self.Nc):
            attach_F, attach_P = np.where(self.connection_table == j)
            self.attach_F_list.append(attach_F)
            self.attach_P_list.append(-attach_P)
    
    def initialize_filament_location_from_starter(self, delaunay, connection_table, connection_theta, 
        f_x_array, f_y_array):
        tri = delaunay
        self.delaunay = tri
        connection_list = []
        for simplex in tri.simplices:
            connection_list.append(np.array([simplex[2], simplex[0]]))
            connection_list.append(np.array([simplex[0], simplex[1]]))
            connection_list.append(np.array([simplex[1], simplex[2]]))        
        new_connection_list = np.unique(np.sort(connection_list), axis = 0)
        self.Nf_Delaunay = len(new_connection_list)

        # connection table: (Nf, 2) = [id_vac1, id_vac2]
        # connection vertex: (Nf, 2) = [id_v_attach1, id_v_attach2]
        self.connection_table = np.copy(connection_table)
        
        # determine the shorter one, to avoid spontaneous rupture of ER
        connection_vertex_1 = (connection_theta/(2*PI)*self.Nv).astype(int)
        connection_vertex_2 = (connection_vertex_1 + 1)%self.Nv
        f_x_array1 = np.copy(f_x_array)
        f_x_array2 = np.copy(f_x_array)
        f_y_array1 = np.copy(f_y_array)
        f_y_array2 = np.copy(f_y_array)
        f_x_array1[:, 0] = self.r_matrix[0, connection_table[:, 0], connection_vertex_1[:, 0]]
        f_x_array1[:, self.Np-1] = self.r_matrix[0, connection_table[:, 1], connection_vertex_1[:, 1]]
        f_y_array1[:, 0] = self.r_matrix[1, connection_table[:, 0], connection_vertex_1[:, 0]]
        f_y_array1[:, self.Np-1] = self.r_matrix[1, connection_table[:, 1], connection_vertex_1[:, 1]]
        f_x_array2[:, 0] = self.r_matrix[0, connection_table[:, 0], connection_vertex_2[:, 0]]
        f_x_array2[:, self.Np-1] = self.r_matrix[0, connection_table[:, 1], connection_vertex_2[:, 1]]
        f_y_array2[:, 0] = self.r_matrix[1, connection_table[:, 0], connection_vertex_2[:, 0]]
        f_y_array2[:, self.Np-1] = self.r_matrix[1, connection_table[:, 1], connection_vertex_2[:, 1]]
        shorter_0 = (((f_x_array1[:,0]-f_x_array1[:,1])**2 + (f_y_array1[:,0]-f_y_array1[:,1])**2) 
             < ((f_x_array2[:,0]-f_x_array2[:,1])**2 + (f_y_array2[:,0]-f_y_array2[:,1])**2))
        shorter_Np = (((f_x_array1[:,self.Np-1]-f_x_array1[:,self.Np-2])**2 + (f_y_array1[:,self.Np-1]-f_y_array1[:,self.Np-2])**2) 
              < ((f_x_array2[:,self.Np-1]-f_x_array2[:,self.Np-2])**2 + (f_y_array2[:,self.Np-1]-f_y_array2[:,self.Np-2])**2))
        connection_vertex = np.zeros_like(connection_vertex_1)
        connection_vertex[:,0] = connection_vertex_1[:,0]*shorter_0 + connection_vertex_2[:,0]*(1-shorter_0)
        connection_vertex[:,1] = connection_vertex_1[:,1]*shorter_Np + connection_vertex_2[:,1]*(1-shorter_Np)
        self.connection_vertex = connection_vertex

        #self.connection_vertex = self.connection_vertex.astype(int)
        self.f_x_array = f_x_array
        self.f_y_array = f_y_array
        self.f_x_array[:, 0] = self.r_matrix[0, connection_table[:, 0], connection_vertex[:, 0]]
        self.f_x_array[:, self.Np-1] = self.r_matrix[0, connection_table[:, 1], connection_vertex[:, 1]]
        self.f_y_array[:, 0] = self.r_matrix[1, connection_table[:, 0], connection_vertex[:, 0]]
        self.f_y_array[:, self.Np-1] = self.r_matrix[1, connection_table[:, 1], connection_vertex[:, 1]]
        self.get_separation_vectors()
        self.r_expanded_0[self.Nc*self.Nv*self.dim:(self.Nc*self.Nv*self.dim + self.Nf*self.Np)] = self.f_x_array.flatten()
        self.r_expanded_0[(self.Nc*self.Nv*self.dim + self.Nf*self.Np):(self.Nc*self.Nv*self.dim + 2*self.Nf*self.Np)] = self.f_y_array.flatten()
        self.r_expanded[self.Nc*self.Nv*self.dim:(self.Nc*self.Nv*self.dim + 2*self.Nf*self.Np)] = self.r_expanded_0[
            self.Nc*self.Nv*self.dim:(self.Nc*self.Nv*self.dim + 2*self.Nf*self.Np)]
        self.r_vac0 = self.radius[self.connection_table[:, 0]].reshape((self.Nf,1))
        self.r_vac1 = self.radius[self.connection_table[:, 1]].reshape((self.Nf,1))

        self.attach_F_list = []
        self.attach_P_list = []
        for j in range(self.Nc):
            attach_F, attach_P = np.where(self.connection_table == j)
            self.attach_F_list.append(attach_F)
            self.attach_P_list.append(-attach_P)

    def plot_system(self, colormap = cm.viridis, plot_canvas = True, plot_colloid = True, 
        plot_string = True, plot_bead = True, lw = 2, show_break = True, alpha_colloid = 1, alpha_bead = 1, alpha_string = 1):
        """
        Plot the colloid-filamentous system
        """
        # update array
        self.r = self.r_expanded[0:self.dim*self.Nc*self.Nv]
        self.r_matrix = self.reshape_to_matrix(self.r)
        #self.theta = self.r_expanded[self.dim*self.Nc:(self.dim+1)*self.Nc]
        
        self.f_x_array = np.reshape(
            self.r_expanded[self.dim*self.Nc*self.Nv:(self.dim*self.Nc*self.Nv+self.Nf*self.Np)], (self.Nf, self.Np), order = 'C')
        self.f_y_array = np.reshape(
            self.r_expanded[(self.dim*self.Nc*self.Nv+self.Nf*self.Np):(self.dim*self.Nc*self.Nv+2*self.Nf*self.Np)], 
            (self.Nf, self.Np), order = 'C')

        if plot_canvas:
            self.canvas.plot_canvas(lw = lw)
        else:
            pass

        ER_break = (np.sum(self.ER_intact, axis = 1) < self.Np-1)

        if self.periodic_bc:
            self.r[0:self.Nc*self.Nv] = (
                self.r[0:self.Nc*self.Nv] - self.xmin)%(2*self.half_box_size_x) + self.xmin
            self.r[self.Nc*self.Nv:2*self.Nc*self.Nv] = (
                self.r[self.Nc*self.Nv:2*self.Nc*self.Nv] - self.ymin)%(2*self.half_box_size_y) + self.ymin
            self.realign_filaments()

            # plot
            if plot_colloid:
                for i, xy in enumerate(zip(self.r[0:self.Nc*self.Nv], self.r[self.Nc*self.Nv:2*self.Nc*self.Nv])):
                    for ix in range(-1, 2):
                        for iy in range(-1, 2):
                            cir = pylab.Circle((xy[0] + ix*2*self.half_box_size_x, xy[1] + iy*2*self.half_box_size_y), 
                                radius=self.delta_all[i]/2,  fc='r')
                            pylab.gca().add_patch(cir)
            else:
                pass

            if plot_string:
                if self.Nf > 0:
                    for i in range(self.Nf):
                        if ER_break[i]:
                            for j in range(self.Np-1):
                                for ix in range(-1, 2):
                                    for iy in range(-1, 2):
                                        if show_break:
                                            if self.ER_intact[i,j] == 0:
                                                plt.plot(self.f_x_array[i, j:j+2] + ix*2*self.half_box_size_x, 
                                                    self.f_y_array[i, j:j+2] + iy*2*self.half_box_size_y, 
                                                    color = 'lightcoral', linestyle = '--')
                                            else:
                                                plt.plot(self.f_x_array[i, j:j+2] + ix*2*self.half_box_size_x, 
                                                    self.f_y_array[i, j:j+2] + iy*2*self.half_box_size_y, color = 'green')
                                        else:
                                            plt.plot(self.f_x_array[i, j:j+2] + ix*2*self.half_box_size_x, 
                                                self.f_y_array[i, j:j+2] + iy*2*self.half_box_size_y, color = 'green', 
                                                alpha = self.ER_intact[i,j])
                        else:
                            for ix in range(-1, 2):
                                for iy in range(-1, 2):
                                    plt.plot(self.f_x_array[i, :] + ix*2*self.half_box_size_x, 
                                        self.f_y_array[i, :] + iy*2*self.half_box_size_y, color = 'green')
            else:
                pass

            plt.xlim([np.min(self.canvas.xy[:, 0]), np.max(self.canvas.xy[:, 0])])
            plt.ylim([np.min(self.canvas.xy[:, 0]), np.max(self.canvas.xy[:, 0])])
                            
        else:
            if plot_colloid:        
                for i in range(self.Nc):
                    plt.plot(self.r_matrix[0, i, range(-1, self.Nv)], self.r_matrix[1, i, range(-1, self.Nv)], 
                        color = 'black', linewidth = 1)
                    plt.fill(self.r_matrix[0, i, range(-1, self.Nv)], self.r_matrix[1, i, range(-1, self.Nv)], 
                        color = 'silver', alpha=alpha_colloid)
            else:
                pass
            if plot_bead:
                for i, xy in enumerate(zip(self.r[0:self.Nc*self.Nv], self.r[self.Nc*self.Nv:2*self.Nc*self.Nv])):
                    cir = pylab.Circle((xy[0], xy[1]), 
                                radius=self.delta_all[i]/2,  fc='silver', alpha = alpha_bead)
                    pylab.gca().add_patch(cir)
            else:
                pass

            if plot_string:
                if self.Nf > 0:
                    for i in range(self.Nf):
                        if ER_break[i]:
                            for j in range(self.Np-1):
                                if show_break:
                                    if self.ER_intact[i,j] == 0:
                                        plt.plot(self.f_x_array[i, j:j+2], self.f_y_array[i, j:j+2], color = 'lightcoral',
                                            linestyle = '--', alpha = alpha_string)
                                    else:
                                        plt.plot(self.f_x_array[i, j:j+2], self.f_y_array[i, j:j+2], color = 'green', 
                                            alpha = alpha_string)
                                else:
                                    plt.plot(self.f_x_array[i, j:j+2], self.f_y_array[i, j:j+2], color = 'green', 
                                        alpha = self.ER_intact[i,j]*alpha_string)
                        else:
                            plt.plot(self.f_x_array[i, :], self.f_y_array[i, :], color = 'green', alpha = alpha_string)
            else:
                pass
        plt.gca().set_aspect('equal')
    
    def reshape_to_array(self, Matrix):
        return Matrix.flatten()
    
    def reshape_to_matrix(self, Array):
        return np.reshape(Array, (self.dim, self.Nc, self.Nv))
    
    def update_center_location(self):
        self.r_center_matrix = np.mean(self.r_matrix, axis = 2).T

    def _compute_a(self):
        '''
        compute area of polygons using shoelace formula
        '''
        self.area_arr = np.einsum('ij->i',
            self.r_matrix[0, :, :]*self.r_matrix[1, :, self.axis_1_minus_1].T 
            - self.r_matrix[0, :, self.axis_1_minus_1].T*self.r_matrix[1, :, :]).reshape((self.Nc, 1))/2

    def _compute_segment_length(self):
        self.lx_arr = self.r_matrix[0, :, self.axis_1_minus_1].T - self.r_matrix[0, :, :]
        self.ly_arr = self.r_matrix[1, :, self.axis_1_minus_1].T - self.r_matrix[1, :, :]
        self.l_arr = np.sqrt(self.lx_arr**2 + self.ly_arr**2)
        #self.perimeters_arr = np.einsum('ij->i', self.l_arr)

    def get_distance_bet_colloid(self):
        """
        In the soft particle version, this is not actually computing the distance between colloid, but just a rough screening
        to find possible interaction pairs. The x & y projections of the colloid is used to include or exclude interactions.
        """
        if self.periodic_bc:
            print("periodic BC is disabled")
            """
            self.colloid_delta_x_arr = self.r_matrix[:, 0, None] - self.r_matrix[:, 0]
            self.colloid_delta_y_arr = self.r_matrix[:, 1, None] - self.r_matrix[:, 1]
            self.colloid_delta_x_arr[self.colloid_delta_x_arr > self.half_box_size_x] -= 2*self.half_box_size_x
            self.colloid_delta_x_arr[self.colloid_delta_x_arr < -self.half_box_size_x] += 2*self.half_box_size_x
            self.colloid_delta_y_arr[self.colloid_delta_y_arr > self.half_box_size_y] -= 2*self.half_box_size_y
            self.colloid_delta_y_arr[self.colloid_delta_y_arr < -self.half_box_size_y] += 2*self.half_box_size_y
            self.colloid_delta_r_arr[:, :, 0] = self.colloid_delta_x_arr
            self.colloid_delta_r_arr[:, :, 1] = self.colloid_delta_y_arr
            self.colloid_dist_arr = np.sqrt(self.colloid_delta_x_arr**2 + self.colloid_delta_y_arr**2)
            self.interaction_true_table = (self.effective_diameter - self.colloid_dist_arr > 0)
            self.interaction_true_table[range(self.Nc), range(self.Nc)] = False
            self.z_colloid = np.sum(self.interaction_true_table, axis = 1)
            self.colloid_dist_arr[range(self.Nc), range(self.Nc)] = 1 # to avoid division by 0 on diagonal terms"""

        else:
            # find colloids that interact with each other
            #self.colloid_delta_r_arr = self.r_center_matrix[:, None, :] - self.r_center_matrix[None, :, :]
            #self.colloid_dist_arr = np.sqrt((self.colloid_delta_r_arr**2.0).sum(axis=2))
            #self.colloid_dist_arr = pairwise_distances(self.r_center_matrix, self.r_center_matrix)
            #self.interaction_true_table = (2*self.effective_diameter - self.colloid_dist_arr > 0) 
            # larger range to avoid missing as the particles are now deformable
            self.xy_vacuole_min = np.min(self.r_matrix, axis = 2).T
            self.xy_vacuole_max = np.max(self.r_matrix, axis = 2).T
            self.X_pair = (self.xy_vacuole_min[:,0:1] >= self.xy_vacuole_min[:,0] - 2*self.delt*self.bidisperse)*(
                self.xy_vacuole_min[:,0:1] <= self.xy_vacuole_max[:,0] + 2*self.delt*self.bidisperse)
            self.Y_pair = (self.xy_vacuole_min[:,1:2] >= self.xy_vacuole_min[:,1] - 2*self.delt*self.bidisperse)*(
                self.xy_vacuole_min[:,1:2] <= self.xy_vacuole_max[:,1] + 2*self.delt*self.bidisperse)
            self.interaction_true_table = (self.X_pair + self.X_pair.T)*(self.Y_pair + self.Y_pair.T)
            self.interaction_true_table[range(self.Nc), range(self.Nc)] = False
            #self.z_colloid = np.sum(self.interaction_true_table, axis = 1)
            #self.colloid_dist_arr[range(self.Nc), range(self.Nc)] = 1 # to avoid division by 0 on diagonal terms

    def realign_filaments(self):
        """
        realign filaments for better plotting result in periodic BC
        """
        if self.periodic_bc:
            # (Nf, Np-1)
            self.dx = self.f_x_array[:, 1:self.Np] - self.f_x_array[:, 0:self.Np-1]
            self.dy = self.f_y_array[:, 1:self.Np] - self.f_y_array[:, 0:self.Np-1]
            self.dx[self.dx > self.half_box_size_x] -= 2*self.half_box_size_x
            self.dx[self.dx < -self.half_box_size_x] += 2*self.half_box_size_x
            self.dy[self.dy > self.half_box_size_y] -= 2*self.half_box_size_y
            self.dy[self.dy < -self.half_box_size_y] += 2*self.half_box_size_y
            
            for i in range(self.Np - 1):
                self.f_x_array[:, i+1] = self.f_x_array[:, i] + self.dx[:, i]
                self.f_y_array[:, i+1] = self.f_y_array[:, i] + self.dy[:, i]

        else:
            pass

    def get_separation_vectors(self):
        """
        calculate the pair-wise separation vector of a single filament
        """
        if self.periodic_bc:
            # (Nf, Np-1)
            self.dx = self.f_x_array[:, 1:self.Np] - self.f_x_array[:, 0:self.Np-1]
            self.dy = self.f_y_array[:, 1:self.Np] - self.f_y_array[:, 0:self.Np-1]
            self.dx[self.dx > self.half_box_size_x] -= 2*self.half_box_size_x
            self.dx[self.dx < -self.half_box_size_x] += 2*self.half_box_size_x
            self.dy[self.dy > self.half_box_size_y] -= 2*self.half_box_size_y
            self.dy[self.dy < -self.half_box_size_y] += 2*self.half_box_size_y
            
            # (Nf, Np-1)
            # Lengths of the separation vectors
            self.dr = (self.dx**2 + self.dy**2)**(1/2)
            
            # (Nf, Np-1)
            self.dx_hat = self.dx/self.dr
            self.dy_hat = self.dy/self.dr
        else:
            # (Nf, Np-1)
            self.dx = self.f_x_array[:, 1:self.Np] - self.f_x_array[:, 0:self.Np-1]
            self.dy = self.f_y_array[:, 1:self.Np] - self.f_y_array[:, 0:self.Np-1]
            #self.dz = self.r[2*self.Np+1:3*self.Np] - self.r[2*self.Np:3*self.Np-1]
            
            # (Nf, Np-1)
            # Lengths of the separation vectors
            #self.dr = (self.dx**2 + self.dy**2 + self.dz**2)**(1/2)
            self.dr = (self.dx**2 + self.dy**2)**(1/2)
            
            # (Nf, Np-1)
            self.dx_hat = self.dx/self.dr
            self.dy_hat = self.dy/self.dr
            #self.dz_hat = self.dz/self.dr
            
            # rows: dimensions, columns : particles
            # Shape : dim x Np-1
            # Unit separation vectors 
            #self.dr_hat = np.vstack((self.dx_hat, self.dy_hat, self.dz_hat))
            #self.dr_hat = np.vstack((self.dx_hat, self.dy_hat))
            #self.dr_hat = np.array(self.dr_hat, dtype = np.double)

    def change_r_expanded(self, r_expanded):
        self.r_expanded = r_expanded
        self.r = r_expanded[0:self.dim*self.Nc*self.Nv]
        self.r_matrix = self.reshape_to_matrix(self.r)
        #self.update_vacuole_polygons()
        self.update_center_location()
        
        self.f_x_array = np.reshape(
            r_expanded[self.dim*self.Nc*self.Nv: (self.dim*self.Nc*self.Nv+self.Nf*self.Np)], 
            (self.Nf, self.Np), order = 'C')
        self.f_y_array = np.reshape(
            r_expanded[(self.dim*self.Nc*self.Nv+self.Nf*self.Np): (self.dim*self.Nc*self.Nv+2*self.Nf*self.Np)], 
            (self.Nf, self.Np), order = 'C')

    def _Fa(self):
        """
        area expansion/contraction component of the soft colloid.
        """
        self._compute_a()
        self.Fax = -self.ka/2*(self.area_arr - self.area0_arr)*(
            self.r_matrix[1, :, self.axis_1_minus_1].T - self.r_matrix[1, :, self.axis_1_plus_1].T)
        self.Fay = -self.ka/2*(self.area_arr - self.area0_arr)*(
            self.r_matrix[0, :, self.axis_1_plus_1].T - self.r_matrix[0, :, self.axis_1_minus_1].T)

    def _Fb(self):
        self._compute_segment_length()
        self.Fbx = self.vacuole_bending_prefactor*(
            3*self.lx_arr - 3*self.lx_arr[:, self.axis_1_plus_1]
            + self.lx_arr[:, self.axis_1_plus_2] - self.lx_arr[:, self.axis_1_minus_1])
        self.Fby = self.vacuole_bending_prefactor*(
            3*self.ly_arr - 3*self.ly_arr[:, self.axis_1_plus_1] 
            + self.ly_arr[:, self.axis_1_plus_2] - self.ly_arr[:, self.axis_1_minus_1])

    def _Fbf(self):
        """
        bending component of the filament
        """
        # reset forces
        self.Fbfx = self.Fbfx*0
        self.Fbfy = self.Fbfy*0

        # compute forces
        self.Fbfx[:, 0] = self.filament_bending_prefactor*(
            (self.dx[:,0] - self.dx[:,1])*self.ER_intact[:,0]*self.ER_intact[:,1])
        self.Fbfy[:, 0] = self.filament_bending_prefactor*(
            (self.dy[:,0] - self.dy[:,1])*self.ER_intact[:,0]*self.ER_intact[:,1])
        self.Fbfx[:, 1] = self.filament_bending_prefactor*(
            -2*(self.dx[:,0]-self.dx[:,1]*self.ER_intact[:,0]*self.ER_intact[:,1])
            +(self.dx[:,1] - self.dx[:,2]*self.ER_intact[:,1]*self.ER_intact[:,2]))
        self.Fbfy[:, 1] = self.filament_bending_prefactor*(
            -2*(self.dy[:,0]-self.dy[:,1]*self.ER_intact[:,0]*self.ER_intact[:,1])
            +(self.dy[:,1] - self.dy[:,2]*self.ER_intact[:,1]*self.ER_intact[:,2]))
        self.Fbfx[:, 2:self.Np-2] = self.filament_bending_prefactor*(
            (self.dx[:, 0:self.Np-4] - self.dx[:, 1:self.Np-3])*self.ER_intact[:,0:self.Np-4]*self.ER_intact[:,1:self.Np-3]
            -2*(self.dx[:, 1:self.Np-3] - self.dx[:, 2:self.Np-2])*self.ER_intact[:,1:self.Np-3]*self.ER_intact[:,2:self.Np-2]
            +(self.dx[:, 2:self.Np-2] - self.dx[:, 3:self.Np-1])*self.ER_intact[:,2:self.Np-2]*self.ER_intact[:,3:self.Np-1])
        self.Fbfy[:, 2:self.Np-2] = self.filament_bending_prefactor*(
            (self.dy[:, 0:self.Np-4] - self.dy[:, 1:self.Np-3])*self.ER_intact[:,0:self.Np-4]*self.ER_intact[:,1:self.Np-3]
            -2*(self.dy[:, 1:self.Np-3] - self.dy[:, 2:self.Np-2])*self.ER_intact[:,1:self.Np-3]*self.ER_intact[:,2:self.Np-2]
            +(self.dy[:, 2:self.Np-2] - self.dy[:, 3:self.Np-1])*self.ER_intact[:,2:self.Np-2]*self.ER_intact[:,3:self.Np-1])
        self.Fbfx[:, self.Np-1] = self.filament_bending_prefactor*(
            (self.dx[:,self.Np-3] - self.dx[:,self.Np-2])*self.ER_intact[:,self.Np-3]*self.ER_intact[:,self.Np-2])
        self.Fbfy[:, self.Np-1] = self.filament_bending_prefactor*(
            (self.dy[:,self.Np-3] - self.dy[:,self.Np-2])*self.ER_intact[:,self.Np-3]*self.ER_intact[:,self.Np-2])
        self.Fbfx[:, self.Np-2] = self.filament_bending_prefactor*(
            (self.dx[:,self.Np-4] - self.dx[:,self.Np-3])*self.ER_intact[:,self.Np-4]*self.ER_intact[:,self.Np-3]
            -2*(self.dx[:,self.Np-3] - self.dx[:,self.Np-2])*self.ER_intact[:,self.Np-3]*self.ER_intact[:,self.Np-2])
        self.Fbfy[:, self.Np-2] = self.filament_bending_prefactor*(
            (self.dy[:,self.Np-4] - self.dy[:,self.Np-3])*self.ER_intact[:,self.Np-4]*self.ER_intact[:,self.Np-3]
            -2*(self.dy[:,self.Np-3] - self.dy[:,self.Np-2])*self.ER_intact[:,self.Np-3]*self.ER_intact[:,self.Np-2])

    def _Flc(self):
        '''
        contractility component of the colloidal surface
        '''
        self._compute_segment_length()
        neg = self.l_arr - self.l0_arr < 0
        neg_plus1 = self.l_arr[:, self.axis_1_plus_1] - self.l0_arr < 0
        self.Flcx = (-self.kl_c_neg*self.Nv*(-(self.l_arr - self.l0_arr)/self.l_arr*self.lx_arr*neg + 
                      (self.l_arr[:, self.axis_1_plus_1] - self.l0_arr)/self.l_arr[:, self.axis_1_plus_1]*self.lx_arr[:, self.axis_1_plus_1]*neg_plus1) 
               -self.kl_c_pos*self.Nv*(-(self.l_arr - self.l0_arr)/self.l_arr*self.lx_arr*(~neg) + 
                      (self.l_arr[:, self.axis_1_plus_1] - self.l0_arr)/self.l_arr[:, self.axis_1_plus_1]*self.lx_arr[:, self.axis_1_plus_1]*(
                        ~neg_plus1)))
        self.Flcy = (-self.kl_c_neg*self.Nv*(-(self.l_arr - self.l0_arr)/self.l_arr*self.ly_arr*neg + 
                      (self.l_arr[:, self.axis_1_plus_1] - self.l0_arr)/self.l_arr[:, self.axis_1_plus_1]*self.ly_arr[:, self.axis_1_plus_1]*neg_plus1) 
               -self.kl_c_pos*self.Nv*(-(self.l_arr - self.l0_arr)/self.l_arr*self.ly_arr*(~neg) + 
                      (self.l_arr[:, self.axis_1_plus_1] - self.l0_arr)/self.l_arr[:, self.axis_1_plus_1]*self.ly_arr[:, self.axis_1_plus_1]*(
                        ~neg_plus1)))

    def _Fl(self):
        '''
        contractility component of filaments
        '''
        # reset forces
        self.Flx = self.Flx*0
        self.Fly = self.Fly*0

        # compute forces
        self.Fl_expanded = self.kl*(self.Np-1)*(self.dr - self.dLf)/self.dr

        self.Flx[:, 1:self.Np-1] = -(self.Fl_expanded[:, 0:self.Np-2]*self.dx[:, 0:self.Np-2]*self.ER_intact[:, 0:self.Np-2] 
            - self.Fl_expanded[:, 1:]*self.dx[:, 1:]*self.ER_intact[:, 1:])
        self.Fly[:, 1:self.Np-1] = -(self.Fl_expanded[:, 0:self.Np-2]*self.dy[:, 0:self.Np-2]*self.ER_intact[:, 0:self.Np-2]
            - self.Fl_expanded[:, 1:]*self.dy[:, 1:]*self.ER_intact[:, 1:])
        self.Flx[:, 0] = (self.Fl_expanded[:, 0]*self.dx[:, 0]*self.ER_intact[:,0])
        self.Flx[:, self.Np-1] = -(self.Fl_expanded[:, self.Np-2]*self.dx[:, self.Np-2]*self.ER_intact[:, self.Np-2])
        self.Fly[:, 0] = (self.Fl_expanded[:, 0]*self.dy[:, 0]*self.ER_intact[:,0])
        self.Fly[:, self.Np-1] = -(self.Fl_expanded[:, self.Np-2]*self.dy[:, self.Np-2]*self.ER_intact[:, self.Np-2])

    def _Fr(self):
        '''
        repulsive component of filament with respect to the connecting vacuole, to avoid filament overriding the vacuoles.
        '''
        # reset forces
        self.Frx = self.Frx * 0
        self.Fry = self.Fry * 0

        if self.Nf == 0:
            pass
        else:
            # compute forces
            if self.full_repulsion:
                fp_r_array = np.hstack([self.f_x_array.reshape((self.Nf*self.Np,1)), self.f_y_array.reshape((self.Nf*self.Np,1))])
                fp_c_interaction_true_table = ((fp_r_array[:,0:1] >= self.xy_vacuole_min[:,0] - 2*self.delt*self.bidisperse)
                               *(fp_r_array[:,0:1] <= self.xy_vacuole_max[:,0] + 2*self.delt*self.bidisperse)
                               *(fp_r_array[:,1:2] >= self.xy_vacuole_min[:,1] - 2*self.delt*self.bidisperse)
                               *(fp_r_array[:,1:2] <= self.xy_vacuole_max[:,1] + 2*self.delt*self.bidisperse))
                fp_c_interaction_true_table[np.arange(0, self.Nf*self.Np, self.Np), self.connection_table[:,0]] = False
                fp_c_interaction_true_table[np.arange(self.Np-1, self.Nf*self.Np, self.Np), self.connection_table[:,1]] = False

                fp_all, c_all = np.where(fp_c_interaction_true_table)

                # point is within canvas
                xy_x = self.r_matrix[0, c_all, :].T
                xy_y = self.r_matrix[1, c_all, :].T
                xy2_x = xy_x[self.axis_1_plus_1, :]
                xy2_y = xy_y[self.axis_1_plus_1, :]
                xs = fp_r_array[fp_all, 0]
                ys = fp_r_array[fp_all, 1]

                v1x = xy_x - xs
                v1y = xy_y - ys
                v2x = xy2_x - xs
                v2y = xy2_y - ys

                ang = angle_xy(v1x, v1y, v2x, v2y)
                within_vacuole = (((np.abs(np.einsum('ij->j', ang)) - (2*PI))*(10**9)).astype(int)/(10.**9) == 0)

                DELTA_eff = self.delta_wall[0,c_all]
                DX = (xs[None, :] - xy_x)
                DY = (ys[None, :] - xy_y)
                DR = np.sqrt(DX**2 + DY**2)
                DELTA_eff_M_DR = DELTA_eff[None, :] - DR
                TF = (DELTA_eff_M_DR > 0)
                vertex_all, pair_all = np.where(TF)

                fx = self.kr*(TF*DX/(DR+1E-10)*(DELTA_eff_M_DR))*((-1)**(within_vacuole))
                fy = self.kr*(TF*DY/(DR+1E-10)*(DELTA_eff_M_DR))*((-1)**(within_vacuole))

                f_collide, p_collide = divmod(fp_all.reshape((-1,1)), self.Np)
                np.add.at(self.Frx, [f_collide, p_collide], np.sum(fx, axis = 0, keepdims = True).T)
                np.add.at(self.Fry, [f_collide, p_collide], np.sum(fy, axis = 0, keepdims = True).T)
                np.add.at(self.Fv, [c_all[pair_all], vertex_all, 0], -fx[vertex_all, pair_all])
                np.add.at(self.Fv, [c_all[pair_all], vertex_all, 1], -fy[vertex_all, pair_all])

                pair_collide, = np.where(within_vacuole)
                # find min dist to vac border
                xy_x2 = xy_x[:, pair_collide]
                xy_y2 = xy_y[:, pair_collide]
                xy2_x2 = xy2_x[:, pair_collide]
                xy2_y2 = xy2_y[:, pair_collide]
                xs2 = xs[pair_collide, None]
                ys2 = ys[pair_collide, None]
                a = xy2_y2 - xy_y2
                b = - (xy2_x2 - xy_x2)
                c = xy_y2*(xy2_x2 - xy_x2) - xy_x2*(xy2_y2 - xy_y2)
                M = a**2 + b**2
                N = b*xs2.T - a*ys2.T
                dist_perp = np.abs(a*xs2.T + b*ys2.T + c)/np.sqrt(M)
                xp = (b*N - a*c)/M
                yp = (a*(-N) - b*c)/M
                fx = self.kr*(xp - xs2.T)
                fy = self.kr*(yp - ys2.T)

                idx_edge = np.argmin(dist_perp, axis = 0)
                n_site = len(xs2)
                f_x = fx[idx_edge, range(n_site)].reshape((n_site, 1))
                f_y = fy[idx_edge, range(n_site)].reshape((n_site, 1))

                # add force
                f_collide, p_collide = divmod(fp_all[pair_collide].reshape((-1,1)), self.Np)

                np.add.at(self.Frx, [f_collide, p_collide], f_x)
                np.add.at(self.Fry, [f_collide, p_collide], f_y)
                np.add.at(self.Fv, [c_all[pair_collide].reshape((-1,1)), np.vstack([idx_edge-1, idx_edge]).T, 0], -f_x/2)
                np.add.at(self.Fv, [c_all[pair_collide].reshape((-1,1)), np.vstack([idx_edge-1, idx_edge]).T, 1], -f_y/2)

                if self.periodic_bc:
                    print("Periodic BC is not supported")
                    """
                    self.dx_to_vac[self.dx_to_vac > self.half_box_size_x] -= 2*self.half_box_size_x
                    self.dx_to_vac[self.dx_to_vac < -self.half_box_size_x] += 2*self.half_box_size_x
                    self.dy_to_vac[self.dy_to_vac > self.half_box_size_y] -= 2*self.half_box_size_y
                    self.dy_to_vac[self.dy_to_vac < -self.half_box_size_y] += 2*self.half_box_size_y"""

                #self.dr_to_vac = (self.dx_to_vac**2 + self.dy_to_vac**2)**(0.5)
                #self.Frx_expanded = (self.radius[None, :, None] - self.dr_to_vac)*(
                #    self.radius[None, :, None] > self.dr_to_vac)/self.dr_to_vac*self.dx_to_vac
                #self.Fry_expanded = (self.radius[None, :, None] - self.dr_to_vac)*(
                #    self.radius[None, :, None] > self.dr_to_vac)/self.dr_to_vac*self.dy_to_vac
                
                #self.Frx[:, 1:self.Np-1] += self.kr*np.sum(self.Frx_expanded, axis = 1)
                #self.Fry[:, 1:self.Np-1] += self.kr*np.sum(self.Fry_expanded, axis = 1)
                #self.F[0:self.Nc] -= self.kr*np.sum(self.Frx_expanded, axis = (0,2))
                #self.F[self.Nc:2*self.Nc] -= self.kr*np.sum(self.Fry_expanded, axis = (0,2))

            else:
                # DISABLED
                print('non full repulsion version is disabled')
                """
                self.dx_to_vac0 = self.f_x_array[:, 1:self.Np-1] - self.r_matrix[self.connection_table[:, 0], 0:1]
                self.dx_to_vac1 = self.f_x_array[:, 1:self.Np-1] - self.r_matrix[self.connection_table[:, 1], 0:1]
                self.dy_to_vac0 = self.f_y_array[:, 1:self.Np-1] - self.r_matrix[self.connection_table[:, 0], 1:2]
                self.dy_to_vac1 = self.f_y_array[:, 1:self.Np-1] - self.r_matrix[self.connection_table[:, 1], 1:2]

                if self.periodic_bc:
                    self.dx_to_vac0[self.dx_to_vac0 > self.half_box_size_x] -= 2*self.half_box_size_x
                    self.dx_to_vac0[self.dx_to_vac0 < -self.half_box_size_x] += 2*self.half_box_size_x
                    self.dx_to_vac1[self.dx_to_vac1 > self.half_box_size_x] -= 2*self.half_box_size_x
                    self.dx_to_vac1[self.dx_to_vac1 < -self.half_box_size_x] += 2*self.half_box_size_x
                    self.dy_to_vac0[self.dy_to_vac0 > self.half_box_size_y] -= 2*self.half_box_size_y
                    self.dy_to_vac0[self.dy_to_vac0 < -self.half_box_size_y] += 2*self.half_box_size_y
                    self.dy_to_vac1[self.dy_to_vac1 > self.half_box_size_y] -= 2*self.half_box_size_y
                    self.dy_to_vac1[self.dy_to_vac1 < -self.half_box_size_y] += 2*self.half_box_size_y

                self.dr_to_vac0 = (self.dx_to_vac0**2 + self.dy_to_vac0**2)**(0.5)
                self.dr_to_vac1 = (self.dx_to_vac1**2 + self.dy_to_vac1**2)**(0.5)
                
                self.Frx[:, 1:self.Np-1] += self.kr*(self.r_vac0 - self.dr_to_vac0)*(
                    self.r_vac0 > self.dr_to_vac0)/self.dr_to_vac0*self.dx_to_vac0
                self.Frx[:, 1:self.Np-1] += self.kr*(self.r_vac1 - self.dr_to_vac1)*(
                    self.r_vac1 > self.dr_to_vac1)/self.dr_to_vac1*self.dx_to_vac1
                self.Fry[:, 1:self.Np-1] += self.kr*(self.r_vac0 - self.dr_to_vac0)*(
                    self.r_vac0 > self.dr_to_vac0)/self.dr_to_vac0*self.dy_to_vac0
                self.Fry[:, 1:self.Np-1] += self.kr*(self.r_vac1 - self.dr_to_vac1)*(
                    self.r_vac1 > self.dr_to_vac1)/self.dr_to_vac1*self.dy_to_vac1"""
    
    def _Fgamma(self):
        '''
        tension component
        '''
        self.Fgx = -self.gamma*(-self.lx_arr/self.l_arr + self.lx_arr[:, self.axis_1_plus_1]/self.l_arr[:, self.axis_1_plus_1])
        self.Fgy = -self.gamma*(-self.ly_arr/self.l_arr + self.ly_arr[:, self.axis_1_plus_1]/self.l_arr[:, self.axis_1_plus_1])

    def compute_potential_energy(self, r_expanded):
        # Set the current filament state
        self.drEdt = self.drEdt * 0
        self.drdt = self.drdt * 0
        self.dfil_x_dt = self.dfil_x_dt * 0
        self.dfil_y_dt = self.dfil_y_dt * 0
        
        self.r = r_expanded[0:self.dim*self.Nc*self.Nv]
        self.r_matrix = self.reshape_to_matrix(self.r)

        self._compute_a()
        self._compute_segment_length()
        
        self.f_x_array = np.reshape(
            r_expanded[self.dim*self.Nc*self.Nv:(self.dim*self.Nc*self.Nv+self.Nf*self.Np)], 
            (self.Nf, self.Np), order = 'C')
        self.f_y_array = np.reshape(
            r_expanded[(self.dim*self.Nc*self.Nv+self.Nf*self.Np):(self.dim*self.Nc*self.Nv+2*self.Nf*self.Np)], 
            (self.Nf, self.Np), order = 'C')
        
        # boundary condition on attachment site: free to rotate, but not free to move
        self.f_x_array[:, 0] = self.r_matrix[0, self.connection_table[:, 0], self.connection_vertex[:, 0]]
        self.f_x_array[:, self.Np-1] = self.r_matrix[0, self.connection_table[:, 1], self.connection_vertex[:, 1]]
        self.f_y_array[:, 0] = self.r_matrix[1, self.connection_table[:, 0], self.connection_vertex[:, 0]]
        self.f_y_array[:, self.Np-1] = self.r_matrix[1, self.connection_table[:, 1], self.connection_vertex[:, 1]]
        
        if self.periodic_bc:
            print("periodic BC is currently disabled")
            """
            self.r[0:self.Nc] = (self.r[0:self.Nc] - self.xmin)%(2*self.half_box_size_x) + self.xmin
            self.r[self.Nc:2*self.Nc] = (self.r[self.Nc:2*self.Nc] - self.ymin)%(2*self.half_box_size_y) + self.ymin
            self.r_matrix = self.reshape_to_matrix(self.r)
            self.f_x_array = (self.f_x_array - self.xmin)%(2*self.half_box_size_x) + self.xmin
            self.f_y_array = (self.f_y_array - self.ymin)%(2*self.half_box_size_y) + self.ymin

            # calculate geometric quantities
            self.get_separation_vectors()
            self.get_distance_bet_colloid()

            # repulsive energy among vertices of colloids
            U_rep_colloid = 0
            i_all, j_all = np.where(np.triu(self.interaction_true_table))
            for i, j in zip(i_all, j_all):
                vertex_delta_x_arr =(self.r_matrix[0, i, :].T)[:, None] - (self.r_matrix[0, j, :].T)
                vertex_delta_y_arr =(self.r_matrix[1, i, :].T)[:, None] - (self.r_matrix[1, j, :].T)
                vertex_delta_x_arr[vertex_delta_x_arr > self.half_box_size_x] -= 2*self.half_box_size_x
                vertex_delta_x_arr[vertex_delta_x_arr < -self.half_box_size_x] += 2*self.half_box_size_x
                vertex_delta_y_arr[vertex_delta_y_arr > self.half_box_size_y] -= 2*self.half_box_size_y
                vertex_delta_y_arr[vertex_delta_y_arr < -self.half_box_size_y] += 2*self.half_box_size_y
                vertex_delta_r_arr = np.zeros((self.Nv, self.Nv, self.dim))
                vertex_delta_r_arr[:, :, 0] = vertex_delta_x_arr
                vertex_delta_r_arr[:, :, 1] = vertex_delta_y_arr
                vertex_dist_arr = np.sqrt((vertex_delta_r_arr**2.0).sum(axis=2))
                vertex_interaction_true_table = (self.effective_diam_delta[i,j] - vertex_dist_arr > 0)
                U_rep_colloid += 1/2*self.kr*np.sum(
                    (self.effective_diam_delta[i,j] - vertex_dist_arr)**2*vertex_interaction_true_table)

            Ua = self._Ua() # area change of vacuoles
            Ub = self._Ub() # bending of vacuole surface
            Ulc = self._Ulc() # stretching of vacuole membrane
            Ugamma = self._Ugamma() # surface tension of vacuoles
            Ul = self._Ul() # stretching of strings
            Ur = self._Ur() # repulsion among strings and vacuoles
            Urb = 0 # repulsion of vacuoles and boundaries
            
            U_total = U_rep_colloid + Ua + Ub + Ulc + Ugamma + Ul + Ur + Urb
            U_total_average = U_total/self.Nc
            U_colloid_average = (U_rep_colloid + Ua + Ub + Ulc + Ugamma + Urb)/self.Nc
            result = {
                  'U_rep_colloid': U_rep_colloid,
                  'Ua': Ua,
                  'Ub': Ub,
                  'Ulc': Ulc,
                  'Ugamma': Ugamma,
                  'Ul': Ul, 
                  'Ur': Ur, 
                  'Urb': Urb, 
                  'U_total': U_total, 
                  'U_total_average': U_total_average, 
                  'U_colloid_average': U_colloid_average
                  }"""

        else:
            # calculate geometric quantities
            self.get_separation_vectors()
            self.get_distance_bet_colloid()

            # repulsive energy among vertices of colloids
            U_rep_colloid = 0
            #i_all, j_all = np.where(np.triu(self.interaction_true_table))
            i_all, j_all = np.where(self.interaction_true_table) # interaction true table: Nc x Nc
            #i_all & j_all: length of interaction particle pairs(n_possible_interaction).

            # point is within canvas, force from i to j
            xy_x = self.r_matrix[0, i_all, :].T # shape: (Nv, n_possible_interaction), i
            xy_y = self.r_matrix[1, i_all, :].T
            xy2_x = xy_x[self.axis_1_plus_1, :] # i
            xy2_y = xy_y[self.axis_1_plus_1, :]
            xs = self.r_matrix[0, j_all, :].T # j
            ys = self.r_matrix[1, j_all, :].T

            v1x = xy_x[None, :, :] - xs[:, None, :]
            v1y = xy_y[None, :, :] - ys[:, None, :]
            v2x = xy2_x[None, :, :] - xs[:, None, :]
            v2y = xy2_y[None, :, :] - ys[:, None, :]

            ang = angle_xy(v1x, v1y, v2x, v2y)
            result = (((np.abs(np.einsum('ijk->ik', ang)) - (2*PI))*(10**9)).astype(int)/(10.**9) == 0) 
            # shape: (Nv, n_possible_interaction), which vertex of j is inside the polygon of i

            collision_pair, = np.where(np.einsum('ij->j', result) != 0) # shape: n_collision
            idx_vertex, idx_pair = np.where(result) 
            # the vertex in j that collide with i inside and the id of the pair

            DELTA_eff = self.effective_diam_delta_simple[i_all, j_all]
            DX = (xs[:, None, :] - xy_x[None, :, :])
            DY = (ys[:, None, :] - xy_y[None, :, :])
            DR = np.sqrt(DX**2 + DY**2)
            DELTA_eff_M_DR = DELTA_eff[None, None, :] - DR
            TF = (DELTA_eff_M_DR > 0)
            U_rep_colloid = 1/2*self.kr*np.sum(TF*DELTA_eff_M_DR**2)

            """
            # find the force that need to account
            n_collision = len(idx_pair)
            xy_x2 = xy_x[:, idx_pair] # shape: (Nv, n_collision), for i
            xy_y2 = xy_y[:, idx_pair]
            xy2_x2 = xy2_x[:, idx_pair] # i
            xy2_y2 = xy2_y[:, idx_pair]
            xs2 = xs[idx_vertex, idx_pair].reshape((n_collision,1)) # j
            ys2 = ys[idx_vertex, idx_pair].reshape((n_collision,1))
            a = xy2_y2 - xy_y2
            b = - (xy2_x2 - xy_x2)
            c = xy_y2*(xy2_x2 - xy_x2) - xy_x2*(xy2_y2 - xy_y2)
            M = a**2 + b**2
            N = b*xs2.T - a*ys2.T
            dist_perp = np.abs(a*xs2.T + b*ys2.T + c)/np.sqrt(M)
            xp = (b*N - a*c)/M # shape: (Nv, n_collision)
            yp = (a*(-N) - b*c)/M
            fx = (xp - xs2.T) # shape: (Nv, n_collision)
            fy = (yp - ys2.T)

            # use cosine to determine which edges should be included for energy
            dx_vert_to_center = xs2 - np.mean(xs[:, idx_pair], axis = 0).reshape((n_collision, 1))
            dy_vert_to_center = ys2 - np.mean(ys[:, idx_pair], axis = 0).reshape((n_collision, 1))
            dx_vert_to_xp = xs2 - xp.T
            dy_vert_to_yp = ys2 - yp.T
            included = (dx_vert_to_center*dx_vert_to_xp + dy_vert_to_center*dy_vert_to_yp > 0).T
            #U_rep_colloid += 1/2*self.kr/self.Nv**2*np.sum((fx*included)**2 + (fy*included)**2)
            U_rep_colloid += 1/2*self.kr*np.sum((fx*included)**2 + (fy*included)**2)
            #normalized by Nv**2 to ensure not increasing U just by increasing discretization points
            """

            Ua = self._Ua() # area change of vacuoles
            Ub = self._Ub() # bending of vacuole surface
            Ulc = self._Ulc() # stretching of vacuole membrane
            Ugamma = self._Ugamma() # surface tension of vacuoles
            Urb = self._Urb() # repulsion of vacuoles and boundaries
            Urbf = self._Urbf() # repulsion of strings and boundaries
            Ul = self._Ul() # stretching of strings
            Ubf = self._Ubf() # bending of filaments
            Ur = self._Ur() # repulsion among strings and vacuoles      
            
            U_total = U_rep_colloid + Ua + Ub + Ulc + Ugamma + Ul + Ur + Urb + Ubf + Urbf
            U_total_average = U_total/self.Nc
            U_colloid_average = (U_rep_colloid + Ua + Ub + Ulc + Ugamma + Urb)/self.Nc
            result = {
                  'U_rep_colloid': U_rep_colloid,
                  'Ua': Ua,
                  'Ub': Ub,
                  'Ubf': Ubf,
                  'Ulc': Ulc,
                  'Ugamma': Ugamma,
                  'Ul': Ul, 
                  'Ur': Ur, 
                  'Urb': Urb, 
                  'Urbf': Urbf,
                  'U_total': U_total, 
                  'U_total_average': U_total_average, 
                  'U_colloid_average': U_colloid_average
                  }
        return result
    
    def _Ua(self):
        '''
        potential energy from the compressibility component
        '''
        self._compute_a()
        Ua = self.ka/2*np.sum((self.area_arr - self.area0_arr)**2)
        return Ua

    def _Ub(self):
        '''
        potential energy from the bending component of vacuole surface
        '''
        lx_arr_normalized = self.lx_arr/self.l_arr
        ly_arr_normalized = self.ly_arr/self.l_arr
        
        denominator = 2*np.sqrt((lx_arr_normalized - lx_arr_normalized[:, self.axis_1_plus_1])**2 + 
                (ly_arr_normalized - ly_arr_normalized[:, self.axis_1_plus_1])**2)
        Ub = self.kb/(2*self.Nv)*np.sum((denominator/(self.l_arr + self.l_arr[:, self.axis_1_plus_1]))**2)
        return Ub

    def _Ubf(self):
        """
        potential energy from the bending component of the filament
        """
        Ubf = self.filament_bending_prefactor/2*np.sum(
            (self.dx[:, 0:self.Np-2] - self.dx[:, 1:self.Np-1])**2*self.ER_intact[:, 0:self.Np-2]*self.ER_intact[:, 1:self.Np-1] + 
            (self.dy[:, 0:self.Np-2] - self.dy[:, 1:self.Np-1])**2*self.ER_intact[:, 0:self.Np-2]*self.ER_intact[:, 1:self.Np-1])
        return Ubf

    def _Ulc(self):
        """
        potential energy from the stretching/compressing of colloid membrane
        """
        Ul = (self.kl_c_pos*self.Nv/2*np.sum((self.l_arr - self.l0_arr)**2*(self.l_arr > self.l0_arr)) 
            + self.kl_c_neg*self.Nv/2*np.sum((self.l_arr - self.l0_arr)**2*(self.l_arr < self.l0_arr)))
        return Ul

    def _Ugamma(self):
        '''
        tension component
        '''
        Ug = self.gamma*np.sum(self.l_arr)
        return Ug

    def _Ul(self):
        '''
        potential energy from the inextensibility of strings
        '''
        Ul = self.kl*(self.Np-1)/2*np.sum((self.dr - self.dLf)**2*self.ER_intact)
        return Ul
    
    def _Ur(self):
        '''
        repulsive component of filament with respect to the conecting vacuole, to avoid filament overriding the vacuoles.
        '''
        if self.Nf == 0:
            Ur = 0
        else:
            if self.full_repulsion:
                Ur = 0

                fp_r_array = np.hstack([self.f_x_array.reshape((-1,1)), self.f_y_array.reshape((-1,1))])
                fp_c_interaction_true_table = ((fp_r_array[:,0:1] >= self.xy_vacuole_min[:,0] - 2*self.delt*self.bidisperse)
                               *(fp_r_array[:,0:1] <= self.xy_vacuole_max[:,0] + 2*self.delt*self.bidisperse)
                               *(fp_r_array[:,1:2] >= self.xy_vacuole_min[:,1] - 2*self.delt*self.bidisperse)
                               *(fp_r_array[:,1:2] <= self.xy_vacuole_max[:,1] + 2*self.delt*self.bidisperse))
                fp_c_interaction_true_table[np.arange(0, self.Nf*self.Np, self.Np), self.connection_table[:,0]] = False
                fp_c_interaction_true_table[np.arange(self.Np-1, self.Nf*self.Np, self.Np), self.connection_table[:,1]] = False

                fp_all, c_all = np.where(fp_c_interaction_true_table)

                # point is within canvas
                xy_x = self.r_matrix[0, c_all, :].T
                xy_y = self.r_matrix[1, c_all, :].T
                xy2_x = xy_x[self.axis_1_plus_1, :]
                xy2_y = xy_y[self.axis_1_plus_1, :]
                xs = fp_r_array[fp_all, 0]
                ys = fp_r_array[fp_all, 1]

                v1x = xy_x - xs
                v1y = xy_y - ys
                v2x = xy2_x - xs
                v2y = xy2_y - ys

                ang = angle_xy(v1x, v1y, v2x, v2y)
                within_vacuole = (((np.abs(np.sum(ang, axis = 0)) - (2*PI))*(10**9)).astype(int)/(10.**9) == 0)

                DELTA_eff = self.delta_wall[0,c_all]
                DX = (xs[None, :] - xy_x)
                DY = (ys[None, :] - xy_y)
                DR = np.sqrt(DX**2 + DY**2)
                DELTA_eff_M_DR = DELTA_eff[None, :] - DR
                TF = (DELTA_eff_M_DR > 0)
                #vertex_all, pair_all = np.where(TF)

                Ur += 1/2*self.kr*np.sum(DELTA_eff_M_DR**2*TF)

                
                pair_collide, = np.where(within_vacuole)
                f_collide, p_collide = divmod(fp_all[pair_collide], self.Np)
                unique_collision_pair = np.unique(np.vstack([f_collide, c_all[pair_collide]]).T, axis = 0)
                string_colloid_collision_table = np.zeros((self.Nf, self.Nc))
                string_colloid_collision_table[unique_collision_pair[:,0], unique_collision_pair[:,1]] = 1
                self.z_filaments = np.sum(string_colloid_collision_table, axis = 0)

                # find min dist to vac border
                xy_x2 = xy_x[:, pair_collide]
                xy_y2 = xy_y[:, pair_collide]
                xy2_x2 = xy2_x[:, pair_collide]
                xy2_y2 = xy2_y[:, pair_collide]
                xs2 = xs[pair_collide, None]
                ys2 = ys[pair_collide, None]
                a = xy2_y2 - xy_y2
                b = - (xy2_x2 - xy_x2)
                c = xy_y2*(xy2_x2 - xy_x2) - xy_x2*(xy2_y2 - xy_y2)
                dist_perp = np.abs(a*xs2.T + b*ys2.T + c)/np.sqrt(a**2 + b**2)
                xp = (b*(b*xs2.T - a*ys2.T) - a*c)/(a**2 + b**2)
                yp = (a*(- b*xs2.T + a*ys2.T) - b*c)/(a**2 + b**2)
                fx = (xp - xs2.T)
                fy = (yp - ys2.T)

                idx_edge = np.argmin(dist_perp, axis = 0)
                n_site = len(xs2)
                f_x = fx[idx_edge, range(n_site)].reshape((n_site, 1))
                f_y = fy[idx_edge, range(n_site)].reshape((n_site, 1))

                #Ur += 1/2*self.kr/self.Np*np.sum(f_x**2 + f_y**2)
                Ur += 1/2*self.kr*np.sum(f_x**2 + f_y**2)
                """
                self.dx_to_vac = self.f_x_array[:, None, 1:self.Np-1] - self.r_matrix[:, 0:1][None, :, :]
                self.dy_to_vac = self.f_y_array[:, None, 1:self.Np-1] - self.r_matrix[:, 1:2][None, :, :]

                if self.periodic_bc:
                    self.dx_to_vac[self.dx_to_vac > self.half_box_size_x] -= 2*self.half_box_size_x
                    self.dx_to_vac[self.dx_to_vac < -self.half_box_size_x] += 2*self.half_box_size_x
                    self.dy_to_vac[self.dy_to_vac > self.half_box_size_y] -= 2*self.half_box_size_y
                    self.dy_to_vac[self.dy_to_vac < -self.half_box_size_y] += 2*self.half_box_size_y

                else:
                    pass

                self.dr_to_vac = (self.dx_to_vac**2 + self.dy_to_vac**2)**(0.5)
                primitive_interaction = (np.sum(self.radius[None, :, None] > self.dr_to_vac, axis = 2) > 0)
                filtered_interaction = np.copy(primitive_interaction)
                for ic, connection in enumerate(self.connection_table):
                    filtered_interaction[ic, connection] = False
                self.z_filaments = np.sum(primitive_interaction, axis = 0)
                self.z_filaments_filtered = np.sum(filtered_interaction, axis = 0)
                Ur = 1/2*self.kr*np.sum((self.radius[None, :, None] - self.dr_to_vac)**2*(self.radius[None, :, None] > self.dr_to_vac))"""

            else:
                print('non full repulsion version is disabled')
        return Ur
    
    def _Urb(self):
        '''
        repulsive component from the wall to the vacuoles
        '''
        if self.periodic_bc:
            Urb = 0
        else:
            Urb, within_canvas = self.canvas.find_wall_potential_vectorized_heuristic(
                self.r_matrix[0,:,:].reshape((self.Nc*self.Nv,1)), 
                self.r_matrix[1,:,:].reshape((self.Nc*self.Nv,1)), 
                self.delta_wall)
            #Urb = self.kr_b*self.kr/self.Nv*np.sum(Urb)
            Urb = self.kr_b*self.kr*np.sum(Urb)

            # compute z_wall
            #wall_collision = U_all*within_canvas > 0 
            #wall_collision[range(self.Nc*self.Nv), min_dist_loc] = (1-within_canvas).flatten()
            #cv_collide, edge_collide = np.where(wall_collision)
            #c_collide, v_collide = divmod(cv_collide, self.Nv)
            #unique_wall_collision = np.unique(np.vstack([c_collide, edge_collide]).T, axis = 0)
            #wall_collision_table = np.zeros((self.Nc, self.canvas.n_corners))
            #wall_collision_table[unique_wall_collision[:,0], unique_wall_collision[:,1]] = 1
            #self.z_wall = np.sum(wall_collision_table, axis = 1)
        return Urb

    def _Urbf(self):
        """
        repulsive component from the wall to the strings
        """
        if self.periodic_bc:
            Urbf = 0
        else:
            if self.kr_bf > 0:
                Urbf, _ = self.canvas.find_wall_potential_vectorized_heuristic(
                    self.f_x_array.reshape((self.Nf*self.Np, 1)), 
                    self.f_y_array.reshape((self.Nf*self.Np, 1)), 
                    self.delta_string)
                Urbf = self.kr_bf*self.kr*np.sum(Urbf)
            else:
                Urbf = 0
        return Urbf
    
    def _KE(self, r_expanded, t):
        self.rhs_cython(r_expanded, t)
        KE = 1/2*np.sum((self.drEdt)**2)/len(self.drEdt)
        KE_velocity_ave = 1/2*np.sum((self.drEdt[0:self.dim*self.Nc*self.Nv])**2)/self.Nc
        result = {'KE': KE, 'KE_velocity_ave': KE_velocity_ave}
        return result
    
    def plot_forces(self, r_expanded, t, plot_string = True, plot_colloid = False):
        """
        Plot the forces acting on the colloids and filaments
        """
        self.rhs_cython(r_expanded, t)
        self.plot_system()

        if self.periodic_bc:
            if plot_string:
                for ix in range(-1, 2):
                    for iy in range(-1, 2):
                        plt.quiver(self.f_x_array.flatten() + ix*2*self.half_box_size_x, 
                            self.f_y_array.flatten() + iy*2*self.half_box_size_y, 
                            (self.Flx + self.Frx + self.Fbfx).flatten(), (self.Fly + self.Fry + self.Fbfy).flatten())
            if plot_colloid:
                for ix in range(-1, 2):
                    for iy in range(-1, 2):
                        plt.quiver(self.r[0:self.Nc] + ix*2*self.half_box_size_x, 
                            self.r[self.Nc:2*self.Nc] + iy*2*self.half_box_size_y, self.F[0:self.Nc], self.F[self.Nc:])
        else:
            if plot_string:
                plt.quiver(self.f_x_array.flatten(), self.f_y_array.flatten(), 
                       (self.Flx + self.Frx + self.Fbfx).flatten(), (self.Fly + self.Fry + self.Fbfy).flatten())
            if plot_colloid:
                plt.quiver(self.r[0:self.Nc], self.r[self.Nc:2*self.Nc], self.F[0:self.Nc], self.F[self.Nc:])
    
    def compute_ridge_point(self, r_expanded = None):
        """
        compute the connection map using scipy.spatial.voronoi
        """
        if r_expanded:
            self.change_r_expanded(r_expanded)
            vor = Voronoi(self.r_center_matrix)
        else:
            vor = Voronoi(self.r_center_matrix)
        return (np.sort(vor.ridge_points)).tolist()

    def rhs_cython(self, r_expanded, t):
        #Set the current filament state
        #self.drEdt = self.drEdt * 0
        #self.drdt = self.drdt * 0
        #self.dfil_x_dt = self.dfil_x_dt * 0
        #self.dfil_y_dt = self.dfil_y_dt * 0
        self.Fv = self.Fv*0
        
        if self.periodic_bc:
            print("Periodic BC is disabled")
            """
            self.r = r_expanded[0:self.dim*self.Nc]
            self.r[0:self.Nc] = (self.r[0:self.Nc] - self.xmin)%(2*self.half_box_size_x) + self.xmin
            self.r[self.Nc:2*self.Nc] = (self.r[self.Nc:2*self.Nc] - self.ymin)%(2*self.half_box_size_y) + self.ymin
            self.r_matrix = self.reshape_to_matrix(self.r)

            self.theta = r_expanded[self.dim*self.Nc:(self.dim+1)*self.Nc]
            
            self.f_x_array = np.reshape(
                r_expanded[(self.dim+1)*self.Nc: ((self.dim+1)*self.Nc+self.Nf*self.Np)], (self.Nf, self.Np), order = 'C')
            self.f_y_array = np.reshape(
                r_expanded[((self.dim+1)*self.Nc+self.Nf*self.Np): ((self.dim+1)*self.Nc+2*self.Nf*self.Np)], 
                (self.Nf, self.Np), order = 'C')
            
            # boundary condition on attachment site: free to rotate, but not free to move
            self.f_x_array[:, 0] = self.r_matrix[0, self.connection_table[:, 0], self.connection_vertex[:, 0]]
            self.f_x_array[:, self.Np-1] = self.r_matrix[0, self.connection_table[:, 1], self.connection_vertex[:, 1]]
            self.f_y_array[:, 0] = self.r_matrix[1, self.connection_table[:, 0], self.connection_vertex[:, 0]]
            self.f_y_array[:, self.Np-1] = self.r_matrix[1, self.connection_table[:, 1], self.connection_vertex[:, 1]]

            # calculate geometric quantities
            self.get_separation_vectors()
            self.get_distance_bet_colloid()

            # Forces on filaments
            self._Fl()
            self._Fr()

            # Forces on vacuoles among vacuoles
            self.rough_repulsive_force = ((self.effective_diameter - self.colloid_dist_arr)
                /self.colloid_dist_arr*self.interaction_true_table)

            self.F[0:self.Nc] += self.kr*np.sum(self.colloid_delta_r_arr[:,:,0]*self.rough_repulsive_force, axis = 1)
            self.F[self.Nc:2*self.Nc] += self.kr*np.sum(self.colloid_delta_r_arr[:,:,1]*self.rough_repulsive_force, axis = 1)

            # no force from canvas

            # Forces on vacuoles from the end of the filaments
            for i in range(self.Nf):
                self.F[0+self.connection_table[i, 0]] += self.Flx[i, 0]
                self.F[0+self.connection_table[i, 1]] += self.Flx[i, self.Np-1]
                self.F[self.Nc+self.connection_table[i, 0]] += self.Fly[i, 0]
                self.F[self.Nc+self.connection_table[i, 1]] += self.Fly[i, self.Np-1]
            
            self.drdt[0:self.Nc] = self.F[0:self.Nc]/self.six_pi_eta_r
            self.drdt[self.Nc:] = self.F[self.Nc:]/self.six_pi_eta_r

            # Torque on vacuoles from the end of the filaments
            #self.Trq_temp0 = self.radius[self.connection_table[:, 0]]*(
            #        -self.Flx[:, 0]*np.sin(self.connection_theta[:, 0] + self.theta[self.connection_table[:, 0]]) 
            #        + self.Fly[:, 0]*np.cos(self.connection_theta[:, 0] + self.theta[self.connection_table[:, 0]]))
            #self.Trq_temp1 = self.radius[self.connection_table[:, 1]]*(
            #        -self.Flx[:, self.Np-1]*np.sin(self.connection_theta[:, 1] + self.theta[self.connection_table[:, 1]]) 
            #        + self.Fly[:, self.Np-1]*np.cos(self.connection_theta[:, 1] + self.theta[self.connection_table[:, 1]]))
            #for i in range(self.Nf):
            #    self.Trq[self.connection_table[i, 0]] += self.Trq_temp0[i]
            #    self.Trq[self.connection_table[i, 1]] += self.Trq_temp1[i]
            #self.dthetadt = self.Trq/self.eight_pi_eta_r_cube

            # Forces on filaments
            self.Flx[:, 0] = 0
            self.Flx[:, self.Np-1] = 0
            self.Fly[:, 0] = 0
            self.Fly[:, self.Np-1] = 0
            
            self.dfil_x_dt = (self.Flx + self.Frx).flatten()/(6*PI*self.eta*self.dLf)
            self.dfil_y_dt = (self.Fly + self.Fry).flatten()/(6*PI*self.eta*self.dLf)
            
            # assemble
            # self.drEdt = np.hstack([self.drdt, self.dthetadt.flatten(), self.dfil_x_dt, self.dfil_y_dt])
            self.drEdt[0:self.dim*self.Nc] = self.drdt
            self.drEdt[self.dim*self.Nc:(self.dim+1)*self.Nc] = self.dthetadt.flatten()
            self.drEdt[(self.dim+1)*self.Nc:(self.dim+1)*self.Nc+self.Np*self.Nf] = self.dfil_x_dt
            self.drEdt[(self.dim+1)*self.Nc+self.Np*self.Nf:(self.dim+1)*self.Nc+self.Np*self.Nf*2] = self.dfil_y_dt"""

        else:
            self.r = r_expanded[0:self.dim*self.Nc*self.Nv]
            self.r_matrix = self.reshape_to_matrix(self.r)
            
            self.f_x_array = np.reshape(
                r_expanded[self.dim*self.Nc*self.Nv:(self.dim*self.Nc*self.Nv+self.Nf*self.Np)], 
                (self.Nf, self.Np), order = 'C')
            self.f_y_array = np.reshape(
                r_expanded[(self.dim*self.Nc*self.Nv+self.Nf*self.Np):(self.dim*self.Nc*self.Nv+2*self.Nf*self.Np)], 
                (self.Nf, self.Np), order = 'C')

            # boundary condition on attachment site: free to rotate, but not free to move
            self.f_x_array[:, 0] = self.r_matrix[0, self.connection_table[:, 0], self.connection_vertex[:, 0]]
            self.f_x_array[:, self.Np-1] = self.r_matrix[0, self.connection_table[:, 1], self.connection_vertex[:, 1]]
            self.f_y_array[:, 0] = self.r_matrix[1, self.connection_table[:, 0], self.connection_vertex[:, 0]]
            self.f_y_array[:, self.Np-1] = self.r_matrix[1, self.connection_table[:, 1], self.connection_vertex[:, 1]]
            r_expanded[self.dim*self.Nc*self.Nv:(self.dim*self.Nc*self.Nv+self.Nf*self.Np)] = self.f_x_array.flatten()
            r_expanded[(self.dim*self.Nc*self.Nv+self.Nf*self.Np):(self.dim*self.Nc*self.Nv+2*self.Nf*self.Np)] = self.f_y_array.flatten()
            
            # calculate geometric quantities
            self.get_separation_vectors()
            self.get_distance_bet_colloid()
            
            # Forces on vacuoles
            self._Fa()
            self._Fb()
            self._Flc()
            self._Fgamma()

            # Forces on filaments
            self._Fl()
            self._Fr()
            self._Fbf()

            # Forces on vacuoles among vacuoles 
            i_all, j_all = np.where(self.interaction_true_table)          
            #i_all, j_all = np.where(np.triu(self.interaction_true_table))
            
            # point is within canvas
            xy_x = self.r_matrix[0, i_all, :].T
            xy_y = self.r_matrix[1, i_all, :].T
            xy2_x = xy_x[self.axis_1_plus_1, :]
            xy2_y = xy_y[self.axis_1_plus_1, :]
            xs = self.r_matrix[0, j_all, :].T
            ys = self.r_matrix[1, j_all, :].T

            v1x = xy_x[None, :, :] - xs[:, None, :]
            v1y = xy_y[None, :, :] - ys[:, None, :]
            v2x = xy2_x[None, :, :] - xs[:, None, :]
            v2y = xy2_y[None, :, :] - ys[:, None, :]

            ang = angle_xy(v1x, v1y, v2x, v2y)
            result = (((np.abs(np.einsum('ijk->ik', ang)) - (2*PI))*(10**9)).astype(int)/(10.**9) == 0)

            collision_pair, = np.where(np.einsum('ij->j', result) != 0)
            idx_vertex, idx_pair = np.where(result)

            DELTA_eff = self.effective_diam_delta_simple[i_all, j_all]
            DX = (xs[:, None, :] - xy_x[None, :, :])
            DY = (ys[:, None, :] - xy_y[None, :, :])
            DR = np.sqrt(DX**2 + DY**2)
            DELTA_eff_M_DR = DELTA_eff[None, None, :] - DR
            TF = (DELTA_eff_M_DR > 0)
            fx = self.kr*np.sum(TF*DX/(DR+1E-10)*(DELTA_eff_M_DR), axis = 1)
            fy = self.kr*np.sum(TF*DY/(DR+1E-10)*(DELTA_eff_M_DR), axis = 1)
            fx[idx_vertex, idx_pair] = fx[idx_vertex, idx_pair]*(-1)
            fy[idx_vertex, idx_pair] = fy[idx_vertex, idx_pair]*(-1)
            np.add.at(self.Fv, 
                [np.tile(j_all, (self.Nv,1)).T.flatten(), np.tile(range(self.Nv), len(j_all)), 0], 
                       fx.T.flatten())
            np.add.at(self.Fv, 
                [np.tile(j_all, (self.Nv,1)).T.flatten(), np.tile(range(self.Nv), len(j_all)), 1], 
                        fy.T.flatten())

            # Forces on vacuoles from the boundaries
            fx, fy = self.canvas.find_wall_force_vectorized_heuristic(
                self.r[0:self.Nc*self.Nv].reshape((self.Nc*self.Nv,1)), 
                self.r[self.Nc*self.Nv:2*self.Nc*self.Nv].reshape((self.Nc*self.Nv,1)), 
                self.delta_wall)
            #self.Fv[:, :, 0] += self.kr_b*self.kr/self.Nv*fx.reshape((self.Nc, self.Nv))
            #self.Fv[:, :, 1] += self.kr_b*self.kr/self.Nv*fy.reshape((self.Nc, self.Nv))
            self.Fv[:, :, 0] += self.kr_b*self.kr*fx.reshape((self.Nc, self.Nv))
            self.Fv[:, :, 1] += self.kr_b*self.kr*fy.reshape((self.Nc, self.Nv))

            # Forces on ER from the boundaries
            if self.kr_bf > 0:
                ffx, ffy = self.canvas.find_wall_force_vectorized_heuristic(
                    self.f_x_array.reshape((self.Nf*self.Np, 1)),
                    self.f_y_array.reshape((self.Nf*self.Np, 1)),
                    self.delta_string)
                ffx[0:self.Nf*self.Np:self.Np] = 0 # no boundary force on the connection site, because the vertex has experienced that already
                ffx[self.Np-1:self.Nf*self.Np:self.Np] = 0
                ffy[0:self.Nf*self.Np:self.Np] = 0
                ffy[self.Np-1:self.Nf*self.Np:self.Np] = 0
                #ffx = self.kr_bf*self.kr/self.Np*ffx.flatten()
                #ffy = self.kr_bf*self.kr/self.Np*ffy.flatten()
                ffx = self.kr_bf*self.kr*ffx.flatten()
                ffy = self.kr_bf*self.kr*ffy.flatten()
            else:
                ffx = 0
                ffy = 0
            
            # Forces on vacuoles from the end of the filaments
            self.Fv[self.connection_table[:, 0], self.connection_vertex[:, 0], 0] += self.Flx[:, 0] + self.Fbfx[:, 0] + self.Frx[:, 0]
            self.Fv[self.connection_table[:, 1], self.connection_vertex[:, 1], 0] += self.Flx[:, self.Np-1] + self.Fbfx[:, self.Np-1] + self.Frx[:, self.Np-1]
            self.Fv[self.connection_table[:, 0], self.connection_vertex[:, 0], 1] += self.Fly[:, 0] + self.Fbfy[:, 0] + self.Fry[:, 0]
            self.Fv[self.connection_table[:, 1], self.connection_vertex[:, 1], 1] += self.Fly[:, self.Np-1] + self.Fbfy[:, self.Np-1] + self.Fry[:, self.Np-1]
            
            self.drdt[0:self.Nc*self.Nv] = ((self.Fax + self.Fbx + self.Flcx + self.Fgx + self.Fv[:, :, 0])/self.six_pi_eta_r).flatten()
            self.drdt[self.Nc*self.Nv:] = ((self.Fay + self.Fby + self.Flcy + self.Fgy + self.Fv[:, :, 1])/self.six_pi_eta_r).flatten()
            
            # Forces on filaments
            self.Flx[:, 0] = 0
            self.Flx[:, self.Np-1] = 0
            self.Fly[:, 0] = 0
            self.Fly[:, self.Np-1] = 0
            self.Fbfx[:, 0] = 0
            self.Fbfx[:, self.Np-1] = 0
            self.Fbfy[:, 0] = 0
            self.Fbfy[:, self.Np-1] = 0
            
            self.dfil_x_dt = ((self.Flx + self.Frx + self.Fbfx).flatten() + ffx)/(self.eta*self.dLf)
            self.dfil_y_dt = ((self.Fly + self.Fry + self.Fbfy).flatten() + ffx)/(self.eta*self.dLf)

            drdt_m = self.reshape_to_matrix(self.drdt)
            dfil_x_dt_m = np.reshape(self.dfil_x_dt, (self.Nf, self.Np))
            dfil_y_dt_m = np.reshape(self.dfil_y_dt, (self.Nf, self.Np))
            dfil_x_dt_m[:, 0] = drdt_m[0, self.connection_table[:, 0], self.connection_vertex[:, 0]]
            dfil_x_dt_m[:, self.Np-1] = drdt_m[0, self.connection_table[:, 1], self.connection_vertex[:, 1]]
            dfil_y_dt_m[:, 0] = drdt_m[1, self.connection_table[:, 0], self.connection_vertex[:, 0]]
            dfil_y_dt_m[:, self.Np-1] = drdt_m[1, self.connection_table[:, 1], self.connection_vertex[:, 1]]
            
            # assemble
            # self.drEdt = np.hstack([self.drdt, self.dfil_x_dt, self.dfil_y_dt])
            self.drEdt[0:self.dim*self.Nc*self.Nv] = self.drdt
            self.drEdt[self.dim*self.Nc*self.Nv:self.dim*self.Nc*self.Nv+self.Np*self.Nf] = dfil_x_dt_m.flatten()
            self.drEdt[self.dim*self.Nc*self.Nv+self.Np*self.Nf:self.dim*self.Nc*self.Nv+self.Np*self.Nf*2] = dfil_y_dt_m.flatten()
        
    def simulate(self, Tf = 100, t0 = 0, Npts = 10, stop_tol = 1E-5, atol=1E-7, rtol=1E-6, sim_type = 'point', 
                 init_condition = {'shape':'line', 'angle':0}, activity_profile = None, scale_factor = 1, 
                 activity_timescale = 0, save = False, method = 'RK45', use_odespy = False,
                 path = '/Users/jrchang612/Spirostomum_model/rock_string_model', note = '', overwrite = False, pid = 0):
        
        # Set the seed for the random number generator
        np.random.seed(pid)
        self.seed = pid
        self.save = save
        self.overwrite = overwrite
        self.method = method
        self.Npts = Npts
        #---------------------------------------------------------------------------------
        def rhs0(t, r_expanded):
            ''' 
            Pass the current time from the ode-solver, 
            so as to implement time-varying conditions
            '''
            self.rhs_cython(r_expanded, t)
            self.time_now = t
            self.pbar.update(100*(self.time_now - self.time_prev)/Tf)
            self.time_prev = self.time_now
            return self.drEdt

        def rhs_odespy(r_expanded, t):
            ''' 
            Pass the current time from the ode-solver, 
            so as to implement time-varying conditions
            '''
            self.rhs_cython(r_expanded, t)
            self.time_now = t
            self.pbar.update(100*(self.time_now - self.time_prev)/Tf)
            self.time_prev = self.time_now
            return self.drEdt

        """def terminate(u, t, step):
            # Termination criterion based on bond-angle
            if(step >0 and np.any(self.cosAngle[1:-1] < 0)):
                return True
            else:
                return False"""
        
        self.time_now = 0
        self.time_prev = 0

        self.activity_timescale = activity_timescale
        # Set the scale-factor
        self.scale_factor = scale_factor
        #---------------------------------------------------------------------------------
        #Allocate a Path and folder to save the results
        subfolder = datetime.now().strftime('%Y-%m-%d')

        # Create sub-folder by date
        self.path = os.path.join(path, subfolder)

        if(not os.path.exists(self.path)):
            os.makedirs(self.path)

        self.folder = 'SimResults_Nc_{}_Nv_{}_Np_{}_Nf_{}_volfracAct_{}_volfrac_{}_filfrac_{}_solver_{}'.format\
                            (self.Nc, self.Nv, self.Np, self.Nf, 
                                round(self.vol_frac_actual, 2),
                                round(self.vol_frac_initial, 2), round(self.filament_frac_initial, 2), 
                             self.method) + note

        self.saveFolder = os.path.join(self.path, self.folder)
        #---------------------------------------------------------------------------------
        # Set the activity profile
        self.activity_profile = activity_profile

        print('Running the filament simulation ....')

        start_time = time.time()
        tqdm_text = "Param: {} Progress: ".format(self.kr).zfill(1)

        # Stagger the start of the simulations to avoid issues with concurrent writing to disk
        time.sleep(pid)
        
        with tqdm(total = 100, desc=tqdm_text, position=pid+1) as self.pbar:
            # printProgressBar(0, Tf, prefix = 'Progress:', suffix = 'Complete', length = 50)

            # integrate the resulting equation using odespy
            T, N = Tf, Npts;  
            time_points = np.linspace(t0, t0+T, N+1);  ## intervals at which output is returned by integrator. 

            if use_odespy:
                if self.method == 'LSODA':
                    solver = odespy.lsoda_scipy(rhs_odespy, atol=atol, rtol=rtol)
                    solver.set_initial_condition(self.r_expanded) # Initial conditions
                    self.R, self.Time = solver.solve(time_points)
                    self.R = self.R.T
                else:
                    solver = odespy.Vode(rhs_odespy, method = self.method, atol=atol, rtol=rtol, order=5) # initialize the odespy solver
                    solver.set_initial_condition(self.r_expanded) # Initial conditions
                    self.R, self.Time = solver.solve(time_points)
                    self.R = self.R.T
            else:
                Sol = solve_ivp(rhs0, [t0, t0+T], self.r_expanded, method=self.method, t_eval=time_points)
                self.R = Sol['y']
                self.Time = Sol['t']
                
            self.cpu_time = time.time() - start_time
            if(self.save):
                print('Saving results...')
                self.save_data()

    def load_data(self, file = None, print_session = True):
        if print_session:
            print('Loading Simulation data from disk .......')
        if(file is not None):
            self.simFolder, self.simFile = os.path.split(file)
            if(file[-4:] == 'hdf5'):
                print('Loading hdf5 file')
                with h5py.File(file, "r") as f:
                    if('simulation data' in f.keys()): # Load the simulation data (newer method)
                        
                        dset = f['simulation data']
                        self.Time = dset["Time"][:]
                        self.R = dset["Position"][:]
                        self.radius = dset["radius"][:]
                        self.effective_diameter = self.radius + self.radius.reshape((self.Nc,1))
                        self.six_pi_eta_r = self.eta*self.radius.reshape((-1,1))*(PI/self.Nv)
                        self.delta_all = (np.tile(self.radius*2*PI/self.Nv, (self.Nv, 1)).T).flatten()
                        self.delta_wall = (self.delta_all/2).reshape((1, self.Nc*self.Nv))
                        self.effective_diam_delta = self.delta_all + self.delta_all.reshape((self.Nc*self.Nv,1))
                        self.effective_diam_delta_simple = self.radius*PI/self.Nv + self.radius.reshape((self.Nc,1))*PI/self.Nv
                        self.area0_arr = (1/2*self.Nv*np.sin(2*PI/self.Nv)*self.radius**2).reshape((self.Nc,1))
                        self.l0_arr = (self.radius*2*np.sin(PI/self.Nv)).reshape((-1,1))
                        self.vacuole_bending_prefactor = self.kb/(self.Nv*self.l0_arr**4)

                        self.ER_intact = dset["ER_intact"][:]
                        self.connection_table = dset["connection table"][:]
                        self.connection_vertex = dset["connection vertex"][:]
                        self.change_canvas(dset["canvas_xy"][:])

                        # Load the metadata:
                        self.dim = dset.attrs['dimensions']
                        self.Nc = dset.attrs['N colloids']
                        self.Nv = dset.attrs['N vertices']
                        self.Nf = dset.attrs['N filaments']
                        self.Np = dset.attrs['N particles per filament']
                        self.Rc = dset.attrs['baseline radius']
                        self.bidisperse = dset.attrs['bidisperse']
                        self.mLf = dset.attrs['normalized length of strings']
                        self.v_char = dset.attrs['v_char']
                        self.Ca = dset.attrs['Ca']
                        self.St = dset.attrs['St']
                        self.Re_R = dset.attrs['Re_R']
                        self.Stk = dset.attrs['Stk']
                        self.K1 = dset.attrs['K1']
                        self.K2_pos = dset.attrs['K2_pos']
                        self.K2_neg = dset.attrs['K2_neg']
                        self.K3 = dset.attrs['K3']
                        self.K4 = dset.attrs['K4']
                        self.kr_b = dset.attrs['repulsive constant from wall']
                        try:
                            self.kr_bf = dset.attrs['kr_bf']
                        except:
                            self.kr_bf = 0
                        self.rho = dset.attrs['rho']
                        self.Aspect_ratio = dset.attrs['Aspect_ratio']
                        self.Length_radius_ratio = dset.attrs['Length_radius_ratio']
                        self.seed = dset.attrs['seed']
                        self.random_init = dset.attrs['random initiation']
                        self.full_repulsion = dset.attrs['full repulsion']
                        
                        self.periodic_bc = dset.attrs['periodic bc']
                        try:
                            self.half_box_size_x = dset.attrs['half box size x']
                            self.half_box_size_y = dset.attrs['half box size y']
                        except:
                            self.half_box_size = dset.attrs['half box size']

                        self.delt = dset.attrs['delta']
                        self.l0 = dset.attrs['baseline distance between vertices']
                        self.eta = dset.attrs['viscosity']
                        self.boundary_forces_external = dset.attrs['boundary_forces_external']
                        self.Lx0 = dset.attrs['Lx0']
                        self.Ly0 = dset.attrs['Ly0']
                        self.T_contr = dset.attrs['T_contr']

                        self.kr = dset.attrs['repulsive constant']
                        self.kl_c_pos = dset.attrs['area expansion coefficient of vacuole surface']
                        self.kl_c_neg = dset.attrs['area compression coefficient of vacuole surface'] 
                        self.ka = dset.attrs['volume expansion modulus of vacuole']
                        self.gamma = dset.attrs['surface tension of vacuole']
                        self.kb = dset.attrs['bending modulus of vacuole surface']
                        self.Lf = dset.attrs['length of strings'] 
                        self.dLf = dset.attrs['segment length of string']
                        self.kl = dset.attrs['spring constant of string']
                        self.kbf = dset.attrs['bending modulus of filaments']
    
    def save_data(self):
        """
        Implement a save module based on HDF5 format
        """
        copy_number = 0
        self.saveFile = 'SimResults_{0:04d}.hdf5'.format(copy_number)

        if(self.save):
            if(not os.path.exists(self.saveFolder)):
                os.makedirs(self.saveFolder)

            # Choose a new copy number for multiple simulations with the same parameters
            while(os.path.exists(os.path.join(self.saveFolder, self.saveFile)) and self.overwrite == False):
                copy_number+=1
                self.saveFile = 'SimResults_{0:04d}.hdf5'.format(copy_number)


        with h5py.File(os.path.join(self.saveFolder, self.saveFile), "w") as f:

            dset = f.create_group("simulation data")
            dset.create_dataset("Time", data = self.Time)
            dset.create_dataset("Position", data = self.R)
            dset.create_dataset('radius', data = self.radius)
            dset.create_dataset('connection table', data = self.connection_table)
            dset.create_dataset('connection vertex', data = self.connection_vertex)
            dset.create_dataset('canvas_xy', data = self.canvas.xy)
            dset.create_dataset('ER_intact', data = self.ER_intact)

            dset.attrs['dimensions'] = self.dim
            dset.attrs['N colloids'] = self.Nc
            dset.attrs['N vertices'] = self.Nv
            dset.attrs['N filaments'] = self.Nf
            dset.attrs['N particles per filament'] = self.Np
            dset.attrs['baseline radius'] = self.Rc
            dset.attrs['bidisperse'] = self.bidisperse
            dset.attrs['normalized length of strings'] = self.mLf
            dset.attrs['v_char'] = self.v_char
            dset.attrs['Ca'] = self.Ca
            dset.attrs['St'] = self.St
            dset.attrs['Re_R'] = self.Re_R
            dset.attrs['Stk'] = self.Stk
            dset.attrs['K1'] = self.K1
            dset.attrs['K2_pos'] = self.K2_pos
            dset.attrs['K2_neg'] = self.K2_neg
            dset.attrs['K3'] = self.K3
            dset.attrs['K4'] = self.K4
            dset.attrs['repulsive constant from wall'] = self.kr_b
            dset.attrs['kr_bf'] = self.kr_bf
            dset.attrs['rho'] = self.rho
            dset.attrs['Aspect_ratio'] = self.Aspect_ratio
            dset.attrs['Length_radius_ratio'] = self.Length_radius_ratio
            dset.attrs['seed'] = self.seed
            dset.attrs['random initiation'] = self.random_init
            dset.attrs['periodic bc'] = self.periodic_bc
            dset.attrs['full repulsion'] = self.full_repulsion

            dset.attrs['N filaments Delaunay'] = self.Nf_Delaunay
            dset.attrs['volume fraction'] = self.vol_frac
            dset.attrs['filament fraction'] = self.filament_frac_initial
            
            dset.attrs['delta'] = self.delt
            dset.attrs['baseline distance between vertices'] = self.l0
            dset.attrs['viscosity'] = self.eta
            dset.attrs['boundary_forces_external'] = self.boundary_forces_external
            dset.attrs['Lx0'] = self.Lx0
            dset.attrs['Ly0'] = self.Ly0
            dset.attrs['T_contr'] = self.T_contr
            dset.attrs['repulsive constant'] = self.kr
            dset.attrs['area expansion coefficient of vacuole surface'] = self.kl_c_pos
            dset.attrs['area compression coefficient of vacuole surface'] = self.kl_c_neg
            dset.attrs['volume expansion modulus of vacuole'] = self.ka
            dset.attrs['surface tension of vacuole'] = self.gamma
            dset.attrs['bending modulus of vacuole surface'] = self.kb
            dset.attrs['length of strings'] = self.Lf
            dset.attrs['segment length of string'] = self.dLf
            dset.attrs['spring constant of string'] = self.kl
            dset.attrs['bending modulus of filaments'] = self.kbf
            
            dset.attrs['half box size x'] = self.half_box_size_x
            dset.attrs['half box size y'] = self.half_box_size_y

            dset.attrs['aspect_ratio_disabled'] = self.aspect_ratio_disabled

            if(self.activity_profile is not None):
                dset.create_dataset("activity profile", data = self.activity_profile(self.Time))

        # Save user readable metadata in the same folder
        self.metadata = open(os.path.join(self.saveFolder, 'metadata.csv'), 'w+')
        self.metadata.write('Dimensions,'+
                            'N colloids,'+
                            'N vertices,'+
                            'N filaments,'+
                            'N particles per filament,'+
                            'baseline radius,'+
                            'bidisperse,'+
                            'normalized length of strings,'
                            'v_char,'+
                            'Ca,'+
                            'St,'+
                            'Re_R,'+
                            'Stk,'+
                            'K1,'+
                            'K2_pos,'+
                            'K2_neg,'+
                            'K3,'+
                            'K4,'+
                            'repulsive constant,'+
                            'repulsive constant from wall,'+
                            'kr_bf,'+
                            'rho,'+
                            'Aspect_ratio,'+
                            'Length_radius_ratio,'+
                            'ODE solver method,'+
                            'boundary_forces_external,'+
                            'T_contr,'+
                            'spring constant of string,'+
                            'viscosity,'+
                            'length of strings,'+
                            'segment length of string,'+
                            'initial volume fraction,'+
                            'filament fraction,'+
                            'maximum filament number,'+
                            'volume fraction,'+
                            'periodic boundary,'+
                            'full repulsion,'+
                            'random initiation,'+
                            'Simulation time,'+
                            'Seed,'+
                            'aspect_ratio_disabled,'+
                            'CPU time (s)\n')
        self.metadata.write(str(self.dim)+','+
                            str(self.Nc)+','+
                            str(self.Nv)+','+
                            str(self.Nf)+','+
                            str(self.Np)+','+
                            str(self.Rc)+','+
                            str(self.bidisperse)+','+
                            str(self.mLf)+','+
                            str(self.v_char)+','+
                            str(self.Ca)+','+
                            str(self.St)+','+
                            str(self.Re_R)+','+
                            str(self.Stk)+','+
                            str(self.K1)+','+
                            str(self.K2_pos)+','+
                            str(self.K2_neg)+','+
                            str(self.K3)+','+
                            str(self.K4)+','+
                            str(self.kr)+','+
                            str(self.kr_b)+','+
                            str(self.kr_bf)+','+
                            str(self.rho)+','+
                            str(self.Aspect_ratio)+','+
                            str(self.Length_radius_ratio)+','+
                            self.method+','+
                            str(self.boundary_forces_external)+','+
                            str(self.T_contr)+','+
                            str(self.kl)+','+
                            str(self.eta)+','+
                            str(self.Lf)+','+
                            str(self.dLf)+','+
                            str(self.vol_frac_initial)+','+
                            str(self.filament_frac_initial)+','+
                            str(self.Nf_Delaunay)+','+
                            str(self.vol_frac)+','+
                            str(self.periodic_bc)+','+
                            str(self.full_repulsion)+','+
                            str(self.random_init)+','+
                            str(self.Time[self.Npts])+','+
                            str(self.seed)+','+
                            str(self.aspect_ratio_disabled)+','+
                            str(self.cpu_time))
        self.metadata.close()
