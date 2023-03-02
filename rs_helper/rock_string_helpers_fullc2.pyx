
from __future__ import division
#cimport cython
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
from scipy.spatial import Delaunay
from scipy.spatial import Voronoi
from scipy.sparse import coo_matrix
import matplotlib.colors as mcolors
import matplotlib.cm as cm
PI = 3.14159265359
#@cython.cdivision(True)
#@cython.nonecheck(False)
#@cython.wraparound(False) 
#@cython.boundscheck(False)



def angle(v1, v2):
    """
    ## This is the vectorized version function which can compute the angle between 2 arrays of vectors.
    Both v1 and v2 should be of shape (n, 2). The length of v1 & v2 should be the same.
    This function returns a (n,) vector, in which the i-th element is the angle between vectors v1[i, :] and v2[i, :].
    """
    cosine = (v1[:,0]*v2[:,0] + v1[:,1]*v2[:,1])/(np.sqrt(v1[:,0]**2+v1[:,1]**2) * np.sqrt(v2[:,0]**2 + v2[:,1]**2))
    irreasonable = (cosine > 1) + (cosine < -1)
    result = 100*irreasonable + np.arccos(cosine)*(~irreasonable)
    return result

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

class PolygonClass(object):
    def __init__(self, xy):

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

        self.xy = reorder_points(xy)
        self.n_corners = len(xy)
        self.edges = []
        for i in range(self.n_corners):
            self.edges.append(self.xy[[i-1, i], :])
        self.area = self.polygon_area(self.xy)

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
        def point_is_on_corner_of_canvas(self, xs, ys):
            """
            Check if a point is one of the corner of the canvas.
            ## This is vectorized version
            """
            result = 0
            for pt in self.xy:
                result += (xs == pt[0])*(ys == pt[1])
            return result

        result_corner = point_is_on_corner_of_canvas(self, xs, ys)

        sum_angle = 0
        p = np.hstack([xs, ys])
        for edge in self.edges:
            e1, e2 = edge
            v1 = e1 - p
            v2 = e2 - p
            sum_angle += angle(v1, v2)

        result_angle = (((sum_angle - 2*math.pi)*(10**9)).astype(int)/(10.**9) == 0).reshape((-1, 1))
        result = result_angle + result_corner

        return result

    def find_min_dist_to_vac_border(self, xs, ys):
        """
        find the minimum distance of the site to the borders of the vacuole. (used for repulsive calculation)
        """
        def find_min_dist_to_edge(xs, ys, edge):
            """
            find the minimum distance of the site to one specific edge of a polygon.
            ## This is vectorized version
            """
            [[x1, y1],[x2, y2]] = edge
            # convert to ax + by + c = 0
            a = (y2 - y1); b = - (x2 - x1); c = y1*(x2 - x1) - x1*(y2 - y1)
            dist_perp = np.abs(a*xs + b*ys + c)/np.sqrt(a**2 + b**2)
            xp = (b*(b*xs - a*ys) - a*c)/(a**2 + b**2)
            yp = (a*(- b*xs + a*ys) - b*c)/(a**2 + b**2)
            
            fx = (xp - xs)
            fy = (yp - ys)

            return dist_perp, fx, fy

        min_dist_list = []
        fx_list = []
        fy_list = []
        n_site = len(xs)
        for edge in self.edges:
            min_dist, fx, fy = find_min_dist_to_edge(xs, ys, edge)
            min_dist_list.append(min_dist)
            fx_list.append(fx)
            fy_list.append(fy)
        
        min_dist_all = np.hstack(min_dist_list)
        fx_all = np.hstack(fx_list)
        fy_all = np.hstack(fy_list)

        min_dist_loc = np.argmin(np.abs(min_dist_all), axis = 1)
        f_x = fx_all[range(n_site), min_dist_loc].reshape((n_site, 1))
        f_y = fy_all[range(n_site), min_dist_loc].reshape((n_site, 1))
                  
        return f_x, f_y
    
    def find_wall_force(self, xs, ys, delta):
        """
        Find the interaction forces from all the walls onto the particles
        """
        
        def find_min_dist_to_edge(xs, ys, edge, within_canvas):
            """
            find the minimum distance of the site to one specific edge of a polygon.
            ## This is vectorized version
            """
            [[x1, y1],[x2, y2]] = edge
            # convert to ax + by + c = 0
            a = (y2 - y1); b = - (x2 - x1); c = y1*(x2 - x1) - x1*(y2 - y1)
            dist_perp = np.abs(a*xs + b*ys + c)/np.sqrt(a**2 + b**2)
            xp = (b*(b*xs - a*ys) - a*c)/(a**2 + b**2)
            yp = (a*(- b*xs + a*ys) - b*c)/(a**2 + b**2)
            
            TF = is_on_segment(xp, yp, edge)
            #min_dist = dist_perp*TF + np.minimum(np.sqrt((x1 - xs)**2 + (y1 - ys)**2), np.sqrt((x2 - xs)**2 + (y2 - ys)**2))*(~TF)
            min_dist = dist_perp*TF + np.minimum(np.sqrt((x1 - xs)**2 + (y1 - ys)**2), np.sqrt((x2 - xs)**2 + (y2 - ys)**2))*(1-TF)
            
            
            #dist_perp = dist_perp*within_canvas - dist_perp*(1-within_canvas)
            dist_perp = min_dist*within_canvas - min_dist*(1-within_canvas)
            TF_delta = (dist_perp < delta)
            
            Fx = (delta - dist_perp)*(xs - xp)/dist_perp*TF_delta
            Fy = (delta - dist_perp)*(ys - yp)/dist_perp*TF_delta

            return min_dist, Fx, Fy
        
        min_dist_list = []
        Fx_list = []
        Fy_list = []
        n_site = len(xs)
        within_canvas = self.point_is_within_canvas(xs, ys)
        for edge in self.edges:
            min_dist, Fx, Fy = find_min_dist_to_edge(xs, ys, edge, within_canvas)
            min_dist_list.append(min_dist)
            Fx_list.append(Fx)
            Fy_list.append(Fy)
        
        min_dist_all = np.hstack(min_dist_list)
        Fx_all = np.hstack(Fx_list)
        Fy_all = np.hstack(Fy_list)
        Fx_sum = np.sum(Fx_all, axis = 1, keepdims = True)
        Fy_sum = np.sum(Fy_all, axis = 1, keepdims = True)
        min_dist_loc = np.argmin(np.abs(min_dist_all), axis = 1)
        
        f_x = Fx_sum*within_canvas + Fx_all[range(n_site), min_dist_loc].reshape((n_site, 1))*(1-within_canvas)
        f_y = Fy_sum*within_canvas + Fy_all[range(n_site), min_dist_loc].reshape((n_site, 1))*(1-within_canvas)
                  
        return f_x, f_y

    def find_wall_potential_for_plot(self, xs, ys, delta):
        """
        Find the repulsive potential between particles swith the wall
        """
        def find_potential_to_edge_for_plot(xs, ys, edge, within_canvas):
            """
            find the minimum distance of the site to one specific edge of a polygon.
            ## This is vectorized version
            """
            [[x1, y1],[x2, y2]] = edge
            # convert to ax + by + c = 0
            a = (y2 - y1); b = - (x2 - x1); c = y1*(x2 - x1) - x1*(y2 - y1)
            dist_perp = np.abs(a*xs + b*ys + c)/np.sqrt(a**2 + b**2)
            xp = (b*(b*xs - a*ys) - a*c)/(a**2 + b**2)
            yp = (a*(- b*xs + a*ys) - b*c)/(a**2 + b**2)
            
            TF = is_on_segment(xp, yp, edge)
            min_dist = dist_perp*TF + np.minimum(np.sqrt((x1 - xs)**2 + (y1 - ys)**2), np.sqrt((x2 - xs)**2 + (y2 - ys)**2))*(~TF)
            
            dist_perp = dist_perp*within_canvas - dist_perp*(1-within_canvas)
            TF_delta = (dist_perp < delta)
            
            U = 1/2*(delta - dist_perp)**2*TF_delta

            return min_dist, U, xp, yp
        
        min_dist_list = []
        U_list = []
        xp_list = []
        yp_list = []
        within_canvas = self.point_is_within_canvas(xs, ys)
        within_canvas_TF = (within_canvas).astype(bool).flatten()
        
        for edge in self.edges:
            min_dist, U, xp, yp = find_potential_to_edge_for_plot(xs, ys, edge, within_canvas)
            min_dist_list.append(min_dist)
            U_list.append(U)
            xp_list.append(xp)
            yp_list.append(yp)
        
        min_dist_all = np.hstack(min_dist_list)
        min_dist_loc = np.argmin(min_dist_all, axis = 1)
        
        U_all = np.hstack(U_list)
        xp_all = np.hstack(xp_list)
        yp_all = np.hstack(yp_list)
        U_filtered = np.zeros_like(U_all)
        U_filtered[~within_canvas_TF, min_dist_loc[~within_canvas_TF]] = \
        U_all[~within_canvas_TF, min_dist_loc[~within_canvas_TF]]
        U_filtered[within_canvas_TF, :] = U_all[within_canvas_TF, :]
        z_wall = np.sum(U_filtered > 0, axis = 1, keepdims = True)
        return U_filtered, xp_all, yp_all, z_wall

    def find_force_on_walls(self, xs, ys, delta):
        """
        Find the interaction forces from particles to each wall edge
        """
        
        def find_min_dist_to_edge(xs, ys, edge, within_canvas):
            """
            find the minimum distance of the site to one specific edge of a polygon.
            ## This is vectorized version
            """
            [[x1, y1],[x2, y2]] = edge
            # convert to ax + by + c = 0
            a = (y2 - y1); b = - (x2 - x1); c = y1*(x2 - x1) - x1*(y2 - y1)
            dist_perp = np.abs(a*xs + b*ys + c)/np.sqrt(a**2 + b**2)
            xp = (b*(b*xs - a*ys) - a*c)/(a**2 + b**2)
            yp = (a*(- b*xs + a*ys) - b*c)/(a**2 + b**2)
            
            TF = is_on_segment(xp, yp, edge)
            #min_dist = dist_perp*TF + np.minimum(np.sqrt((x1 - xs)**2 + (y1 - ys)**2), np.sqrt((x2 - xs)**2 + (y2 - ys)**2))*(~TF)
            min_dist = dist_perp*TF + np.minimum(np.sqrt((x1 - xs)**2 + (y1 - ys)**2), np.sqrt((x2 - xs)**2 + (y2 - ys)**2))*(1-TF)
            
            
            #dist_perp = dist_perp*within_canvas - dist_perp*(1-within_canvas)
            dist_perp = min_dist*within_canvas - min_dist*(1-within_canvas)
            TF_delta = (dist_perp < delta)
            
            Fx = (delta - dist_perp)*(xs - xp)/dist_perp*TF_delta
            Fy = (delta - dist_perp)*(ys - yp)/dist_perp*TF_delta

            return min_dist, Fx, Fy
        
        min_dist_list = []
        Fx_list = []
        Fy_list = []
        n_site = len(xs)
        within_canvas = self.point_is_within_canvas(xs, ys)
        for edge in self.edges:
            min_dist, Fx, Fy = find_min_dist_to_edge(xs, ys, edge, within_canvas)
            min_dist_list.append(min_dist)
            Fx_list.append(Fx)
            Fy_list.append(Fy)
        
        min_dist_all = np.hstack(min_dist_list)
        Fx_all = np.hstack(Fx_list)
        Fy_all = np.hstack(Fy_list)
        min_dist_loc = np.argmin(np.abs(min_dist_all), axis = 1)

        Fx_filtered = Fx_all*within_canvas
        Fy_filtered = Fy_all*within_canvas

        Fx_filtered[range(n_site), min_dist_loc] = Fx_all[range(n_site), min_dist_loc]
        Fy_filtered[range(n_site), min_dist_loc] = Fy_all[range(n_site), min_dist_loc]

        Fx_sum = -np.sum(Fx_filtered, axis = 0, keepdims = True)
        Fy_sum = -np.sum(Fy_filtered, axis = 0, keepdims = True)
                          
        return Fx_sum, Fy_sum
    
    def find_wall_potential(self, xs, ys, delta):
        """
        Find the repulsive potential between particles swith the wall
        """
        def find_potential_to_edge(xs, ys, edge, within_canvas):
            """
            find the minimum distance of the site to one specific edge of a polygon.
            ## This is vectorized version
            """
            [[x1, y1],[x2, y2]] = edge
            # convert to ax + by + c = 0
            a = (y2 - y1); b = - (x2 - x1); c = y1*(x2 - x1) - x1*(y2 - y1)
            dist_perp = np.abs(a*xs + b*ys + c)/np.sqrt(a**2 + b**2)
            xp = (b*(b*xs - a*ys) - a*c)/(a**2 + b**2)
            yp = (a*(- b*xs + a*ys) - b*c)/(a**2 + b**2)
            
            TF = is_on_segment(xp, yp, edge)
            min_dist = dist_perp*TF + np.minimum(np.sqrt((x1 - xs)**2 + (y1 - ys)**2), np.sqrt((x2 - xs)**2 + (y2 - ys)**2))*(~TF)
            
            dist_perp = dist_perp*within_canvas - dist_perp*(1-within_canvas)
            TF_delta = (dist_perp < delta)
            
            U = 1/2*(delta - dist_perp)**2*TF_delta

            return min_dist, U
        
        min_dist_list = []
        U_list = []
        within_canvas = self.point_is_within_canvas(xs, ys)
        
        for edge in self.edges:
            min_dist, U = find_potential_to_edge(xs, ys, edge, within_canvas)
            min_dist_list.append(min_dist)
            U_list.append(U)
        
        min_dist_all = np.hstack(min_dist_list)
        min_dist_loc = np.argmin(min_dist_all, axis = 1)
        
        U_all = np.hstack(U_list)
        U_sum = np.sum(U_all, axis = 1, keepdims = True)
        n_site = len(xs)
        U_final = U_sum*within_canvas + U_all[range(n_site), min_dist_loc].reshape((n_site, 1))*(1-within_canvas)
        
        return U_final
    
    def plot_canvas(self):
        fig = plt.figure(figsize = (5, 5), dpi = 200)
        ax = fig.add_axes([0, 0, 1, 1])

        # plot canvas
        for edge_canvas in self.edges:
            ax.plot(edge_canvas[:, 0], edge_canvas[:, 1], 
                'black', lw = 2, solid_capstyle = 'round', zorder = 2)


class Colloid_and_Filament(object):
    """
    This is the main class of colloid(c) and filament(f) (Np particle per filament) simulations
    """
    def __init__(self, dim = 2, Nc = 10, Np = 10, Nf = 0, Rc = 1, bidisperse = 1.4, 
                 Rep = 1, kr = 1, kr_b = 1, Contr = 1, Inext = 10, mLf = 3.3,
                 periodic_bc = False, full_repulsion = True, seed = 0,
                 canvas_initial_xy = np.array([[-15, -15], [15, -15], [15, 15], [-15, 15]]), random_init = True):
        """
        Rep = repulsive number = k_r/(eta*v)
        Contr = contractility number = k_l*(Lf/10)/(6*pi*eta*Rc*v_char)
        Inext = inextensibility of the string
        """
        # Main parameters
        self.dim = dim
        self.Nc = Nc
        self.Np = Np
        self.Nf = Nf
        self.Rc = Rc
        self.bidisperse = bidisperse
        self.Rep = Rep
        self.kr = kr
        self.kr_b = kr_b # stronger/weaker repulsion from boundary
        self.v_char = 1
        self.eta = self.kr/(self.Rep*self.v_char)
        self.mLf = mLf
        self.Lf = self.Rc*self.bidisperse*self.mLf # length of each filament
        self.dLf = self.Lf/(self.Np-1)
        self.Contr = Contr
        self.Inext = Inext
        self.seed = seed
        self.kl = self.Contr*(6*PI*self.eta*self.Rc*self.v_char)/(self.Lf/self.Inext)
        self.periodic_bc = periodic_bc
        self.full_repulsion = full_repulsion

        # Set up canvas
        self.canvas_initial = PolygonClass(canvas_initial_xy)
        self.canvas = self.canvas_initial
        self.xmin = np.min(self.canvas.xy[:,0])
        self.xmax = np.max(self.canvas.xy[:,0])
        self.ymin = np.min(self.canvas.xy[:,1])
        self.ymax = np.max(self.canvas.xy[:,1])
        self.half_box_size_x = (self.xmax - self.xmin)/2
        self.half_box_size_y = (self.ymax - self.ymin)/2

        # Initialize arrays for storing particle positions, activity strengths etc.
        self.allocate_arrays()
        
        # Other parameters
        self.cpu_time = 0
        
        # Initialize the colloids
        if random_init:
            self.initialize_colloid_location()
        else:
            self.initialize_colloid_location_nonrandom()
        vol_frac_initial = np.sum(PI*self.radius**2)/self.canvas_initial.area
        self.vol_frac_initial = vol_frac_initial
        self.vol_frac = self.vol_frac_initial

        # Initialize the filament
        self.initialize_filament_location()
        self.filament_frac_initial = self.Nf/self.Nf_Delaunay
    
    def print_help(self):
        """
        print help information of the class
        """
        print('test message!')
    
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
    
    def allocate_arrays(self):
        # Initiate positions, orientations, forces etc of the particles
        self.r = np.zeros(self.Nc*self.dim, dtype = np.double)
        self.r_expanded = np.zeros(self.Nc*(self.dim+1)+self.dim*self.Nf*self.Np, dtype = np.double)
        self.r_matrix = np.zeros((self.Nc, self.dim), dtype = np.double)
        self.theta = np.zeros(self.Nc, dtype = np.double)
        self.p = np.zeros(self.Nc*self.dim, dtype = np.double)
        
        self.r0 = np.zeros(self.Nc*self.dim, dtype = np.double)
        self.r_expanded_0 = np.zeros(self.Nc*(self.dim+1)+self.dim*self.Nf*self.Np, dtype = np.double)
        self.p0 = np.zeros(self.Nc*self.dim, dtype = np.double)
        
        self.radius = np.zeros(self.Nc, dtype = np.double)
        self.effective_diameter = np.zeros((self.Nc, self.Nc), dtype = np.double)
        self.interaction_true_table = np.zeros((self.Nc, self.Nc), dtype = bool)
        self.colloid_dist_arr = np.zeros((self.Nc, self.Nc), dtype = np.double)
        self.colloid_delta_r_arr = np.zeros((self.Nc, self.Nc, self.dim), dtype = np.double)
        self.colloid_delta_x_arr = np.zeros((self.Nc, self.Nc), dtype = np.double)
        self.colloid_delta_y_arr = np.zeros((self.Nc, self.Nc), dtype = np.double)
        self.rough_repulsive_force = np.zeros((self.Nc, self.Nc), dtype = np.double)
        
        # filaments
        self.dx = np.zeros((self.Nf, self.Np-1), dtype = np.double)
        self.dy = np.zeros((self.Nf, self.Np-1), dtype = np.double)
        self.dr = np.zeros((self.Nf, self.Np-1), dtype = np.double)
        self.dx_hat = np.zeros((self.Nf, self.Np-1), dtype = np.double)
        self.dy_hat = np.zeros((self.Nf, self.Np-1), dtype = np.double)
        self.f_x_array = np.zeros((self.Nf, self.Np), dtype = np.double)
        self.f_y_array = np.zeros((self.Nf, self.Np), dtype = np.double)
        self.connection_table = np.zeros((self.Nf, 2), dtype = int) # i, j 
        self.connection_theta = np.zeros((self.Nf, 2), dtype = np.double) # theta_i, theta_j
        self.Flx = np.zeros((self.Nf, self.Np), dtype = np.double)
        self.Fly = np.zeros((self.Nf, self.Np), dtype = np.double)
        self.Frx = np.zeros((self.Nf, self.Np), dtype = np.double)
        self.Fry = np.zeros((self.Nf, self.Np), dtype = np.double)
        self.Frx_expanded = np.zeros((self.Nf, self.Nc, self.Np-2), dtype = np.double)
        self.Fry_expanded = np.zeros((self.Nf, self.Nc, self.Np-2), dtype = np.double)

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
        self.drdt = np.zeros(self.Nc*self.dim, dtype = np.double)
        self.dthetadt = np.zeros(self.Nc, dtype = np.double)
        self.dfil_x_dt = np.zeros(self.Nf*self.Np, dtype = np.double)
        self.dfil_y_dt = np.zeros(self.Nf*self.Np, dtype = np.double)
        self.drEdt = np.zeros(self.Nc*(self.dim+1)+2*self.Nf*self.Np, dtype = np.double)
        
        self.cosAngle = np.ones(self.Nc, dtype = np.double)
        self.t_hat = np.zeros((self.dim,self.Nc), dtype = np.double)

        self.F = np.zeros(self.Nc*self.dim, dtype = np.double) # force on vacuoles
        self.Trq = np.zeros(self.Nc, dtype = np.double) # torque
        self.Trq_temp0 = np.zeros(self.Nf, dtype = np.double)
        self.Trq_temp1 = np.zeros(self.Nf, dtype = np.double)
    
    def initialize_colloid_location(self):
        np.random.seed(self.seed)
        self.r0 = np.reshape(self.canvas_initial.random_points_in_canvas(self.Nc), (-1), order = 'F')
        self.radius = np.array([self.Rc]*int(self.Nc/2)+[self.Rc*self.bidisperse]*int(self.Nc/2))
        np.random.shuffle(self.radius)
        self.six_pi_eta_r = 6*PI*self.eta*self.radius
        self.eight_pi_eta_r_cube = 8*PI*self.eta*self.radius**3
        self.effective_diameter = self.radius + self.radius.reshape((self.Nc,1))
        self.r = self.r0
        self.r_matrix = self.reshape_to_matrix(self.r)
        self.r_expanded_0[0:self.Nc*self.dim] = self.r0
        self.r_expanded[0:self.Nc*self.dim] = self.r
    
    def initialize_colloid_location_nonrandom(self):
        """
        Initialize the colloid location in a square grid shape.
        """
        np.random.seed(self.seed)
        n_grid = int(np.sqrt(self.Nc))+1
        #L = np.max(np.max(self.canvas_initial.xy))
        loc_X = np.linspace(self.xmin + self.Rc, self.xmax - self.Rc, n_grid)
        loc_Y = np.linspace(self.ymin + self.Rc, self.ymax - self.Rc, n_grid)
        X, Y = np.meshgrid(loc_X, loc_Y)
        xy = np.hstack([X.reshape((-1,1)), Y.reshape((-1,1))])
        i_choice = np.random.choice(np.arange(len(xy)), self.Nc, replace = False)
        sites = xy[i_choice, :]
        self.r_matrix = sites
        self.r0 = np.reshape(sites, (-1), order = 'F')
        self.radius = np.array([self.Rc]*int(self.Nc/2)+[self.Rc*self.bidisperse]*int(self.Nc/2))
        np.random.shuffle(self.radius)
        self.six_pi_eta_r = 6*PI*self.eta*self.radius
        self.eight_pi_eta_r_cube = 8*PI*self.eta*self.radius**3
        self.effective_diameter = self.radius + self.radius.reshape((self.Nc,1))
        self.r = self.r0
        self.r_expanded_0[0:self.Nc*self.dim] = self.r0
        self.r_expanded[0:self.Nc*self.dim] = self.r
    
    def initialize_filament_location(self):
        tri = Delaunay(self.r_matrix)
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
        
        # connection table: (Nf, 4) = [id_vac1, id_vac2, theta_attach1, theta_attach2]
        self.connection_table[:, 0:2] = self.connection_list[list_id_selected, :]
        self.connection_theta[:, 0] = 2*PI*np.random.rand(self.Nf)
        self.connection_theta[:, 1] = (self.connection_theta[:, 0] + PI)%(2*PI)
        self.f_x_array = np.linspace(self.r_matrix[self.connection_table[:,0], 0] 
                                     + self.radius[self.connection_table[:,0]]*np.cos(self.connection_theta[:,0]), 
                                     self.r_matrix[self.connection_table[:,1], 0] 
                                     + self.radius[self.connection_table[:,1]]*np.cos(self.connection_theta[:,1]), 
                                     self.Np).T
        self.f_y_array = np.linspace(self.r_matrix[self.connection_table[:,0], 1] 
                                     + self.radius[self.connection_table[:,0]]*np.sin(self.connection_theta[:,0]), 
                                     self.r_matrix[self.connection_table[:,1], 1] 
                                     + self.radius[self.connection_table[:,1]]*np.sin(self.connection_theta[:,1]), 
                                     self.Np).T
        self.get_separation_vectors()
        self.r_expanded_0[self.Nc*(self.dim+1):(self.Nc*(self.dim+1) + self.Nf*self.Np)] = self.f_x_array.flatten()
        self.r_expanded_0[(self.Nc*(self.dim+1) + self.Nf*self.Np):(self.Nc*(self.dim+1) + 2*self.Nf*self.Np)] = self.f_y_array.flatten()
        self.r_expanded[self.Nc*(self.dim+1):(self.Nc*(self.dim+1) + self.Nf*self.Np)] = self.r_expanded_0[
            self.Nc*(self.dim+1):(self.Nc*(self.dim+1) + self.Nf*self.Np)]
        self.r_expanded[(self.Nc*(self.dim+1) + self.Nf*self.Np):(self.Nc*(self.dim+1) + 2*self.Nf*self.Np)] = self.r_expanded_0[
            (self.Nc*(self.dim+1) + self.Nf*self.Np):(self.Nc*(self.dim+1) + 2*self.Nf*self.Np)]
        self.r_vac0 = self.radius[self.connection_table[:, 0]].reshape((self.Nf,1))
        self.r_vac1 = self.radius[self.connection_table[:, 1]].reshape((self.Nf,1))
    
    def plot_system(self, plot_energy = False, colormap = cm.viridis, alpha_colloid = 1, alpha_fil = 1, 
        string_color = None, vac_color = 'silver', edge_color = 'black'):
        """
        Plot the colloid-filamentous system
        """
        # update array
        self.r = self.r_expanded[0:self.dim*self.Nc]
        self.r_matrix = self.reshape_to_matrix(self.r)
        self.theta = self.r_expanded[self.dim*self.Nc:(self.dim+1)*self.Nc]
        
        self.f_x_array = np.reshape(
            self.r_expanded[(self.dim+1)*self.Nc: ((self.dim+1)*self.Nc+self.Nf*self.Np)], (self.Nf, self.Np), order = 'C')
        self.f_y_array = np.reshape(
            self.r_expanded[((self.dim+1)*self.Nc+self.Nf*self.Np): ((self.dim+1)*self.Nc+2*self.Nf*self.Np)], 
            (self.Nf, self.Np), order = 'C')

        if self.periodic_bc:
            self.r[0:self.Nc] = (self.r[0:self.Nc] - self.xmin)%(2*self.half_box_size_x) + self.xmin
            self.r[self.Nc:2*self.Nc] = (self.r[self.Nc:2*self.Nc] - self.ymin)%(2*self.half_box_size_y) + self.ymin
            self.realign_filaments()

            # plot
            self.canvas.plot_canvas()
            for i, xy in enumerate(self.r.reshape((self.Nc, self.dim), order = 'F')):
                for ix in range(-1, 2):
                    for iy in range(-1, 2):
                        cir = pylab.Circle((xy[0] + ix*2*self.half_box_size_x, xy[1] + iy*2*self.half_box_size_y), 
                            radius=self.radius[i], fc=vac_color, ec = edge_color, alpha = alpha_colloid)
                        pylab.gca().add_patch(cir)

            if string_color:
                for i in range(self.Nf):
                    for ix in range(-1, 2):
                        for iy in range(-1, 2):
                            plt.plot(self.f_x_array[i, :] + ix*2*self.half_box_size_x, 
                                self.f_y_array[i, :] + iy*2*self.half_box_size_y, 
                                color = string_color, alpha = alpha_fil)
            else:
                for i in range(self.Nf):
                    for ix in range(-1, 2):
                        for iy in range(-1, 2):
                            plt.plot(self.f_x_array[i, :] + ix*2*self.half_box_size_x, 
                                self.f_y_array[i, :] + iy*2*self.half_box_size_y, 
                                color = 'C'+str(i), alpha = alpha_fil)

            plt.xlim([np.min(self.canvas.xy[:, 0]), np.max(self.canvas.xy[:, 0])])
            plt.ylim([np.min(self.canvas.xy[:, 0]), np.max(self.canvas.xy[:, 0])])
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
                            
        else:        
            # plot
            self.canvas.plot_canvas()
            for i, xy in enumerate(self.r.reshape((self.Nc, self.dim), order = 'F')):
                cir = pylab.Circle((xy[0], xy[1]), radius = self.radius[i], fc = vac_color, ec = edge_color, alpha = alpha_colloid)
                pylab.gca().add_patch(cir)
            if string_color:
                for i in range(self.Nf):
                    plt.plot(self.f_x_array[i, :], self.f_y_array[i, :], color = string_color, alpha = alpha_fil)
            else:
                for i in range(self.Nf):
                    plt.plot(self.f_x_array[i, :], self.f_y_array[i, :], alpha = alpha_fil)

            if plot_energy:
                # energy between colloids
                self.get_distance_bet_colloid()
                coo = coo_matrix(np.triu((self.effective_diameter - self.colloid_dist_arr))**2
                    *self.interaction_true_table)
                # energy from boundary repulsions
                U_filtered, xp_all, yp_all, _ = self.canvas.find_wall_potential_for_plot(
                    self.r[0:self.Nc].reshape((self.Nc,1)), self.r[self.Nc:].reshape((self.Nc,1)), 
                    self.radius.reshape((self.Nc,1)))
                coo2 = coo_matrix(U_filtered)
                normalize = mcolors.Normalize(vmin=0, vmax=np.max([np.max(coo.data), np.max(coo2.data)]))
                ax = plt.gca()
                for i, data in enumerate(coo.data):
                    row = coo.row[i]
                    col = coo.col[i]
                    normed = normalize(data)
                    color = colormap(normed)
                    ax.plot(self.r_matrix[[row,col],0], self.r_matrix[[row,col],1], 
                        color=color, alpha = normed)
                for i, data in enumerate(coo2.data):
                    row = coo2.row[i]
                    col = coo2.col[i]
                    normed = normalize(data)
                    color = colormap(normed)
                    ax.plot([self.r_matrix[row, 0], xp_all[row, col]], 
                        [self.r_matrix[row, 1], yp_all[row, col]], 
                        color=color, alpha = normed)

    
    def reshape_to_array(self, Matrix):
        """
        Takes a matrix of shape (dim, Np) and reshapes to an array (dim*Nc, 1) 
        where the convention is [x1, x2 , x3 ... X_Nc, y1, y2, .... y_Nc, z1, z2, .... z_Nc]
        """
        nrows, ncols = np.shape(Matrix)
        return np.squeeze(np.reshape(Matrix, (nrows*ncols,1), order = 'F'))
    
    def reshape_to_matrix(self, Array):
        """Takes an array of shape (dim*N, 1) and reshapes to a Matrix  of shape (N, dim) and 
        where the array convention is [x1, x2 , x3 ... x_Nc, y1, y2, .... y_Nc, z1, z2, .... z_Nc]
        and matrix convention is |x1   y1   z1  |
                                 |x2   y2   z2  |
                                 |...  ...  ... |
                                 |x_Nc y_Nc z_Nc|
        """
        array_len = len(Array)
        nrows = int(array_len/self.dim)
        return np.reshape(Array, (nrows, self.dim), order = 'F')
    
    def get_distance_bet_colloid(self):
        """
        Get the distance matrix among Nc colloids. colloid_dist_arr is a (Nc, Nc) matrix.
        """
        if self.periodic_bc:
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
            self.colloid_dist_arr[range(self.Nc), range(self.Nc)] = 1 # to avoid division by 0 on diagonal terms

        else:
            self.colloid_delta_r_arr = self.r_matrix[:, None, :] - self.r_matrix[None, :, :]
            self.colloid_dist_arr = np.sqrt((self.colloid_delta_r_arr**2.0).sum(axis=2))
            self.interaction_true_table = (self.effective_diameter - self.colloid_dist_arr > 0)
            self.interaction_true_table[range(self.Nc), range(self.Nc)] = False
            self.z_colloid = np.sum(self.interaction_true_table, axis = 1)
            self.colloid_dist_arr[range(self.Nc), range(self.Nc)] = 1 # to avoid division by 0 on diagonal terms

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
    
    def initialize_filament(self):
        self.initialize_filament_shape()
        # Initialize the bending-stiffness array
        self.initialize_bending_stiffness()
        self.get_separation_vectors()
        #self.t_hat_array = self.reshape_to_array(self.t_hat)
        # Initialize the particle orientations to be along the local tangent vector
        #self.p = self.t_hat_array
        # Orientation vectors of particles depend on local tangent vector
        #self.p0 = self.p

    def change_r_expanded(self, r_expanded):
        self.r_expanded = r_expanded
        self.r = r_expanded[0:self.dim*self.Nc]
        self.r_matrix = self.reshape_to_matrix(self.r)
        self.theta = r_expanded[self.dim*self.Nc:(self.dim+1)*self.Nc]
        
        self.f_x_array = np.reshape(
            r_expanded[(self.dim+1)*self.Nc: ((self.dim+1)*self.Nc+self.Nf*self.Np)], (self.Nf, self.Np), order = 'C')
        self.f_y_array = np.reshape(
            r_expanded[((self.dim+1)*self.Nc+self.Nf*self.Np): ((self.dim+1)*self.Nc+2*self.Nf*self.Np)], (self.Nf, self.Np), order = 'C')


    def _Fl(self):
        '''
        contractility component of filaments
        '''
        # reset forces
        self.Flx = self.Flx*0
        self.Fly = self.Fly*0

        # compute forces
        self.Fl_expanded = self.kl*self.Np*(self.dr - self.dLf)/self.dr

        self.Flx[:, 1:self.Np-1] = -(self.Fl_expanded[:, 0:self.Np-2]*self.dx[:, 0:self.Np-2] 
            - self.Fl_expanded[:, 1:]*self.dx[:, 1:])
        self.Fly[:, 1:self.Np-1] = -(self.Fl_expanded[:, 0:self.Np-2]*self.dy[:, 0:self.Np-2] 
            - self.Fl_expanded[:, 1:]*self.dy[:, 1:])
        self.Flx[:, 0] = (self.Fl_expanded[:, 0]*self.dx[:, 0])
        self.Flx[:, self.Np-1] = -(self.Fl_expanded[:, self.Np-2]*self.dx[:, self.Np-2])
        self.Fly[:, 0] = (self.Fl_expanded[:, 0]*self.dy[:, 0])
        self.Fly[:, self.Np-1] = -(self.Fl_expanded[:, self.Np-2]*self.dy[:, self.Np-2])

    def _Fr(self):
        '''
        repulsive component of filament with respect to the conectinng vacuole, to avoid filament overriding the vacuoles.
        '''
        # reset forces
        self.Frx = self.Frx * 0
        self.Fry = self.Fry * 0

        # compute forces
        if self.full_repulsion:
            
            self.dx_to_vac = self.f_x_array[:, None, 1:self.Np-1] - self.r_matrix[:, 0:1][None, :, :]
            self.dy_to_vac = self.f_y_array[:, None, 1:self.Np-1] - self.r_matrix[:, 1:2][None, :, :]

            if self.periodic_bc:
                self.dx_to_vac[self.dx_to_vac > self.half_box_size_x] -= 2*self.half_box_size_x
                self.dx_to_vac[self.dx_to_vac < -self.half_box_size_x] += 2*self.half_box_size_x
                self.dy_to_vac[self.dy_to_vac > self.half_box_size_y] -= 2*self.half_box_size_y
                self.dy_to_vac[self.dy_to_vac < -self.half_box_size_y] += 2*self.half_box_size_y

            self.dr_to_vac = (self.dx_to_vac**2 + self.dy_to_vac**2)**(0.5)
            self.Frx_expanded = (self.radius[None, :, None] - self.dr_to_vac)*(
                self.radius[None, :, None] > self.dr_to_vac)/self.dr_to_vac*self.dx_to_vac
            self.Fry_expanded = (self.radius[None, :, None] - self.dr_to_vac)*(
                self.radius[None, :, None] > self.dr_to_vac)/self.dr_to_vac*self.dy_to_vac
            
            self.Frx[:, 1:self.Np-1] += self.kr*np.sum(self.Frx_expanded, axis = 1)
            self.Fry[:, 1:self.Np-1] += self.kr*np.sum(self.Fry_expanded, axis = 1)
            self.F[0:self.Nc] -= self.kr*np.sum(self.Frx_expanded, axis = (0,2))
            self.F[self.Nc:2*self.Nc] -= self.kr*np.sum(self.Fry_expanded, axis = (0,2))

        else:
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
                self.r_vac1 > self.dr_to_vac1)/self.dr_to_vac1*self.dy_to_vac1
    
    def compute_potential_energy(self, r_expanded):
        # Set the current filament state
        self.drEdt = self.drEdt * 0
        self.drdt = self.drdt * 0
        self.dthetadt = self.dthetadt * 0
        self.dfil_x_dt = self.dfil_x_dt * 0
        self.dfil_y_dt = self.dfil_y_dt * 0
        
        self.r = r_expanded[0:self.dim*self.Nc]
        self.r_matrix = self.reshape_to_matrix(self.r)
        self.theta = r_expanded[self.dim*self.Nc:(self.dim+1)*self.Nc]
        
        self.f_x_array = np.reshape(
            r_expanded[(self.dim+1)*self.Nc: ((self.dim+1)*self.Nc+self.Nf*self.Np)], (self.Nf, self.Np), order = 'C')
        self.f_y_array = np.reshape(
            r_expanded[((self.dim+1)*self.Nc+self.Nf*self.Np): ((self.dim+1)*self.Nc+2*self.Nf*self.Np)], (self.Nf, self.Np), order = 'C')
        
        # boundary condition on attachment site: free to rotate, but not free to move
        self.f_x_array[:, 0] = (self.r_matrix[self.connection_table[:, 0], 0] + 
                                self.radius[self.connection_table[:, 0]]*np.cos(
                                    self.connection_theta[:, 0] + self.theta[self.connection_table[:, 0]]))
        self.f_x_array[:, self.Np-1] = (self.r_matrix[self.connection_table[:, 1], 0] + 
                                 self.radius[self.connection_table[:, 1]]*np.cos(
                                    self.connection_theta[:, 1] + self.theta[self.connection_table[:, 1]]))
        self.f_y_array[:, 0] = (self.r_matrix[self.connection_table[:, 0], 1] + 
                                self.radius[self.connection_table[:, 0]]*np.sin(
                                    self.connection_theta[:, 0] + self.theta[self.connection_table[:, 0]]))
        self.f_y_array[:, self.Np-1] = (self.r_matrix[self.connection_table[:, 1], 1] + 
                                 self.radius[self.connection_table[:, 1]]*np.sin(
                                    self.connection_theta[:, 1] + self.theta[self.connection_table[:, 1]]))
        
        if self.periodic_bc:
            self.r[0:self.Nc] = (self.r[0:self.Nc] - self.xmin)%(2*self.half_box_size_x) + self.xmin
            self.r[self.Nc:2*self.Nc] = (self.r[self.Nc:2*self.Nc] - self.ymin)%(2*self.half_box_size_y) + self.ymin
            self.r_matrix = self.reshape_to_matrix(self.r)
            self.f_x_array = (self.f_x_array - self.xmin)%(2*self.half_box_size_x) + self.xmin
            self.f_y_array = (self.f_y_array - self.ymin)%(2*self.half_box_size_y) + self.ymin

            # calculate geometric quantities
            self.get_separation_vectors()
            self.get_distance_bet_colloid()

            U_rep_colloid = 1/2*self.kr*np.sum(
                np.triu((self.effective_diameter - self.colloid_dist_arr))**2*self.interaction_true_table)
            
            Ul = self._Ul()
            Ur = self._Ur()
            Urb = 0
            
            U_total = U_rep_colloid + Ul + Ur + Urb
            U_total_average = U_total/self.Nc
            U_colloid_average = (U_rep_colloid + Urb)/self.Nc
            
            result = {'Ul': Ul, 'Ur': Ur, 'Urb': Urb, 'U_rep_colloid': U_rep_colloid, 
                  'U_total': U_total, 'U_total_average': U_total_average, 'U_colloid_average': U_colloid_average}

        else:
            # calculate geometric quantities
            self.get_separation_vectors()
            self.get_distance_bet_colloid()
            
            U_rep_colloid = 1/2*self.kr*np.sum(
                np.triu((self.effective_diameter - self.colloid_dist_arr))**2*self.interaction_true_table)
            
            Ul = self._Ul()
            Ur = self._Ur()
            Urb = self._Urb()
            
            U_total = U_rep_colloid + Ul + Ur + Urb
            U_total_average = U_total/self.Nc
            U_colloid_average = (U_rep_colloid + Urb)/self.Nc
            
            result = {'Ul': Ul, 'Ur': Ur, 'Urb': Urb, 'U_rep_colloid': U_rep_colloid, 
                  'U_total': U_total, 'U_total_average': U_total_average, 'U_colloid_average': U_colloid_average}
        return result
    
    def _Ul(self):
        '''
        potential energy from the inextensibility of strings
        '''
        Ul = self.kl*self.Np/2*np.sum((self.dr - self.dLf)**2)
        return Ul
    
    def _Ur(self):
        '''
        repulsive component of filament with respect to the conectinng vacuole, to avoid filament overriding the vacuoles.
        '''
        if self.full_repulsion:
            
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
            Ur = 1/2*self.kr*np.sum((self.radius[None, :, None] - self.dr_to_vac)**2*(self.radius[None, :, None] > self.dr_to_vac))

        else:
            # compute distance
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
            else:
                pass
                
            self.dr_to_vac0 = (self.dx_to_vac0**2 + self.dy_to_vac0**2)**(0.5)
            self.dr_to_vac1 = (self.dx_to_vac1**2 + self.dy_to_vac1**2)**(0.5)
            
            Ur = 1/2*self.kr*np.sum((self.r_vac0 - self.dr_to_vac0)**2*(self.r_vac0 > self.dr_to_vac0) 
                + (self.r_vac1 - self.dr_to_vac1)**2*(self.r_vac1 > self.dr_to_vac1))
            
        return Ur
    
    def _Urb(self):
        '''
        repulsive component from the wall to the vacuoles
        '''
        if self.periodic_bc:
            Urb = 0
        else:
            Urb = np.sum(self.canvas.find_wall_potential(
                self.r[0:self.Nc].reshape((self.Nc,1)), self.r[self.Nc:].reshape((self.Nc,1)), self.radius.reshape((self.Nc,1))))
        return Urb
    
    def _KE(self, r_expanded, t):
        self.rhs_cython(r_expanded, t)
        KE = 1/2*np.sum((self.drEdt)**2)/len(self.drEdt)
        KE_colloid = 1/2*np.sum((self.drEdt[0:(self.dim+1)*self.Nc])**2)/len(self.drEdt)
        KE_velocity_ave = 1/2*np.sum((self.drEdt[0:self.dim*self.Nc])**2)/self.Nc
        KE_colloid_ave = KE_colloid/self.Nc
        result = {'KE': KE, 'KE_colloid': KE_colloid, 'KE_colloid_ave': KE_colloid_ave, 'KE_velocity_ave': KE_velocity_ave}
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
                            (self.Flx + self.Frx).flatten(), (self.Fly + self.Fry).flatten())
            if plot_colloid:
                for ix in range(-1, 2):
                    for iy in range(-1, 2):
                        plt.quiver(self.r[0:self.Nc] + ix*2*self.half_box_size_x, 
                            self.r[self.Nc:2*self.Nc] + iy*2*self.half_box_size_y, self.F[0:self.Nc], self.F[self.Nc:])
        else:
            if plot_string:
                plt.quiver(self.f_x_array.flatten(), self.f_y_array.flatten(), 
                       (self.Flx + self.Frx).flatten(), (self.Fly + self.Fry).flatten())
            if plot_colloid:
                plt.quiver(self.r[0:self.Nc], self.r[self.Nc:2*self.Nc], self.F[0:self.Nc], self.F[self.Nc:])
    
    def compute_ridge_point(self, r_expanded = None):
        """
        compute the connection map using scipy.spatial.voronoi
        """
        if r_expanded:
            self.change_r_expanded(r_expanded)
            vor = Voronoi(self.r_matrix)
        else:
            vor = Voronoi(self.r_matrix)

        rp = (np.sort(vor.ridge_points)).tolist()
        dist_rp = np.sqrt(np.sum(
            (vor.points[vor.ridge_points[:,0], :] - vor.points[vor.ridge_points[:,1], :])**2, 
            axis = 1))
        threshold = self.Rc*self.bidisperse*4
        [drop_list] = np.where(dist_rp > threshold)
        drop_list = np.flip(drop_list)
        for drop in drop_list:
            rp.pop(drop)
        return rp

    def rhs_cython(self, r_expanded, t):
        # Set the current filament state
        #self.drEdt = self.drEdt * 0
        #self.drdt = self.drdt * 0
        #self.dthetadt = self.dthetadt * 0
        #self.dfil_x_dt = self.dfil_x_dt * 0
        #self.dfil_y_dt = self.dfil_y_dt * 0
        self.F = self.F*0
        self.Trq = self.Trq*0
        
        if self.periodic_bc:
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
            self.f_x_array[:, 0] = (self.r_matrix[self.connection_table[:, 0], 0] + 
                                    self.radius[self.connection_table[:, 0]]*np.cos(
                                        self.connection_theta[:, 0] + self.theta[self.connection_table[:, 0]]))
            self.f_x_array[:, self.Np-1] = (self.r_matrix[self.connection_table[:, 1], 0] + 
                                     self.radius[self.connection_table[:, 1]]*np.cos(
                                        self.connection_theta[:, 1] + self.theta[self.connection_table[:, 1]]))
            self.f_y_array[:, 0] = (self.r_matrix[self.connection_table[:, 0], 1] + 
                                    self.radius[self.connection_table[:, 0]]*np.sin(
                                        self.connection_theta[:, 0] + self.theta[self.connection_table[:, 0]]))
            self.f_y_array[:, self.Np-1] = (self.r_matrix[self.connection_table[:, 1], 1] + 
                                     self.radius[self.connection_table[:, 1]]*np.sin(
                                        self.connection_theta[:, 1] + self.theta[self.connection_table[:, 1]]))

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
            self.Trq_temp0 = self.radius[self.connection_table[:, 0]]*(
                    -self.Flx[:, 0]*np.sin(self.connection_theta[:, 0] + self.theta[self.connection_table[:, 0]]) 
                    + self.Fly[:, 0]*np.cos(self.connection_theta[:, 0] + self.theta[self.connection_table[:, 0]]))
            self.Trq_temp1 = self.radius[self.connection_table[:, 1]]*(
                    -self.Flx[:, self.Np-1]*np.sin(self.connection_theta[:, 1] + self.theta[self.connection_table[:, 1]]) 
                    + self.Fly[:, self.Np-1]*np.cos(self.connection_theta[:, 1] + self.theta[self.connection_table[:, 1]]))
            for i in range(self.Nf):
                self.Trq[self.connection_table[i, 0]] += self.Trq_temp0[i]
                self.Trq[self.connection_table[i, 1]] += self.Trq_temp1[i]
            self.dthetadt = self.Trq/self.eight_pi_eta_r_cube

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
            self.drEdt[(self.dim+1)*self.Nc+self.Np*self.Nf:(self.dim+1)*self.Nc+self.Np*self.Nf*2] = self.dfil_y_dt

        else:
            self.r = r_expanded[0:self.dim*self.Nc]
            self.r_matrix = self.reshape_to_matrix(self.r)
            self.theta = r_expanded[self.dim*self.Nc:(self.dim+1)*self.Nc]
            
            self.f_x_array = np.reshape(
                r_expanded[(self.dim+1)*self.Nc: ((self.dim+1)*self.Nc+self.Nf*self.Np)], (self.Nf, self.Np), order = 'C')
            self.f_y_array = np.reshape(
                r_expanded[((self.dim+1)*self.Nc+self.Nf*self.Np): ((self.dim+1)*self.Nc+2*self.Nf*self.Np)], (self.Nf, self.Np), order = 'C')

            # boundary condition on attachment site: free to rotate, but not free to move
            self.f_x_array[:, 0] = (self.r_matrix[self.connection_table[:, 0], 0] + 
                                    self.radius[self.connection_table[:, 0]]*np.cos(
                                        self.connection_theta[:, 0] + self.theta[self.connection_table[:, 0]]))
            self.f_x_array[:, self.Np-1] = (self.r_matrix[self.connection_table[:, 1], 0] + 
                                     self.radius[self.connection_table[:, 1]]*np.cos(
                                        self.connection_theta[:, 1] + self.theta[self.connection_table[:, 1]]))
            self.f_y_array[:, 0] = (self.r_matrix[self.connection_table[:, 0], 1] + 
                                    self.radius[self.connection_table[:, 0]]*np.sin(
                                        self.connection_theta[:, 0] + self.theta[self.connection_table[:, 0]]))
            self.f_y_array[:, self.Np-1] = (self.r_matrix[self.connection_table[:, 1], 1] + 
                                     self.radius[self.connection_table[:, 1]]*np.sin(
                                        self.connection_theta[:, 1] + self.theta[self.connection_table[:, 1]]))
            
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
            
            fx, fy = self.canvas.find_wall_force(
                self.r[0:self.Nc].reshape((self.Nc,1)), 
                self.r[self.Nc:].reshape((self.Nc,1)), 
                self.radius.reshape((self.Nc,1)))
            self.F[0:self.Nc] += self.kr_b*self.kr*fx.flatten()
            self.F[self.Nc:] += self.kr_b*self.kr*fy.flatten()
            
            # Forces on vacuoles from the end of the filaments
            for i in range(self.Nf):
                self.F[0+self.connection_table[i, 0]] += self.Flx[i, 0]
                self.F[0+self.connection_table[i, 1]] += self.Flx[i, self.Np-1]
                self.F[self.Nc+self.connection_table[i, 0]] += self.Fly[i, 0]
                self.F[self.Nc+self.connection_table[i, 1]] += self.Fly[i, self.Np-1]
            
            self.drdt[0:self.Nc] = self.F[0:self.Nc]/self.six_pi_eta_r
            self.drdt[self.Nc:] = self.F[self.Nc:]/self.six_pi_eta_r
            
            # Torque on vacuoles from the end of the filaments
            self.Trq_temp0 = self.radius[self.connection_table[:, 0]]*(
                    -self.Flx[:, 0]*np.sin(self.connection_theta[:, 0] + self.theta[self.connection_table[:, 0]]) 
                    + self.Fly[:, 0]*np.cos(self.connection_theta[:, 0] + self.theta[self.connection_table[:, 0]]))
            self.Trq_temp1 = self.radius[self.connection_table[:, 1]]*(
                    -self.Flx[:, self.Np-1]*np.sin(self.connection_theta[:, 1] + self.theta[self.connection_table[:, 1]]) 
                    + self.Fly[:, self.Np-1]*np.cos(self.connection_theta[:, 1] + self.theta[self.connection_table[:, 1]]))
            for i in range(self.Nf):
                self.Trq[self.connection_table[i, 0]] += self.Trq_temp0[i]
                self.Trq[self.connection_table[i, 1]] += self.Trq_temp1[i]
            self.dthetadt = self.Trq/self.eight_pi_eta_r_cube
            
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
            self.drEdt[(self.dim+1)*self.Nc+self.Np*self.Nf:(self.dim+1)*self.Nc+self.Np*self.Nf*2] = self.dfil_y_dt
        
    def simulate(self, Tf = 100, t0 = 0, Npts = 10, stop_tol = 1E-5, sim_type = 'point', 
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

        self.folder = 'SimResults_Nc_{}_Np_{}_Nf_{}_volfrac_{}_filfrac_{}_solver_{}'.format\
                            (self.Nc, self.Np, self.Nf, round(self.vol_frac_initial, 2), round(self.filament_frac_initial, 2), 
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
                    solver = odespy.lsoda_scipy(rhs_odespy, atol=1E-7, rtol=1E-6)
                    solver.set_initial_condition(self.r_expanded) # Initial conditions
                    self.R, self.Time = solver.solve(time_points)
                    self.R = self.R.T
                else:
                    solver = odespy.Vode(rhs_odespy, method = self.method, atol=1E-7, rtol=1E-6, order=5) # initialize the odespy solver
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
            if(file[-4:] == 'hdf5'):  # Newer data format (.hdf5)
                print('Loading hdf5 file')
                with h5py.File(file, "r") as f:
                    if('simulation data' in f.keys()): # Load the simulation data (newer method)
                        
                        dset = f['simulation data']
                        self.Time = dset["Time"][:]
                        self.R = dset["Position"][:]
                        self.radius = dset["radius"][:]
                        self.six_pi_eta_r = 6*PI*self.eta*self.radius
                        self.eight_pi_eta_r_cube = 8*PI*self.eta*self.radius**3
                        self.effective_diameter = self.radius + self.radius.reshape((self.Nc,1))
                        self.connection_table = dset["connection table"][:]
                        self.connection_theta = dset["connection theta"][:]
                        self.change_canvas(dset["canvas_xy"][:])

                        # Load the metadata:
                        self.Nc = dset.attrs['N colloids']
                        self.Np = dset.attrs['N particles per filament']
                        self.Rc = dset.attrs['baseline radius']
                        self.bidisperse = dset.attrs['bidisperse']
                        self.Rep = dset.attrs['repulsive number']
                        self.kr = dset.attrs['repulsive constant']
                        self.kr_b = dset.attrs['repulsive constant from wall']
                        self.kl = dset.attrs['spring constant of string']
                        self.Lf = dset.attrs['length of strings'] 
                        self.dLf = dset.attrs['segment length of string']
                        self.eta = dset.attrs['viscosity']
                        self.Contr = dset.attrs['contractility']
                        self.Inext = dset.attrs['inextensibility'] 

                        try:
                            self.periodic_bc = dset.attrs['periodic bc']
                            try:
                                self.half_box_size_x = dset.attrs['half box size x']
                                self.half_box_size_y = dset.attrs['half box size y']
                            except:
                                self.half_box_size = dset.attrs['half box size']

                            try:
                                self.full_repulsion = dset.attrs['full repulsion']
                            except:
                                self.full_repulsion = False

                            try:
                                self.mLf = dset.attrs['normalized length of strings']
                            except:
                                self.mLf = 3.3

                        except:
                            self.periodic_bc = False
    
    def save_data(self):
        """
        Implement a save module based on HDF5 format
        """
        copy_number = 0
        self.saveFile = 'SimResults_{0:03d}.hdf5'.format(copy_number)

        if(self.save):
            if(not os.path.exists(self.saveFolder)):
                os.makedirs(self.saveFolder)

            # Choose a new copy number for multiple simulations with the same parameters
            while(os.path.exists(os.path.join(self.saveFolder, self.saveFile)) and self.overwrite == False):
                copy_number+=1
                self.saveFile = 'SimResults_{0:03d}.hdf5'.format(copy_number)


        with h5py.File(os.path.join(self.saveFolder, self.saveFile), "w") as f:

            dset = f.create_group("simulation data")
            dset.create_dataset("Time", data = self.Time)
            dset.create_dataset("Position", data = self.R)
            dset.create_dataset('radius', data = self.radius)
            dset.create_dataset('connection table', data = self.connection_table)
            dset.create_dataset('connection theta', data = self.connection_theta)
            dset.create_dataset('canvas_xy', data = self.canvas.xy)
            dset.attrs['N colloids'] = self.Nc
            dset.attrs['N particles per filament'] = self.Np
            dset.attrs['N filaments'] = self.Nf
            dset.attrs['baseline radius'] = self.Rc
            dset.attrs['bidisperse'] = self.bidisperse
            dset.attrs['repulsive number'] = self.Rep
            dset.attrs['repulsive constant'] = self.kr
            dset.attrs['repulsive constant from wall'] = self.kr_b
            dset.attrs['spring constant of string'] = self.kl
            dset.attrs['normalized length of strings'] = self.mLf
            dset.attrs['length of strings'] = self.Lf
            dset.attrs['segment length of string'] = self.dLf
            dset.attrs['viscosity'] = self.eta
            dset.attrs['contractility'] = self.Contr
            dset.attrs['inextensibility'] = self.Inext
            dset.attrs['volume fraction'] = self.vol_frac
            dset.attrs['filament fraction'] = self.filament_frac_initial
            dset.attrs['periodic bc'] = self.periodic_bc
            dset.attrs['half box size x'] = self.half_box_size_x
            dset.attrs['half box size y'] = self.half_box_size_y
            dset.attrs['full repulsion'] = self.full_repulsion
            
            if(self.activity_profile is not None):
                dset.create_dataset("activity profile", data = self.activity_profile(self.Time))

        # Save user readable metadata in the same folder
        self.metadata = open(os.path.join(self.saveFolder, 'metadata.csv'), 'w+')
        self.metadata.write('Dimensions,'+
                            'N colloids,'+
                            'N particles per filament,'+
                            'N filaments,'+
                            'baseline radius,'+
                            'bidisperse,'+
                            'repulsive number,'+
                            'repulsive constant,'+
                            'repulsive constant from wall,'+
                            'contractility,'+
                            'inextensibility,'+
                            'ODE solver method,'+
                            'spring constant of string,'+
                            'characteristic velocity,'+
                            'viscosity,'+
                            'normalized length of strings,'
                            'length of strings,'+
                            'segment length of string,'+
                            'initial volume fraction,'+
                            'filament fraction,'+
                            'maximum filament number,'+
                            'volume fraction,'+
                            'periodic boundary,'+
                            'full repulsion,'+
                            'seed,'+
                            'Simulation time,'+
                            'Seed,'+
                            'CPU time (s)\n')
        self.metadata.write(str(self.dim)+','+
                            str(self.Nc)+','+
                            str(self.Np)+','+
                            str(self.Nf)+','+
                            str(self.Rc)+','+
                            str(self.bidisperse)+','+
                            str(self.Rep)+','+
                            str(self.kr)+','+
                            str(self.kr_b)+','+
                            str(self.Contr)+','+
                            str(self.Inext)+','+
                            self.method+','+
                            str(self.kl)+','+
                            str(self.v_char)+','+
                            str(self.eta)+','+
                            str(self.mLf)+','+
                            str(self.Lf)+','+
                            str(self.dLf)+','+
                            str(self.vol_frac_initial)+','+
                            str(self.filament_frac_initial)+','+
                            str(self.Nf_Delaunay)+','+
                            str(self.vol_frac)+','+
                            str(self.periodic_bc)+','+
                            str(self.full_repulsion)+','+
                            str(self.seed)+','+
                            str(self.Time[self.Npts])+','+
                            str(self.seed)+','+
                            str(self.cpu_time))
        self.metadata.close()
