from __future__ import division
#sys.path.remove('/share/software/user/open/py-numpy/1.14.3_py27/lib/python2.7/site-packages')
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
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.sparse import coo_matrix
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns
import sys
cwd = os.getcwd()
sys.path.insert(1, cwd)
#from rs_helper import rock_string_soft_helpers_SSmethod_ERbreak_fullc_canvasVar as rsssb
from rs_helper import rock_string_soft_analysis_helper_SSmethod as rsssh
from matplotlib import animation, rc
from IPython.display import HTML
from moviepy.editor import VideoFileClip, concatenate_videoclips

env_var = os.environ
folder_name = env_var['FOLDERNAME']
try:
	copy_number = env_var['CPNUMBER']
except:
	copy_number = 0
ER_break = bool(env_var['ER_BREAK'])

exist = True
while exist:
	file_name = '/scratch/users/jrc612/' + folder_name + 'Anim_{0:04d}.mp4'.format(copy_number)
	if os.path.exists(file_name):
		copy_number += 1
	else:
		exist = False

rsssh.animate_folder_result(
    '/scratch/users/jrc612/'+folder_name, 
    copy_number = copy_number, lw = 1, ER_break = ER_break, jamming = True)
