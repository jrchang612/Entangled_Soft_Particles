cimport cython
import numpy as np
cimport numpy as np
np.import_array()
DTYPE = np.double
ctypedef np.double_t DTYPE_t
from libc.math cimport atan2
from cython.parallel import prange

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)

cpdef angle_xy(np.ndarray v1x, np.ndarray v1y, np.ndarray v2x, np.ndarray v2y, int n_site):
	cdef np.ndarray ang = np.zeros([n_site], dtype = DTYPE)
	#cdef int ii
	#for ii in range(n_site):
		#ang[ii] = atan2(-v2x[ii]*v1y[ii] + v2y[ii]*v1x[ii], v2x[ii]*v1x[ii] + v2y[ii]*v1y[ii])
	ang = np.arctan2(-v2x[0:n_site]*v1y[0:n_site] + v2y[0:n_site]*v1x[0:n_site], v2x[0:n_site]*v1x[0:n_site] + v2y[0:n_site]*v1y[0:n_site])
	return ang
