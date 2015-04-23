
import numpy as np
cimport numpy as np

np.import_array()

# cdefine the signature of the c function
cdef extern from "paralp.h":
    void paralp(double *obj, double *mat, double *rhs, int *m0 , 
                int *n0, double *opt, int *status, double *lambda_min, 
                double *rhs_bar, double *obj_bar)
    
def mainfunc(np.ndarray[double, ndim=1, mode="c"] obj not None,
             np.ndarray[double, ndim=2, mode="c"] mat not None,
             np.ndarray[double, ndim=1, mode="c"] rhs not None,
             np.ndarray[double, ndim=1, mode="c"] obj_bar not None,
             np.ndarray[double, ndim=1, mode="c"] rhs_bar not None,
             double lambda_min):   

    #Dimensions    
    cdef int m  = len(rhs)
    cdef int n  = len(obj)
    cdef int m0 = mat.shape[0]
    cdef int n0 = mat.shape[1]
    cdef int m1 = len(rhs_bar)
    cdef int n1 = len(obj_bar)
    
    #Define output
    cdef np.ndarray opt = np.zeros((len(obj),), dtype = np.float64, order='C')
    cdef int status = 0
    
    if (m != m0 or n != n0 or m != m1 or n != n1):
        raise ValueError("Dimensions do not match.")
    
    if ((obj_bar < 0.).any() or (rhs_bar < 0.).any()):
        raise ValueError("The pertubation vector obj_bar and rhs_bar must be nonnegative.")
        
    #Call external C function
    paralp(<double*> np.PyArray_DATA(obj),
           <double*> np.PyArray_DATA(mat),
           <double*> np.PyArray_DATA(rhs),
           &m0,
           &n0,
           <double*> np.PyArray_DATA(opt),
           &status,
           <double*> &lambda_min,
           <double*> np.PyArray_DATA(rhs_bar),
           <double*> np.PyArray_DATA(obj_bar))

    if (status == 0): 
        #print "optimal solution found! \n"
        return opt

    elif (status == 1):
        print "The problem is infeasible! \n"
    
    elif (status == 2):
        print "The problem is unbounded! \n"