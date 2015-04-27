
import numpy as np
cimport numpy as np

np.import_array()

# cdefine the signature of the c function
cdef extern from "parametric2.h":
    void parametric2(double *SigmaInput_xx, double *SigmaInput_xy, int *m1, double *mu_input, 
                    double *lambdamin, int *nlambda, int *maxnlambda, double *iicov);
    
def mainfunc(np.ndarray[double, ndim=2, mode="c"] SigmaInput_xx not None,
             np.ndarray[double, ndim=2, mode="c"] SigmaInput_xy not None,
             double lambdamin, 
             int nlambda):   

    #Dimensions    
    cdef int m1 = SigmaInput_xx.shape[1]
    
    #Define output
    cdef int maxnlambda = 0
    cdef np.ndarray mu_input = np.zeros((m1,nlambda), dtype = np.float64, order='C')
    cdef np.ndarray iicov = np.zeros((m1*m1,nlambda), dtype = np.float64, order='C')

    #Call external C function
    parametric2(<double*> np.PyArray_DATA(SigmaInput_xx),
               <double*> np.PyArray_DATA(SigmaInput_xy),
               &m1,
               <double*> np.PyArray_DATA(mu_input),
               <double*> &lambdamin,
               &nlambda,
               &maxnlambda,
               <double*> np.PyArray_DATA(iicov)
              )
    
    return (SigmaInput_xx,SigmaInput_xy,mu_input,maxnlambda,iicov)