
import numpy as np
cimport numpy as np

np.import_array()

# cdefine the signature of the c function
cdef extern from "parametric.h":
    void parametric(double *SigmaInput, int *m1, double *mu_input, 
                    double *lambdamin, int *nlambda, int *maxnlambda, double *iicov);
    
def mainfunc(np.ndarray[double, ndim=2, mode="c"] SigmaInput not None, 
             double lambdamin, 
             int nlambda):   

    #Dimensions    
    cdef int m1 = SigmaInput.shape[1]
 
    #Define output
    cdef int maxnlambda = 0
    cdef np.ndarray SigmaOutput = SigmaInput.copy()
    cdef np.ndarray mu_input = np.zeros((nlambda,m1), dtype = np.float64)
    cdef np.ndarray iicov = np.zeros((nlambda,m1*m1), dtype = np.float64)
    
    #Call external C function
    parametric(<double*> np.PyArray_DATA(SigmaOutput),
               &m1,
               <double*> np.PyArray_DATA(mu_input),
               <double*> &lambdamin,
               &nlambda,
               &maxnlambda,
               <double*> np.PyArray_DATA(iicov))
    
    return (SigmaOutput,mu_input,maxnlambda,iicov)