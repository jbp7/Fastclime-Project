
import readline
import fastclime as fc
import numpy as np

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
fastclime = importr('fastclime')
grdevices = importr('grDevices')

##For some tests, we simulate some random data
# and check that the Python output matches that in R
#Generate random data using fastclime.generator in R
n = 200
d = 50
lambda_min = 0.1
nlambda = 50
L = fastclime.fastclime_generator(n, d)
x = np.array(L.rx2('data'))
Pyout = fc.fastclime_R(x,lambda_min,nlambda)
Rout = fastclime.fastclime(L.rx2('data'),lambda_min,nlambda)
##################################################################

############################# 
#Unit tests for is_Hermitian
#############################

#Check that square matrix is successful
def test_square():
    A = np.eye(2)
    assert fc.is_Hermitian(A)

#Check that non-square matrix fails
def test_nonsquare():
    A = np.arange(6).reshape((3,2))
    np.testing.assert_equal(fc.is_Hermitian(A),False)

#Check that square symmetric matrix with missing values is successful
def test_square_miss():
    A = np.eye(3)
    A[1,:] = np.nan
    assert fc.is_Hermitian(A)

############################# 
#Unit tests for is_Hermitian
#############################

#Check that square matrix is successful
def test_square():
    A = np.eye(2)
    assert fc.is_Hermitian(A)

#Check that non-square matrix fails
def test_nonsquare():
    A = np.arange(6).reshape((3,2))
    np.testing.assert_equal(fc.is_Hermitian(A),False)

#Check that square symmetric matrix with missing values is successful
def test_square_miss():
    A = np.eye(3)
    A[1,:] = np.nan
    assert fc.is_Hermitian(A)
    
############################# 
#Unit tests for symmetrize
#############################

#Check that non-square matrix fails
def test_nonsquare():
    A = np.arange(6).reshape((3,2))
    np.testing.assert_raises(TypeError,fc.symmetrize,A)
    
#Check that inputting np.array([1,0],[5,1]) with min rule gives I_2
def test_I2min():
    A = np.array([[1,0],[5,1]],dtype=float)
    np.testing.assert_equal(fc.symmetrize(A,rule="min"),np.eye(2))

#Check that inputting np.array([1,0],[-5,1]) with max rule gives I_2
def test_I2max():
    A = np.array([[1,0],[-5,1]],dtype=float)
    np.testing.assert_equal(fc.symmetrize(A,rule="max"),np.eye(2))

#Check that a positive semi-definite matrix flags user warning
def test_psd():
    Sigma = np.eye(3)
    Omega = np.array([[1,0,0],[0,0,0],[0,0,1]])
    np.testing.assert_warns(UserWarning,fc.loglik,Sigma,Omega)
    
############################# 
#Unit tests for loglik
#############################    
    
#Check that absolute value of log likelihood is <= Inf for at least
#   a positive semidefinite precision matrix
#PosDef Omega
def test_loglikval1():
    np.random.seed(1234)
    A = np.random.normal(0,1,(3,4))
    Sigma = np.corrcoef(A.T)
    Omega = np.linalg.inv(Sigma)
    assert fc.loglik(Sigma,Omega)<=float('Inf')

#PSD Omega
def test_loglikval2():
    Sigma = np.eye(3,dtype=float)
    Omega = np.array([[1,0,0],[0,0,0],[0,0,1]],dtype=float)
    assert fc.loglik(Sigma,Omega)<=float('Inf')
    
#Check that identity matrices (I_3) for both covariance and 
#   precision ma[list()]trices yield correct likelihood value of -3.0
def test_loglikval3():
    Sigma = np.eye(3)
    Omega = np.eye(3)
    np.testing.assert_equal(fc.loglik(Sigma,Omega),-3.0)
    
############################# 
#Unit tests for fastclime_lambda
#############################    
    
#Using the simulated data above, check Python and R output
def test_fc_lambda():
    lambda_val = 0.2
    icovPy = fc.fastclime_lambda(Pyout.lambdamtx,Pyout.icovlist,lambda_val)
    icovR = fastclime.fastclime_lambda(Rout.rx2('lambdamtx'),Rout.rx2('icovlist'),lambda_val)
    np.testing.assert_allclose(icovPy,np.array(icovR.rx2('icov')))

############################## 
#Unit tests for fastclime_select
#############################    
 
#This test checks that the function is doing something REASONABLE under every
#possible metric

#Check that selected opt_lambda is between lambda_min and 1 (not including 1)

def test_opt_lambda_aic():
    res = fc.fastclime_select(x,Pyout.lambdamtx,Pyout.icovlist,metric="AIC")
    assert (lambda_min <= res.opt_lambda < 1.0) 

def test_opt_lambda_bic():
    res = fc.fastclime_select(x,Pyout.lambdamtx,Pyout.icovlist,metric="BIC")
    assert (lambda_min <= res.opt_lambda < 1.0) 

#Check that selected opt_icov is not identity

def test_opt_icov_aic():
    res = fc.fastclime_select(x,Pyout.lambdamtx,Pyout.icovlist,metric="AIC")
    assert np.allclose(res.opt_lambda,np.eye(d))==False

def test_opt_icov_bic():
    res = fc.fastclime_select(x,Pyout.lambdamtx,Pyout.icovlist,metric="BIC")
    assert np.allclose(res.opt_lambda,np.eye(d))==False

############################# 
#Unit tests for fastclime_R
#############################

#Check that function throws an error when input is not ndarray
def test_input():
    inarray = list(np.arange(6).reshape(2,3)) #a list 
    np.testing.assert_raises(TypeError,fc.fastclime_R,inarray)

#For the simulated data above check that the following output 
#are equivalent between Python and R

#Check that x is the same
def test_x():
    np.testing.assert_allclose(Pyout.x,np.array(Rout.rx2('data')))
    
#Check that cov_input is the same
def test_cov_input():
    np.testing.assert_allclose(Pyout.cov_input,np.array(Rout.rx2('cov.input')))

#Check that Sigmahat is the same
def test_Sigmahat():
    np.testing.assert_allclose(Pyout.Sigmahat,np.array(Rout.rx2('sigmahat')))

#Check that maxnlambda is the same
def test_maxnlambda():
    np.testing.assert_allclose(Pyout.maxnlambda,np.array(Rout.rx2('maxnlambda')))

#Check that lambdamtx is the same
def test_lambdamtx():
    np.testing.assert_allclose(Pyout.lambdamtx,np.array(Rout.rx2('lambdamtx')))

#Check that each icov in icovlist is the same
def test_icovlist():
    for i in range(Pyout.maxnlambda):
        np.testing.assert_allclose(Pyout.icovlist[:,:,i],np.array(Rout.rx2('icovlist')[i]))