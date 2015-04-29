
import fastclime as fc
import numpy as np

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
