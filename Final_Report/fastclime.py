
def is_Hermitian(m):
    """
    Checks if a given matrix is Hermitian 
    (symmetric)

    Parameters:
    -----------------------------------------
    m   :  A 2-D array

    Returns:
    -----------------------------------------
    logical (True if symmetric)
    """
    import numpy as np
    
    #Set missing to zero prior to checking symmetry
    m[np.isnan(m)] = 0.

    try:
        return np.allclose(np.transpose(m,(1,0)), m)
    except:
        return False
    
def symmetrize(m,rule="min"):
    """
    Symmetrizes a given square matrix with 
    respect to main diagonal, based on a rule

    Parameters:
    -----------------------------------------
    m   :  A square matrix

    rule:  criterion for symmetrizing m
           -"min" computes the minimum of m(i,j) and m(j,i)
           -"max" computes the maximum of m(i,j) and m(j,i)
           where i and j are row and column indices

    Returns:
    -----------------------------------------
    A symmetric square matrix
    """
    import numpy as np
    
    if (m.shape[0] != m.shape[1]):
        raise TypeError("Input matrix must be square.")
        
    if (rule == "min"):
        min_mat =  np.fmin(np.triu(m),np.tril(m).T)
        return np.triu(min_mat,1) + min_mat.T

    elif (rule == "max"):
        max_mat =  np.fmax(np.triu(m),np.tril(m).T)
        return np.triu(max_mat,1) + max_mat.T
    
    else:
        raise ValueError("Specify rule as min or max.")

def loglik(Sigma,Omega):
    """
    Computes the log likelihood given precision matrix
    estimates

    Parameters:
    -----------------------------------------
    Sigma:  covariance matrix

    Omega:  precision matrix 

    Returns:
    -----------------------------------------
    log likelihood value
    """
    import warnings
    import numpy as np
    
    #Check if precision matrix estimate is positive definite
    if (np.linalg.det(Omega) <= 0.0):
        warnings.warn("Precision matrix estimate is not positive definite.")
    
    loglik = np.log(np.linalg.det(Omega)) - np.trace(Sigma.dot(Omega))
    
    if np.isfinite(loglik):
        return loglik
    else:
        return float('Inf')
    
def fastclime_lambda(lambdamtx,icovlist,lambda_val):
    """
    Selects the precision matrix for a given lambda

    Parameters:
    ------------------------------------------------------
    lambdamtx : The sequence of regularization parameters for each column, 
                it is a nlambda by d matrix. It will be filled with 0 when 
                the program finds the required lambda.min value for that 
                column. 
    
    icovlist  : A nlambda list of d by d precision matrices as an 
                alternative graph path (numerical path) corresponding 
                to lambdamtx. 
               
    lambda_val : user-selected regularization tuning parameter 
                 (must be greater than lambda_min)

    Returns  : 
    ------------------------------------------------------
    Precision matrix corresponding to lambda_val
    """ 
    import numpy as np
    import warnings
    
    d = icovlist[:,:,0].shape[1]
    maxnlambda = lambdamtx.shape[0]
    status = 0
    seq = np.empty((d,),dtype=int)
    icov = np.empty((d,d),dtype=float)
    
    for i in range(d):
        temp_lambda = np.where(lambdamtx[:,i]>lambda_val)[0]
        seq[i] = len(temp_lambda)
    
        if((seq[i]+1)>maxnlambda):
            status=1
            icov[:,i]=icovlist[:,:,seq[i]][:,i]
        else:
            icov[:,i]=icovlist[:,:,seq[i]][:,i]
            
    icov = (icov + icov.T)/2.0
    
    if (status == 1):
        warnings.warn("Some columns do not reach the required lambda.\n You may want to increase lambda_min or use a large nlambda. \n")
    
    del temp_lambda, seq, d#, threshold
    
    return icov

class ReturnSelect(object):
    def __init__(self,opt_lambda, opt_icov):
        self.opt_lambda = opt_lambda
        self.opt_icov = opt_icov
    
    
def fastclime_select(x,lambdamtx,icovlist,metric="BIC"):
    """
    Computes optimal regularization tuning parameter, 
    lambda using AIC or BIC metric

    Parameters:
    ------------------------------------------------------ 
    x         : data matrix
    
    lambdamtx : The sequence of regularization parameters for each column, 
                it is a nlambda by d matrix. It will be filled with 0 when 
                the program finds the required lambda.min value for that 
                column. This parameter is required for fastclime_lambda.
    
    icovlist  : A nlambda list of d by d precision matrices as an 
                alternative graph path (numerical path) corresponding 
                to lambdamtx. This parameter is also required for
                fastclime_lambda.
               
    metric    : selection criterion. AIC and BIC are available.
                When n < d, the degrees of freedom for AIC and BIC are 
                adjusted based on the number of non-zero elements in 
                the estimate precision matrix.

    Returns   : 
    ------------------------------------------------------
    optimal lambda parameter and corresponding precision
    matrix
    
    References: 1) H. Pang, et al. (2014) The fastclime Package for Linear 
                Programming and Large-Scale Precision Matrix Estimation in R

    
    """
    import numpy as np
    
    SigmaInput = np.corrcoef(x.T)
        
    #Dimensions
    n = x.shape[0]
    d = SigmaInput.shape[1]
    nl = icovlist.shape[2]
        
    if (metric=="AIC"):
        AIC = np.empty((nl,),dtype=float)

        for i in range(nl):
            if (d>n):
                m = np.sum(np.absolute(icovlist[:,:,1])>1.0e-5,dtype=int)
                df = d + m*(m-1.0)/2.0
            else: 
                df = d
                
            AIC[i]=-2.0*loglik(SigmaInput,icovlist[:,:,i]) + df*2.0
        
        opt_index = np.where(AIC[2:]==min(AIC[2:][np.where(AIC[2:]!=-np.inf)]))[0]+2
        opt_lambda = np.max(lambdamtx[opt_index,:])

    if (metric=="BIC"):
        BIC = np.empty((nl,),dtype=float)

        for i in range(nl):
            if (d>n):
                m = np.sum(np.absolute(icovlist[:,:,1])>1e-5,dtype=int)
                df = d + m*(m-1.0)/2.0
            else: 
                df = d
                
            BIC[i]=-2.0*loglik(SigmaInput,icovlist[:,:,i]) + df*np.log(n)
        
        opt_index = np.where(BIC[2:]==min(BIC[2:][np.where(BIC[2:]!=-np.inf)]))[0]+2
        opt_lambda = np.max(lambdamtx[opt_index,:])
    
    opt_icov = symmetrize(fastclime_lambda(lambdamtx,icovlist,opt_lambda),rule="min")
    
    return ReturnSelect(opt_lambda, opt_icov)

class fastclime_obj(object):
    def __init__(self, x, cov_input, Sigmahat, maxnlambda, lambdamtx, icovlist):
        self.x = x
        self.cov_input = cov_input
        self.Sigmahat = Sigmahat  
        self.maxnlambda = maxnlambda
        self.lambdamtx = lambdamtx
        self.icovlist = icovlist
    
def fastclime_R(x,lambda_min=0.1,nlambda=50):
    """
    Main function for CLIME estimation of the precision
    matrix

    Parameters:
    ---------------------------------------------------
    x          :  There are 2 options: (1) x is an n by d data matrix 
                  (2) a d by d sample covariance matrix. The program 
                  automatically identifies the input matrix by checking the
                  symmetry. The program automatically normalizes the data
                  to mean 0 and standard deviation 1 along each column.

    lambda_min :  This is the smallest value of lambda you would 
                  like the solver to explorer
    
    nlambda    :  maximum path length. Note if d is large and nlambda 
                  is also large, it is possible that the program
                  will fail to allocate memory for the path.

    Returns:
    ---------------------------------------------------
    x         : Input matrix
    
    cov_input : Indicator for sample covariance
    
    Sigmahat  : normalized empirical covariance matrix
    
    maxnlambda: The length of the path. If the program finds 
                lambda.min in less than nlambda iterations for 
                all columns, then the acutal maximum lenth for 
                all columns will be returned. Otherwise it equals 
                nlambda.
    
    lambdamtx : The sequence of regularization parameters for each column, 
                it is a nlambda by d matrix. It will be filled with 0 when 
                the program finds the required lambda.min value for that 
                column. This parameter is required for fastclime_lambda.
    
    icovlist  : A nlambda list of d by d precision matrices as an 
                alternative graph path (numerical path) corresponding 
                to lambdamtx. This parameter is also required for
                fastclime_lambda.
    
    References: 1) T. Cai, et al. (2011) A constrained l1 minimization 
                approach to sparse precision matrix estimation.
     
                2) H. Pang, et al. (2014) The fastclime Package for Linear 
                Programming and Large-Scale Precision Matrix Estimation in R
    """
    import numpy as np
    import parametric as param
    
    if (isinstance(x,np.ndarray)==False):
        raise TypeError("Input must be ndarray.")
        
    cov_input = 1
    SigmaInput = x.copy()
    
    #Check if matrix is symmetric
    #If not, create normalized covariance matrix = correlation matrix
    if not is_Hermitian(SigmaInput):
        SigmaInput = np.corrcoef(SigmaInput.T)
        cov_input = 0
    
    #Run parametric simplex linear solver
    Sigmahat, mu, maxnlambda, iicov = param.mainfunc(SigmaInput,lambda_min,nlambda)
      
    #Process output
    maxnlambda+=1
    
    #Reshape the array in Fortran order
    #and then slice the array to extract only the top maxnlambda rows
    lambdamtx = mu.T.reshape(nlambda, -1, order='F')[:maxnlambda,:]
    mu = None
    
    #Take each row of iicov and convert it to a d x d matrix
    d = Sigmahat.shape[1]
    icovlist = np.empty((d, d, maxnlambda)) 
   
    for i in range(maxnlambda):
        icovlist[:,:,i] = iicov[:,i].reshape((d,d)).T
    
    return fastclime_obj(x, cov_input, Sigmahat, maxnlambda, lambdamtx, icovlist)

def fastclime_est_select(x,lambda_min=0.1,nlambda=50,metric="BIC"):
    """
    Performs CLIME estimation and automatically selects the 
    optimal regularization parameter based on a given metric
    (default = BIC)

    Parameters:
    ------------------------------------------------------ 
    x         : data matrix
    
    lambda_min :  This is the smallest value of lambda you would 
                  like the solver to explorer
    
    nlambda    :  maximum path length. Note if d is large and nlambda 
                  is also large, it is possible that the program
                  will fail to allocate memory for the path.
               
    metric    : selection criterion. AIC and BIC are available.
                When n < d, the degrees of freedom for AIC and BIC are 
                adjusted based on the number of non-zero elements in 
                the estimate precision matrix.

    Returns   : 
    ------------------------------------------------------
    optimal lambda parameter and corresponding precision
    matrix
    
    References: 1) H. Pang, et al. (2014) The fastclime Package for Linear 
                Programming and Large-Scale Precision Matrix Estimation in R

    
    """

    #Get CLIME estimates of the regularization path
    fcres = fastclime_R(x,lambda_min,nlambda)
    
    #Get icov corresponding to best selected regularization parameter
    fcres_select = fastclime_select(x,fcres.lambdamtx,fcres.icovlist,metric)
    
    return ReturnSelect(fcres_select.opt_lambda,fcres_select.opt_icov)
    