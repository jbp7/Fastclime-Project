
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
    Symmetrizes a given square matrix based on a rule

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
        raise ValueError("Input matrix must be square.")
        
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
    Sigma:  empirical covariance matrix

    Omega:  precision matrix (estimate or ground truth)

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
    Selects the precision matrix and solution path for a 
    given parameter lambda

    Parameters:
    ------------------------------------------------------
    lambdamtx : A nlambda x d array containing the regularization path 
                of tuning parameters for each column of the estimated 
                precision matrix
    
    icovlist  : A 3-D array with nlambda d x d precision matrices
               estimated from fastclime
               
    lambda_val : user-selected regularization tuning parameter

    Returns  : 
    ------------------------------------------------------
    Precision matrix corresponding to lambda_val
    """ 
    import numpy as np
    import warnings
    
    d = icovlist[:,:,0].shape[0]
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
            icov[:,i]=icovlist[:,:,seq[i]+1][:,i]
            
    icov = (icov + icov.T)/2.0
    
    if (status == 1):
        warnings.warn("Some columns do not reach the required lambda.\n You may want to increase lambda_min or use a large nlambda. \n")
    
    del temp_lambda, seq, d#, threshold
    
    return icov
    
    
def fastclime_select(x,lambdamtx,icovlist,metric="stars",rep_num=20,
                    stars_thresh=0.1,stars_subsample_ratio=None):
    """
    Computes optimal regularization tuning parameter, 
    lambda using AIC or BIC metric or using stability approach
    to regularization selection (stars)

    Parameters:
    ------------------------------------------------------ 
    x         : data matrix
    
    lambdamtx : A nlambda x d array containing the regularization path 
                of tuning parameters for each column of the estimated 
                precision matrix
    
    icovlist  : A 3-D array with nlambda d x d precision matrices
               estimated from fastclime
               
    metric    : selection criterion. AIC, BIC and stars are available.
                When n < d, the degrees of freedom for AIC and BIC are 
                adjusted based on the number of non-zero elements in 
                the estimate precision matrix.
    
    rep_num   : The number of subsamplings. The default value is 20. 
                Only applicable when metric = "stars".
    
    stars_thresh : The variability threshold in stars. The default 
                   value is 0.1. Only applicable when metric = "stars".
    
    stars_subsample_ratio : The subsampling ratio. The default 
                            value is 10*sqrt(n)/n when n>144 and 0.8 
                            when n<=144, where n is the sample size. 
                            Only applicable when metric = "stars"

    Returns   : 
    ------------------------------------------------------
    optimal lambda parameter and corresponding precision
    matrix
    
    References: 1) H. Pang, et al. (2013) The fastclime Package for Linear 
                Programming and Large-Scale Precision Matrix Estimation in R
    
                2) H. Liu, et al. (2010) Stability Approach to Regularization
                Selection (StARS) for High Dimensional Graphical Models
    
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
                m = np.sum(np.absolute(icovlist[:,:,1])>1e-5,dtype=int)
                df = d + m*(m-1.0)/2.0
            else 
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
            else 
                df = d
                
            BIC[i]=-2.0*loglik(SigmaInput,icovlist[:,:,i]) + df*np.log(n)
        
        opt_index = np.where(BIC[2:]==min(BIC[2:][np.where(BIC[2:]!=-np.inf)]))[0]+2
        opt_lambda = np.max(lambdamtx[opt_index,:])

    if (metric=="stars"):

        if (stars_subsample_ratio is None):
            stars_subsample_ratio = [10.0*np.sqrt(n)/n,0.8][n<=144]

        merge = np.zeros((d,d,nl),dtype=float)

        print "Conducting subsampling...in progress. \n"
        for i in range(rep_num):
            rows = np.floor(float(n)*stars_subsample_ratio)
            rand_sample = np.random.permutation(x)[:rows,:]

            tmp = fastclime_main(rand_sample)[5]

            for i in range(nl):
                merge[:,:,i]+=tmp[:,:,i]

            del rand_sample, tmp
        print "Conducting subsampling...done. \n"

        variability = np.empty((nl,),dtype=float)
        for i in range(nl):
            merge[:,:,i]/=float(rep_num)
            variability[i] = 4.0*np.sum(merge[:,:,i].dot(1.0-merge[:,:,i]))/(d*(d-1.0))

        opt_index = max(np.where(variability[variability>=float(stars_thresh)] == max(variability))[0]-1,1)
        opt_lambda = np.max(lambdamtx[opt_index,:])
        
    opt_icov = fastclime_lambda(lambdamtx,icovlist,opt_lambda)
        
    return opt_lambda, opt_icov

        
def fastclime_main(x,lambda_min=0.1,nlambda=50):
    """
    Main function for CLIME estimation of the precision
    matrix

    Parameters:
    ---------------------------------------------------
    x          :  There are 2 options: (1) x is an n by d data matrix 
                  (2) a d by d sample covariance matrix. The program 
                  automatically identifies the input matrix by checking the
                  symmetry.

    lambda_min :  precision matrix (estimate or ground truth)
    
    nlambda    :  maximum path length

    Returns:
    ---------------------------------------------------
    x         :
    
    cov_input : Indicator for sample covariance
    
    Sigmahat  : empirical covariance matrix
    
    maxnlambda: The length of the path. If the program finds 
                lambda.min in less than nlambda iterations for 
                all columns, then the acutal maximum lenth for 
                all columns will be returned. Otherwise it equals 
                nlambda.
    
    lambdamtx : A nlambda x d array containing the regularization path 
                of tuning parameters for each column of the estimated 
                precision matrix
    
    icovlist  : A 3-D array with nlambda d x d precision matrices
               estimated from fastclime
    
    References: 1) T. Cai, et al. (2011) A constrained l1 minimization 
                approach to sparse precision matrix estimation.
     
                2) H. Pang, et al. (2013) The fastclime Package for Linear 
                Programming and Large-Scale Precision Matrix Estimation in R
    """
    import numpy as np
    import parametric as param
    
    if (isinstance(x,np.ndarray)==False):
        raise ValueError("Input must be ndarray.")
        
    cov_input = 1
    SigmaInput = x.copy()
    
    #Check if matrix is symmetric
    #If not, create scaled covariance matrix
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
   
    #Symmetrize output precision matrices 
    for i in range(maxnlambda):
        #icovlist[:,:,i] = symmetrize(iicov[:,i].reshape((d,d)).T,"min")
        icovlist[:,:,i] = iicov[:,i].reshape((d,d)).T
    
    return x, cov_input, Sigmahat, maxnlambda, lambdamtx, icovlist
    