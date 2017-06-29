from scipy.optimize import minimize
import numpy as np
def markowitz_optimization():
    n_sec = 5
    return_series = np.random.standard_normal((n_sec,10))
    ret =np.mean(return_series, axis=1)
    cov = np.cov(return_series)
    weights = np.random.rand(n_sec,1)
    
    def mean_variance(w):
        return-(np.dot(ret.T,w) -np.dot(w.T,np.dot(cov,w)) -np.sqrt(np.dot(w.T,w)))
    
    def c1(w):
        return(w)
    
    def c2(w):
        return(np.sum(w)-1)
    
    def c3(w):
        return(0.3-w)
    
    constraints =({'type':'ineq','fun':c1},
                 {'type':'ineq','fun':c2},
                 {'type':'ineq','fun':c3})
    
    results = minimize(mean_variance, weights, constraints = constraints)
    return results

import cvxopt as opt
from cvxopt import blas, solvers

def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)

def random_portfolio(returns):
    ''' 
    Returns the mean and standard deviation of returns for a random portfolio
    '''

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))
    
    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)
    
    # This recursion reduces outliers to keep plots pretty
    if sigma > 2:
        return random_portfolio(returns)
    return mu, sigma

def optimal_portfolio(returns):
    n = len(returns)
    returns = np.asmatrix(returns)
    
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), portfolios, returns, risks

def random_portfolio_sim(returns, similarity):
    ''' 
    Returns the mean and standard deviation of returns for a random portfolio
    '''

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(similarity)
    
    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)
    
    # This recursion reduces outliers to keep plots pretty
    if sigma > 2:
        return random_portfolio(returns)
    return mu, sigma

def optimal_portfolio_sim(returns,similarity):
    n = len(returns)
    returns = np.asmatrix(returns)
    
    N = 100
    mus = [100**(10.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(similarity)
    pbar = opt.matrix(np.mean(returns, axis=1))
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), portfolios, returns, risks

from scipy import spatial
def calc_similarity(data,encoder):
    size= len(data)-1
    similarity = np.ndarray((size,size))
    for i in range(size):
        dat1=encoder.predict(data[i+1])
        for j in range(size):
            dat2=encoder.predict(data[j+1])                            
            similarity[i,j] = np.mean([1 - spatial.distance.cosine(dat1[k,:], \
                dat2[k,:]) for k in range(124)])
    return similarity

def test_portfolio(returns, weights):
    ''' 
    Returns the mean and standard deviation of returns for a random portfolio
    '''
    w=[]
    p = np.asmatrix(np.mean(returns, axis=1))
    C = np.asmatrix(np.cov(returns))
    for i in range(len(weights)):
        w.append(np.array(weights[i]))
    w = np.asmatrix(np.array(w))
    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)
    return mu, sigma

def test_portfolio1(returns, weights):
    ''' 
    Returns the mean and standard deviation of returns for a random portfolio
    '''
    p = np.asmatrix(np.mean(returns, axis=1))
    C = np.asmatrix(np.cov(returns))
    w = np.asmatrix(np.array(weights))
    mu = np.dot(w.T, p.T)
    sigma = np.sqrt(np.dot(w.T, np.dot(C,w)))
    return mu, sigma