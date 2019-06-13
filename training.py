import numpy as np
import cvxpy as cvx
from warnings import warn
import matplotlib.pyplot as plt
import sls


## Training the error model

def fit_error_model(dataset, C, P, alpha, norm=np.inf, verbose=False): 
    # solve for optimal perception given data
    if norm == np.inf: norm = 'inf'
    assert alpha >= 0. and alpha <= 1.
    
    w_c = cvx.Variable()
    w_eta = cvx.Variable()
    cons = [w_c >= 0., w_eta >= 0]

    xds, zds = dataset
    x_pred = zds.dot(P.T)
    for t in range(len(xds)):
        z = cvx.norm(C.dot(xds[t]) - x_pred[t], norm) - w_c * cvx.norm(xds[t], norm) - w_eta
        cons.append(z <= 0.)
    obj = alpha * w_c  + (1-alpha) * w_eta
        
    problem = cvx.Problem(cvx.Minimize(obj), cons)
    problem.solve('MOSEK', verbose=verbose)
    if verbose: 
        print('problem status:', problem.status)
    return obj.value, (w_c.value, w_eta.value)


def learn_robust_perception(dataset, C, alpha, reg, norm=np.inf, verbose=False): 
    # solve for optimal perception given data
    if norm == np.inf: norm = 'inf'
    assert alpha >= 0. and alpha <= 1.
    
    P = cvx.Variable((C.shape[0], zds.shape[1]))
    w_c = cvx.Variable()
    w_eta = cvx.Variable()

    xds, zds = dataset
    x_pred = zds * P.T
    constr = [w_c >= 0., w_eta >= 0]
    for t in range(len(xds)):
        z = cvx.norm(C.dot(xds[t]) - x_pred[t], norm) - w_c * cvx.norm(xds[t], norm) - w_eta
        constr.append(z <= 0.)
    obj = alpha * w_c  + (1-alpha) * w_eta + reg * cvx.norm(P, norm)

    problem = cvx.Problem(cvx.Minimize(obj), cons)
    problem.solve('MOSEK', verbose=verbose)
    if verbose: 
        print('problem status:', problem.status)
    
    return P.value, (w_c.value, w_eta.value)


def train_P(dataset, C, alpha, reg, robust, norm=np.inf):
    x, z = dataset
    print("alpha: {}, reg: {}".format(alpha, reg))
    if robust:
        P, err_tr = learn_robust_perception_weighted(z, x, C, alpha, reg, norm=norm, verbose=False)
    else: # use ridge regression
        ridge = linear_model.Ridge(alpha=reg)
        ridge.fit(z, x.dot(C.T))
        P = ridge.coef_
        err_tr = fit_error_model(dataset, C, P, alpha)[1]
    return P, err_tr


def compute_alpha(xval, norm=np.inf):
    avg_xnorm = np.linalg.norm(xval, axis=1, ord=norm).mean()
    return 1/(1+1/avg_xnorm)


def find_best_reg(trainset, testset, alpha, lower, upper, max_iters, tolerance, norm=np.inf):
    ## using golden section search to find best regularization
    tradeoff = np.mean(np.linalg.norm(xtrain, axis=1, ord=norm))
    alpha = 1 / (1/tradeoff + 1) # alpha = 1 puts all weight on eps_C

    debug_log = []
    def reg_gs_helper(reg):
        val, errs, pnorm = evaluate_reg(trainset, testset, alpha, reg, norm=norm)
        debug_log.append((errs, pnorm))
        return val

    best_reg, regs, costs = golden_section_search(reg_gs_helper, lower, upper, max_iters, tolerance)
    print("alpha: {}, best_reg: {}".format(alpha, best_reg))
    return best_reg
 
    
def evaluate_reg(trainset, testset, C, alpha, reg, norm=np.inf):
    # cross validation to find best regularization for learning P
    xtrain, ztrain = trainset
    xtest, ztest = testset
    errs_tr, P, status = learn_robust_perception_weighted(ztrain, xtrain, C, alpha,
                                                          reg=reg, verbose=False, norm=norm)
    print("errs train: {}".format(errs_tr))
    errs_test, _, status = learn_robust_perception_weighted(ztest, xtest, C, alpha, P=P,
                                                          reg=reg, verbose=False, norm=norm)
    Pnorm = np.linalg.norm(P, ord=norm)
    print("errs test: {}, ||P||: {}".format(errs_test, Pnorm))
    eps_C, eps_eta = errs_test
    return eps_C * alpha + eps_eta * (1-alpha), errs_test, Pnorm
       
    
def golden_section_search(fnc, lower, upper, max_iters, tolerance):
    gr = (np.sqrt(5) + 1) / 2 

    c = upper - (upper - lower) / gr
    d = lower + (upper - lower) / gr 
    locs = []
    vals = []
    for i in range(max_iters):
        if abs(c - d) <= tolerance:
            break
        cval = fnc(c); locs.append(c); vals.append(cval)
        dval = fnc(d); locs.append(d); vals.append(dval)
        print("f({})={} and f({})={}".format(c, cval, d, dval))
        if cval < dval:
            upper = d
        else:
            lower = c
        c = upper - (upper - lower) / gr
        d = lower + (upper - lower) / gr
    return (upper + lower)/2, locs, vals


