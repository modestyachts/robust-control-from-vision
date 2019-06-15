import numpy as np
import scipy.linalg
import cvxpy as cvx


#### NORMS #####

def h2_norm(coefs):
    return np.sqrt(sum([np.linalg.norm(coef, 'fro')**2 for coef in coefs]))

def l1_norm(coefs, ret_idx=False):
    row_l1s = sum([np.linalg.norm(coef, ord=1, axis=1) for coef in coefs])
    ret = (np.max(row_l1s), np.argmax(row_l1s)) if ret_idx else np.max(row_l1s)
    return ret


def h2_norm_cvx(coefs):
    return sum([cvx.norm(coef, 'fro')**2 for coef in coefs])


def l1_norm_cvx(coefs):
    coef_row_11_norms = sum([cvx.norm(coef, p=1, axis=1) for coef in coefs])
    return cvx.max(coef_row_11_norms)


#### CONSTRAINTS ####

def realizability_constraints(sys_coefs, A, B, C):
    Phi_xx, Phi_ux, Phi_xy, Phi_uy = sys_coefs
    T = len(Phi_xx) - 2 # coefs for components 0 to T+1
    constr = [Phi_xx[0] == 0, Phi_ux[0] == 0, Phi_xy[0] == 0] # strict properness
    constr += [Phi_xx[1] == np.eye(A.shape[0])]

    for t in range(1, T+1):
        constr += [Phi_xx[t+1] == A * Phi_xx[t] + B * Phi_ux[t],
                   Phi_xx[t+1] == Phi_xx[t] * A + Phi_xy[t] * C]
    for t in range(T+1):
        constr += [Phi_xy[t+1] == A * Phi_xy[t] + B * Phi_uy[t],
                   Phi_ux[t+1] == Phi_ux[t] * A + Phi_uy[t] * C]

    # finite horizon for things that affect cost
    constr += [Phi_xx[T+1] == 0, Phi_ux[T+1] == 0, 
               Phi_xy[T+1] == 0, Phi_uy[T+1] == 0]
    return constr

    
def hinf_constraint(coefs, gamma, l1_trans=False):
    # constrain the H_inf norm of the response [coefs] <= gamma
    # using the equivalent SDP formulation from equation 4.42 of Dumitrescu
    # returns a list a cvx expression constraints
    T = len(coefs) - 1
    m, n = coefs[0].shape # m >= n, enforcing constraint on transpose system
    
    if l1_trans or T > 70:
        print('using l1 of transpose instead of h inf norm')
        ht_l1_norm = cvx.max(sum([cvx.norm(coef.T, p=1, axis=1) for coef in coefs]))
        return [ht_l1_norm <= gamma], ht_l1_norm
    
    constr = []
    hcoefs = cvx.bmat([[coef.T] for coef in coefs])
    Q = cvx.Variable((n*(T+1), n*(T+1)), PSD=True)
    # constraint 4.39, part 1
    # k == 0, diagonal of Q sums to gamma**2 I_n
    constr.append(
        sum([Q[n*t:n*(t+1), n*t:n*(t+1)] for t in range(T+1)]) == gamma**2*np.eye(n))
    # k > 0, k-th off-diagonal of Q sums to 0
    for k in range(1, T+1):
        constr.append(
            sum([Q[n*t:n*(t+1), n*(t+k):n*(t+1+k)] for t in range(T+1-k)]) == np.zeros((n, n)))
    # constraint 4.39, constrain to be PSD
    constr.append(
        cvx.bmat([[Q, hcoefs], [hcoefs.T, np.eye(m)]]) == cvx.Variable((n*(T+1)+m, n*(T+1)+m), PSD=True))
    return constr, (hcoefs, Q)


def l1_constraint(coefs, gamma):
    # constrain the l1 norm of response [coefs] <= gamma
    # returns a list of cvx expression constraints
    coefs_l1 = l1_norm_cvx(coefs)
    return [coefs_l1 <= gamma]


#### SYNTHESIS ####

def synthesize_perception_controller(A, B, C, Q, R, T, norm, wx=1, wy=1,
                                     Ld=None, wc=None, solver='MOSEK', verbose=False):
    srQ, srR = scipy.linalg.sqrtm(Q), scipy.linalg.sqrtm(R)
    nx, nu = B.shape
    ny = C.shape[0]
    
    alpha = np.sqrt(2) if norm.upper() == 'H2' else 1
    if Ld is None or wc is None: # nominal controller, no robustness constraint
        Phi_xy_upper = None
    else: # robustness constraint on norm of Phi_xy
        Phi_xy_upper = 1/(wc + alpha * (Ld + wc))
    
    # 0th to (T+1)-th spectral elements of transfer matrices as defined in SLS paper
    Phi_xx = [cvx.Variable((nx, nx)) for t in range(T+2)]
    Phi_ux = [cvx.Variable((nu, nx)) for t in range(T+2)]
    Phi_xy = [cvx.Variable((nx, ny)) for t in range(T+2)]
    Phi_uy = [cvx.Variable((nu, ny)) for t in range(T+2)]

    out_coefs = [cvx.bmat([[srQ* Phi_xx[t]* wx, srQ* Phi_xy[t]* wy],
                           [srR* Phi_ux[t]* wx, srR* Phi_uy[t]* wy]]) for t in range(T+1)]
    constr = []
    if norm.upper() == 'H2':
        out_norm = h2_norm_cvx(out_coefs)
        
        if Phi_xy_upper is not None: # constrain L2 -> L2 (H_inf) norm of Phi_xy
            hinf_constrs, aux = hinf_constraint(Phi_xy[:-1], Phi_xy_upper)
            constr = hinf_constrs
        
    elif norm.upper() == 'L1' or norm.upper() == 'INF->INF':
        out_norm = l1_norm_cvx(out_coefs)
        
        if Phi_xy_upper is not None: # constrain L_inf -> L_inf (L1) norm of Phi_xy
            constr = l1_constraint(Phi_xy, Phi_xy_upper)
        
    else:
        raise NotImplementedError("norm not supported")
            
    constr.extend(realizability_constraints((Phi_xx, Phi_ux, Phi_xy, Phi_uy), A, B, C))

    problem = cvx.Problem(cvx.Minimize(out_norm), constr)
    problem.solve(solver=solver, verbose=verbose)
    return ([np.asarray(list(map(lambda e: e.value, X))) for X in [Phi_xx, Phi_ux, Phi_xy, Phi_uy]],
            (out_norm.value, problem.status))


#### SIMULATE CLOSED LOOP ####


def simulate_output_feedback_loop(x0, H, A, B, C, Phi_ux, Phi_uy, 
                                  obs_fnc, proc_fnc, ref=None):
    xs = [x0]; ys = []; es = []; zs = []; us = [];
    if ref is None:
        ref = [np.zeros(C.shape[0]) for _ in range(H)]
    Hmax = len(Phi_ux)
    for k in range(H-1):
        y, z = obs_fnc(xs[k])
        ys.append(y); zs.append(z)
        es.append(y - C.dot(ref[k]))
        u = np.zeros(B.shape[1])
        us.append(u)
        for j in range(min(Hmax, k+1)):
            u += -Phi_ux[j].dot(B.dot(us[k-j])) + Phi_uy[j].dot(es[k-j])
        xs.append(proc_fnc(xs[k], us[k]))
    return np.array(xs), np.array(us), np.array(ys), zs


#### UPPER BOUND ON COST ####


def control_upper_bound(SLS_controller, srQ, srR, eps_eta, eps_C, H=None, norm='l1'):
    assert norm == 'l1', "Only Implemented l1 norm so far"
    if H is None: H = np.eye(srQ.shape[0])
    Rs, Ms, Ns, Ls = SLS_controller
    xnorm,_ = l1_norm([srQ.dot(R.dot(H)+N*eps_eta) for (R,N) in zip(Rs, Ns)])
    unorm,_ = l1_norm([srR.dot(M.dot(H)+L*eps_eta) for (M,L) in zip(Ms, Ls)])
    xetanorm,_ = l1_norm(Ns)
    nominalcost = max(xnorm, unorm)
    xetacost,_ = l1_norm([srQ.dot(N) for N in Ns])
    uetacost,_ = l1_norm([srR.dot(L) for L in Ls])
    etanorm = max(xetacost, uetacost)
    if eps_C * xetanorm >= 1:
        print('eps_C * xetanorm=', eps_C * xetanorm)
        return np.inf
    return nominalcost + eps_C * xnorm / (1 - eps_C * xetanorm) * etanorm


