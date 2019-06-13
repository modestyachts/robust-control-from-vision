import numpy as np
import scipy.linalg


def optimal_k(A, B, R, P):
    return scipy.linalg.inv(B.T.dot(P).dot(B) + R).dot(B.T.dot(P).dot(A))

def lqr_inf_horizon(A, B, Q, R):
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = optimal_k(A, B, R, P)
    return K, P

def kalman_gain(C, V, S):
    return S.dot(C.T).dot(scipy.linalg.inv(C.dot(S).dot(C.T) + V))

def lqg_inf_horizon(A, C, W, V):
    S = scipy.linalg.solve_discrete_are(A.T, C.T, W, V)
    L = kalman_gain(C, V, S)
    return L, S

def lqg_sim(x0, T, A, B, C, Q, R, W, V, wx, wy):
    K = lqr_inf_horizon(A, B, Q, R)[0]
    L = lqg_inf_horizon(A, C, W, V)[0]
    ys = []
    xs = [x0]
    xhs = []
    us = []
    x_proj = x0 # np.zeros(x0.shape); cheating by starting it with true state
    for k in range(T):
        ys.append(C.dot(xs[k]) + wy[:,k])
        xhs.append(x_proj + L.dot(ys[k] - C.dot(x_proj)))
        us.append(-K.dot(xhs[k]))
        xs.append(A.dot(xs[k]) + B.dot(us[k]) + wx[:,k])
        x_proj = A.dot(xhs[k]) + B.dot(us[k])
    return np.array(xs).T, np.array(us).T, np.array(ys).T, np.array(xhs).T

def lqr_parameters(q, ndims=2):
    Q1 = np.array([[q**2, 0], [0, 0]])
    R1 = np.array([[1]])
    Q = scipy.linalg.block_diag(*[Q1]*ndims)
    R = scipy.linalg.block_diag(*[R1]*ndims)
    return Q, R
    