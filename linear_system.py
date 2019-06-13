import numpy as np
import scipy.linalg


def double_integrator_dynamics(dt, ndims=2, q=None):
    A1 = np.array([[1, dt],[0, 1]])
    B1 = np.array([[0], [dt]])
    A = scipy.linalg.block_diag(*[A1]*ndims)
    B = scipy.linalg.block_diag(*[B1]*ndims)
    C1 = np.array([[1, 0]])
    C = scipy.linalg.block_diag(*[C1]*ndims)
    return A, B, C


class LinearDynamics(object):

    def __init__(self, A, B, process_noise=None):
        self.A = A
        self.B = B
        self.process_noise = process_noise

        self.nx, self.nu = B.shape
        self.reset()

    def reset(self):
        self.t = self.t0 = 0
        self.xs = []
        self.us = []

    def init_trajectory(self, x0, t0=0):
        # t0 is the time to start the system
        self.xs = [np.asarray(x0)]
        self.us = []
        self.t0 = t0
        self.t = 0
    
    def forward_step(self, u):
        t = self.t
        self.t = t+1

        x = self.xs[t]
        if t >= self.t0:
            u = np.asarray(u)
        else:
            u = np.zeros(self.nu)
        xnext = self.A.dot(x) + self.B.dot(u)
        if self.process_noise is not None:
            xnext += self.process_noise()

        self.us.append(u)
        self.xs.append(xnext)
        return xnext
