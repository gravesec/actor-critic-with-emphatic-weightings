import numpy as np


class DeterministicPolicy:

    def __init__(self, u):
        self.num_features = u.shape[0]
        self.u = u

    def pi(self, x_t, a_t=None):
        mu = self.u.dot(x_t)
        if a_t is not None:
            return float(np.isclose(a_t,mu))
        return mu

    def pi_params(self, x_t):
        mu = self.u.dot(x_t)
        sig = 0.0
        return [mu, sig]

    def grad_pi(self, x_t):
        return x_t
