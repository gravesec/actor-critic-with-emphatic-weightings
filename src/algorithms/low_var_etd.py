import numpy as np


# TODO: The linear one's probably based on an old implementation
class LinearLowVarETD:

    def __init__(self, num_features, alpha, lambda_c):
        self.num_features = num_features

        self.alpha = alpha
        self.lambda_c = lambda_c

        self.e = np.zeros(self.num_features)
        self.v = np.zeros(self.num_features)

    def learn(self, delta_t, x_t, gamma_t, i_t, x_tp1, gamma_tp1, rho_t, F_t):
        M = self.lambda_c * i_t + (1. - self.lambda_c) * F_t
        self.e = rho_t * (gamma_t * self.lambda_c * self.e + M * x_t)
        self.v += self.alpha * delta_t * self.e

    def estimate(self, x):
        return self.v.dot(x)


class BinaryLowVarETD:

    def __init__(self, num_features, alpha_c, lambda_c):
        self.alpha_c = alpha_c
        self.lambda_c = lambda_c
        self.e = np.zeros(num_features)
        self.v = np.zeros(num_features)

    def learn(self, delta_t, indices_t, gamma_t, i_t, indices_tp1, gamma_tp1, rho_t, F_t, r_tp1=None, rho_tp1=None, a_t=None, a_tp1=None):
        M = self.lambda_c * i_t + (1. - self.lambda_c) * F_t
        self.e *= rho_t * gamma_t * self.lambda_c
        self.e[indices_t] += M * rho_t
        self.v += self.alpha_c * delta_t * self.e

    def estimate(self, indices, a=None):
        return self.v[indices].sum()
