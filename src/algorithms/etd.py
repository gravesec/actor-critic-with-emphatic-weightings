import numpy as np

class ETD:

    def __init__(self, num_features, alpha, lamda):
        self.num_features = num_features

        self.alpha = alpha
        self.lamda = lamda

        self.i = 1. # Fixed interest
        self.e = np.zeros((self.num_features))
        self.v = np.zeros((self.num_features))
        self.F = 0.
        self.rho_tm1 = 1.

    def learn(self, delta_t, x_t, gamma_t, x_tp1, gamma_tp1, rho_t):
        self.F = self.rho_tm1 * gamma_t * self.F + self.i
        M = self.lamda * self.i + (1. - self.lamda) * self.F
        self.e = rho_t * (gamma_t * self.lamda * self.e + M * x_t)
        self.v += self.alpha * delta_t * self.e

    def estimate(self, x):
        return self.v.dot(x)
