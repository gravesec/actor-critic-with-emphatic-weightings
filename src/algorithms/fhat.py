import numpy as np


class FHat:

    def __init__(self, num_features, alpha):
        self.num_features = num_features

        self.alpha = alpha
        self.lamda = lamda

        self.f = np.zeros(self.num_features)

    def learn(self, x_t, gamma_t, x_tm1, rho_tm1, i_t):
        target = i_t + gamma_t * rho_tm1 * self.estimate(x_tm1)
        delta_t = self.estimate(x_t) - target
        self.f += self.alpha * delta_t * x_t

        ## Leaving ETD stuff here for reference
        # self.F = self.rho_tm1 * gamma_t * self.F + self.i
        # M = self.lamda * self.i + (1. - self.lamda) * self.F
        # self.e = rho_t * (gamma_t * self.lamda * self.e + M * x_t)
        # self.v += self.alpha * delta_t * self.e
    def estimate(self, x):
        return self.f.dot(x)



class BinaryFHat:

    def __init__(self, num_features, alpha):
        self.num_features = num_features

        self.alpha = alpha
        self.lamda = lamda

        self.f = np.zeros(self.num_features)

    def learn(self, indices_t, gamma_t, indices_tm1, rho_tm1, i_t):
        target = i_t + gamma_t * rho_tm1 * self.f[indices_tm1].sum()
        delta_t = self.f[indices_t].sum() - target
        self.f[indices_t] += self.alpha * delta_t

    def estimate(self, x):
        return self.f[indices].sum()
