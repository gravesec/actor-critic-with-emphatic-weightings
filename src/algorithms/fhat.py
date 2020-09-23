import numpy as np


class LinearFHat:

    def __init__(self, num_features, alpha, normalize=False):
        self.num_features = num_features
        self.alpha = alpha
        self.normalize = normalize
        self.f = np.zeros(self.num_features)

    def learn(self, x_t, gamma_t, x_tm1, rho_tm1, i_t):
        f_hat = self.estimate(x_tm1)
        if self.normalize:
            target = (1 - gamma_t) * i_t + gamma_t * rho_tm1 * f_hat
        else:
            target = i_t + gamma_t * rho_tm1 * f_hat
        delta_t = target - self.estimate(x_t)
        self.f += self.alpha * delta_t * x_t

    def estimate(self, x):
        return self.f.dot(x)



class BinaryFHat:

    def __init__(self, num_features, alpha, normalize=False):
        self.num_features = num_features
        self.alpha = alpha
        self.normalize = normalize
        self.f = np.zeros(self.num_features)

    def learn(self, indices_t, gamma_t, indices_tm1, rho_tm1, i_t):
        f_hat = self.estimate(indices_tm1)
        if self.normalize:
            target = (1 - gamma_t) * i_t + gamma_t * rho_tm1 * f_hat
        else:
            target = i_t + gamma_t * rho_tm1 * f_hat
        delta_t = target - self.estimate(indices_t)
        self.f[indices_t] += self.alpha * delta_t

    def estimate(self, indices):
        return self.f[indices].sum()
