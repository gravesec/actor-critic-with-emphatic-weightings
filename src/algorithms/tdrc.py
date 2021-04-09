import numpy as np


class LinearTDRC:

    def __init__(self, num_features, alpha, lambda_c):
        self.num_features = num_features
        self.alpha_w = alpha
        self.alpha_v = self.alpha_w
        self.beta = 1.0
        self.lambda_c = lambda_c
        self.w = np.zeros(num_features)
        self.v = np.zeros(num_features)
        self.z = np.zeros(num_features)

    def learn(self, delta_t, x_t, gamma_t, x_tp1, gamma_tp1, rho_t):
        self.z = rho_t * (gamma_t * self.lambda_c * self.z + x_t)
        self.w += self.alpha_w * delta_t * self.z - self.alpha_w * gamma_tp1 * (1 - self.lambda_c) * self.z.dot(self.v) * x_tp1
        self.v += self.alpha_v * delta_t * self.z - self.alpha_v * self.beta * self.v - self.alpha_v * self.v.dot(x_t) * x_t

    def estimate(self, x_t):
        return self.w.dot(x_t)


class BinaryTDRC:

    def __init__(self, num_features, alpha , lambda_c):
        self.num_features = num_features
        self.alpha_w = alpha
        self.alpha_v = self.alpha_w
        self.beta = 1.0
        self.lambda_c = lambda_c
        self.w = np.zeros(num_features)
        self.v = np.zeros(num_features)
        self.z = np.zeros(num_features)

    def learn(self, delta_t, indices_t, gamma_t, indices_tp1, gamma_tp1, rho_t):
        self.z *= rho_t * gamma_t * self.lambda_c
        self.z[indices_t] += rho_t
        self.w += self.alpha_w * delta_t * self.z
        self.w[indices_tp1] -= self.alpha_w * gamma_tp1 * (1 - self.lambda_c) * self.z.dot(self.v)
        v_dot_x = self.v[indices_t].sum()
        self.v += ((self.alpha_v * delta_t * self.z) - (self.alpha_v * self.beta * self.v))
        self.v[indices_t] -= self.alpha_v * v_dot_x

    def estimate(self, indices):
        return self.w[indices].sum()
