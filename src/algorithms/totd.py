import numpy as np


class LinearTOTD:
    def __init__(self, num_features, alpha_w, lambda_e):
        self.alpha_w = alpha_w
        self.lambda_e = lambda_e
        self.e = np.zeros(num_features)
        self.w = np.zeros(num_features)
        self.v_old = 0

    def learn(self, delta_t, v_t, x_t, gamma_t, v_tp1):
        self.e = gamma_t * self.lambda_e * self.e + (1 - self.alpha_w * gamma_t * self.lambda_e * self.e.dot(x_t)) * x_t
        self.w = self.w + self.alpha_w * (delta_t + v_t - self.v_old) * self.e - self.alpha_w * (v_t - self.v_old) * x_t
        self.v_old = v_tp1

    def estimate(self, x_t):
        return self.w.dot(x_t)


class BinaryTOTD:
    def __init__(self, num_features, alpha_w, lambda_e):
        self.alpha_w = alpha_w
        self.lambda_e = lambda_e
        self.e = np.zeros(num_features)
        self.w = np.zeros(num_features)
        self.v_old = 0

    def learn(self, delta_t, v_t, indices_t, gamma_t, v_tp1):
        e_dot_x = self.e[indices_t].sum()
        self.e *= gamma_t * self.lambda_e
        self.e[indices_t] += 1 - self.alpha_w * gamma_t * self.lambda_e * e_dot_x
        self.w += self.alpha_w * (delta_t + v_t - self.v_old) * self.e
        self.w[indices_t] -= self.alpha_w * (v_t - self.v_old)
        self.v_old = v_tp1

    def estimate(self, indices_t):
        return self.w[indices_t].sum()
