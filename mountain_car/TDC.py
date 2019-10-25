import numpy as np


class TDC:

    def __init__(self, num_features, alpha_c, lambda_c):
        self.alpha_c = alpha_c
        self.alpha_w = alpha_c / 10.
        self.lambda_c = lambda_c
        self.e = np.zeros(num_features)
        self.v = np.zeros(num_features)
        self.w = np.zeros(num_features)

    def learn(self, delta_t, indices_t, gamma_t, indices_tp1, gamma_tp1, rho_t):
        self.e *= rho_t * gamma_t * self.lambda_c
        self.e[indices_t] += rho_t
        self.v += self.alpha_c * delta_t * self.e
        self.v[indices_tp1] -= self.alpha_c * gamma_tp1 * (1 - self.lambda_c) * self.e.dot(self.w)
        self.w[indices_t] -= self.alpha_w * self.w[indices_t].sum()
        self.w += self.alpha_w * delta_t * self.e

    def estimate(self, indices):
        return self.v[indices].sum()


if __name__ == '__main__':
    # Tests:
