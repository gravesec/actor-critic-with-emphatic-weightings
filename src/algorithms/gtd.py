import numpy as np

class GTD:

    def __init__(self, num_features, alpha_v, alpha_w, lamda):
        self.num_features = num_features

        self.alpha_v = alpha_v
        self.alpha_w = alpha_w
        self.lamda = lamda

        self.e = np.zeros((self.num_features))
        self.v = np.zeros((self.num_features))
        self.w = np.zeros((self.num_features))

    def learn(self, delta_t, x_t, gamma_t, x_tp1, gamma_tp1, rho_t):
        self.e = rho_t * (gamma_t * self.lamda * self.e + x_t)
        self.v += self.alpha_v * (delta_t * self.e - gamma_tp1 * (1 - self.lamda) * self.e.dot(self.w) * x_tp1)
        self.w += self.alpha_w * (delta_t * self.e - x_t.dot(self.w) * x_t)

    def estimate(self, x):
        return self.v.dot(x)
