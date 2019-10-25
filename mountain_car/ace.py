import numpy as np
from mountain_car.tiles3 import tiles


# TODO: Implement action-value estimates instead of using TD error (which is higher variance due to the sampling of next states)
# TODO: derive eligibility traces for actor?


class TileCoder:
    """
    Linear function approximator.
    """
    def __init__(self, min_values, max_values, num_tiles, num_tilings, num_features, bias_unit=False):
        self.num_tiles = np.array(num_tiles)
        self.scale_factor = self.num_tiles / (np.array(max_values) - np.array(min_values))
        self.num_tilings = num_tilings
        self.bias_unit = bias_unit
        self.num_features = num_features
        self.num_active_features = self.num_tilings + self.bias_unit

    def indices(self, observations):
        indices = tiles(int(self.num_features - self.bias_unit), self.num_tilings, list(np.array(observations) * self.scale_factor))
        if self.bias_unit:
            indices.append(self.num_features - 1)  # Add bias unit.
        return np.array(indices, dtype=np.intp)

    def features(self, indices):
        features = np.zeros(self.num_features)
        features[indices] = 1.
        return features


class ACE:

    def __init__(self, num_actions, num_features):
        self.theta = np.zeros((num_actions, num_features))
        self.F = 0.
        self.rho_tm1 = 1.
        self.psi_s_a = np.zeros((num_actions, num_features))
        self.psi_s_b = np.zeros((num_actions, num_features))

    def pi(self, indices):
        logits = self.theta[:, indices].sum(axis=1)
        # Converts potential overflows of the largest probability into underflows of the lowest probability:
        logits = logits - logits.max()
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)

    def grad_log_pi(self, indices, a_t):
        self.psi_s_a.fill(0.)
        self.psi_s_a[a_t, indices] = 1.
        pi = np.expand_dims(self.pi(indices), axis=1)  # Add dimension to enable broadcasting.
        self.psi_s_b.fill(0.)
        self.psi_s_b[:, indices] = 1.
        return self.psi_s_a - pi * self.psi_s_b

    def learn(self, gamma_t, i_t, eta_t, alpha_t, rho_t, delta_t, indices_t, a_t):
        self.F = self.rho_tm1 * gamma_t * self.F + i_t
        M_t = (1 - eta_t) * i_t + eta_t * self.F
        self.theta += alpha_t * rho_t * M_t * delta_t * self.grad_log_pi(indices_t, a_t)
        self.rho_tm1 = rho_t


class GTD:

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


class TOETD:
    """
    True Online Emphatic Temporal Difference learning algorithm based on code by Ashique Rupam Mahmood.
    """
    def __init__(self, n, I, alpha):
        self.ep = np.zeros(n)
        self.theta = np.zeros(n)
        self.prevtheta = np.zeros(n)

        self.H = 0.
        self.M = alpha*I
        self.prevI = I
        self.prevgm = 0
        self.prevlm = 0

    def learn(self, phi, delta, rho, gm, lm, I, alpha):
        self.ep = rho*(self.prevgm*self.prevlm*self.ep + self.M*(1-rho*self.prevgm*self.prevlm*np.dot(self.ep, phi))*phi)
        Delta = delta*self.ep + np.dot(self.theta - self.prevtheta, phi)*(self.ep - rho*self.M*phi)
        self.prevtheta = self.theta.copy()
        self.theta += Delta
        self.H = rho*gm*(self.H + self.prevI)
        self.M = alpha*(I + (1-lm)*self.H)
        self.prevgm = gm
        self.prevlm = lm
        self.prevI = I

    def estimate(self, indices):
        return self.theta[indices].sum()
