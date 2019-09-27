import numpy as np
from .tiles3 import tiles


# TODO: Implement action-value estimates instead of using TD error (which is higher variance due to the sampling of next states)
# TODO: Consider using Q-estimation instead of TD error (which has higher variance due to sampling next states)
# TODO: derive eligibility traces for actor?


class TileCoder:

    def __init__(self, min_values, max_values, num_tiles, num_tilings, num_features, bias_unit=False):
        self.num_tiles = np.array(num_tiles)
        self.scale_factor = self.num_tiles / (np.array(max_values) - np.array(min_values))
        self.num_tilings = num_tilings
        self.bias_unit = bias_unit
        self.num_features = num_features
        self.num_active_features = self.num_tilings + self.bias_unit

    def indices(self, observations):
        return np.array(tiles(int(self.num_features - self.bias_unit), self.num_tilings, list(np.array(observations) * self.scale_factor)), dtype=np.intp)

    def features(self, indices):
        features = np.zeros(self.num_features)
        features[indices] = 1.
        if self.bias_unit:
            features[self.num_features - 1] = 1.
        return features


class TOETD:
    '''
    True Online Emphatic Temporal Difference learning algorithm written by Ashique Rupam Mahmood.
    '''

    def __init__(self, n, I, alpha):
        self.ep = np.zeros(n)
        self.theta = np.zeros(n)
        self.prevtheta = np.zeros(n)
        self.H = 0.
        self.M = alpha*I
        self.prevI = I
        self.prevgm = 0
        self.prevlm = 0

    def learn(self, phi, phiPrime, R, rho, gm, lm, I, alpha):
        delta = R + gm * np.dot(self.theta, phiPrime) - np.dot(self.theta, phi)
        self.ep = rho * (self.prevgm * self.prevlm * self.ep + self.M * (1 - rho * self.prevgm * self.prevlm * np.dot(self.ep, phi)) * phi)
        Delta = delta * self.ep + np.dot(self.theta - self.prevtheta, phi) * (self.ep - rho * self.M * phi)
        self.prevtheta = self.theta.copy()
        self.theta += Delta
        self.H = rho * gm * (self.H + self.prevI)
        self.M = alpha * (I + (1 - lm) * self.H)
        self.prevgm = gm
        self.prevlm = lm
        self.prevI = I

    def estimate(self, phi):
        return np.dot(self.theta, phi)


class ACE:

    def __init__(self, num_actions, num_features):
        self.theta = np.zeros((num_actions, num_features))
        self.F = 0.
        self.rho_tm1 = 1.

        self.psi_s_a = np.zeros((num_actions, num_features))
        self.psi_s_b = np.zeros((num_actions, num_features))

    def pi(self, x_t):
        prefs = self.theta.dot(x_t)
        # Converts potential overflows of the largest probability into underflows of the lowest probability:
        prefs = prefs - prefs.max()
        exp_prefs = np.exp(prefs)
        return exp_prefs / np.sum(exp_prefs)

    def grad_log_pi(self, x_t, a_t):
        self.psi_s_a.fill(0.)
        self.psi_s_a[a_t] = x_t
        probs = self.pi(x_t).reshape(self.num_actions, 1) # reshape to enable broadcasting.
        self.psi_s_b[:] = x_t
        return self.psi_s_a - probs * self.psi_s_b

    def learn(self, gamma_t, i_t, eta_t, alpha_t, rho_t, delta_t, x_t, a_t):
        self.F = self.rho_tm1 * gamma_t * self.F + i_t
        M_t = (1 - eta_t) * i_t + eta_t * self.F

        self.theta += alpha_t * rho_t * M_t * delta_t * self.grad_log_pi(x_t, a_t)

        self.rho_tm1 = rho_t
