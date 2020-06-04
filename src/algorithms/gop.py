import numpy as np


class BinaryGOP:
    """Doesn't work. For some reason increases value estimates for state-actions not taken?!"""

    def __init__(self, num_actions, num_features, alpha_v, alpha_w, lamda):
        self.num_actions = num_actions
        self.num_features = num_features
        self.alpha_v = alpha_v
        self.alpha_w = alpha_w
        self.lamda = lamda
        self.e = np.zeros((self.num_actions, self.num_features))
        self.e_w = np.zeros((self.num_actions, self.num_features))
        self.v = np.zeros((self.num_actions, self.num_features))
        self.w = np.zeros((self.num_actions, self.num_features))

    def learn(self, indices_t, a_t, gamma_t, rho_t, r_tp1, indices_tp1, gamma_tp1, pi_tp1):
        v_tp1 = pi_tp1.dot(self.estimate(indices_tp1))
        q_t = self.estimate(indices_t, a_t)
        delta_t = r_tp1 + gamma_tp1 * v_tp1 - q_t

        # Update eligibility trace:
        self.e *= gamma_t * self.lamda * rho_t
        self.e[a_t, indices_t] += 1

        # Update Dutch trace:
        e_w_dot_phi = self.e_w[a_t, indices_t].sum()
        self.e_w *= gamma_t * self.lamda * rho_t
        self.e_w[a_t, indices_t] += self.alpha_w * (1 - gamma_t * self.lamda * rho_t * e_w_dot_phi)

        # Compute this inner product before updating w, to use when updating v:
        w_T_e = np.ravel(self.w).dot(np.ravel(self.e))

        # Update w:
        w_dot_phi = self.w[a_t, indices_t].sum()
        self.w += delta_t * self.e_w
        self.w[a_t, indices_t] -= self.alpha_w * w_dot_phi

        # Update v:
        for a in range(self.num_actions):
            self.v[a, indices_tp1] -= self.alpha_v * w_T_e * gamma_tp1 * pi_tp1[a]
        self.v[a_t, indices_t] += self.alpha_v * w_T_e

    def estimate(self, indices, action=None):
        """Return value estimates for the given observation and action, or for all possible actions if 'action' is None."""
        return self.v[:, indices].sum(axis=1) if action is None else self.v[action, indices].sum()