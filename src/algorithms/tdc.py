import numpy as np


class LinearTDC:

    def __init__(self, num_features, alpha_v, alpha_w, lambda_c):
        self.alpha_v = alpha_v
        self.alpha_w = alpha_w
        self.lambda_c = lambda_c
        self.e = np.zeros(num_features)
        self.v = np.zeros(num_features)
        self.w = np.zeros(num_features)

    def learn(self, delta_t, x_t, gamma_t, x_tp1, gamma_tp1, rho_t):
        self.e = rho_t * (gamma_t * self.lambda_c * self.e + x_t)
        self.v += self.alpha_v * (delta_t * self.e - gamma_tp1 * (1 - self.lambda_c) * self.e.dot(self.w) * x_tp1)
        self.w += self.alpha_w * (delta_t * self.e - x_t.dot(self.w) * x_t)

    def estimate(self, x_t):
        return self.v.dot(x_t)


class BinaryTDC:

    def __init__(self, num_features, alpha_v, alpha_w, lambda_c, q_value_mode=False, num_action=None):
        self.alpha_v = alpha_v
        self.alpha_w = alpha_w
        self.lambda_c = lambda_c
        self.e = np.zeros(num_features)
        self.v = np.zeros(num_features)
        self.w = np.zeros(num_features)
        self.q_value_mode = q_value_mode
        if self.q_value_mode:
            self.num_action = num_action
            self.num_features = num_features
            temp_dim = self.num_features*self.num_action
            self.e_q = np.zeros(temp_dim)
            self.v_q = np.zeros(temp_dim)
            self.w_q = np.zeros(temp_dim)

    def learn(self, delta_t, indices_t, gamma_t, indices_tp1, gamma_tp1, rho_t, r_tp1=None, rho_tp1=None, a_t=None, a_tp1=None):
        if self.q_value_mode:
            delta_t = r_tp1 + gamma_tp1 * self.v[indices_tp1].sum() - self.v[indices_t].sum()
        self.e *= rho_t * gamma_t * self.lambda_c
        self.e[indices_t] += rho_t
        self.v += self.alpha_v * delta_t * self.e
        self.v[indices_tp1] -= self.alpha_v * gamma_tp1 * (1 - self.lambda_c) * self.e.dot(self.w)
        self.w[indices_t] -= self.alpha_w * self.w[indices_t].sum()
        self.w += self.alpha_w * delta_t * self.e

        if self.q_value_mode:
            indices_t_q = a_t * self.num_features + indices_t
            indices_tp1_q = a_tp1 * self.num_features + indices_tp1
            delta_t_q = r_tp1 + gamma_tp1 * rho_tp1 * self.v_q[indices_tp1_q].sum() - self.v_q[indices_t_q].sum()
            self.e_q *= rho_t * gamma_t * self.lambda_c
            self.e_q[indices_t_q] += 1
            self.v_q += self.alpha_v * delta_t_q * self.e_q
            self.v_q[indices_tp1_q] -= self.alpha_v * gamma_tp1 * (1 - self.lambda_c) * self.e_q.dot(self.w_q)
            self.w_q[indices_t_q] -= self.alpha_w * self.w_q[indices_t_q].sum()
            self.w_q += self.alpha_w * delta_t_q * self.e_q

    def estimate(self, indices, q_mode=False, a=None):
        if q_mode:
            indices_new = a * self.num_features + indices
            return self.v_q[indices_new].sum()
        else:
            return self.v[indices].sum()


class BinaryTOTDC:

    def __init__(self, num_actions, num_features, alpha, beta, lamda):
        self.num_actions = num_actions
        self.num_features = num_features
        self.alpha = alpha
        self.beta = beta
        self.lamda = lamda
        self.e = np.zeros((num_actions, num_features))
        self.e_v = np.zeros((num_actions, num_features))
        self.e_w = np.zeros((num_actions, num_features))
        self.v = np.zeros((num_actions, num_features))
        self.w = np.zeros((num_actions, num_features))

    def learn(self, q_old, rho_tm1, delta_t, indices_t, a_t, gamma_t, rho_t, indices_tp1, a_tp1, gamma_tp1):
        q_t = self.estimate(indices_t, a_t)

        self.e *= rho_t * gamma_t * self.lamda
        self.e[a_t, indices_t] += rho_t

        e_v_dot_phi = self.e_v[a_t, indices_t].sum()
        self.e_v *= rho_t * gamma_t * self.lamda
        self.e_v[a_t, indices_t] += rho_t * self.alpha * (1 - rho_t * gamma_t * self.lamda * e_v_dot_phi)

        e_w_dot_phi = self.e_w[a_t, indices_t].sum()
        self.e_w *= rho_tm1 * gamma_t * self.lamda
        self.e_w[a_t, indices_t] += self.beta * (1 - rho_tm1 * gamma_t * self.lamda * e_w_dot_phi)

        w_dot_e = np.ravel(self.w).dot(np.ravel(self.e))
        self.v += delta_t * self.e_v + self.e_v * (q_t - q_old)
        self.v[a_t, indices_t] -= self.alpha * rho_t * (q_t - q_old)
        self.v[a_tp1, indices_tp1] -= self.alpha * gamma_tp1 * (1 - self.lamda) * w_dot_e

        self.w += rho_t * delta_t * self.e_w
        self.w[a_t, indices_t] -= self.beta * self.w[a_t, indices_t].sum()

    def estimate(self, indices, action=None):
        """Return value estimates for the given observation and action, or for all possible actions if 'action' is None."""
        return self.v[:, indices].sum(axis=1) if action is None else self.v[action, indices].sum()
