import numpy as np


# TODO: The linear one's probably based on an old implementation
class LinearLowVarETD:

    def __init__(self, num_features, alpha, lamda, fhat):
        self.num_features = num_features

        self.alpha = alpha
        self.lamda = lamda

        self.fhat = fhat

        self.i = 1. # Fixed interest
        self.e = np.zeros(self.num_features)
        self.v = np.zeros(self.num_features)

    def learn(self, delta_t, x_t, gamma_t, x_tp1, gamma_tp1, rho_t):
        self.F = self.fhat.estimate(x_t)
        M = self.lamda * self.i + (1. - self.lamda) * self.F
        self.e = rho_t * (gamma_t * self.lamda * self.e + M * x_t)
        self.v += self.alpha * delta_t * self.e

    def estimate(self, x):
        return self.v.dot(x)


class BinaryLowVarETD:

    def __init__(self, num_features, alpha_c, lambda_c, fhat, q_value_mode=False, num_action=None):
        self.alpha_v = alpha_c
        self.lamda_c = lamda_c
        self.e = np.zeros(num_features)
        self.v = np.zeros(num_features)

        self.fhat = fhat

        self.q_value_mode = q_value_mode
        if self.q_value_mode:
            self.num_action = num_action
            self.num_features = num_features
            temp_dim = self.num_features*self.num_action
            self.e_q = np.zeros(temp_dim)
            self.v_q = np.zeros(temp_dim)

    def learn(self, delta_t, indices_t, gamma_t, indices_tp1, gamma_tp1, rho_t, r_tp1=None, rho_tp1=None, a_t=None, a_tp1=None):
        if self.q_value_mode:
            delta_t = r_tp1 + gamma_tp1 * self.v[indices_tp1].sum() - self.v[indices_t].sum()

        self.F = self.fhat.estimate(x_t)
        M = self.lamda_c * self.i + (1. - self.lamda_c) * self.F

        self.e *= rho_t * gamma_t * self.lambda_c
        self.e[indices_t] += M * rho_t
        self.v += self.alpha_v * delta_t * self.e

        # # TODO: Should I keep this?!
        # self.v[indices_tp1] -= self.alpha_v * gamma_tp1 * (1 - self.lambda_c) * self.e.dot(self.w)

        if self.q_value_mode:
            raise NotImplementedError
            # indices_t_q = a_t * self.num_features + indices_t
            # indices_tp1_q = a_tp1 * self.num_features + indices_tp1
            # delta_t_q = r_tp1 + gamma_tp1 * rho_tp1 * self.v_q[indices_tp1_q].sum() - self.v_q[indices_t_q].sum()
            # self.e_q *= rho_t * gamma_t * self.lambda_c
            # self.e_q[indices_t_q] += 1
            # self.v_q += self.alpha_v * delta_t_q * self.e_q
            # self.v_q[indices_tp1_q] -= self.alpha_v * gamma_tp1 * (1 - self.lambda_c) * self.e_q.dot(self.w_q)
            # self.w_q[indices_t_q] -= self.alpha_w * self.w_q[indices_t_q].sum()
            # self.w_q += self.alpha_w * delta_t_q * self.e_q

    def estimate(self, indices, q_mode=False, a=None):
        if q_mode:
            indices_new = a * self.num_features + indices
            return self.v_q[indices_new].sum()
        else:
            return self.v[indices].sum()
