import numpy as np


class LinearTDC:

    def __init__(self, num_features, alpha_w, alpha_v, lambda_c):
        self.num_features = num_features
        self.alpha_w = alpha_w
        self.alpha_v = alpha_v
        self.lambda_c = lambda_c
        self.w = np.zeros(num_features)
        self.v = np.zeros(num_features)
        self.z = np.zeros(num_features)

    def learn(self, delta_t, x_t, gamma_t, x_tp1, gamma_tp1, rho_t):
        self.z = rho_t * (gamma_t * self.lambda_c * self.z + x_t)
        self.w += self.alpha_w * delta_t * self.z - self.alpha_w * gamma_tp1 * (1 - self.lambda_c) * self.z.dot(self.v) * x_tp1
        self.v += self.alpha_v * delta_t * self.z - self.alpha_v * self.v.dot(x_t) * x_t

    def estimate(self, x_t):
        return self.w.dot(x_t)


class BinaryTDC:

    def __init__(self, num_features, alpha_w, alpha_v, lambda_c):
        self.num_features = num_features
        self.alpha_w = alpha_w
        self.alpha_v = alpha_v
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
        self.v += self.alpha_v * delta_t * self.z
        self.v[indices_t] -= self.alpha_v * v_dot_x

    def estimate(self, indices):
        return self.w[indices].sum()


class BinaryGQ:

    def __init__(self, num_actions, num_features, alpha_w, alpha_v, lambda_c):
        self.num_actions = num_actions
        self.num_features = num_features
        self.alpha_w = alpha_w
        self.alpha_v = alpha_v
        self.lambda_c = lambda_c
        self.w = np.zeros((num_actions, num_features))
        self.v = np.zeros((num_actions, num_features))
        self.z = np.zeros((num_actions, num_features))

    def learn(self, indices_t, a_t, rho_t, gamma_t, r_tp1, indices_tp1, pi_tp1, gamma_tp1):
        delta_t = r_tp1 + gamma_tp1 * pi_tp1.dot(self.estimate(indices_tp1)) - self.estimate(indices_t, a_t)
        self.z *= rho_t * gamma_t * self.lambda_c
        self.z[a_t, indices_t] += rho_t
        z_dot_v = np.ravel(self.z).dot(np.ravel(self.v))
        self.w += self.alpha_w * delta_t * self.z
        for a_tp1 in range(self.num_actions):
            self.w[a_tp1, indices_tp1] -= pi_tp1[a_tp1] * self.alpha_w * gamma_tp1 * (1 - self.lambda_c) * z_dot_v
        v_dot_x = self.v[a_t, indices_t].sum()
        self.v += self.alpha_v * delta_t * self.z
        self.v[a_t, indices_t] -= self.alpha_v * v_dot_x

    def estimate(self, indices, action=None):
        return self.w[:, indices].sum(axis=1) if action is None else self.w[action, indices].sum()


class BinaryTOGQ:
    # Currently isn't quite right. Might need to actually derive TOGQ instead of modifying TOGTD.
    def __init__(self, num_actions, num_features, alpha_w, alpha_v, lambda_c):
        self.num_actions = num_actions
        self.num_features = num_features
        self.alpha_w = alpha_w
        self.alpha_v = alpha_v
        self.lambda_c = lambda_c
        self.w = np.zeros((num_actions, num_features))
        self.v = np.zeros((num_actions, num_features))
        self.z = np.zeros((num_actions, num_features))
        self.z_w = np.zeros((num_actions, num_features))
        self.z_v = np.zeros((num_actions, num_features))

    def learn(self, q_old, rho_tm1, indices_t, a_t, gamma_t, rho_t, r_tp1, indices_tp1, pi_tp1, gamma_tp1):
        q_t = self.estimate(indices_t, a_t)
        delta_t = r_tp1 + gamma_tp1 * pi_tp1.dot(self.estimate(indices_tp1)) - q_t

        # Update trace for gradient correction:
        self.z *= rho_t * gamma_t * self.lambda_c
        self.z[a_t, indices_t] += rho_t

        # Update trace for main weights:
        z_w_dot_x = self.z_w[a_t, indices_t].sum()
        self.z_w *= rho_t * gamma_t * self.lambda_c
        self.z_w[a_t, indices_t] += rho_t * self.alpha_w * (1 - rho_t * gamma_t * self.lambda_c * z_w_dot_x)

        # Update trace for auxiliary weights:
        z_v_dot_x = self.z_v[a_t, indices_t].sum()
        self.z_v *= rho_tm1 * gamma_t * self.lambda_c
        self.z_v[a_t, indices_t] += self.alpha_v * (1 - rho_tm1 * gamma_t * self.lambda_c * z_v_dot_x)

        # Update main weights:
        v_dot_z = np.ravel(self.v).dot(np.ravel(self.z))
        self.w += delta_t * self.z_w + self.z_w * (q_t - q_old)
        self.w[a_t, indices_t] -= self.alpha_w * rho_t * (q_t - q_old)
        for a_tp1 in range(self.num_actions):
            self.w[a_tp1, indices_tp1] -= pi_tp1[a_tp1] * self.alpha_w * gamma_tp1 * (1 - self.lambda_c) * v_dot_z

        # Update auxiliary weights:
        v_dot_x = self.v[a_t, indices_t].sum()
        self.v += rho_t * delta_t * self.z_v
        self.v[a_t, indices_t] -= self.alpha_v * v_dot_x

    def estimate(self, indices, action=None):
        return self.w[:, indices].sum(axis=1) if action is None else self.w[action, indices].sum()
