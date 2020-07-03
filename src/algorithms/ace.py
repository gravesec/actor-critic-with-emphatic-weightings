import numpy as np


class LinearACE:

    def __init__(self, num_actions, num_features, alpha):
        self.num_actions = num_actions
        self.num_features = num_features
        self.alpha = alpha
        self.theta = np.zeros((num_actions, num_features))
        self.psi_s_a = np.zeros((num_actions, num_features))
        self.psi_s_b = np.zeros((num_actions, num_features))

    def pi(self, features):
        preferences = self.theta.dot(features)
        # Converts potential overflows of the largest probability into underflows of the lowest probability:
        preferences = preferences - preferences.max()
        exp_preferences = np.exp(preferences)
        return exp_preferences / np.sum(exp_preferences)

    def grad_log_pi(self, features, a_t):
        self.psi_s_a.fill(0.)
        self.psi_s_a[a_t] = features
        pi = np.expand_dims(self.pi(features), axis=1)  # Add dimension to enable broadcasting.
        self.psi_s_b.fill(0.)
        self.psi_s_b[:] = features
        return self.psi_s_a - pi * self.psi_s_b

    def learn(self, features, a_t, delta_t, m_t, rho_t):
        self.theta += self.alpha * rho_t * m_t * delta_t * self.grad_log_pi(features, a_t)


class BinaryACE:

    def __init__(self, num_actions, num_features, alpha):
        self.num_actions = num_actions
        self.num_features = num_features
        self.alpha = alpha
        self.theta = np.zeros((num_actions, num_features))
        self.grad_log_pi = np.zeros((num_actions, num_features))

    def pi(self, indices):
        preferences = self.theta[:, indices].sum(axis=1)
        preferences = preferences - preferences.max()  # Converts potential overflows of the largest probability into underflows of the lowest probability.
        exp_preferences = np.exp(preferences)
        return exp_preferences / np.sum(exp_preferences)

    def learn(self, indices_t, a_t, delta_t, m_t, rho_t):
    #     M_t = (1 - eta_t) * i_t + eta_t * f_t
        pi = self.pi(indices_t)
        for a in range(self.theta.shape[0]):
            self.theta[a, indices_t] += self.alpha * rho_t * m_t * delta_t * (1 - pi[a] if a == a_t else 0 - pi[a])

    def all_actions_learn(self, indices_t, q_t, m_t):
        pi = self.pi(indices_t)
        # For each action,
        for a in range(self.num_actions):
            # Compute grad log pi:
            for b in range(self.num_actions):
                self.grad_log_pi[b, indices_t] += (1 - pi[b] if b == a else 0 - pi[b])
            self.theta += self.alpha * m_t * q_t[a] * pi[a] * self.grad_log_pi  # Update policy weights.
            self.grad_log_pi.fill(0.)  # Clear grad log pi for next iteration.
