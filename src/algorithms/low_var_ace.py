import numpy as np


# TODO: Derive and implement eligibility traces for actor?
class LinearLowVarACE:

    def __init__(self, num_actions, num_features):
        self.theta = np.zeros((num_actions, num_features))


        self.psi_s_a = np.zeros((num_actions, num_features))
        self.psi_s_b = np.zeros((num_actions, num_features))

    def pi(self, features):
        logits = self.theta.dot(features)
        # Converts potential overflows of the largest probability into underflows of the lowest probability:
        logits = logits - logits.max()
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)

    def grad_log_pi(self, features, a_t):
        self.psi_s_a.fill(0.)
        self.psi_s_a[a_t] = features
        pi = np.expand_dims(self.pi(features), axis=1)  # Add dimension to enable broadcasting.
        self.psi_s_b.fill(0.)
        self.psi_s_b[:] = features
        return self.psi_s_a - pi * self.psi_s_b

    def learn(self, gamma_t, i_t, eta_t, alpha_t, rho_t, delta_t, features, a_t, F_t):
        M_t = (1 - eta_t) * i_t + eta_t * F_t
        self.theta += alpha_t * rho_t * M_t * delta_t * self.grad_log_pi(features, a_t)
        self.rho_tm1 = rho_t

class BinaryLowVarACE:

    def __init__(self, num_actions, num_features):
        self.num_actions = num_actions
        self.theta = np.zeros((num_actions, num_features))

    def pi(self, indices):
        logits = self.theta[:, indices].sum(axis=1)
        # Converts potential overflows of the largest probability into underflows of the lowest probability:
        logits = logits - logits.max()
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)

    def learn(self, gamma_t, i_t, eta_t, alpha_t, rho_t, delta_t, indices_t, a_t, F_t):
        M_t = (1 - eta_t) * i_t + eta_t * F_t
        pi = self.pi(indices_t)
        for a in range(self.theta.shape[0]):
            self.theta[a, indices_t] += alpha_t * rho_t * M_t * delta_t * (1 - pi[a] if a == a_t else 0 - pi[a])
        self.rho_tm1 = rho_t

    def all_actions_learn(self, q_t, indices_t, gamma_t, i_t, eta_t, alpha_t, rho_t, F_t):
        M_t = (1 - eta_t) * i_t + eta_t * F_t
        pi = self.pi(indices_t)
        for a in range(self.num_actions):
            for b in range(self.num_actions):
                self.theta[b, indices_t] += alpha_t * M_t * q_t[a] * pi[a] * (1 - pi[b] if b == a else 0 - pi[b])
        self.rho_tm1 = rho_t