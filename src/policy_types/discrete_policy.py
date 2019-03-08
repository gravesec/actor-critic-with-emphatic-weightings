import numpy as np


class DiscretePolicy:

    def __init__(self, u):
        self.num_actions, self.num_features = u.shape
        self.u = u

    def pi(self, x_t, a_t=None):
        prefs = self.u.dot(x_t)
        # Converts potential overflows of the largest probability into underflows of the lowest probability:
        prefs = prefs - prefs.max()
        exp_prefs = np.exp(prefs)
        probs = exp_prefs / np.sum(exp_prefs)
        return probs if a_t is None else probs[a_t]

    def grad_log_pi(self, x_t, a_t):
        psi_s_a = np.zeros((self.num_actions, self.num_features))
        psi_s_a[a_t] = x_t
        probs = self.pi(x_t).reshape(self.num_actions, 1) # reshape to enable broadcasting.
        psi_s_b = np.empty((self.num_actions, self.num_features))
        psi_s_b[:] = x_t
        return psi_s_a - probs * psi_s_b
