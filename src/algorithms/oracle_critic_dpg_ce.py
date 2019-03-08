import numpy as np
import scipy
from scipy.special import expit

class OracleCriticDPGCE:
    ''' Returns the true state values only for DPG counterexample.'''

    def __init__(self, tc, policy):
        self.tc = tc
        self.policy = policy
        self.num_states = 3
        # Assuming MDP with states from 0 to |S|-1
        self.features = [tc.features(s) for s in range(self.num_states)]


    def estimate(self, pi=None):
        ''' Gets the state instead of features. '''

        # find pi from policy and tc
        if pi is None:
            pi = np.array([self.policy.pi(x) for x in self.features])

        v = np.zeros(3)
        v[0] = expit(-pi[0]) * (2*expit(-pi[1])) + (1.0 - expit(-pi[0])) * expit(pi[2])
        v[1] = 2 * expit(-pi[1])
        v[2] = expit(pi[2])

        return v

    def estimate_grad(self, pi=None):
        ''' Gets the state instead of features. Returns gradient of Q. '''

        # find pi from policy and tc
        if pi is None:
            pi = np.array([self.policy.pi(x) for x in self.features])

        grad = np.zeros(3)
        grad[0] = -(2*expit(-pi[1]) - expit(pi[2])) * expit(-pi[0]) * (1.0 - expit(-pi[0]))
        grad[1] = -2 * expit(-pi[1]) * (1.0 - expit(-pi[1]))
        grad[2] = expit(pi[2]) * (1.0 - expit(pi[2]))

        return grad

    def steady_distribution(self, pi=None):
        # find pi from policy and tc
        if pi is None:
            pi = np.array([self.policy.pi(x) for x in self.features])

        p_pi = np.zeros((self.num_states,self.num_states))
        p_pi[0,1] = expit(-pi[0])
        p_pi[0,2] = 1.0 - expit(-pi[0])
        p_pi[1,0] = 1.0
        p_pi[2,0] = 1.0
        vals, vecs, _ = scipy.linalg.eig(p_pi, left=True)
        distribution = vecs[:,np.isclose(vals,1.)][:,0]
        distribution /= distribution.sum()
        return distribution

    def true_Mt(self, d_mu, pi=None):
        ''' Not implemented. Returns the true emphatic weighting to be used in the updates.
        It is m(s)/d_mu(s). The division is needed because updates are according to d_mu.'''

        return None
