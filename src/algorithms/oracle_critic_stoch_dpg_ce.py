import numpy as np
import scipy
from scipy.special import expit
from scipy.stats import norm

class OracleCriticStochDPGCE:
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
            pi = np.array([self.policy.pi_params(x) for x in self.features])

        v = np.zeros(self.num_states)
        v[2] = self._sig_gauss_integral(pi[2,0],pi[2,1])
        v[1] = 2.0 * (1.0 - self._sig_gauss_integral(pi[1,0],pi[1,1]))
        sig_gauss_s0 = self._sig_gauss_integral(pi[0,0],pi[0,1])
        v[0] = v[1] * (1.0 - sig_gauss_s0) + v[2] * sig_gauss_s0

        return v

    def estimate_grad(self, pi=None):
        ''' Gets the state instead of features. Returns gradient of Q. Assumes deterministic policy.'''

        # find pi from policy and tc
        if pi is None:
            pi_params = [self.policy.pi_params(x) for x in self.features]
            # Make sure it's deterministic.
            assert all(np.isclose(np.array([param[1] for param in pi_params]), 0))
            pi = np.array([param[0] for param in pi_params])

        grad = np.zeros(3)
        grad[0] = -(2*expit(-pi[1]) - expit(pi[2])) * expit(-pi[0]) * (1.0 - expit(-pi[0]))
        grad[1] = -2 * expit(-pi[1]) * (1.0 - expit(-pi[1]))
        grad[2] = expit(pi[2]) * (1.0 - expit(pi[2]))

        return grad


    def steady_distribution(self, pi=None):
        # find pi from policy and tc
        if pi is None:
            pi = np.array([self.policy.pi_params(x) for x in self.features])

        p_pi = np.zeros((self.num_states,self.num_states))
        sig_gauss_s0 = self._sig_gauss_integral(pi[0,0],pi[0,1])
        p_pi[0,1] = 1.0 - sig_gauss_s0
        p_pi[0,2] = sig_gauss_s0
        p_pi[1,0] = 1.0
        p_pi[2,0] = 1.0
        vals, vecs, _ = scipy.linalg.eig(p_pi, left=True)
        distribution = vecs[:,np.isclose(vals,1.)][:,0]
        distribution /= distribution.sum()
        return distribution

    def true_Mt(self, d_mu, pi=None):
        ''' Not implemented. Returns the true emphatic weighting to be used in the updates.
        It is m(s)/d_mu(s). The division is needed because updates are according to d_mu.'''
        # TODO: Add interest

         # find pi from policy and tc
        if pi is None:
            pi = np.array([self.policy.pi_params(x) for x in self.features])

        # find m
        p_pi = np.zeros((self.num_states,self.num_states))
        sig_gauss_s0 = self._sig_gauss_integral(pi[0,0],pi[0,1])
        p_pi[0,1] = 1.0 - sig_gauss_s0
        p_pi[0,2] = sig_gauss_s0
        p_pi[1,0] = 1.0
        p_pi[2,0] = 1.0
        m = np.zeros(d_mu.shape)
        m[0] = d_mu[0]
        m[1] = d_mu[1] + p_pi[0,1]*d_mu[0]
        m[2] = d_mu[2] + p_pi[0,2]*d_mu[0]

        # print('---')
        # print(d_mu)
        # print(pi)
        # print(p_pi)
        # print(m)
        # print(m/d_mu)

        return m/d_mu


    def _sig_gauss_integral(self, mean, std):
        ''' Estimates the integral of sigmoid*gaussian, which frequently
        appears in the dpg counterexample.
        Ref: https://math.stackexchange.com/questions/207861/expected-value-of-applying-the-sigmoid-function-to-a-normal-distribution'''
        return norm.cdf( (np.pi/8.0)*mean / np.sqrt(1.0 + ((np.pi/8.0)**2 * std**2) ))
