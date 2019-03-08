import numpy as np
from blackhc import mdp
import scipy

class OracleCritic:
    ''' Returns the true state values for simple MDPs.
    Assuming start state is index 0 and there's only one terminal state at the end.'''

    def __init__(self, env, tc, policy):
        p, r, gamma = self._extract_info(env)
        self.tc = tc
        self.policy = policy
        self.num_states = p.shape[0]
        # Assuming MDP with states from 0 to |S|-1
        self.features = [tc.features(s) for s in range(self.num_states)]

        self.p = p
        self.p_r = np.sum(p*r, axis=2)
        self.p_gamma = p*gamma

    def estimate(self, pi=None):
        ''' Gets the state instead of features '''

        # find pi from policy and tc
        if pi is None:
            pi = np.array([self.policy.pi(x) for x in self.features])

        # find v_pi
        r_pi = np.sum(pi*self.p_r, axis=1)
        p_pi_gamma = np.sum(pi[...,None]*self.p_gamma, axis=1)
        v = np.linalg.pinv(np.eye(p_pi_gamma.shape[0]) - p_pi_gamma).dot(r_pi)

        # print('---')
        # print(pi)
        # print(v)

        return v

    def steady_distribution(self, pi=None):
        # find pi from policy and tc
        if pi is None:
            pi = np.array([self.policy.pi(x) for x in self.features])

        p_pi = np.sum(pi[...,None]*self.p, axis=1)
        vals, vecs, _ = scipy.linalg.eig(p_pi, left=True)
        distribution = vecs[:,np.isclose(vals,1.)][:,0]
        distribution /= distribution.sum()
        return distribution.real

    def true_Mt(self, d_mu, pi=None):
        ''' Returns the true emphatic weighting to be used in the updates.
        It is m(s)/d_mu(s). The division is needed because updates are according to d_mu.'''
        # TODO: Add interest

         # find pi from policy and tc
        if pi is None:
            pi = np.array([self.policy.pi(x) for x in self.features])

        # find m
        p_pi_gamma = np.sum(pi[...,None]*self.p_gamma, axis=1)
        m = np.linalg.pinv(np.eye(p_pi_gamma.shape[0]) - p_pi_gamma).T.dot(d_mu)

        # print('---')
        # print(d_mu)
        # print(pi)
        # print(p_pi_gamma)
        # print(m)
        # print(m/d_mu)

        return m/d_mu

    def _extract_info(self, env):
        ''' Extracts transition and reward and discounting matrices from mdp '''

        transitions = list(mdp.Transitions(env).next_states.items())
        rewards = list(mdp.Transitions(env).rewards.items())

        num_states = len(env.states) - 1
        num_actions = len(env.actions)

        p = np.zeros((num_states, num_actions, num_states))
        r = np.zeros((num_states, num_actions, num_states))
        gamma = np.ones((num_states, num_actions, num_states))

        for trans in transitions:
            s = trans[0][0].index
            a = trans[0][1].index
            for k, v in trans[1].items():
                sp = k.index
                if k.terminal_state:
                    sp = 0
                    gamma[s][a][sp] = 0
                # print(s,a,'->',sp,v)
                p[s][a][sp] = v

        # print('-')
        for reward in rewards:
            if reward[0][0].terminal_state:
                continue
            s = reward[0][0].index
            a = reward[0][1].index
            rew = 0
            weight_sum = 0
            for k, v in reward[1].items():
                rew += k*v
                weight_sum += v
            rew /= weight_sum
            # print(s,a,'->',rew)
            r[s][a][:] = rew

        # print('-')
        # print(p)
        # print('-')
        # print(gamma)
        # print('-')
        # print(r)

        return p, r, gamma
