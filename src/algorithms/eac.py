import numpy as np
from src.algorithms.gtd import GTD
from src.policy_types.discrete_policy import DiscretePolicy
from src.policy_types.continuous_policy import ContinuousPolicy


class EAC:

    def __init__(self, num_actions, num_features, alpha_u, alpha_v, alpha_w, lamda_v, lamda_a, use_extra_weights=False, policy_type='discrete'):
        self.actor = EACActor(num_actions, num_features, alpha_u, lamda_a, policy_type)
        self.critic = GTD(num_features, alpha_v, alpha_w, lamda_v)

        self.use_extra_weights = use_extra_weights
        if self.use_extra_weights:
            self.w_tilde = GTD(num_features, alpha_v, alpha_w, 0.)
            self.w_tilde.v = np.ones((num_features)) #TODO: This initialization is bad in other domains.

    def learn(self, x_t, gamma_t, a_t, r_tp1, x_tp1, gamma_tp1, rho_t, v_t=None, v_tp1=None, M=None):
        # Compute temporal difference error:
        if v_t is None:
            v_t = self.critic.estimate(x_t)
        if v_tp1 is None:
            v_tp1 = self.critic.estimate(x_tp1)
        delta_t = r_tp1 + gamma_tp1 * v_tp1 - v_t

        # Update learned weights:
        if self.use_extra_weights:
            # Fixed interest
            w_tilde_delta_tp1 = 1 + gamma_t * self.w_tilde.estimate(x_t) - self.w_tilde.estimate(x_tp1)
            self.w_tilde.learn(w_tilde_delta_tp1, x_tp1, gamma_tp1, x_t, gamma_t, rho_t)
            self.actor.learn(delta_t, x_t, gamma_t, a_t, rho_t, self.w_tilde.estimate(x_t), M=1.0)
        else:
            self.actor.learn(delta_t, x_t, gamma_t, a_t, rho_t, M=M)
        self.critic.learn(delta_t, x_t, gamma_t, x_tp1, gamma_tp1, rho_t)

class EACActor:

    def __init__(self, num_actions, num_features, alpha_u, lamda_a, policy_type='discrete'):

        # Called theta in the draft
        if policy_type == 'discrete':
            init = np.zeros((num_actions, num_features))
            # init[0,:] += np.log(9.0)
            self.policy = DiscretePolicy(init)
        else:
            self.policy = ContinuousPolicy(np.tile(np.array([0.0, np.log(1.0)])[:,None], (num_features)))

        self.i = 1. # Fixed interest
        self.alpha_u = alpha_u
        self.lamda_a = lamda_a

        self.F = 0.
        self.rho_tm1 = 1. # TODO: Write rho_0 in the pseudocode?

    def learn(self, delta_t, x_t, gamma_t, a_t, rho_t, w_tilde_x_t=1., M=None):

        # No eligibility trace and no Q estimation for now
        self.F = self.rho_tm1 * gamma_t * self.F + self.i
        if M is None:
            M = (1 - self.lamda_a) * self.i + self.lamda_a * self.F

        self.policy.u += self.alpha_u * M * rho_t * delta_t * self.policy.grad_log_pi(x_t, a_t) * w_tilde_x_t

        self.rho_tm1 = rho_t
