import numpy as np
from src.algorithms.tdc import LinearTDC
from src.policy_types.deterministic_policy import DeterministicPolicy


class DPG:

    def __init__(self, num_features, alpha_u, alpha_v, alpha_w, lamda_v, lamda_a, use_extra_weights=False):
        self.actor = DPGActor(num_features, alpha_u, lamda_a)
        self.critic = None

        assert use_extra_weights == False # TODO: Add this later
        self.use_extra_weights = use_extra_weights
        if self.use_extra_weights:
            self.w_tilde = LinearTDC(num_features, alpha_v, alpha_w, 0.)

    def learn(self, x_t, gamma_t, a_t, r_tp1, x_tp1, gamma_tp1, rho_t, v_t=None, v_tp1=None, grad_t=None, M=None):
        # Compute temporal difference error:
        if v_t is None:
            v_t = self.critic.estimate(x_t)
        if v_tp1 is None:
            v_tp1 = self.critic.estimate(x_tp1)
        delta_t = r_tp1 + gamma_tp1 * v_tp1 - v_t

        # Update learned weights:
        if self.use_extra_weights: # TODO
            w_tilde_delta_t = 1 + gamma_tp1 * self.w_tilde.estimate(x_tp1) - self.w_tilde.estimate(x_t)
            self.w_tilde.learn(w_tilde_delta_t, x_t, gamma_t, x_tp1, gamma_tp1, rho_t)
            self.actor.learn(delta_t, x_t, gamma_t, a_t, rho_t, self.w_tilde.estimate(x_t), M=M)
        else:
            self.actor.learn(x_t, gamma_t, a_t, rho_t, grad_t, M=M)
        # self.critic.learn(delta_t, x_t, gamma_t, x_tp1, gamma_tp1, rho_t)

class DPGActor:

    def __init__(self, num_features, alpha_u, lamda_a):

        # Called theta in the draft
        self.policy = DeterministicPolicy(np.zeros((num_features)))

        self.i = 1. # Fixed interest
        self.alpha_u = alpha_u
        self.lamda_a = lamda_a

        self.F = 0.
        self.rho_tm1 = 1.

    def learn(self, x_t, gamma_t, a_t, rho_t, grad_t, w_tilde_x_t=1., M=None):

        #self.F = self.rho_tm1 * gamma_t * self.F + self.i
        if M is None:
            M = 1.0#self.lamda_a * self.i + (1. - self.lamda_a) * self.F

        # No w_tilde_x for now
        self.policy.u += self.alpha_u * M * grad_t * self.policy.grad_pi(x_t)

        self.rho_tm1 = rho_t
