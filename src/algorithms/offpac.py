import numpy as np
from src.algorithms.gtd import GTD
from src.algorithms.td import TD
from src.policy_types.discrete_policy import DiscretePolicy


class OffPAC:

    def __init__(self, num_actions, num_features, alpha_u, alpha_v, alpha_w, lamda_v, use_extra_weights=False):
        self.actor = OffPACActor(num_actions, num_features, alpha_u, use_extra_weights)
        self.critic = GTD(num_features, alpha_v, alpha_w, lamda_v)

    def learn(self, x_t, gamma_t, a_t, r_tp1, x_tp1, gamma_tp1, rho_t):
        # Compute temporal difference error:
        v_t = self.critic.estimate(x_t)
        v_tp1 = self.critic.estimate(x_tp1)
        delta_t = r_tp1 + gamma_tp1 * v_tp1 - v_t

        # Update learned weights:
        self.critic.learn(delta_t, x_t, gamma_t, x_tp1, gamma_tp1, rho_t)
        self.actor.learn(delta_t, x_t, gamma_t, a_t, rho_t)

class OffPACActor:

    def __init__(self, num_actions, num_features, alpha_u, use_extra_weights=False):
        self.policy = DiscretePolicy(np.zeros((num_actions, num_features)))

        self.alpha_u = alpha_u
        self.use_extra_weights = use_extra_weights
        
        if self.use_extra_weights:
            self.extra_weights = TD(num_features, alpha_u, 0.)
        
    def learn(self, delta_t, x_t, gamma_t, a_t, rho_t):
        if self.use_extra_weights:
            self.extra_weights.learn(delta_t, x_t, gamma_t)
            self.policy.u += self.alpha_u * delta_t * rho_t * self.policy.grad_log_pi(x_t, a_t) * self.extra_weights.w.dot(x_t)
        else:
            self.policy.u += self.alpha_u * delta_t * rho_t * self.policy.grad_log_pi(x_t, a_t)
