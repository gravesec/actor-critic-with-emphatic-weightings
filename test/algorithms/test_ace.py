import numpy as np
import unittest
from src.algorithms.ace import ACE


class ACETests(unittest.TestCase):

    def test_learn_one(self):
        actor = ACE(2, 3)
        indices_t = np.array([0, 1, 2], dtype=np.intp)
        a_t = 0
        gamma_t = 1.
        i_t = 1.
        eta_t = 1.
        alpha_t = .1
        rho_t = 1.  # on-policy
        delta_t = 1  # action was better than expected

        actor.learn(gamma_t, i_t, eta_t, alpha_t, rho_t, delta_t, indices_t, a_t)

        # If the action taken turned out to be better than expected, one would expect the probability of taking it in the future to increase:
        self.assertGreater(actor.pi(indices_t)[a_t], .5)

    def test_learn_two(self):
        actor = ACE(2, 3)
        indices_t = np.array([0, 1, 2], dtype=np.intp)
        a_t = 0
        gamma_t = 1.
        i_t = 1.
        eta_t = 1.
        alpha_t = .1
        rho_t = 1.  # on-policy
        delta_t = 1  # action was better than expected

        # Two time-steps with better than expected actions:
        actor.learn(gamma_t, i_t, eta_t, alpha_t, rho_t, delta_t, indices_t, a_t)
        actor.learn(gamma_t, i_t, eta_t, alpha_t, rho_t, delta_t, indices_t, a_t)

        # If the action taken turned out to be better than expected, one would expect the probability of taking it in the future to increase:
        self.assertGreater(actor.pi(indices_t)[a_t], .5)

    # def test_pi_equal(self):
    #     opa = EACActor(2, 1, None, None)
    #     x_t = np.array([[1.]])
    #
    #     pi = opa.policy.pi(x_t)
    #
    #     expected = np.full((2, 1), 1/2)
    #     np.testing.assert_array_almost_equal(pi, expected)
    #
    # def test_grad_log_pi_simple(self):
    #     # Equiprobable random policy with 2 actions and 3 features:
    #     opa = EACActor(2, 3, None, None)
    #     x_t = np.array([0., 1., 1.])
    #     a_t = 0
    #
    #     # Execute test:
    #     grad_log_pi = opa.policy.grad_log_pi(x_t, a_t)
    #
    #     # To increase the chance of taking action a_t, one would think the weights corresponding to a_t should all be increased (if eligible):
    #     assert (grad_log_pi[a_t] >= 0.).all()
    #     # And the weights for the other action should all be decreased (if eligible):
    #     assert (grad_log_pi[1] <= 0.).all()
    #
    # def test_grad_log_pi_complex(self):
    #     opa = EACActor(3, 4, None, None)
    #     opa.u = np.array([[.0, .01, .05, 0.1], [.2, .25, .3, .3], [.51, .52, .55, .6]])
    #     x_t = np.array([.0, .1, .25, .5])
    #     a_t = 0
    #
    #     grad_log_pi = opa.policy.grad_log_pi(x_t, a_t)
    #
    #     # To increase the chance of taking action a_t, one would think the weights corresponding to a_t should all be increased (if eligible):
    #     assert (grad_log_pi[a_t] >= 0.).all()
    #     # One would also think that the weights corresponding to a_t should be increased in proportion to their contribution (due to the feature vector):
    #     assert np.all(grad_log_pi[a_t, 1:] >= grad_log_pi[a_t, :-1])


if __name__ == '__main__':
    unittest.main()
