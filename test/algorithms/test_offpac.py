import numpy as np
import unittest
from src.algorithms.offpac import OffPAC, OffPACActor


class OffPACActorTests(unittest.TestCase):

    def test_learn_simple(self):
        opa = OffPACActor(2, 3, .1, .9)
        x_t = np.array([1., 1., 1.])
        a_t = 0
        gamma_t = 1.
        rho_t = 1. # on-policy
        delta_t = 1 # action was better than expected

        opa.learn(delta_t, x_t, gamma_t, a_t, rho_t)

        # If the action taken turned out to be better than expected, one would expect the probability of taking it in the future to increase:
        assert opa.policy.pi(x_t, a_t) > 0.5
        # One would also expect the eligibility traces to have been updated:
        assert np.all(opa.e!=0.)

    def test_learn_complex(self):
        opa = OffPACActor(2, 3, .1, .99)
        x_t = np.array([1., 1., 1.])
        a_t = 0
        gamma_t = 1.
        rho_t = 1.  # on-policy
        delta_t = 1  # action was better than expected

        # Two time-steps with better than expected actions:
        opa.learn(delta_t, x_t, gamma_t, a_t, rho_t)
        opa.learn(delta_t, x_t, gamma_t, a_t, rho_t)

        # If the action taken turned out to be better than expected, one would expect the probability of taking it in the future to increase:
        assert opa.policy.pi(x_t, a_t) > 0.5
        # One would also expect the eligibility traces to begin piling up:
        assert np.all(opa.e!=0.)
        assert np.all(opa.e[a_t] > 0.5)

    def test_pi_equal(self):
        opa = OffPACActor(2, 1, None, None)
        x_t = np.array([[1.]])

        pi = opa.policy.pi(x_t)

        expected = np.full((2, 1), 1/2)
        np.testing.assert_array_almost_equal(pi, expected)

    def test_grad_log_pi_simple(self):
        # Equiprobable random policy with 2 actions and 3 features:
        opa = OffPACActor(2, 3, None, None)
        x_t = np.array([0., 1., 1.])
        a_t = 0

        # Execute test:
        grad_log_pi = opa.policy.grad_log_pi(x_t, a_t)

        # To increase the chance of taking action a_t, one would think the weights corresponding to a_t should all be increased (if eligible):
        assert (grad_log_pi[a_t] >= 0.).all()
        # And the weights for the other action should all be decreased (if eligible):
        assert (grad_log_pi[1] <= 0.).all()

    def test_grad_log_pi_complex(self):
        opa = OffPACActor(3, 4, None, None)
        opa.u = np.array([[.0, .01, .05, 0.1], [.2, .25, .3, .3], [.51, .52, .55, .6]])
        x_t = np.array([.0, .1, .25, .5])
        a_t = 0

        grad_log_pi = opa.policy.grad_log_pi(x_t, a_t)

        # To increase the chance of taking action a_t, one would think the weights corresponding to a_t should all be increased (if eligible):
        assert (grad_log_pi[a_t] >= 0.).all()
        # One would also think that the weights corresponding to a_t should be increased in proportion to their contribution (due to the feature vector):
        assert np.all(grad_log_pi[a_t, 1:] >= grad_log_pi[a_t, :-1])

if __name__ == '__main__':
    unittest.main()
