import unittest
import numpy as np
from src.algorithms.gtd import GTD


class GDTTests(unittest.TestCase):

    def test_learn_simple(self):
        gtd = GTD(3, .1, .01, .99)
        x_t = np.array([1., 0., 0.])
        gamma_t = 1.
        x_tp1 = np.array([0., 1., 0.])
        gamma_tp1 = 1.
        rho_t = 1. # on-policy
        delta_t = 1 # better than expected

        gtd.learn(delta_t, x_t, gamma_t, x_tp1, gamma_tp1, rho_t)

        # If the state value for x_t is better than expected, one would expect the state value estimate for x_t to increase:
        assert gtd.estimate(x_t) > 0.

if __name__ == '__main__':
    unittest.main()
