import numpy as np
import unittest
from src.algorithms.ace import ACE
from src.function_approximation.tile_coder import TileCoder


class ACETests(unittest.TestCase):

    def test_learn_one(self):
        ace = ACE(2, 3)
        indices_t = np.array([2], dtype=np.intp)
        a_t = 0
        delta_t = 1  # action was better than expected

        before = ace.pi(indices_t)
        ace.learn(gamma_t=.9, i_t=1, eta_t=1., alpha_t=.1, rho_t=1., delta_t=delta_t, indices_t=indices_t, a_t=a_t)
        after = ace.pi(indices_t)

        # If the action taken turned out to be better than expected, one would expect the probability of taking it in the future to increase:
        self.assertGreater(after[a_t], before[a_t])

    def test_learn_two(self):
        ace = ACE(2, 3)
        indices_t = np.array([0, 1, 2], dtype=np.intp)
        a_t = 0
        delta_t = -1  # action was worse than expected

        before = ace.pi(indices_t)
        ace.learn(gamma_t=.9, i_t=1., eta_t=1., alpha_t=.1, rho_t=1., delta_t=delta_t, indices_t=indices_t, a_t=a_t)
        after = ace.pi(indices_t)
        # If the action taken turned out to be better than expected, one would expect the probability of taking it in the future to increase:
        self.assertLess(after[a_t], before[a_t])

    def test_ace(self):
        tc = TileCoder(min_state_values, max_state_values, [int(num_tiles), int(num_tiles)], int(num_tilings), num_features, int(bias_unit))
        actor = ACE()


if __name__ == '__main__':
    unittest.main()
