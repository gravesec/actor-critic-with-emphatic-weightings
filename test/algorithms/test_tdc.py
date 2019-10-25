import unittest
import numpy as np
from tqdm import tqdm
from src.algorithms.tdc import LinearTDC
from test.algorithms.bairds_counterexample import BairdsCounterexample


class TDCTests(unittest.TestCase):

    def test_TDC_simple(self):
        tdc = LinearTDC(3, .1, .99)
        indices_t = np.array([0])
        indices_tp1 = np.array([1])
        delta_t = 1  # better than expected.
        rho_t = 1.  # on-policy.
        tdc.learn(delta_t, indices_t, .9, indices_tp1, .9, rho_t)

        self.assertGreater(tdc.estimate(indices_t), 0.)

    def test_tdc_bairds_counterexample(self):
        np.random.seed(1663983993)
        num_timesteps = 1000
        env = BairdsCounterexample
        rmspbe = np.empty(num_timesteps, dtype=float)
        tdc = LinearTDC(env.num_features, .005, .05)
        x_t = env.reset()
        for t in tqdm(range(num_timesteps)):
            a_t = np.random.choice(env.actions, p=env.mu)

            # Compute and store RMSPBE:
            be = np.zeros(env.num_states)
            for s_t in range(env.num_states):
                for s_tp1 in range(env.num_states):
                    if s_tp1 == env.num_states - 1:
                        be[s_t] = env.discount_rate * tdc.estimate(env.features[s_tp1]) - tdc.estimate(env.features[s_t])
            pbe = np.matmul()
            rmspbe[t] = np.sqrt(np.dot(np.square(pbe), d_mu))

        # plt.plot(rmspbe, label='RMSPBE')
        self.assertLess(rmspbe[-1], .1)


if __name__ == '__main__':
    unittest.main()