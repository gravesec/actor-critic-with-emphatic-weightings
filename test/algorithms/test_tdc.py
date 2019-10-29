import unittest
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from src.algorithms.tdc import LinearTDC, BinaryTDC
from test.algorithms.bairds_counterexample import BairdsCounterexample


class TDCTests(unittest.TestCase):

    def test_linear_tdc_simple(self):
        tdc = LinearTDC(3, .1, .01, .99)
        x_t = np.array([1, 0, 1])
        v_0 = tdc.estimate(x_t)  # Initial estimate of value of x_t.
        x_tp1 = np.array([0, 1, 1])
        delta_t = 1  # next state was better than expected.
        tdc.learn(delta_t, x_t, 1., x_tp1, 1., rho_t=1)
        # New estimate of value of x_t should be higher than initial estimate:
        self.assertGreater(tdc.estimate(x_t), v_0)

    def test_linear_tdc_bairds_counterexample(self):
        bce = BairdsCounterexample
        np.random.seed(1663983993)
        num_timesteps = 1000

        tdc = LinearTDC(bce.num_features, .005, .05, .0)
        # Non-standard weight initialization:
        tdc.v = np.ones(bce.num_features)
        tdc.v[6] = 10

        mspbe = np.full(num_timesteps, np.nan)
        x_t = bce.reset()
        for t in tqdm(range(num_timesteps)):
            a_t = np.random.choice(bce.actions, p=bce.mu)
            r_tp1, x_tp1 = bce.step(x_t, a_t)

            rho_t = bce.rho[a_t]
            delta_t = r_tp1 + bce.discount_rate * tdc.estimate(x_tp1) - tdc.estimate(x_t)
            tdc.learn(delta_t, x_t, bce.discount_rate, x_tp1, bce.discount_rate, rho_t)
            x_t = x_tp1

            # Compute and store RMSPBE:
            mspbe[t] = bce.mspbe(tdc)

        plt.plot(mspbe, label='RMSPBE')
        plt.title('Linear TDC on Baird\'s Counterexample')
        plt.xlabel('Timesteps')
        plt.ylabel('MSPBE')
        plt.savefig('linear_tdc_bairds_counterexample.png')
        self.assertLess(mspbe[-1], .1)


if __name__ == '__main__':
    unittest.main()