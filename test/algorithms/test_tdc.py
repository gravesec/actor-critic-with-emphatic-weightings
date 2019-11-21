import unittest
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from src.algorithms.tdc import LinearTDC, BinaryTDC
from src.misc_envs.bairds_counterexample import BairdsCounterexample
from src.misc_envs.collision import Collision


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

    def test_binary_tdc(self):
        env = Collision
        np.random.seed(1730740995)
        num_timesteps = 1000
        num_runs = 50

        state_visit_counts = np.zeros((num_runs, Collision.num_states))
        estimated_state_values = np.full((num_runs, num_timesteps, Collision.num_states), np.nan)
        for run_num in tqdm(range(num_runs)):
            agent = BinaryTDC(env.num_features, .01, .001, 0.9)
            indices = env.indices()
            s_t = env.init()
            indices_t = indices[s_t]
            gamma_t = 0.
            for t in range(num_timesteps):
                a_t = np.random.choice(env.actions, p=env.mu[s_t])
                r_tp1, s_tp1 = env.sample(s_t, a_t)
                if s_tp1 is None:
                    gamma_tp1 = 0.
                    s_tp1 = env.init()
                else:
                    gamma_tp1 = env.gamma
                state_visit_counts[run_num, s_tp1] += 1
                indices_tp1 = indices[s_tp1]
                rho_t = env.rho[s_t, a_t]
                delta_t = r_tp1 + gamma_tp1 * agent.estimate(indices_tp1) - agent.estimate(indices_t)
                agent.learn(delta_t, indices_t, gamma_t, indices_tp1, gamma_tp1, rho_t)
                indices_t = indices_tp1
                s_t = s_tp1
                gamma_t = gamma_tp1

                for state in range(Collision.num_states):
                    estimated_state_values[run_num, t, state] = agent.estimate(indices[state])

        d_mu = np.mean(state_visit_counts / num_timesteps, axis=0)
        msve = np.mean(np.sum(d_mu * np.square(estimated_state_values - Collision.true_state_values), axis=2), axis=0)
        plt.plot(msve, label='MSVE')
        plt.title('Binary TDC on the Collision problem')
        plt.xlabel('Timesteps')
        plt.ylabel('MSVE')
        plt.savefig('binary_tdc_collision.png')
        self.assertLess(msve[-1], 0.4)


if __name__ == '__main__':
    unittest.main()