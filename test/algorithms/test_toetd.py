import unittest
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from src.algorithms.toetd import LinearTOETD
from test.algorithms.collision import Collision


class TOETDTests(unittest.TestCase):

    def test_linear_toetd(self):
        env = Collision
        np.random.seed(1730740995)
        num_timesteps = 10000
        num_runs = 50

        state_visit_counts = np.zeros((num_runs, Collision.num_states))
        estimated_state_values = np.full((num_runs, num_timesteps, Collision.num_states), np.nan)
        for run_num in tqdm(range(num_runs)):
            agent = LinearTOETD(env.num_features, 1., .001)
            features = env.features()
            s_t = env.init()
            x_t = features[s_t]
            for t in range(num_timesteps):
                a_t = np.random.choice(env.actions, p=env.mu[s_t])
                r_tp1, s_tp1 = env.sample(s_t, a_t)
                if s_tp1 is None:
                    gamma_tp1 = 0.
                    s_tp1 = env.init()
                else:
                    gamma_tp1 = env.gamma
                state_visit_counts[run_num, s_tp1] += 1
                x_tp1 = features[s_tp1]
                rho_t = env.rho[s_t, a_t]
                delta_t = r_tp1 + gamma_tp1 * agent.estimate(x_tp1) - agent.estimate(x_t)
                agent.learn(x_t, delta_t, rho_t, gamma_tp1, .9, 1., .001)
                x_t = x_tp1
                s_t = s_tp1

                for state in range(Collision.num_states):
                    estimated_state_values[run_num, t, state] = agent.estimate(features[state])

        d_mu = np.mean(state_visit_counts / num_timesteps, axis=0)
        msve = np.mean(np.sum(d_mu * np.square(estimated_state_values - Collision.true_state_values), axis=2), axis=0)
        plt.plot(msve, label='MSVE')
        plt.title('True Online Emphatic TD on the Collision problem')
        plt.xlabel('Timesteps')
        plt.ylabel('MSVE')
        plt.savefig('linear_toetd_collision.png')
        self.assertLess(msve[-1], 0.4)


if __name__ == '__main__':
    unittest.main()