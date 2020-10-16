import unittest
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from src.algorithms.fhat import LinearFHat, BinaryFHat
from src.algorithms.etd import LinearETD, BinaryETD
from src.environments.collision import Collision


class LowVarETDTests(unittest.TestCase):

    def test_low_var_etd(self):
        env = Collision
        np.random.seed(1730740995)
        num_timesteps = 1000
        num_runs = 50

        i_t = 1.
        i_tp1 = 1.
        gamma_t = 0.

        state_visit_counts = np.zeros((num_runs, Collision.num_states))
        estimated_state_values = np.full((num_runs, num_timesteps, Collision.num_states), np.nan)
        for run_num in tqdm(range(num_runs)):
            fhat = LinearFHat(env.num_features, .001)
            agent = LinearETD(env.num_features, .01, .0)
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

                F_t = fhat.estimate(x_t)

                agent.learn(delta_t, x_t, gamma_t, i_t, x_tp1, gamma_tp1, rho_t, F_t)
                fhat.learn(x_tp1, gamma_tp1, x_t, rho_t, i_tp1)

                x_t = x_tp1
                s_t = s_tp1

                gamma_t = gamma_tp1

                for state in range(Collision.num_states):
                    estimated_state_values[run_num, t, state] = agent.estimate(features[state])

        d_mu = np.mean(state_visit_counts / num_timesteps, axis=0)
        msve = np.mean(np.sum(d_mu * np.square(estimated_state_values - Collision.true_state_values), axis=2), axis=0)
        plt.plot(msve, label='MSVE')
        plt.title('Linear True Online Emphatic TD on the Collision problem')
        plt.xlabel('Timesteps')
        plt.ylabel('MSVE')
        plt.savefig('linear_toetd_collision.png')
        self.assertLess(msve[-1], 0.5)

    def test_binary_low_var_etd(self):
        env = Collision
        np.random.seed(1730740995)
        num_timesteps = 1000
        num_runs = 50

        i_t = 1.
        i_tp1 = 1.
        gamma_t = 0.

        state_visit_counts = np.zeros((num_runs, Collision.num_states))
        estimated_state_values = np.full((num_runs, num_timesteps, Collision.num_states), np.nan)
        for run_num in tqdm(range(num_runs)):
            fhat = BinaryFHat(env.num_features, .001)
            agent = BinaryETD(env.num_features, .01, .0)
            indices = env.indices()
            s_t = env.init()
            x_t = indices[s_t]
            for t in range(num_timesteps):
                a_t = np.random.choice(env.actions, p=env.mu[s_t])
                r_tp1, s_tp1 = env.sample(s_t, a_t)
                if s_tp1 is None:
                    gamma_tp1 = 0.
                    s_tp1 = env.init()
                else:
                    gamma_tp1 = env.gamma
                state_visit_counts[run_num, s_tp1] += 1
                x_tp1 = indices[s_tp1]
                rho_t = env.rho[s_t, a_t]
                delta_t = r_tp1 + gamma_tp1 * agent.estimate(x_tp1) - agent.estimate(x_t)

                F_t = fhat.estimate(x_t)

                agent.learn(delta_t, x_t, gamma_t, i_t, rho_t, F_t)
                fhat.learn(x_tp1, gamma_tp1, x_t, rho_t, i_tp1)

                x_t = x_tp1
                s_t = s_tp1

                for state in range(Collision.num_states):
                    estimated_state_values[run_num, t, state] = agent.estimate(indices[state])

                gamma_t = gamma_tp1

        d_mu = np.mean(state_visit_counts / num_timesteps, axis=0)
        msve = np.mean(np.sum(d_mu * np.square(estimated_state_values - Collision.true_state_values), axis=2), axis=0)
        plt.plot(msve, label='MSVE')
        plt.title('Binary True Online Emphatic TD on the Collision problem')
        plt.xlabel('Timesteps')
        plt.ylabel('MSVE')
        plt.savefig('binary_toetd_collision.png')
        self.assertLess(msve[-1], 0.5)


if __name__ == '__main__':
    unittest.main()
