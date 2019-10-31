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
        num_timesteps = 2000

        agent = LinearTOETD(env.num_features, 1., .001)
        features = env.features()

        msve = np.full(num_timesteps, np.nan)
        gamma_t = env.gamma
        s_t = env.init()
        x_t = features[s_t]
        for t in range(num_timesteps):
            a_t = np.random.choice(env.actions, p=env.mu[s_t])
            r_tp1, s_tp1 = env.sample(s_t, a_t)
            x_tp1 = features[s_tp1]

            # TODO: Handle s_tp1==None (aka add transition-dependent discounting):
            if s_tp1 is None:
                gamma_tp1 = 0
                s_tp1 = env.init()

            rho_t = env.rho[a_t]
            delta_t = r_tp1 + env.discount_rate * agent.estimate(x_tp1) - agent.estimate(x_t)
            agent.learn(x_t, delta_t, rho_t, env.discount_rate, .0, 1., .03)
            x_t = x_tp1
            # Calculate and store error:
            # msve[t] = ...

        plt.plot(msve, label='MSVE')
        plt.title('True Online Emphatic TD on the Collision problem')
        plt.xlabel('Timesteps')
        plt.ylabel('MSVE')
        plt.savefig('linear_toetd_collision.png')


if __name__ == '__main__':
    unittest.main()