import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import copy
from gym.envs.classic_control import rendering
from scipy.special import expit

class DPGCEEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        self.current_state = 0

        self._seed()
        self.viewer = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        if self.current_state == 0:
            reward = 0
            done = False

            threshold = expit(-action)
            random_number = self.np_random.uniform(low=0.0, high=1.0, size=(1,))
            if random_number < threshold:
                next_state = 1
            else:
                next_state = 2
        else:
            next_state = 0
            done = True
            if self.current_state == 1:
                reward = 2.0*expit(-action)
            else:
                reward = expit(action)

        self.current_state = next_state

        return self.current_state, reward, done, {}

    def _reset(self):
        self.current_state = 0
        return self.current_state

    def _render(self, mode='human', close=False):
        return None
