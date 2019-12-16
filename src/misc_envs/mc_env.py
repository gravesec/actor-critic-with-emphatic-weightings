import math
import numpy as np
from gym.utils import seeding

class MountainCar():
    def __init__(self, normalized=True):
        self.num_action = 3
        self.actions = [-1,0,1]
        self.num_state = 2
        self.state = None

        self.pos_min = -1.2
        self.pos_max = 0.6
        self.pos_range = self.pos_max - self.pos_min
        self.vel_min = -0.07
        self.vel_max = 0.07
        self.vel_range = self.vel_max - self.vel_min

        self.normalized = normalized

        self.wasReset = False

    def seed(self, random_seed):
        # self.np_random = np.random.RandomState()
        # self.np_random(random_seed)

        #defaulting to using gym's seeding as that seems to be more contrived, so I'm guessing principled?
        self.np_random, random_seed = seeding.np_random(random_seed)

    def internal_reset(self):
        if not self.wasReset:
            self.state = np.zeros((2))
            self.state[0] = self.np_random.uniform(low=-0.6, high=-0.4)
            self.state[1] = 0.0
            self.wasReset = True
        return self._get_ob()

    def reset(self):
        self.wasReset = False
        return self.internal_reset()

    def set_state(self, state):
        self.state = self.np_random.uniform(low=0.0, high=0.1, size=(2,))
        self.state[:] = state

    def _get_ob(self):
        if self.normalized:
            s = self.state
            s0 = (s[0] - self.pos_min) * self.pos_range
            s1 = (s[1] - self.vel_min) * self.vel_range
            return np.array([s0, s1])
        else:
            s = self.state
            return np.array([s[0], s[1]])

    def _terminal(self):
        s = self.state
        return bool(s[0] >= self.pos_max)

    def _reward(self,terminal):
        if terminal:
            return 0
        return -1

    def step(self,a):
        s = self.state

        s[1] += (0.001 * self.actions[a]) - (0.0025 * math.cos(3.0 * s[0]));
        if s[1] > self.vel_max:
            s[1] = self.vel_max
        elif s[1] < self.vel_min:
            s[1] = self.vel_min

        s[0] += s[1]

        if s[0] < self.pos_min:
            s[1] = 0.0
            s[0] = self.pos_min

        self.state = s

        terminal = self._terminal()
        reward = self._reward(terminal)

        if terminal:
            self.reset()

        return (self._get_ob(), reward, terminal, {})
