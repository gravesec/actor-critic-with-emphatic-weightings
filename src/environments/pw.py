import numpy as np


class puddle():
    def __init__(self, headX, headY, tailX, tailY, radius, length, axis):
        self.headX = headX
        self.headY = headY
        self.tailX = tailX
        self.tailY = tailY
        self.radius = radius
        self.length = length
        self.axis = axis

    def get_distance(self, xCoor, yCoor):

        if self.axis == 0:
            u = (xCoor - self.tailX)/self.length
        else:
            u = (yCoor - self.tailY)/self.length

        dist = 0.0

        if u < 0.0 or u > 1.0:
            if u < 0.0:
                dist = np.sqrt(np.power((self.tailX - xCoor),2) + np.power((self.tailY - yCoor),2));
            else:
                dist = np.sqrt(np.power((self.headX - xCoor),2) + np.power((self.headY - yCoor),2));
        else:
            x = self.tailX + u * (self.headX - self.tailX);
            y = self.tailY + u * (self.headY - self.tailY);

            dist = np.sqrt(np.power((x - xCoor),2) + np.power((y - yCoor),2));

        if dist < self.radius:
            return (self.radius - dist)
        else:
            return 0

class DummyObject(object):
    def __init__(self):
        return

class puddleworld():
    def __init__(self):

        self.episodic = True
        self.num_action = 4
        self.obs_dim = 2

        self.state = None
        self.puddle1 = puddle(0.45,0.75,0.1,0.75,0.1,0.35,0)
        self.puddle2 = puddle(0.45,0.8,0.45,0.4,0.1,0.4,1)

        self.pworld_min_x = 0.0
        self.pworld_max_x = 1.0
        self.pworld_min_y = 0.0
        self.pworld_max_y = 1.0
        self.pworld_mid_x = (self.pworld_max_x - self.pworld_min_x)/2.0
        self.pworld_mid_y = (self.pworld_max_y - self.pworld_min_y)/2.0

        self.goal_dimension = 0.05
        self.def_displacement = 0.05

        self.sigma = 0.01

        self.goal_x_coor = self.pworld_max_x - self.goal_dimension #1000#
        self.goal_y_coor = self.pworld_max_y - self.goal_dimension #1000#

        self.was_reset = False

        self.time_step = 0

        self.action_space = DummyObject()
        self.action_space.n = 4
        self.observation_space = DummyObject()
        self.observation_space.low = np.array([0.,0.])
        self.observation_space.high = np.array([1.,1.])
        self.observation_space.shape = self.observation_space.high.shape

    def internal_reset(self):

        if not self.was_reset:
            self.state = np.zeros(2)

            # self.state[0] = self.np_random.uniform(low=0, high=self.goal_x_coor)
            # self.state[1] = self.np_random.uniform(low=0, high=self.goal_y_coor)

            self.state[0] = self.np_random.uniform(low=0.4, high=0.45)
            self.state[1] = self.np_random.uniform(low=0.7, high=0.75)

            self.was_reset = True

        return self._get_ob()

    def reset(self):
        self.was_reset = False
        return self.internal_reset()

    def set_state(self, state):
        self.state = np.zeros(2)
        self.state[:] = state

    def _get_ob(self):
        s = np.copy(self.state)
        return s

    def _terminal(self):
        s = self.state
        return bool((s[0] >= self.goal_x_coor) and (s[1] >= self.goal_y_coor))

    def _reward(self,x,y,terminal):
        if terminal:
            return -1.
        reward = -1.
        dist = self.puddle1.get_distance(x, y)
        reward += (-400. * dist)
        dist = self.puddle2.get_distance(x, y)
        reward += (-400. * dist)
        # reward = (reward/81)
        return reward

    def step(self,a):

        s = self.state

        xpos = s[0]
        ypos = s[1]

        nx = self.np_random.normal(scale=self.sigma)
        ny = self.np_random.normal(scale=self.sigma)

        if a == 0: #up
            ypos += (self.def_displacement+ny)
            xpos += nx
        elif a == 1: #down
            ypos -= (self.def_displacement+ny)
            xpos += nx
        elif a == 2: #right
            xpos += (self.def_displacement+nx)
            ypos += ny
        else: #left
            xpos -= (self.def_displacement+nx)
            ypos += ny

        if xpos > self.pworld_max_x:
            xpos = self.pworld_max_x
        elif xpos < self.pworld_min_x:
            xpos = self.pworld_min_x

        if ypos > self.pworld_max_y:
            ypos = self.pworld_max_y
        elif ypos < self.pworld_min_y:
            ypos = self.pworld_min_y

        s[0] = xpos
        s[1] = ypos
        self.state = s

        terminal = self._terminal()
        reward = self._reward(xpos,ypos,terminal)

        if terminal:
            self.reset()

        self.time_step += 1

        return (self._get_ob(), reward, terminal, {})

    @staticmethod
    def state_dim():
        return 2

    @staticmethod
    def num_action():
        return 4

    def seed(self, seed):
        self.np_random = np.random.RandomState(seed)


def init():
    return puddleworld()
