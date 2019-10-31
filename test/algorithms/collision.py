import numpy as np
from scipy.special import comb
from itertools import combinations


class Collision:
    class Action:
        right = 0
        retreat = 1
    actions = np.array([Action.right, Action.retreat])
    num_actions = len(actions)
    num_states = 8
    num_features = 6
    num_active_features = 3
    gamma = .9
    pi = np.array([1., 0.])  # Target policy always goes right.
    mu = np.full((num_states, num_actions), .5)  # Behaviour policy.
    mu[:int(num_states/2)] = np.array([1., 0.])
    rho = np.full((num_states, num_actions), 1.)
    rho[:int(num_states/2)] = np.array([2., 0.])

    # Generate all possible feature vectors to sample from:
    num_possible_feature_vectors = int(comb(num_features, num_active_features))
    possible_feature_vectors = np.zeros((num_possible_feature_vectors, num_features))
    for i, indices in enumerate(combinations(range(num_features), num_active_features)):
        possible_feature_vectors[i, indices] = 1

    @staticmethod
    def init():
        return np.random.randint(Collision.num_states / 2)

    @staticmethod
    def sample(state, action):
        if action == Collision.Action.right:
            reward = 0
            next_state = state + 1
            if next_state == Collision.num_states:
                next_state = None
                r = 1
        elif action == Collision.Action.retreat:
            reward = 0
            next_state = None
        else:
            raise ValueError('Invalid action {}'.format(action))
        return reward, next_state

    @staticmethod
    def features():
        # Return a subset of the possible feature vectors:
        indices = np.random.choice(Collision.num_possible_feature_vectors, Collision.num_states, replace=False)
        return Collision.possible_feature_vectors[indices]