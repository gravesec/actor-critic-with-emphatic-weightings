import numpy as np
from src.function_approximation.tiles3 import tiles


class LongCounterexampleFeatures:

    def __init__(self, middle_steps=3, bias_unit=False):

        self.num_states = middle_steps*2 + 1
        self.bias_unit = bias_unit
        self.num_features = self.num_states - 1
        self.num_active_features = 1

        if self.bias_unit:
            self.num_features += 1
            self.num_active_features += 1

    def indices(self, observations):
        raise NotImplementedError

    def features(self, observations):
        # Create a binary feature vector:
        features = np.zeros(self.num_features)

        if observations < self.num_states - 1:
            features[int(observations)] = 1.
        elif observations == self.num_states - 1:
            features[self.num_states - 2] = 1.

        # TODO: Is it fine that, with no bias unit, the terminal state will have zero features?

        if self.bias_unit:
            features[self.num_features - 1] = 1.

        return features
