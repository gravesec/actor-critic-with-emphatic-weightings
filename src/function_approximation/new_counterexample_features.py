import numpy as np
from src.function_approximation.tiles3 import tiles


class NewCounterexampleFeatures:

    def __init__(self, bias_unit=False):

        self.bias_unit = bias_unit
        self.num_features = 3
        self.num_active_features = 1

        if self.bias_unit:
            self.num_features += 1
            self.num_active_features += 1

    def indices(self, observations):
        raise NotImplementedError

    def features(self, observations):
        # Create a binary feature vector:
        features = np.zeros(3)

        if observations < 3:
            features[int(observations)] = 1.
        elif observations == 3:
            features[1] = 1.

        # TODO: Is it fine that, with no bias unit, the terminal state will have zero features?

        if self.bias_unit:
            features[self.num_features - 1] = 1.

        return features
