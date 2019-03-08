import numpy as np
from src.function_approximation.tiles3 import tiles


class TileCoder:

    def __init__(self, min_values, max_values, num_tiles, num_tilings, num_features, bias_unit=False):
        self.num_tiles = np.array(num_tiles)
        self.scale_factor = self.num_tiles / (np.array(max_values) - np.array(min_values))
        self.num_tilings = num_tilings
        self.bias_unit = bias_unit
        self.num_features = num_features
        self.num_active_features = self.num_tilings

        if self.bias_unit:
            self.num_features += 1
            self.num_active_features += 1

    def indices(self, observations):
        return np.array(tiles(int(self.num_features), self.num_tilings, list(np.array(observations) * self.scale_factor)), dtype=np.intp)

    def features(self, observations):
        # Create a binary feature vector:
        features = np.zeros((self.num_features))

        # Populate the feature vector:
        indices = self.indices(observations)
        features[indices] = 1.

        if self.bias_unit:
            features[self.num_features - 1] = 1.

        return features
