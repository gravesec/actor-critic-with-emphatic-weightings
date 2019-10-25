import numpy as np
from src.function_approximation.tiles3 import tiles


class TileCoder:
    def __init__(self, min_values, max_values, num_tiles, num_tilings, num_features, bias_unit=False):
        self.num_tiles = np.array(num_tiles)
        self.scale_factor = self.num_tiles / (np.array(max_values) - np.array(min_values))
        self.num_tilings = num_tilings
        self.bias_unit = bias_unit
        self.num_features = num_features
        self.num_active_features = self.num_tilings + self.bias_unit

    def indices(self, observations):
        indices = tiles(int(self.num_features - self.bias_unit), self.num_tilings, list(np.array(observations) * self.scale_factor))
        if self.bias_unit:
            indices.append(self.num_features - 1)  # Add bias unit.
        return np.array(indices, dtype=np.intp)

    def features(self, indices):
        features = np.zeros(self.num_features)
        features[indices] = 1.
        return features
