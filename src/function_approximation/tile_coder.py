import numpy as np


class TileCoder:
    def __init__(self, space, num_tiles_per_dim, num_tilings, bias_unit=False):
        self.space = np.asarray(space)
        self.num_tiles_per_dim = np.asarray(num_tiles_per_dim)
        self.num_tilings = int(num_tilings)
        self.bias_unit = bias_unit
        self.num_active_features = self.num_tilings + self.bias_unit
        self.total_num_tiles = self.num_tilings * np.prod(self.num_tiles_per_dim) + self.bias_unit
        self.tile_size = (self.space[:, 1] - self.space[:, 0]) / (self.num_tiles_per_dim - 1)

        # Compute the offset in each dimension for each tiling:
        self.tiling_offsets = np.arange(1, 2 * len(self.space), 2) * np.repeat(np.arange(self.num_tilings), len(self.space)).reshape(self.num_tilings, len(self.space)) / self.num_tilings % 1 * self.tile_size
        # Create an array to help convert n-dimensional tiling coordinates into a 1-dimensional index in the tiling:
        self.coords_to_indices = np.array([np.prod(self.num_tiles_per_dim[0:dim]) for dim in range(len(self.space))])
        # Compute the indices in the feature vector where each tiling starts:
        self.tilings_to_features = np.prod(self.num_tiles_per_dim) * np.arange(self.num_tilings)

    def encode(self, obs):
        # Compute the coordinates in each tiling of the tile containing the observation:
        tiling_coords = ((np.asarray(obs) - self.space[:, 0] + self.tiling_offsets) // self.tile_size).astype(int)
        # Convert the N-dimensional tiling coordinates to a 1-dimensional index in the tiling:
        tiling_indices = np.dot(tiling_coords, self.coords_to_indices)
        # Convert the tiling indices to indices in the feature vector:
        feature_indices = self.tilings_to_features + tiling_indices
        return np.append(feature_indices, self.total_num_tiles - 1) if self.bias_unit else feature_indices
