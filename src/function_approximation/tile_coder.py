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
        # Allow obs to be an array-like with shape (d,) or (n, d)
        # Note that np.asarray(obs)[..., None, :] has shape of either (1, d) or (n, 1, d),
        # and so does tiling_coords
        tiling_coords = ((np.asarray(obs)[..., None, :] - self.space[:, 0] + self.tiling_offsets) // self.tile_size).astype(int)
        # Convert the N-dimensional tiling coordinates to a 1-dimensional index in the tiling:
        # self.coords_to_indices has shape (d,), so tiling_indices has shape(n, k) or (k,)
        tiling_indices = np.dot(tiling_coords, self.coords_to_indices)
        # Convert the tiling indices to indices in the feature vector:
        # self.tilings_to_features has shape (k,)
        feature_indices = self.tilings_to_features + tiling_indices
        if self.bias_unit:
            if feature_indices.ndim == 2:
                feature_indices = np.hstack([
                    feature_indices, 
                    (self.total_num_tiles - 1) * np.ones((len(feature_indices), 1), dtype=int)])
            else:
                feature_indices = np.append(feature_indices, self.total_num_tiles - 1)
        return feature_indices
