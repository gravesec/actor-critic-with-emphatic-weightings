import numpy as np


# TODO: implement bias unit.
class TileCoder:
    def __init__(self, limits=[[-10, 10], [-10, 10]], tiles_per_dim=[6, 6], num_tilings=4, bias_unit=True):
        self.limits = np.asarray(limits)
        self.tiles_per_dim = np.asarray(tiles_per_dim)
        self.num_tilings = num_tilings
        # self.bias_unit = bias_unit

        self.tiling_offsets = np.array([[(tiling_num / num_tilings * 3 ** dimension) % 1 for dimension in range(len(limits))] for tiling_num in range(num_tilings)])
        self.scale_factors = (self.tiles_per_dim - 1) / (self.limits[:, 1] - self.limits[:, 0])
        self.num_tiles = self.num_tilings * np.prod(self.tiles_per_dim)

        self.tile_base_ind = np.prod(self.tiles_per_dim) * np.arange(num_tilings)

        self.hash_vec = np.array([np.prod(tiles_per_dim[0:i]) for i in range(len(limits))])

    def indices(self, obs):
        offset_coords = ((obs - self.limits[:, 0]) * self.scale_factors + self.tiling_offsets).astype(int)
        return self.tile_base_ind + np.dot(offset_coords, self.hash_vec)


class KrisTileCoder:
    def __init__(self, tiles_per_dim, value_limits, tilings, offset=lambda n: 2 * np.arange(n) + 1):
        tiling_dims = np.array(np.ceil(tiles_per_dim), dtype=np.int) + 1
        self._offsets = offset(len(value_limits)) * \
          np.repeat([np.arange(tilings)], len(value_limits), 0).T / float(tilings) % 1
        self._limits = np.array(value_limits)
        self._norm_dims = np.array(tiles_per_dim) / (self._limits[:, 1] - self._limits[:, 0])
        self._tile_base_ind = np.prod(tiling_dims) * np.arange(tilings)
        self._hash_vec = np.array([np.prod(tiling_dims[0:i]) for i in range(len(value_limits))])
        self._n_tiles = tilings * np.prod(tiling_dims)

    def __getitem__(self, x):
        off_coords = ((x - self._limits[:, 0]) * self._norm_dims + self._offsets).astype(int)
        return self._tile_base_ind + np.dot(off_coords, self._hash_vec)


# TODO: update test for tile coder.
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Define a function to approximate:
    def target_function(x1, x2):
        return np.sin(np.sqrt(x1**2 + x2**2)) / np.sqrt(x1**2 + x2**2)

    # A function to add noise:
    def noise(y):
        return y + .1 * np.random.randn()

    # Create a tile coder and weight vector:
    tc = TileCoder(limits=[[-10, 10], [-10, 10]], tiles_per_dim=[6, 6], num_tilings=4, bias_unit=True)
    w = np.zeros(tc.num_observation_features)
    step_size = .2 / tc.num_active_features

    # Use the tile coder to train the weight vector from samples of the target function:
    num_examples = 1000
    for i in range(num_examples):
        x1, x2 = np.random.random_sample(2) * 20 - 10  # Sample the input space uniformly.
        indices = tc.get_indices((x1, x2))  # Compute indices for the input point.
        y_hat = w[indices].sum()  # Generate an estimate from the weights.
        y = noise(target_function(x1, x2))  # Get a sample output from the function.
        w[indices] += step_size * (y - y_hat)  # Update the weight vector.

    # Check the function and the learned approximation:
    resolution = 100
    x1 = np.arange(-10, 10, 20 / resolution)
    x2 = np.arange(-10, 10, 20 / resolution)
    y = np.zeros((resolution, resolution))
    y_hat = np.zeros((resolution, resolution))
    for j in range(len(x1)):
        for k in range(len(x2)):
            y[j, k] = target_function(x1[j], x2[k])  # True value of the function.
            y_hat[j, k] = w[tc.get_indices((x1[j], x2[k]))].sum()  # Learned estimate.

    # Visualize the function and the learned approximation:
    x1, x2 = np.meshgrid(x1, x2)
    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(x1, x2, y_hat, cmap='hot')
    ax.set_zlim(-.25, 1.)
    ax.set_title('Learned estimate after {} examples'.format(num_examples))

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(x1, x2, y, cmap='hot')
    ax.set_zlim(-.25, 1.)
    ax.set_title('True function')

    plt.show()
