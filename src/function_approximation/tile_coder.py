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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Define a function to approximate:
    def target_function(x1, x2):
        return np.sin(np.sqrt(x1**2 + x2**2)) / np.sqrt(x1**2 + x2**2)

    # A function to add noise:
    def noise(y):
        return y + .1 * np.random.randn()

    # Create a tile coder and weight vector:
    tc = TileCoder(space=[[-10, 10], [-10, 10]], num_tiles_per_dim=[11, 11], num_tilings=8, bias_unit=True)

    w = np.zeros(tc.total_num_tiles)
    step_size = .2 / tc.num_active_features

    # Use the tile coder to train the weight vector from samples of the target function:
    num_examples = 1000
    for i in range(num_examples):
        x1, x2 = np.random.random_sample(2) * 20 - 10  # Sample the input space uniformly.
        indices = tc.encode((x1, x2))  # Compute indices for the input point.
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
            y_hat[j, k] = w[tc.encode((x1[j], x2[k]))].sum()  # Learned estimate.

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
