import unittest
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.function_approximation.tile_coder import TileCoder


class TileCoderTests(unittest.TestCase):

    def test_tile_coder(self):
        np.random.seed(2734287609)

        # Define a function to approximate:
        def target_function(x1, x2):
            return np.sin(np.sqrt(x1**2 + x2**2)) / np.sqrt(x1**2 + x2**2)

        # A function to add noise:
        def noise(y):
            return y + .1 * np.random.randn()

        # Create a tile coder and weight vector:
        tc = TileCoder(space=[[-10, 10], [-10, 10]], num_tiles_per_dim=[11, 11], num_tilings=8, bias_unit=True)
        weights = np.zeros(tc.total_num_tiles)
        step_size = .2 / tc.num_active_features

        # Use the tile coder to train the weight vector from noisy samples of the target function:
        num_examples = 1000
        for i in range(num_examples):
            x1, x2 = np.random.random_sample(2) * 20 - 10  # Sample the input space uniformly.
            indices = tc.encode((x1, x2))  # Compute indices for the input point.
            y_hat = weights[indices].sum()  # Generate an estimate from the weights.
            y = noise(target_function(x1, x2))  # Get a noisy sample output from the function.
            weights[indices] += step_size * (y - y_hat)  # Update the weight vector.

        # Check the function and the learned approximation:
        resolution = 100
        x1 = np.arange(-10, 10, 20 / resolution)
        x2 = np.arange(-10, 10, 20 / resolution)
        y = np.zeros((resolution, resolution))
        y_hat = np.zeros((resolution, resolution))
        for j in range(len(x1)):
            for k in range(len(x2)):
                y[j, k] = target_function(x1[j], x2[k])  # True value of the function.
                y_hat[j, k] = weights[tc.encode((x1[j], x2[k]))].sum()  # Learned estimate.

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
        plt.savefig('tile_coder_test.png')

        tolerance = .01
        self.assertLess(np.mean(y - y_hat), tolerance)


if __name__ == '__main__':
    unittest.main()
