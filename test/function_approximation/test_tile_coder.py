import unittest
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.function_approximation.tile_coder import TileCoder


class TileCoderTests(unittest.TestCase):

    def test_tile_coder(self):
        np.random.seed(2734287609)
        num_features = 1024
        weights = np.zeros(num_features)
        tc = TileCoder(min_values=[0, 0], max_values=[2 * np.pi, 2 * np.pi], num_tiles=[8, 8], num_tilings=8, num_features=num_features, bias_unit=True)
        step_size = .1/tc.num_active_features

        def target(x, y):
            return np.sin(x) + np.cos(y) + 0.1 * np.random.randn()

        # Use the tilecoder to train:
        for i in range(10000):
            x, y = 2 * np.pi * np.random.rand(2)
            z = target(x, y)
            indices = tc.indices((x, y))
            weights[indices] += step_size * (z - weights[indices].sum())

        resolution = 100
        x = np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / resolution)
        y = np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / resolution)
        z = np.zeros((resolution, resolution))
        z_hat = np.zeros((resolution, resolution))
        for j in range(len(x)):
            for k in range(len(y)):
                z[j,k] = target(x[j], y[k])
                z_hat[j,k] = weights[tc.indices((x[j],y[k]))].sum()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        x, y = np.meshgrid(x, y)
        surface = ax.plot_surface(x, y, z_hat, cmap='hot')
        ax.plot_surface(x, y, z, color=(0.1, 0.2, 0.5, 0.3))
        plt.savefig('tile_coder_surface.png')
        tolerance = .1
        self.assertLess(np.mean(z - z_hat), tolerance)


if __name__ == '__main__':
    unittest.main()
