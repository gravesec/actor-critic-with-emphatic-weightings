import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mountain_car.generate_experience import min_state_values, max_state_values


def plot_learned_value_function(tile_coder, critic, num_samples_per_dimension=100):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Sample the learned value function:
    positions = np.linspace(-1.2, .6, num_samples_per_dimension)
    velocities = np.linspace(-.07, .07, num_samples_per_dimension)
    value_estimates = np.zeros((num_samples_per_dimension, num_samples_per_dimension))
    for p, position in enumerate(positions):
        for v, velocity in enumerate(velocities):
            indices = tile_coder.indices((position, velocity))
            value_estimates[p, v] = critic.estimate(indices)

    pos, vel = np.meshgrid(positions, velocities)
    # surface = ax.plot_wireframe(pos, vel, value_estimates, rcount=15, ccount=15)
    surface = ax.plot_surface(pos, vel, value_estimates, cmap='hot')
    plt.title('Learned value function on the Mountain Car environment')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.savefig('learned_value_function.png')
    plt.show()


def plot_learned_policy(tile_coder, actor, num_samples_per_dimension=100):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Sample the learned policy:
    positions = np.linspace(-1.2, .6, num_samples_per_dimension)
    velocities = np.linspace(-.07, .07, num_samples_per_dimension)
    learned_policy = np.zeros((num_samples_per_dimension, num_samples_per_dimension, 3))
    for p, position in enumerate(positions):
        for v, velocity in enumerate(velocities):
            indices = tile_coder.indices((position, velocity))
            learned_policy[p, v] = actor.pi(indices)

    pos, vel = np.meshgrid(positions, velocities)
    surface = ax.plot_surface(pos, vel, learned_policy[:,:,2], cmap='hot')
    plt.title('Learned policy on the Mountain Car environment')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.savefig('learned_policy.png')
    plt.show()