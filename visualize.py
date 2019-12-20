import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

position_limits = np.array([-1.2, .6])
velocity_limits = np.array([-.07, .07])


def plot_learned_value_function(critic, tile_coder, num_samples_per_dimension=100):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Sample the learned value function:
    positions = np.linspace(*position_limits, num_samples_per_dimension)
    velocities = np.linspace(*velocity_limits, num_samples_per_dimension)
    value_estimates = np.zeros((num_samples_per_dimension, num_samples_per_dimension))
    for p, position in enumerate(positions):
        for v, velocity in enumerate(velocities):
            indices = tile_coder.indices((position, velocity))
            value_estimates[p, v] = critic.estimate(indices)

    pos, vel = np.meshgrid(positions, velocities)
    ax.plot_surface(pos, vel, value_estimates, cmap='hot')
    plt.title('Learned value function on the Mountain Car environment')
    plt.xlabel('Position')
    plt.xlim(position_limits)
    plt.ylabel('Velocity')
    plt.ylim(velocity_limits)
    plt.savefig('learned_value_function.png')
    plt.show()


def plot_learned_policy(actor, tile_coder, num_samples_per_dimension=100):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Sample the learned policy:
    positions = np.linspace(*position_limits, num_samples_per_dimension)
    velocities = np.linspace(*velocity_limits, num_samples_per_dimension)
    learned_policy = np.zeros((num_samples_per_dimension, num_samples_per_dimension, 3))
    for p, position in enumerate(positions):
        for v, velocity in enumerate(velocities):
            indices = tile_coder.indices((position, velocity))
            learned_policy[p, v] = actor.pi(indices)

    pos, vel = np.meshgrid(positions, velocities)
    ax.plot_surface(pos, vel, learned_policy[:, :, 2], cmap='hot')
    plt.title('Probability of action 2 in learned policy on the Mountain Car environment')
    plt.xlabel('Position')
    plt.xlim(position_limits)
    plt.ylabel('Velocity')
    plt.ylim(velocity_limits)
    plt.savefig('learned_policy.png')
    plt.show()


def plot_visits(transitions):
    fig = plt.figure()
    ax = fig.gca()
    states = []
    for transition in transitions:
        states.append(transition[0])  # Add the first state in every transition.
    states.append(transitions[-1][3])  # Add the last state in the last transition.
    states = np.array(states).T  # Convert to np.array.
    ax.plot(states[0], states[1])

    plt.title('State visits')
    plt.xlabel('Position')
    plt.xlim(position_limits)
    plt.ylabel('Velocity')
    plt.ylim(velocity_limits)
    plt.savefig('state_visits.png')
    plt.show()

