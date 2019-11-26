import gym
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
    ax.plot_surface(pos, vel, value_estimates, cmap='hot')
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
    ax.plot_surface(pos, vel, learned_policy[:, :, 2], cmap='hot')
    plt.title('Probability of action 2 in learned policy on the Mountain Car environment')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.savefig('learned_policy.png')
    plt.show()


def evaluate_policy(tile_coder, actor, num_timesteps=1000):
    env = gym.make('MountainCar-v0').env

    g_t = 0.
    s_t = env.reset()
    for t in range(num_timesteps):

        # Get feature vector for the current state:
        indices_t = tile_coder.indices(s_t)

        # Select an action:
        pi = actor.pi(indices_t)
        a_t = np.random.choice(pi.shape[0], p=pi)

        # Take action a_t, observe next state s_tp1 and reward r_tp1:
        s_tp1, r_tp1, terminal, _ = env.step(a_t)

        # Add reward:
        g_t += r_tp1

        env.render()

        # If done, break the loop:
        if terminal:
            break
    return g_t
