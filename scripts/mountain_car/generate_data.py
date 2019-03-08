# Add the src directory to the path, because python doesn't look in subfolders:
import sys, os
src_path = os.path.join(os.getcwd(), '.')
if src_path not in sys.path:
    sys.path.append(src_path)

import gym
import argparse
import numpy as np
from src import utils
from gym.utils import seeding

environment_name = "MountainCar-v0"
behaviour_policy_name = 'random'
num_actions = 3

if __name__ == '__main__':

    # Parse command line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('run_num', type=int, help='the run number')
    parser.add_argument('num_timesteps', type=int, default=100000, help='the number of timesteps of experience to generate')
    args = parser.parse_args()

    # Initialize the environment and rngs:
    env = gym.make(environment_name).env  # Get the underlying environment to turn off the time limit.
    env_seeds = env.seed()
    rng, alg_seed = seeding.np_random()

    # Generate the required number of timesteps of experience:
    transitions = np.empty(args.num_timesteps, dtype=[('s_t', float, (2,)), ('gamma_t', float, 1), ('a_t', int, 1), ('s_tp1', float, (2,)), ('r_tp1', float, 1), ('gamma_tp1', float, 1)])
    s_t = env.reset()
    gamma_t = 0.
    for t in range(args.num_timesteps):

        # Select an action:
        a_t = rng.choice(env.action_space.n)  # equiprobable random policy

        # Take action a_t, observe next state s_tp1 and reward r_tp1:
        s_tp1, r_tp1, terminal, _ = env.step(a_t)
        gamma_tp1 = not terminal

        # The agent should be reset to the starting state after the terminal transition:
        if terminal:
            s_tp1 = env.reset()

        # Store the transition:
        transitions[t] = (s_t, gamma_t, a_t, s_tp1, r_tp1, gamma_tp1)

        # Update counters:
        s_t = s_tp1
        gamma_t = gamma_tp1

    # Save the data for later:
    file_path = 'output/{}/behaviour_policies/{}/runs/{}/data.npz'.format(environment_name, behaviour_policy_name, args.run_num)
    utils.create_directory(file_path)
    np.savez_compressed(file_path, transitions=transitions, env_seeds=env_seeds, alg_seed=alg_seed)
