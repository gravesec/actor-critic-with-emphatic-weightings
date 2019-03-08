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

from src.misc_envs.dpg_ce_env import DPGCEEnv

import numpy as np

# Set up the experiment:
environment_name = 'dpg_counterexample'
max_num_timesteps = 1000
behavior_params = [1.0,1.0] # mu and std. TODO: DPG oracle assumes std=0
behaviour_policy_name = 'random_'+str(behavior_params[0])+'_'+str(behavior_params[1])

if __name__ == '__main__':

    # Parse command line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('run_num', type=int, help='the run number')
    parser.add_argument('num_episodes', type=int, default=100, help='the number of episodes per run')
    args = parser.parse_args()

    # Initialize the environment and rngs:
    env = DPGCEEnv()
    env_seeds = env._seed()
    rng, alg_seed = seeding.np_random()

    # Generate the required number of episodes:
    episodes = []
    for episode_num in range(args.num_episodes):
        transitions = []

        # Start an episode:
        s_t = env._reset() # Reset and step and render without underscore don't work without registering.

        # Play out the episode:
        for t in range(max_num_timesteps): # Because the env is not registered.

            # Select an action:
            a_t = np.random.normal(behavior_params[0], behavior_params[1]) # TODO: Add std maybe.

            # Take action a, observe reward r', and next state s':
            s_tp1, r_tp1, terminal, _ = env._step(a_t)

            # Store the transition:
            transitions.append((s_t, a_t, r_tp1, s_tp1))

            # Update counters:
            s_t = s_tp1

            if terminal:
                break

        # print(np.array(transitions, dtype=[('s', float, (1,)), ('a', int, 1), ('r_prime', float, 1), ('s_prime', float, (1,))]))
        # Store the episode:
        episodes.append(np.array(transitions, dtype=[('s', int, 1), ('a', float, 1), ('r_prime', float, 1), ('s_prime', int, 1)]))

    # Save the data for later:
    file_path = 'output/{}/behaviour_policies/{}/runs/{}/data.npz'.format(environment_name, behaviour_policy_name, args.run_num)
    utils.save_data_to_file(episodes, file_path)
