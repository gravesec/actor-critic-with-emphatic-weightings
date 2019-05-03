import os
import gym
import argparse
import numpy as np
from gym.utils import seeding
from joblib import Parallel, delayed
from src import utils


output_directory_name = 'data'
environment_name = 'MountainCar-v0'
behaviour_policy_name = 'random_policy'
num_actions = 3
transition_dtype = [('s_t', float, (2,)), ('gamma_t', float, 1), ('a_t', int, 1), ('s_tp1', float, (2,)), ('r_tp1', float, 1), ('gamma_tp1', float, 1)]


def generate_experience(experience, run_num, num_timesteps, random_seed):
    
    # Initialize the environment:
    env = gym.make(environment_name).env  # Get the underlying environment object to bypass the built-in timestep limit.

    # Configure random state for the run:
    env.seed(random_seed)
    rng = env.np_random
    
    # Generate the required timesteps of experience:
    transitions = np.empty(num_timesteps, dtype=transition_dtype)
    s_t = env.reset()
    gamma_t = 0.
    for t in range(num_timesteps):

        # Select an action:
        a_t = rng.choice(env.action_space.n)  # equiprobable random policy

        # Take action a_t, observe next state s_tp1 and reward r_tp1:
        s_tp1, r_tp1, terminal, _ = env.step(a_t)
        gamma_tp1 = not terminal

        # The agent is reset to a starting state after a terminal transition:
        if terminal:
            s_tp1 = env.reset()

        # Add the transition:
        transitions[t] = (s_t, gamma_t, a_t, s_tp1, r_tp1, gamma_tp1)

        # Update temporary variables:
        s_t = s_tp1
        gamma_t = gamma_tp1
    
    # Write the generated transitions to file:
    experience[run_num] = transitions


if __name__ == '__main__':

    # Parse command line arguments:
    parser = argparse.ArgumentParser(description='A script to generate experience from a uniform random policy on the Mountain Car environment in parallel.', fromfile_prefix_chars='@')
    parser.add_argument('--num_runs', type=int, default=5, help='The number of runs of experience to generate')
    parser.add_argument('--num_timesteps', type=int, default=100000, help='The number of timesteps of experience to generate per run')
    parser.add_argument('--random_seeds', type=int, nargs='*', default=None, help='The random seeds to use for the corresponding runs')
    parser.add_argument('--num_cpus', type=int, default=-1, help='The number of cpus to use')
    parser.add_argument('--experiment_name', default='experiment1', help='The directory (within "{}") to write files to'.format(os.path.join(output_directory_name, environment_name)))
    parser.add_argument('--args_file', default='generate_experience.args', help='Name of the file to store the command line arguments in')
    args = parser.parse_args()

    # If no random seeds were given, generate them in a reasonable way:
    if args.random_seeds is None:
        args.random_seeds = [seeding.hash_seed(seeding.create_seed(None)) for _ in range(args.num_runs)]
    elif len(args.random_seeds) != args.num_runs:
        parser.error('the number of random seeds should be equal to the number of runs')

    # Create the output directory:
    output_directory = os.path.join(output_directory_name, environment_name, args.experiment_name)
    os.makedirs(output_directory, exist_ok=True)

    # Write the command line arguments to a file (for reproducibility):
    args_file_path = os.path.join(output_directory, args.args_file)
    utils.save_args_to_file(args, args_file_path)

    # Create the memmapped array of experience to be populated in parallel:
    experience_directory = os.path.join(output_directory, behaviour_policy_name)
    os.makedirs(experience_directory, exist_ok=True)
    experience_file_path = os.path.join(experience_directory, 'experience.npy')
    experience = np.memmap(experience_file_path, shape=(args.num_runs, args.num_timesteps), dtype=transition_dtype, mode='w+')

    # Generate the experience in parallel:
    Parallel(n_jobs=args.num_cpus, verbose=10)(
        delayed(generate_experience)(experience, run_num, args.num_timesteps, random_seed) for run_num, random_seed in
        enumerate(args.random_seeds))