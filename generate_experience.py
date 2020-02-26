import os
import gym
import gym_puddle
import random
import argparse
import numpy as np
from src import utils
from pathlib import Path
from joblib import Parallel, delayed


def generate_experience(experience, run_num, random_seed):
    # Check if this run of experience has already been generated:
    if np.count_nonzero(experience[run_num]) > 0:
        return

    # Initialize the environment:
    import gym_puddle  # Re-import the puddleworld env in each subprocess or it sometimes isn't found during creation.
    env = gym.make(args.environment).env
    env.seed(random_seed)
    rng = env.np_random

    # Create the behaviour policy:
    mu = eval(args.behaviour_policy, {'np': np, 'env': env})  # Give the eval'd function access to some objects.

    # Generate the required timesteps of experience:
    s_t = env.reset()
    for t in range(args.num_timesteps):
        # Select an action:
        mu_t = mu(s_t)
        a_t = rng.choice(env.action_space.n, p=mu_t)

        # Take action a_t, observe next state s_tp1 and reward r_tp1:
        s_tp1, r_tp1, terminal, _ = env.step(a_t)

        # The agent is reset to a starting state after a terminal transition:
        if terminal:
            s_tp1 = env.reset()

        # Add the transition:
        experience[run_num, t] = (s_t, a_t, r_tp1, s_tp1, terminal)

        # Update temporary variables:
        s_t = s_tp1


if __name__ == '__main__':

    # Parse command line arguments:
    parser = argparse.ArgumentParser(description='A script to generate experience from the specified behaviour policy on the specified environment in parallel.', fromfile_prefix_chars='@', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experiment_name', type=str, default='experiment', help='The directory to read/write experiment files to/from')
    parser.add_argument('--num_runs', type=int, default=30, help='The number of independent runs of experience to generate')
    parser.add_argument('--num_timesteps', type=int, default=100000, help='The number of timesteps of experience to generate per run')
    parser.add_argument('--random_seed', type=int, default=3139378768, help='The master random seed to use')
    parser.add_argument('--num_cpus', type=int, default=-1, help='The number of cpus to use (-1 means all)')
    parser.add_argument('--backend', type=str, choices=['loky', 'threading'], default='loky', help='The backend to use (\'loky\' for processes or \'threading\' for threads; always use \'loky\' because Python threading is terrible).')
    parser.add_argument('--verbosity', type=int, default=51, help='Controls how verbose the joblib progress reporting is. 0 for none, 51 for all iterations to stdout.')
    parser.add_argument('--behaviour_policy', type=str, default='lambda s: np.ones(env.action_space.n)/env.action_space.n', help='Policy to use. Default is uniform random. Another Example: \'lambda s: np.array([.9, .05, .05]) if s[1] < 0 else np.array([.05, .05, .9]) \' (energy pumping policy w/ 15 percent randomness)')
    parser.add_argument('--environment', type=str, choices=['MountainCar-v0', 'Acrobot-v1', 'PuddleWorld-v0'], default='MountainCar-v0', help='The environment to generate experience from.')
    args = parser.parse_args()

    # Generate the random seed for each run without replacement to prevent the birthday problem:
    random.seed(args.random_seed)
    random_seeds = random.sample(range(2**32), args.num_runs)

    # Save the command line arguments in a format interpretable by argparse:
    experiment_path = Path(args.experiment_name)
    utils.save_args_to_file(args, experiment_path / Path(parser.prog).with_suffix('.args'))

    # Create the memmapped structured array of experience to be populated in parallel:
    env = gym.make(args.environment).env  # Make a dummy env to get shape info for observations.
    transition_dtype = np.dtype([('s_t', float, env.observation_space.shape), ('a_t', int), ('r_tp1', float), ('s_tp1', float, env.observation_space.shape), ('terminal', bool)])
    experience_memmap_path = str(experiment_path / 'experience.npy')
    if os.path.isfile(experience_memmap_path):
        experience_memmap = np.lib.format.open_memmap(experience_memmap_path, shape=(args.num_runs, args.num_timesteps), dtype=transition_dtype, mode='r+')
    else:
        experience_memmap = np.lib.format.open_memmap(experience_memmap_path, shape=(args.num_runs, args.num_timesteps), dtype=transition_dtype, mode='w+')

    # Generate the experience in parallel:
    Parallel(n_jobs=args.num_cpus, verbose=args.verbosity, backend=args.backend)(
        delayed(generate_experience)(
            experience_memmap, run_num, random_seed
        )
        for run_num, random_seed in enumerate(random_seeds)
    )
