import gym
import random
import argparse
import numpy as np

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from src import utils
from pathlib import Path
from joblib import Parallel, delayed
from mc_env import MountainCar

# TODO: Figure out how to do checkpointing (i.e. keep track of progress via a memmap so if the process gets killed it can pick up where it left off).
# TODO: Figure out how to append to a memmap in case we want to do more runs later on (we might get this without any extra work with checkpointing).
# TODO: Consider implementing our own mountain car environment (https://en.wikipedia.org/wiki/Mountain_car_problem) because openai's is slow, stateful, and has bizarre decisions built in like time limits and the inability to get properties of an environment without creating an instantiation of it.
num_actions = 3
min_state_values = np.array([-1.2, -0.07])
max_state_values = np.array([0.6, 0.07])


def generate_experience(experience, run_num, random_seed):

    # Initialize the environment:
    # env = gym.make('MountainCar-v0').env  # Get the underlying environment object to bypass the built-in timestep limit.
    env = MountainCar(normalized=False)  # Get the underlying environment object to bypass the built-in timestep limit.

    # Configure random state for the run:
    env.seed(random_seed)
    rng = env.np_random

    # Create the behaviour policy:
    mu = eval(args.behaviour_policy, {'np': np})  # Give the eval'd function access to numpy.

    # Generate the required timesteps of experience:
    s_t = env.reset()
    for t in range(args.num_timesteps):

        # Select an action:
        mu_t = mu(s_t)
        # a_t = rng.choice(env.action_space.n, p=mu_t)
        a_t = rng.choice(env.num_action, p=mu_t)

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
    parser = argparse.ArgumentParser(description='A script to generate experience from a uniform random policy on the Mountain Car environment in parallel.', fromfile_prefix_chars='@', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experiment_name', type=str, default='experiment', help='The directory to read/write experiment files to/from')
    parser.add_argument('--num_runs', type=int, default=30, help='The number of independent runs of experience to generate')
    parser.add_argument('--num_timesteps', type=int, default=200000, help='The number of timesteps of experience to generate per run')
    parser.add_argument('--random_seed', type=int, default=3139378768, help='The master random seed to use')
    parser.add_argument('--num_cpus', type=int, default=-1, help='The number of cpus to use (-1 means all)')
    parser.add_argument('--backend', type=str, choices=['loky', 'threading'], default='loky', help='The backend to use (\'loky\' for processes or \'threading\' for threads; always use \'loky\' because Python threading is terrible).')
    parser.add_argument('--behaviour_policy', type=str, default='lambda s: np.array([1/3, 1/3, 1/3])', help='Policy to use. Example: \'lambda s: np.array([.9, .05, .05]) if s[1] < 0 else np.array([.05, .05, .9]) \' (energy pumping policy w/ 15 percent randomness)')
    args = parser.parse_args()

    # Generate the random seed for each run without replacement to prevent the birthday problem:
    random.seed(args.random_seed)
    random_seeds = random.sample(range(2**32), args.num_runs)

    # Save the command line arguments in a format interpretable by argparse:
    experiment_path = Path(args.experiment_name)
    utils.save_args_to_file(args, experiment_path / Path(parser.prog).with_suffix('.args'))

    # Create the memmapped structured array of experience to be populated in parallel:
    transition_dtype = np.dtype([('s_t', float, (2,)), ('a_t', int), ('r_tp1', float), ('s_tp1', float, (2,)), ('terminal', bool)])
    experience = np.lib.format.open_memmap(str(experiment_path / 'experience.npy'), shape=(args.num_runs, args.num_timesteps), dtype=transition_dtype, mode='w+')

    # Generate the experience in parallel:
    Parallel(n_jobs=args.num_cpus, verbose=10, backend=args.backend)(
        delayed(generate_experience)(
            experience, run_num, random_seed
        )
        for run_num, random_seed in enumerate(random_seeds)
    )

    # Close the memmap file:
    del experience
