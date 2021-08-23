import os
import gym
import gym_puddle
import random
import argparse
import numpy as np
from src import utils
from src.environments.pw import puddleworld
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed


def generate_experience(experience, run_num, random_seed):
    # Check if this run of experience has already been generated:
    if np.count_nonzero(experience[run_num]) != 0:
        return

    # Initialize the environment:
    if args.environment == 'pw':
        env = puddleworld()
    else:
        import gym_puddle
        env = gym.make(args.environment).unwrapped
    env.seed(random_seed)
    rng = env.np_random

    # Create the behaviour policy:
    mu = eval(args.behaviour_policy, {'np': np, 'env': env})  # Give the eval'd function access to some objects.

    # Generate the required timesteps of experience:
    s_t = env.reset()
    a_t = rng.choice(env.action_space.n, p=mu(s_t))
    for t in range(args.num_timesteps):
        # Take action a_t, observe next state s_tp1 and reward r_tp1:
        s_tp1, r_tp1, terminal, _ = env.step(a_t)

        # The agent is reset to a starting state after a terminal transition:
        if terminal:
            s_tp1 = env.reset()

        a_tp1 = rng.choice(env.action_space.n, p=mu(s_t))

        # Add the transition:
        experience[run_num, t] = (s_t, a_t, r_tp1, s_tp1, a_tp1, terminal)

        # Update temporary variables:
        s_t = s_tp1
        a_t = a_tp1


def generate_experience_test(experience, run_num, random_seed):
    # Check if this run of experience has already been generated:
    if np.count_nonzero(experience[run_num]) != 0:
        return

    # Initialize the environment:
    if args.environment == 'pw':
        env = puddleworld()
    else:
        import gym_puddle  # Re-import the puddleworld env in each subprocess or it sometimes isn't found during creation.
        env = gym.make(args.environment).unwrapped
    env.seed(random_seed)
    rng = env.np_random

    # Create the behaviour policy:
    mu = eval(args.behaviour_policy, {'np': np, 'env': env})  # Give the eval'd function access to some objects.

    # Generate the required timesteps of experience:
    s_t = env.reset()
    a_t = rng.choice(env.action_space.n, p=mu(s_t))
    t = 0
    step = 0
    while t != args.num_timesteps:
        # Take action a_t, observe next state s_tp1 and reward r_tp1:
        s_tp1, r_tp1, terminal, _ = env.step(a_t)

        # The agent is reset to a starting state after a terminal transition:
        if terminal:
            s_tp1 = env.reset()

        a_tp1 = rng.choice(env.action_space.n, p=mu(s_t))

        step += 1
        #adds every 1000th state as an evaluation state
        if step % 1000 == 0:
            # Add the transition:
            experience[run_num, t] = (s_t,)
            step = 0
            t += 1

        # Update temporary variables:
        s_t = s_tp1
        a_t = a_tp1


if __name__ == '__main__':

    # Parse command line arguments:
    parser = argparse.ArgumentParser(description='A script to generate experience from the specified behaviour policy on the specified environment in parallel.', fromfile_prefix_chars='@', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output_dir', type=str, default='experiment', help='The directory to read/write experiment files to/from')
    parser.add_argument('--num_runs', type=int, default=5, help='The number of independent runs of experience to generate')
    parser.add_argument('--num_timesteps', type=int, default=100000, help='The number of timesteps of experience to generate per run')
    parser.add_argument('--random_seed', type=int, default=2937573853, help='The master random seed to use')
    parser.add_argument('--num_cpus', type=int, default=-1, help='The number of cpus to use (-1 means all)')
    parser.add_argument('--behaviour_policy', type=str, default='lambda s: np.ones(env.action_space.n)/env.action_space.n', help='Policy to use. Default is uniform random. Another Example: \'lambda s: np.array([.9, .05, .05]) if s[1] < 0 else np.array([.05, .05, .9]) \' (energy pumping policy w/ 15 percent randomness)')
    parser.add_argument('--environment', type=str, default='MountainCar-v0', help='An OpenAI Gym environment string.')
    parser.add_argument('--test_data', type=int, choices=[0, 1], default=0, help='Whether generating transition data or excursions evaluation start state.')
    args = parser.parse_args()

    # Generate the random seed for each run without replacement to prevent the birthday problem:
    random.seed(args.random_seed)
    random_seeds = random.sample(range(2**32), args.num_runs)

    # Save the command line arguments in a format interpretable by argparse:
    output_dir = Path(args.output_dir)

    if args.test_data:
        #we consider 50 different start states for evaluation
        args.num_timesteps = 50

        utils.save_args_to_file(args, output_dir / 'experience_test.args')

        # Create the memmapped structured array of experience to be populated in parallel:
        if args.environment == 'pw':
            env = puddleworld()
        else:
            env = gym.make(args.environment).unwrapped  # Make a dummy env to get shape info for observations.
        transition_dtype = np.dtype([
            ('s_t', float, env.observation_space.shape)
        ])


        experience_memmap_path = str(output_dir / 'experience_test.npy')
        if os.path.isfile(experience_memmap_path):
            experience_memmap = np.lib.format.open_memmap(experience_memmap_path, mode='r+')
        else:
            experience_memmap = np.lib.format.open_memmap(experience_memmap_path, shape=(args.num_runs, args.num_timesteps), dtype=transition_dtype, mode='w+')

        # Generate the experience in parallel:
        with utils.tqdm_joblib(tqdm(total=args.num_runs)) as progress_bar:
            Parallel(n_jobs=args.num_cpus, verbose=0)(
                delayed(generate_experience_test)(experience_memmap, run_num, random_seed)
                for run_num, random_seed in enumerate(random_seeds)
            )
    else:
        utils.save_args_to_file(args, output_dir / 'experience.args')

        # Create the memmapped structured array of experience to be populated in parallel:
        if args.environment == 'pw':
            env = puddleworld()
        else:
            env = gym.make(args.environment).unwrapped  # Make a dummy env to get shape info for observations.
        transition_dtype = np.dtype([
            ('s_t', float, env.observation_space.shape),
            ('a_t', int),
            ('r_tp1', float),
            ('s_tp1', float, env.observation_space.shape),
            ('a_tp1', int),
            ('terminal', bool)
        ])
        experience_memmap_path = str(output_dir / 'experience.npy')
        if os.path.isfile(experience_memmap_path):
            experience_memmap = np.lib.format.open_memmap(experience_memmap_path, mode='r+')
        else:
            experience_memmap = np.lib.format.open_memmap(experience_memmap_path, shape=(args.num_runs, args.num_timesteps), dtype=transition_dtype, mode='w+')

        # Generate the experience in parallel:
        with utils.tqdm_joblib(tqdm(total=args.num_runs)) as progress_bar:
            Parallel(n_jobs=args.num_cpus, verbose=0)(
                delayed(generate_experience)(experience_memmap, run_num, random_seed)
                for run_num, random_seed in enumerate(random_seeds)
            )
