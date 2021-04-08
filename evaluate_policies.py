import os
import gym_puddle
import gym
import random
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src import utils
from src.utils import tqdm_joblib
from src.algorithms.ace import BinaryACE
from src.function_approximation.tile_coder import TileCoder
from joblib import Parallel, delayed


def evaluate_policy(actor, tc, env=None, rng=np.random, num_timesteps=1000, render=False, state=None):
    env = gym.make('MountainCar-v0').unwrapped if env is None else env
    g_t = 0.
    if state is not None:
        indices_t = tc.encode(state)
    else:
        indices_t = tc.encode(env.reset())
    for t in range(num_timesteps):
        a_t = rng.choice(env.action_space.n, p=actor.pi(indices_t))
        s_tp1, r_tp1, terminal, _ = env.step(a_t)
        indices_t = tc.encode(s_tp1)
        g_t += r_tp1
        if terminal:
            break
        if render:
            env.render()
    return g_t


def evaluate_policy_avg_return(actor, tc, env=None, rng=np.random, num_timesteps=1000, render=False):
    env = gym.make('MountainCar-v0').unwrapped if env is None else env
    g_t = 0.
    indices_t = tc.encode(env.reset())
    num_episodes = 0
    for t in range(num_timesteps):
        a_t = rng.choice(env.action_space.n, p=actor.pi(indices_t))
        s_tp1, r_tp1, terminal, _ = env.step(a_t)
        indices_t = tc.encode(s_tp1)
        g_t += r_tp1
        if terminal:
            num_episodes += 1
        if render:
            env.render()
    if num_episodes > 1:
        return g_t/num_episodes
    else:
        return g_t


def evaluate_policies(policies_memmap, performance_memmap, evaluation_run_num, ace_run_num, config_num, policy_num, random_seed):

    if evaluation_run_num == 0:
        performance_memmap[config_num]['parameters'] = policies_memmap[config_num]['parameters']

    # Load the policy to evaluate:
    configuration = policies_memmap[config_num]['parameters']
    weights = policies_memmap[config_num]['policies'][ace_run_num, policy_num]['weights']
    actor = BinaryACE(weights.shape[0], weights.shape[1], 0.)
    actor.theta = weights

    # Handle situations where the learning process diverged:
    if np.any(np.isnan(weights)):
        # If the weights overflowed, assign NaN as return:
        performance_memmap[config_num]['results'][ace_run_num, policy_num, evaluation_run_num] = np.nan
    else:
        # Set up the environment:
        import gym_puddle  # Re-import the puddleworld env in each subprocess or it sometimes isn't found during creation.
        env = gym.make(args.environment).unwrapped
        env.seed(random_seed)
        rng = env.np_random
        if args.objective == 'episodic':
            # Use the environment's start state:
            s_t = env.reset()
        else:
            raise NotImplementedError

        # Configure the tile coder:
        num_tiles_per_dim = configuration['num_tiles_per_dim']
        num_tilings = configuration['num_tilings']
        bias_unit = configuration['bias_unit']
        tc = TileCoder(np.array([env.observation_space.low, env.observation_space.high]).T, num_tiles_per_dim, num_tilings, bias_unit)

        # Write the total rewards received to file:
        performance_memmap[config_num]['results'][ace_run_num, policy_num, evaluation_run_num] = evaluate_policy(actor, tc, env, rng)


if __name__ == '__main__':

    # Parse command line arguments:
    parser = argparse.ArgumentParser(description='A script to evaluate policies on the given environment in parallel.', fromfile_prefix_chars='@', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experiment_name', type=str, default='experiment', help='The directory to read/write experiment files to/from')
    parser.add_argument('--num_evaluation_runs', type=int, default=5, help='The number of times to evaluate each policy')
    parser.add_argument('--max_timesteps', type=int, default=1000, help='The maximum number of timesteps allowed per run')
    parser.add_argument('--random_seed', type=int, default=1944801619, help='The master random seed to use')
    parser.add_argument('--num_cpus', type=int, default=-1, help='The number of cpus to use (-1 means all)')
    parser.add_argument('--objective', type=str, choices=['excursions', 'alternative_life', 'episodic'], default='episodic', help='Determines the state distribution the starting state is sampled from (excursions: behaviour policy, alternative life: target policy, episodic: environment start state.)')
    parser.add_argument('--environment', type=str, default='MountainCar-v0', help='An OpenAI Gym environment string.')
    args = parser.parse_args()

    # Generate the random seed for each run without replacement to prevent the birthday paradox:
    random.seed(args.random_seed)
    random_seeds = random.sample(range(2**32), args.num_evaluation_runs)

    # Save the command line arguments in a format interpretable by argparse:
    experiment_path = Path(args.experiment_name)
    utils.save_args_to_file(args, experiment_path / Path(parser.prog).with_suffix('.args'))

    # Load the learned policies to evaluate:
    policies_memmap = np.lib.format.open_memmap(str(experiment_path / 'policies.npy'), mode='r')
    num_configurations, num_ace_runs, num_policies = policies_memmap['policies'].shape

    parameters_dtype = policies_memmap['parameters'].dtype
    performance_dtype = np.dtype([
        ('parameters', parameters_dtype),
        ('results', float, (num_ace_runs, num_policies, args.num_evaluation_runs))
    ])

    # Create the memmapped array of results to be populated in parallel:
    performance_memmap_path = str(experiment_path / '{}_performance.npy'.format(args.objective))
    performance_memmap = np.lib.format.open_memmap(performance_memmap_path, shape=(num_configurations,), dtype=performance_dtype, mode='w+')

    # Evaluate the learned policies in parallel:
    with tqdm_joblib(tqdm(total=args.num_evaluation_runs * num_ace_runs * num_configurations * num_policies)) as progress_bar:
        Parallel(n_jobs=args.num_cpus, verbose=0)(
            delayed(evaluate_policies)(
                policies_memmap, performance_memmap,
                evaluation_run_num, ace_run_num, config_num, policy_num, random_seed
            )
            for evaluation_run_num, random_seed in enumerate(random_seeds)
            for ace_run_num in range(num_ace_runs)
            for config_num in range(num_configurations)
            for policy_num in range(num_policies)
        )
