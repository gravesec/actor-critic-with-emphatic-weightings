import gym
import random
import argparse
import numpy as np
from pathlib import Path
from src import utils
from src.algorithms.ace import BinaryACE
from src.function_approximation.tile_coder import TileCoder
from joblib import Parallel, delayed


def evaluate_policy(actor, tc, env=None, rng=np.random, num_timesteps=1000, render=False):
    env = gym.make('MountainCar-v0').env if env is None else env
    g_t = 0.
    indices_t = tc.indices(env.reset())
    for t in range(num_timesteps):
        a_t = rng.choice(env.action_space.n, p=actor.pi(indices_t))
        s_tp1, r_tp1, terminal, _ = env.step(a_t)
        indices_t = tc.indices(s_tp1)
        g_t += r_tp1
        if terminal:
            break
        if render:
            env.render()
    return g_t


def evaluate_policies(performance_memmap, policies_memmap, evaluation_run_num, ace_run_num, config_num, policy_num, random_seed):
    # Load the policy to evaluate:
    configuration = policies_memmap[ace_run_num, config_num]
    num_features = configuration['num_features']
    policy = configuration['policies'][policy_num]
    weights = policy['weights'][:, :num_features]  # trim potential padding.
    actor = BinaryACE(weights.shape[0], weights.shape[1])
    actor.theta = weights

    # Set up the environment:
    import gym_puddle  # Re-import the puddleworld env in each subprocess or it sometimes isn't found during creation.
    env = gym.make(args.environment).env
    env.seed(random_seed)
    rng = env.np_random
    if args.objective == 'episodic':
        # Use the environment's start state:
        s_t = env.reset()
    else:
        raise NotImplementedError

    # Configure the tile coder:
    num_tiles = configuration['num_tiles']
    num_tilings = configuration['num_tilings']
    bias_unit = configuration['bias_unit']
    tc = TileCoder(env.observation_space.low, env.observation_space.high, num_tiles, num_tilings, num_features, bias_unit)

    # Write the total rewards received to file:
    performance_memmap[evaluation_run_num, ace_run_num, config_num, policy_num] = evaluate_policy(actor, tc, env, rng)


if __name__ == '__main__':

    # Parse command line arguments:
    parser = argparse.ArgumentParser(description='A script to evaluate policies on the Mountain Car environment in parallel.', fromfile_prefix_chars='@', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experiment_name', type=str, default='experiment', help='The directory to read/write experiment files to/from')
    parser.add_argument('--num_evaluation_runs', type=int, default=5, help='The number of times to evaluate each policy')
    parser.add_argument('--max_timesteps', type=int, default=1000, help='The maximum number of timesteps allowed per run')
    parser.add_argument('--random_seed', type=int, default=1944801619, help='The master random seed to use')
    parser.add_argument('--num_cpus', type=int, default=-1, help='The number of cpus to use (-1 means all)')
    parser.add_argument('--backend', type=str, choices=['loky', 'threading'], default='loky', help='The backend to use (\'loky\' for processes or \'threading\' for threads; always use \'loky\' because Python threading is terrible).')
    parser.add_argument('--objective', type=str, choices=['excursions', 'alternative_life', 'episodic'], default='episodic', help='Determines the state distribution the starting state is sampled from (excursions: behaviour policy, alternative life: target policy, episodic: environment start state.)')
    parser.add_argument('--environment', type=str, choices=['MountainCar-v0', 'Acrobot-v1', 'PuddleWorld-v0'], default='MountainCar-v0', help='The environment to generate experience from.')
    args = parser.parse_args()

    # Generate the random seed for each run without replacement to prevent the birthday problem:
    random.seed(args.random_seed)
    random_seeds = random.sample(range(2**32), args.num_evaluation_runs)

    # Save the command line arguments in a format interpretable by argparse:
    experiment_path = Path(args.experiment_name)
    utils.save_args_to_file(args, experiment_path / Path(parser.prog).with_suffix('.args'))

    # Load the learned policies to evaluate:
    policies_memmap = np.lib.format.open_memmap(str(experiment_path / 'policies.npy'), mode='r')
    num_ace_runs, num_configurations, num_policies = policies_memmap['policies'].shape

    # Create the memmapped array of results to be populated in parallel:
    performance_memmap = np.lib.format.open_memmap(str(experiment_path / '{}_performance.npy'.format(args.objective)), shape=(args.num_evaluation_runs, num_ace_runs, num_configurations, num_policies), dtype=float, mode='w+')

    # Evaluate the learned policies in parallel:
    Parallel(n_jobs=args.num_cpus, verbose=51, backend=args.backend)(
        delayed(evaluate_policies)(
            performance_memmap, policies_memmap,
            evaluation_run_num, ace_run_num, config_num, policy_num, random_seed
        )
        for evaluation_run_num, random_seed in enumerate(random_seeds)
        for ace_run_num in range(num_ace_runs)
        for config_num in range(num_configurations)
        for policy_num in range(num_policies)
    )

    # Close the memmap file:
    del policies_memmap
    del performance_memmap
