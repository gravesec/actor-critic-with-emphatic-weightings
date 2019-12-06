import gym
import random
import argparse
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed

from src import utils
from src.algorithms.ace import BinaryACE
from src.function_approximation.tile_coder import TileCoder
from mountain_car.generate_experience import num_actions, min_state_values, max_state_values


# TODO: implement our own mountain car environment (https://en.wikipedia.org/wiki/Mountain_car_problem) because openai's is slow, stateful, and has bizarre decisions built in like time limits and the inability to get properties of an environment without creating an instantiation of it.


def evaluate_policy(performance_memmap, policies_memmap, evaluation_run_num, ace_run_num, config_num, policy_num, random_seed):
    # Load the policy to evaluate:
    configuration = policies_memmap[ace_run_num, config_num]
    num_features = configuration['num_features']
    policy = configuration['policies'][policy_num]
    weights = policy['weights'][:, :num_features]  # trim potential padding.
    agent = BinaryACE(weights.shape[0], weights.shape[1])
    agent.theta = weights

    # Configure the tile coder:
    num_tiles = configuration['num_tiles']
    num_tilings = configuration['num_tilings']
    bias_unit = configuration['bias_unit']
    tc = TileCoder(min_state_values, max_state_values, [num_tiles, num_tiles], num_tilings, num_features, bias_unit)

    # Set up the environment:
    env = gym.make('MountainCar-v0').env  # Get the underlying environment object to bypass the built-in timestep limit.
    env.seed(random_seed)
    rng = env.np_random
    if args.objective == 'episodic':
        # Use the mountain_car start state:
        s_t = env.reset()
    elif args.objective == 'excursions':
        # Sample a state from the behaviour policy's state distribution:
        # TODO: sample a state from the generated_experience.npy file? Or generate new experience?
        raise NotImplementedError
    else:
        raise NotImplementedError

    # Evaluate the policy:
    g_t = 0.
    indices_t = tc.indices(s_t)
    for t in range(args.max_timesteps):
        a_t = rng.choice(env.action_space.n, p=agent.pi(indices_t))
        s_tp1, r_tp1, terminal, _ = env.step(a_t)
        indices_t = tc.indices(s_tp1)
        g_t += r_tp1
        if terminal:
            break
        # env.render()

    # Write the total rewards received to file:
    performance_memmap[evaluation_run_num, ace_run_num, config_num, policy_num] = g_t


if __name__ == '__main__':

    # Parse command line arguments:
    parser = argparse.ArgumentParser(description='A script to evaluate policies on the Mountain Car environment in parallel.', fromfile_prefix_chars='@', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experiment_name', type=str, default='experiment', help='The directory to read/write experiment files to/from')
    parser.add_argument('--num_evaluation_runs', type=int, default=5, help='The number of times to evaluate each policy')
    parser.add_argument('--max_timesteps', type=int, default=1000, help='The maximum number of timesteps allowed per run')
    parser.add_argument('--random_seed', type=int, default=1944801619, help='The master random seed to use')
    parser.add_argument('--num_cpus', type=int, default=-1, help='The number of cpus to use (-1 means all)')
    parser.add_argument('--backend', type=str, choices=['loky', 'threading'], default='loky', help='The backend to use (\'loky\' for processes or \'threading\' for threads; always use \'loky\' because Python threading is terrible).')
    parser.add_argument('--objective', type=str, choices=['excursions', 'alternative_life', 'episodic'], default='episodic', help='Determines the state distribution the starting state is sampled from (excursions: behaviour policy, alternative life: target policy, episodic: mountain car start state.)')
    args = parser.parse_args()

    # Generate the random seed for each run without replacement:
    random.seed(args.random_seed)
    random_seeds = random.sample(range(2**32), args.num_evaluation_runs)

    # Save the command line arguments in a format interpretable by argparse:
    experiment_path = Path(args.experiment_name)
    utils.save_args_to_file(args, experiment_path / Path(parser.prog).with_suffix('.args'))

    # Load the learned policies to evaluate:
    policies_memmap = np.lib.format.open_memmap(str(experiment_path / 'policies.npy'), mode='r')
    num_ace_runs, num_configurations, num_policies = policies_memmap['policies'].shape

    # TODO: change this to include configuration information!
    # Create the memmapped array of results to be populated in parallel:
    performance_memmap = np.lib.format.open_memmap(str(experiment_path / '{}_performance.npy'.format(args.objective)), shape=(args.num_evaluation_runs, num_ace_runs, num_configurations, num_policies), dtype=float, mode='w+')

    # Evaluate the learned policies in parallel:
    Parallel(n_jobs=args.num_cpus, verbose=10, backend=args.backend)(
        delayed(evaluate_policy)(
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
