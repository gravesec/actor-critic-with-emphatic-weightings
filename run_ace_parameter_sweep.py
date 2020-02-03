import gym
import gym_puddle
import argparse
import numpy as np
from pathlib import Path
from run_ace import run_ace
from src import utils
from src.algorithms.ace import BinaryACE
from src.algorithms.tdc import BinaryTDC
from src.function_approximation.tile_coder import TileCoder
from joblib import Parallel, delayed


def run_ace_sweep(policies_memmap, experience_memmap,
            run_num,
            gamma_idx, gamma,
            alpha_a_idx, alpha_a,
            alpha_c_idx, alpha_c,
            alpha_c2_idx, alpha_c2,
            lambda_c_idx, lambda_c,
            eta_idx, eta,
            num_tiles_idx, num_tiles,
            num_tilings_idx, num_tilings,
            num_features_idx, num_features,
            bias_unit_idx, bias_unit):

    transitions = experience_memmap[run_num]  # Get the run of experience to learn from.
    policies = run_ace(transitions, gamma, alpha_a, alpha_c, alpha_c2, lambda_c, eta, num_tiles, num_tilings, num_features, bias_unit)  # Run ACE on the experience.
    policies_memmap[run_num, gamma_idx, alpha_a_idx, alpha_c_idx, alpha_c2_idx, lambda_c_idx, eta_idx, num_tiles_idx, num_tilings_idx, num_features_idx, bias_unit_idx] = policies  # Store the learned policies in the memmap.


if __name__ == '__main__':

    # Parse command line arguments:
    parser = argparse.ArgumentParser(description='A script to run ACE (Actor-Critic with Emphatic weightings) in parallel.', fromfile_prefix_chars='@', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experiment_name', type=str, default='experiment', help='The directory to read/write experiment files to/from')
    parser.add_argument('--checkpoint_interval', type=int, default=12500, help='The number of timesteps after which to save the learned policy.')
    parser.add_argument('--num_cpus', type=int, default=-1, help='The number of cpus to use (-1 for all).')
    parser.add_argument('--backend', type=str, choices=['loky', 'threading'], default='loky', help='The backend to use (\'loky\' for processes or \'threading\' for threads; always use \'loky\' because Python threading is terrible).')
    parser.add_argument('--interest_function', type=str, default='lambda s, g=1: 1.', help='Interest function to use. Example: \'lambda s, g=1: 1. if g==0. else 0.\' (episodic interest function)')
    parser.add_argument('--behaviour_policy', type=str, default='lambda s: np.ones(env.action_space.n)/env.action_space.n', help='Policy to use. Default is uniform random. Another Example: \'lambda s: np.array([.9, .05, .05]) if s[1] < 0 else np.array([.05, .05, .9]) \' (energy pumping policy w/ 15 percent randomness)')
    parser.add_argument('--environment', type=str, choices=['MountainCar-v0', 'Acrobot-v1', 'PuddleWorld-v0'], default='MountainCar-v0', help='The environment to run ACE on.')
    parser.add_argument('--gamma', '--discount_rate', type=float, nargs='+', help='Discount rates.')
    parser.add_argument('--alpha_a', '--actor_step_sizes', type=float, nargs='+', help='Step sizes for the actor.')
    parser.add_argument('--alpha_c', '--critic_step_sizes', type=float, nargs='+', help='Step sizes for the critic.')
    parser.add_argument('--alpha_c2', '--critic_step_sizes_2', type=float, nargs='+', help='Step sizes for the second set of weights in the GTD critic.')
    parser.add_argument('--lambda_c', '--critic_trace_decay_rates', type=float, nargs='+', help='Trace decay rates for the critic.')
    parser.add_argument('--eta', '--offpac_ace_tradeoff', type=float, nargs='+', help='Values for the parameter that interpolates between OffPAC (0) and ACE (1).')
    parser.add_argument('--num_tiles', type=int, nargs='+', help='The number of tiles to use in the tile coder.')
    parser.add_argument('--num_tilings', type=int, nargs='+', help='The number of tilings to use in the tile coder.')
    parser.add_argument('--num_features', type=int, help='The number of features to use in the tile coder.')
    parser.add_argument('--bias_unit', type=int, nargs='+', help='Whether or not to include a bias unit in the tile coder.')
    args = parser.parse_args()

    # Save the command line arguments in a format interpretable by argparse:
    experiment_path = Path(args.experiment_name)
    utils.save_args_to_file(args, experiment_path / Path(parser.prog).with_suffix('.args'))

    # Load the input data as a memmap to prevent a copy being loaded into memory in each sub-process:
    experience_memmap = np.lib.format.open_memmap(str(experiment_path / 'experience.npy'), mode='r')
    num_runs, num_timesteps = experience_memmap.shape

    # Create the memmapped array of learned policies that will be populated in parallel:
    env = gym.make(args.environment).env  # Make a dummy env to get shape info.
    num_policies = num_timesteps // args.checkpoint_interval + 1
    max_num_features = max(args.num_features)
    policy_dtype = np.dtype(
        [
            ('timesteps', int),
            ('weights', float, (env.action_space.n, max_num_features))
        ]
    )
    configuration_dtype = np.dtype(
        [
            ('gamma', float),
            ('alpha_a', float),
            ('alpha_c', float),
            ('alpha_c2', float),
            ('lambda_c', float),
            ('eta', float),
            ('num_tiles', int),
            ('num_tilings', int),
            ('num_features', int),
            ('bias_unit', bool),
            ('policies', policy_dtype, num_policies)
        ]
    )
    policies_memmap = np.lib.format.open_memmap(str(experiment_path / 'policies.npy'), shape=(num_runs, len(args.gamma), len(args.alpha_a), len(args.alpha_c), len(args.alpha_c2), len(args.lambda_c), len(args.eta), len(args.num_tiles), len(args.num_tilings), len(args.num_features), len(args.bias_unit)), dtype=configuration_dtype, mode='w+')

    # Run ACE for all combinations of the given parameters in parallel:
    Parallel(n_jobs=args.num_cpus, verbose=51)(
        delayed(run_ace_sweep)(
            policies_memmap, experience_memmap,
            run_num,
            gamma_idx, gamma,
            alpha_a_idx, alpha_a,
            alpha_c_idx, alpha_c,
            alpha_c2_idx, alpha_c2,
            lambda_c_idx, lambda_c,
            eta_idx, eta,
            num_tiles_idx, num_tiles,
            num_tilings_idx, num_tilings,
            num_features_idx, num_features,
            bias_unit_idx, bias_unit
        )
        for run_num in range(num_runs)
        for gamma_idx, gamma in enumerate(args.gamma)
        for alpha_a_idx, alpha_a in enumerate(args.alpha_a)
        for alpha_c_idx, alpha_c in enumerate(args.alpha_c)
        for alpha_c2_idx, alpha_c2 in enumerate(args.alpha_c2)
        for lambda_c_idx, lambda_c in enumerate(args.lambda_c)
        for eta_idx, eta in enumerate(args.eta)
        for num_tiles_idx, num_tiles in enumerate(args.num_tiles)
        for num_tilings_idx, num_tilings in enumerate(args.num_tilings)
        for num_features_idx, num_features in enumerate(args.num_features)
        for bias_unit_idx, bias_unit in enumerate(args.bias_unit)
    )
