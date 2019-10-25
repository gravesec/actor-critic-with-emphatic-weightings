import argparse
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from src.algorithms.ace import ACE
from src.algorithms.toetd import BinaryTOETD
from src.function_approximation.tile_coder import TileCoder
from mountain_car.generate_experience import num_actions, min_state_values, max_state_values


def run_ace_sweep(policies, experience, num_timesteps, checkpoint_interval, num_features, gamma,
                run_num,
                alpha_a_idx, alpha_a,
                alpha_c_idx, alpha_c,
                lambda_c_idx, lambda_c,
                eta_idx, eta,
                num_tiles_idx, num_tiles,
                num_tilings_idx, num_tilings,
                bias_unit_idx, bias_unit):

    # Configure the agent:
    tc = TileCoder(min_state_values, max_state_values, [num_tiles, num_tiles], num_tilings, num_features, bias_unit)
    actor = ACE(num_actions, num_features)
    critic = BinaryTOETD(num_features, 1., alpha_c)

    # Process the experience:
    for t in range(num_timesteps):
        pass

if __name__ == '__main__':

    # Parse command line arguments:
    parser = argparse.ArgumentParser(description='A script to run ACE (Actor-Critic with Emphatic weightings) in parallel.', fromfile_prefix_chars='@')
    parser.add_argument('--experiment_name', default='experiment', help='The directory to read/write experiment files from/to.')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='The number of timesteps after which to save the learned policy.')
    parser.add_argument('--num_cpus', type=int, default=-1, help='The number of cpus to use (-1 for all).')
    parser.add_argument('--backend', type=str, choices=['loky', 'threading'], default='loky', help='The backend to use (\'loky\' for processes or \'threading\' for threads; always use \'loky\').')
    parser.add_argument('--gamma', '--discount_rate', type=float, default=[1.], help='Discount rate.')
    parser.add_argument('--alpha_a', '--actor_step_sizes', type=float, nargs='+', help='Step sizes for the actor.')
    parser.add_argument('--alpha_c', '--critic_step_sizes', type=float, nargs='+', help='Step sizes for the critic.')
    parser.add_argument('--lambda_c', '--critic_trace_decay_rates', type=float, nargs='+', help='Trace decay rates for the critic.')
    parser.add_argument('--eta', '--offpac_ace_tradeoff', type=float, nargs='+', help='Values for the parameter that interpolates between OffPAC (0) and ACE (1).')
    # parser.add_argument('--i', '--interest', type=exec, default='def i(x_t): return 1.', help='Interest function to use.')
    parser.add_argument('--num_tiles', type=int, nargs='+', default=[4], help='The number of tiles to use in the tile coder.')
    parser.add_argument('--num_tilings', type=int, nargs='+', default=[4], help='The number of tilings to use in the tile coder.')
    parser.add_argument('--num_features', type=int, default=1024, help='The number of features to use in the tile coder.')
    parser.add_argument('--bias_unit', type=int, nargs='+', default=[1], help='Whether or not to include a bias unit in the tile coder.')
    args = parser.parse_args()

    # Save the command line arguments in a format interpretable by argparse:
    experiment_path = Path(args.experiment_name)
    with open(experiment_path / Path(parser.prog).with_suffix('.args'), 'w') as args_file:
        for key, value in vars(args).items():
            if isinstance(value, list):  # Special case for list arguments.
                value = '\n'.join(str(i) for i in value)
            args_file.write('--{}\n{}\n'.format(key, value))

    # Load the input data as a memmap to prevent a copy being loaded into memory in each sub-process:
    experience = np.lib.format.open_memmap(str(experiment_path / 'experience.npy'), mode='r')
    num_runs, num_timesteps = experience.shape

    # Create the memmapped array of learned policies to be populated in parallel:
    num_policies = int(num_timesteps / args.checkpoint_interval)
    policies_dtype = np.dtype(
        [
            ('gamma', float),
            ('alpha_a', float),
            ('alpha_c', float),
            ('lambda_c', float),
            ('eta', float),
            ('num_tiles', int),
            ('num_tilings', int),
            ('bias_unit', bool),
            ('timesteps', int),
            ('weights', float, (num_actions, args.num_features))
        ]
    )
    policies_shape = (num_runs, len(args.alpha_a), len(args.alpha_c), len(args.lambda_c), len(args.eta), len(args.num_tiles), len(args.num_tilings), len(args.bias_unit), num_policies)
    policies = np.lib.format.open_memmap(str(experiment_path / 'policies.npy'), shape=policies_shape, dtype=policies_dtype, mode='w+')

    # Run ACE for all combinations of the given parameters in parallel:
    Parallel(n_jobs=args.num_cpus, verbose=10)(
        delayed(run_ace_sweep)(
            policies, experience, num_timesteps, args.checkpoint_interval, args.num_features, args.gamma,
            run_num,
            alpha_a_idx, alpha_a,
            alpha_c_idx, alpha_c,
            lambda_c_idx, lambda_c,
            eta_idx, eta,
            num_tiles_idx, num_tiles,
            num_tilings_idx, num_tilings,
            bias_unit_idx, bias_unit
        )
        for run_num in range(num_runs)
        for alpha_a_idx, alpha_a in enumerate(args.alpha_a)
        for alpha_c_idx, alpha_c in enumerate(args.alpha_c)
        for lambda_c_idx, lambda_c in enumerate(args.lambda_c)
        for eta_idx, eta in enumerate(args.eta)
        for num_tiles_idx, num_tiles in enumerate(args.num_tiles)
        for num_tilings_idx, num_tilings in enumerate(args.num_tilings)
        for bias_unit_idx, bias_unit in enumerate(args.bias_unit)
    )

    # Close the memmap files:
    del experience
    del policies
