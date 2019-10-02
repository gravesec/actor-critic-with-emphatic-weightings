import os
import argparse
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from mountain_car.ace import TileCoder, TOETD, ACE
from mountain_car.generate_experience import transition_dtype


# TODO: Figure out how to do checkpointing (i.e. keep track of progress via a memmap so if the process gets killed it can pick up where it left off).
# TODO: Figure out how to append to a memmap in case we want to do more runs later on (we might get this without any extra work with checkpointing).
# TODO: implement our own mountain car environment (https://en.wikipedia.org/wiki/Mountain_car_problem) because openai's is slow, stateful, and has bizarre decisions built in like time limits and the inability to get properties of an environment without creating an instantiation of it.
num_actions = 3
min_state_values = [-1.2, -0.07]
max_state_values = [0.6, 0.07]


def run_ace(policies, experience, num_timesteps, checkpoint_interval, num_features, run_num, parameters):
    gamma, alpha_a, alpha_c, lambda_c, eta, num_tiles, num_tilings, bias_unit = parameters

    # Configure the agent:
    tc = TileCoder(min_state_values, max_state_values, [num_tiles, num_tiles], num_tilings, num_features, bias_unit)
    actor = ACE(num_actions, num_features)
    critic = TOETD(num_features, 1., alpha_c)

    # Process the experience:
    for t in range(num_timesteps):
        pass


if __name__ == '__main__':

    # Parse command line arguments:
    parser = argparse.ArgumentParser(description='A script to run ACE (Actor-Critic with Emphatic weightings) in parallel.', fromfile_prefix_chars='@', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experiment_name', default='experiment', help='The directory to read/write experiment files to.')
    parser.add_argument('--num_runs', type=int, default=5, help='The number of independent runs of experience.')
    parser.add_argument('--num_timesteps', type=int, default=1000, help='The number of timesteps of experience per run.')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='The number of timesteps after which to save the learned policy.')
    parser.add_argument('--num_cpus', type=int, default=-1, help='The number of cpus to use (-1 for all).')
    parser.add_argument('--backend', type=str, choices=['loky', 'threading'], default='loky', help='The backend to use (\'loky\' for processes or \'threading\' for threads; always use \'loky\').')
    parser.add_argument('--num_features', type=int, default=256, help='The number of features to use in the tile coder.')
    parser.add_argument('--parameters', type=float, nargs=8, action='append', metavar=('DISCOUNT_RATE', 'ACTOR_STEP_SIZE', 'CRITIC_STEP_SIZE', 'CRITIC_TRACE_DECAY_RATE', 'OFFPAC_ACE_TRADEOFF', 'NUM_TILES', 'NUM_TILINGS', 'BIAS_UNIT'), help='Parameters to use for ACE. Can be specified multiple times to run multiple configurations of ACE at once.')
    args = parser.parse_args()

    # Save the command line arguments in a format interpretable by argparse:
    experiment_path = Path(args.experiment_name)
    os.makedirs(experiment_path, exist_ok=True)
    with open(experiment_path / Path(parser.prog).with_suffix('.args'), 'w') as args_file:
        for key, value in vars(args).items():
            if isinstance(value, list):  # Special case for parameters argument.
                for plist in value:
                    args_file.write('--{}\n{}\n'.format(key, '\n'.join(str(i) for i in plist)))
            else:
                args_file.write('--{}\n{}\n'.format(key, value))

    # Create the memmapped array of learned policies to be populated in parallel:
    num_policies = int(args.num_timesteps / args.checkpoint_interval)
    num_configurations = len(args.parameters)
    policies_shape = (args.num_runs, num_configurations, num_policies, num_actions, args.num_features)
    policies = np.memmap(experiment_path / 'policies.npy', shape=policies_shape, mode='w+')

    # Load the input data as a memmap to prevent a copy being loaded into memory in each sub-process:
    experience = np.memmap(experiment_path / 'experience.npy', shape=(args.num_runs, args.num_timesteps), dtype=transition_dtype, mode='r')

    # Run ACE for each set of parameters in parallel:
    Parallel(n_jobs=args.num_cpus, verbose=10)(
        delayed(run_ace)(
            policies, experience, args.num_timesteps, args.checkpoint_interval, args.num_features, run_num, parameters
        )
        for parameters in args.parameters
        for run_num in range(args.num_runs)
    )

    # Close the memmap files:
    del experience
    del policies
