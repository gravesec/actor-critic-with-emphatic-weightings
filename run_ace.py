import os
import gym
import gym_puddle
import argparse
import itertools
import numpy as np
from pathlib import Path
from src import utils
from src.algorithms.ace import BinaryACE
from src.algorithms.tdc import BinaryTDC
from src.function_approximation.tile_coder import TileCoder
from joblib import Parallel, delayed


def run_ace(policies_memmap, experience_memmap, run_num, config_num, parameters):
    # Check if this run and configuration has already been done:
    if policies_memmap[run_num, config_num]['gamma'] != 0:
        return

    gamma, alpha_a, alpha_c, alpha_c2, lambda_c, eta, num_tiles, num_tilings, num_features, bias_unit = parameters

    tc = TileCoder(env.observation_space.low, env.observation_space.high, num_tiles, num_tilings, num_features, bias_unit)
    actor = BinaryACE(env.action_space.n, num_features)
    critic = BinaryTDC(num_features, alpha_c, alpha_c2, lambda_c)

    i = eval(args.interest_function)  # Create the interest function to use.
    mu = eval(args.behaviour_policy, {'np': np, 'env': env})  # Create the behaviour policy and give it access to numpy.

    transitions = experience_memmap[run_num]

    policies = np.zeros(num_policies, dtype=policy_dtype)
    gamma_t = 0.
    indices_t = tc.indices(transitions[0][0])
    for t, transition in enumerate(transitions):
        if t % args.checkpoint_interval == 0:  # Save the learned policy if it's a checkpoint timestep:
            padded_weights = np.zeros_like(policies[t // args.checkpoint_interval][1])
            padded_weights[0:actor.theta.shape[0], 0:actor.theta.shape[1]] = np.copy(actor.theta)
            policies[t // args.checkpoint_interval] = (t, padded_weights)

        # Unpack the stored transition.
        s_t, a_t, r_tp1, s_tp1, terminal = transition
        gamma_tp1 = gamma if not terminal else 0  # Transition-dependent discounting.
        indices_tp1 = tc.indices(s_tp1)
        i_t = i(s_t, gamma_t)
        # Compute importance sampling ratio for the policy:
        pi_t = actor.pi(indices_t)
        mu_t = mu(s_t)
        rho_t = pi_t[a_t] / mu_t[a_t]
        # Compute TD error:
        v_t = critic.estimate(indices_t)
        v_tp1 = critic.estimate(indices_tp1)
        delta_t = r_tp1 + gamma_tp1 * v_tp1 - v_t
        # Update actor:
        actor.learn(gamma_t, i_t, eta, alpha_a, rho_t, delta_t, indices_t, a_t)
        # Update critic:
        critic.learn(delta_t, indices_t, gamma_t, indices_tp1, gamma_tp1, rho_t)

        gamma_t = gamma_tp1
        indices_t = indices_tp1

    # Save the policy after the final timestep:
    padded_weights = np.zeros_like(policies[-1][1])
    padded_weights[0:actor.theta.shape[0], 0:actor.theta.shape[1]] = np.copy(actor.theta)
    policies[-1] = (t+1, padded_weights)

    policies_memmap[run_num, config_num] = (gamma, alpha_a, alpha_c, alpha_c2, lambda_c, eta, num_tiles, num_tilings, num_features, bias_unit, policies)  # Store the learned policies in the memmap.


if __name__ == '__main__':

    # Parse command line arguments:
    parser = argparse.ArgumentParser(description='A script to run ACE (Actor-Critic with Emphatic weightings) in parallel.', fromfile_prefix_chars='@', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experiment_name', type=str, default='experiment', help='The directory to read/write experiment files to/from')
    parser.add_argument('--checkpoint_interval', type=int, default=5000, help='The number of timesteps after which to save the learned policy.')
    parser.add_argument('--num_cpus', type=int, default=-1, help='The number of cpus to use (-1 for all).')
    parser.add_argument('--backend', type=str, choices=['loky', 'threading'], default='loky', help='The backend to use (\'loky\' for processes or \'threading\' for threads; always use \'loky\' because Python threading is terrible).')
    parser.add_argument('--verbosity', type=int, default=51, help='Controls how verbose the joblib progress reporting is. 0 for none, 51 for all iterations to stdout.')
    parser.add_argument('--interest_function', type=str, default='lambda s, g=1: 1.', help='Interest function to use. Example: \'lambda s, g=1: 1. if g==0. else 0.\' (episodic interest function)')
    parser.add_argument('--behaviour_policy', type=str, default='lambda s: np.ones(env.action_space.n)/env.action_space.n', help='Policy to use. Default is uniform random. Another Example: \'lambda s: np.array([.9, .05, .05]) if s[1] < 0 else np.array([.05, .05, .9]) \' (energy pumping policy w/ 15 percent randomness)')
    parser.add_argument('--environment', type=str, choices=['MountainCar-v0', 'Acrobot-v1', 'PuddleWorld-v0'], default='MountainCar-v0', help='The environment to run ACE on.')
    parser.add_argument('--run_mode', type=str, choices=['combinations', 'corresponding'], default='combinations', help='Whether to run all combinations of given parameters, or only corresponding parameters')
    parser.add_argument('--gamma', '--discount_rate', type=float, nargs='+', default=[1.], help='Discount rate.')
    parser.add_argument('--alpha_a', '--actor_step_sizes', type=float, nargs='+', default=[.01], help='Step sizes for the actor.')
    parser.add_argument('--alpha_c', '--critic_step_sizes', type=float, nargs='+', default=[.05], help='Step sizes for the critic.')
    parser.add_argument('--alpha_c2', '--critic_step_sizes_2', type=float, nargs='+', default=[.0001], help='Step sizes for the second set of weights in the GTD critic.')
    parser.add_argument('--lambda_c', '--critic_trace_decay_rates', type=float, nargs='+', default=[0.], help='Trace decay rates for the critic.')
    parser.add_argument('--eta', '--offpac_ace_tradeoff', type=float, nargs='+', default=[0.], help='Values for the parameter that interpolates between OffPAC (0) and ACE (1).')
    parser.add_argument('--num_tiles', type=int, nargs='+', default=[9], help='The number of tiles to use in the tile coder.')
    parser.add_argument('--num_tilings', type=int, nargs='+', default=[9], help='The number of tilings to use in the tile coder.')
    parser.add_argument('--num_features', type=int, nargs='+', default=[10000], help='The number of features to use in the tile coder.')
    parser.add_argument('--bias_unit', type=int, nargs='+', default=[1], help='Whether or not to include a bias unit in the tile coder.')
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
    max_num_features = np.max(args.num_features)
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
    parameters = [args.gamma, args.alpha_a, args.alpha_c, args.alpha_c2, args.lambda_c, args.eta, args.num_tiles, args.num_tilings, args.num_features, args.bias_unit]
    if args.run_mode == 'corresponding':
        assert all(len(parameter) == len(args.alpha_a) for parameter in parameters)
        configurations = list(zip(*parameters))  # Transpose parameters list.
    else:
        configurations = list(itertools.product(*parameters))  # All combinations of parameters.
    num_configurations = len(configurations)

    policies_memmap_path = str(experiment_path / 'policies.npy')
    if os.path.isfile(policies_memmap_path):
        policies_memmap = np.lib.format.open_memmap(policies_memmap_path, shape=(num_runs, num_configurations), dtype=configuration_dtype, mode='r+')
    else:
        policies_memmap = np.lib.format.open_memmap(policies_memmap_path, shape=(num_runs, num_configurations), dtype=configuration_dtype, mode='w+')

    # Run ACE for each configuration in parallel:
    Parallel(n_jobs=args.num_cpus, verbose=args.verbosity)(
        delayed(run_ace)(policies_memmap, experience_memmap, run_num, config_num, parameters)
        for config_num, parameters in enumerate(configurations)
        for run_num in range(num_runs)
    )
