import os
import gym
import gym_puddle
import random
import argparse
import numpy as np
from src import utils
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
from src.algorithms.ace import BinaryACE
from src.algorithms.tdc import BinaryTDC, BinaryGQ
from src.function_approximation.tile_coder import TileCoder
from evaluate_policies import evaluate_policy


def run_ace(experience_memmap, policies_memmap, performance_memmap, run_num, config_num, parameters, random_seed):
    # If this run and configuration has already been done (i.e., previous run timed out), exit early:
    if np.count_nonzero(policies_memmap[config_num]['policies'][run_num]) != 0:
        return

    alpha_a, alpha_w, alpha_v, lambda_c, eta = parameters

    # If this is the first run with a set of parameters, save the parameters:
    if run_num == 0:
        policies_memmap[config_num]['parameters'] = (alpha_a, alpha_w, alpha_v, lambda_c, eta, args.gamma, args.num_tiles_per_dim, args.num_tilings, args.bias_unit)
        performance_memmap[config_num]['parameters'] = (alpha_a, alpha_w, alpha_v, lambda_c, eta, args.gamma, args.num_tiles_per_dim, args.num_tilings, args.bias_unit)

    # Create the environment to evaluate the learned policy in:
    import gym_puddle
    env = gym.make(args.environment).unwrapped
    env.seed(random_seed)
    rng = env.np_random

    actor = BinaryACE(env.action_space.n, tc.total_num_tiles, alpha_a / tc.num_active_features)
    if args.all_actions:
        critic = BinaryGQ(env.action_space.n, tc.total_num_tiles, alpha_w / tc.num_active_features, alpha_v / tc.num_active_features, lambda_c)
    else:
        critic = BinaryTDC(tc.total_num_tiles, alpha_w / tc.num_active_features, alpha_v / tc.num_active_features, lambda_c)

    i = eval(args.interest_function)  # Create the interest function to use.
    mu = eval(args.behaviour_policy, {'np': np, 'env': env})  # Create the behaviour policy and give it access to numpy and the env.

    policies = np.zeros(num_policies, dtype=policy_dtype)
    performance = np.zeros((num_policies, args.num_evaluation_runs), dtype=float)

    np.seterr(divide='raise', over='raise', invalid='raise')
    try:
        transitions = experience_memmap[run_num]
        gamma_t = 0.
        f_t = 0.
        rho_tm1 = 1.
        indices_t = tc.encode(transitions[0][0])
        for t, transition in enumerate(transitions):
            # Save and evaluate the learned policy if it's a checkpoint timestep:
            if t % args.checkpoint_interval == 0:
                performance[t // args.checkpoint_interval] = [evaluate_policy(actor, tc, env, rng, args.max_timesteps) for _ in range(args.num_evaluation_runs)]
                policies[t // args.checkpoint_interval] = (t, np.copy(actor.theta))

            # Unpack the stored transition.
            s_t, a_t, r_tp1, s_tp1, a_tp1, terminal = transition
            gamma_tp1 = args.gamma if not terminal else 0  # Transition-dependent discounting.
            indices_tp1 = tc.encode(s_tp1)
            i_t = i(s_t, gamma_t)
            # Compute importance sampling ratio for the policy:
            pi_t = actor.pi(indices_t)
            mu_t = mu(s_t)
            rho_t = pi_t[a_t] / mu_t[a_t]

            f_t = rho_tm1 * gamma_t * f_t + i_t
            m_t = (1 - eta) * i_t + eta * f_t
            if args.all_actions:
                critic.learn(indices_t, a_t, rho_t, gamma_t, r_tp1, indices_tp1, actor.pi(indices_tp1), gamma_tp1)
                q_t = critic.estimate(indices_t)
                actor.all_actions_learn(indices_t, q_t, m_t)
            else:
                delta_t = r_tp1 + gamma_tp1 * critic.estimate(indices_tp1) - critic.estimate(indices_t)
                critic.learn(delta_t, indices_t, gamma_t, indices_tp1, gamma_tp1, rho_t)
                actor.learn(indices_t, a_t, delta_t, m_t, rho_t)

            gamma_t = gamma_tp1
            indices_t = indices_tp1
            rho_tm1 = rho_t
        # Save and evaluate the policy after the final timestep:
        policies[-1] = (t+1, np.copy(actor.theta))
        performance[-1] = [evaluate_policy(actor, tc, env, rng, args.max_timesteps) for _ in range(args.num_evaluation_runs)]

        # Save the learned policies and their performance to the memmap:
        performance_memmap[config_num]['results'][run_num] = performance
        policies_memmap[config_num]['policies'][run_num] = policies
    except (FloatingPointError, ValueError) as e:
        # Save NaN to indicate the weights overflowed and exit early:
        performance_memmap[config_num]['results'][run_num] = np.full_like(performance, np.NaN)
        policies_memmap[config_num]['policies'][run_num] = np.full_like(policies, np.NaN)
        return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='A script to run ACE (Actor-Critic with Emphatic weightings).', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output_dir', type=str, default='experiment', help='The directory to write experiment files to')
    parser.add_argument('--experience_file', type=str, default='experiment/experience.npy', help='The file to read experience from')
    parser.add_argument('--num_cpus', type=int, default=-1, help='The number of cpus to use (-1 for all).')

    # Policy evaluation parameters:
    parser.add_argument('--checkpoint_interval', type=int, default=5000, help='The number of timesteps after which to save the learned policy.')
    parser.add_argument('--num_evaluation_runs', type=int, default=5, help='The number of times to evaluate each policy')
    parser.add_argument('--max_timesteps', type=int, default=5000, help='The maximum number of timesteps allowed per policy evaluation')
    parser.add_argument('--random_seed', type=int, default=1944801619, help='The master random seed to use')

    # Experiment parameters:
    parser.add_argument('--all_actions', type=int, choices=[0, 1], default=0, help='Use all-actions updates instead of TD error-based updates.')
    parser.add_argument('--interest_function', type=str, default='lambda s, g=1: 1.', help='Interest function to use. Example: \'lambda s, g=1: 1. if g==0. else 0.\' (episodic interest function)')
    parser.add_argument('--behaviour_policy', type=str, default='lambda s: np.ones(env.action_space.n)/env.action_space.n', help='Policy to use. Default is uniform random. Another Example: \'lambda s: np.array([.9, .05, .05]) if s[1] < 0 else np.array([.05, .05, .9]) \' (energy pumping policy w/ 15 percent randomness)')
    parser.add_argument('--environment', type=str, default='MountainCar-v0', help='An OpenAI Gym environment string.')
    parser.add_argument('--gamma', '--discount_rate', type=float, default=.99, help='Discount rate.')
    parser.add_argument('--parameters', type=float, nargs=5, action='append', metavar=('alpha_a', 'alpha_w', 'alpha_v', 'lambda', 'eta'), help='Parameters to use. Can be specified multiple times to run multiple configurations in parallel.')
    parser.add_argument('--num_tiles_per_dim', type=int, nargs='+', default=[5, 5], help='The number of tiles per dimension to use in the tile coder.')
    parser.add_argument('--num_tilings', type=int, default=8, help='The number of tilings to use in the tile coder.')
    parser.add_argument('--bias_unit', type=int, choices=[0, 1], default=1, help='Whether or not to include a bias unit in the tile coder.')
    args = parser.parse_args()

    # Generate the random seed for each run without replacement to prevent the birthday paradox:
    random.seed(args.random_seed)
    random_seeds = random.sample(range(2**32), args.num_evaluation_runs)

    # Save the command line arguments in a format interpretable by argparse:
    output_dir = Path(args.output_dir)
    utils.save_args_to_file(args, output_dir / Path(parser.prog).with_suffix('.args'))

    # Load the input data as a memmap to prevent a copy being loaded into memory in each sub-process:
    experience_memmap = np.lib.format.open_memmap(args.experience_file, mode='r')
    num_runs, num_timesteps = experience_memmap.shape

    # Create the tile coder to be used for all parameter settings:
    dummy_env = gym.make(args.environment).unwrapped  # Make a dummy env to get shape info.
    tc = TileCoder(np.array([dummy_env.observation_space.low, dummy_env.observation_space.high]).T, args.num_tiles_per_dim, args.num_tilings, args.bias_unit)

    # Create the memmapped array of learned policies that will be populated in parallel:
    parameters_dtype = np.dtype([
        ('alpha_a', float),
        ('alpha_w', float),
        ('alpha_v', float),
        ('lambda', float),
        ('eta', float),
        ('gamma', float),
        ('num_tiles_per_dim', int, (len(args.num_tiles_per_dim),)),
        ('num_tilings', int),
        ('bias_unit', bool)
    ])
    policy_dtype = np.dtype([
        ('timesteps', int),
        ('weights', float, (dummy_env.action_space.n, tc.total_num_tiles))
    ])
    num_policies = num_timesteps // args.checkpoint_interval + 1
    configuration_dtype = np.dtype([
        ('parameters', parameters_dtype),
        ('policies', policy_dtype, (num_runs, num_policies))
    ])
    policies_memmap_path = str(output_dir / 'policies.npy')
    if os.path.isfile(policies_memmap_path):
        policies_memmap = np.lib.format.open_memmap(policies_memmap_path, mode='r+')
    else:
        policies_memmap = np.lib.format.open_memmap(policies_memmap_path, shape=(len(args.parameters),), dtype=configuration_dtype, mode='w+')

    # Create the memmapped array of performance results for the learned policies:
    performance_dtype = np.dtype([
        ('parameters', parameters_dtype),
        ('results', float, (num_runs, num_policies, args.num_evaluation_runs))
    ])
    performance_memmap_path = str(output_dir / 'performance.npy')
    if os.path.isfile(performance_memmap_path):
        performance_memmap = np.lib.format.open_memmap(performance_memmap_path, mode='r+')
    else:
        performance_memmap = np.lib.format.open_memmap(performance_memmap_path, shape=(len(args.parameters),), dtype=performance_dtype, mode='w+')

    # Run ACE for each configuration in parallel:
    with utils.tqdm_joblib(tqdm(total=num_runs * len(args.parameters))) as progress_bar:
        Parallel(n_jobs=args.num_cpus, verbose=0)(
            delayed(run_ace)(experience_memmap, policies_memmap, performance_memmap, run_num, config_num, parameters, random_seed)
            for config_num, parameters in enumerate(args.parameters)
            for run_num, random_seed in enumerate(random_seeds)
        )
