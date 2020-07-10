import os
import gym
import gym_puddle
import argparse
import itertools
import numpy as np
from src import utils
from tqdm import tqdm
from pathlib import Path
from src.utils import tqdm_joblib
from joblib import Parallel, delayed
from src.algorithms.ace import BinaryACE
from src.algorithms.tdc import BinaryTDC
from src.function_approximation.tile_coder import TileCoder


def run_ace(experience_memmap, policies_memmap, run_num, config_num, parameters):
    # Check if this run and configuration has already been done:
    if np.count_nonzero(policies_memmap[config_num]['policies'][run_num]) != 0:
        return

    alpha_a, alpha_w, alpha_v, lambda_c, eta = parameters

    if run_num == 0:
        policies_memmap[config_num]['parameters'] = (args.gamma, alpha_a, alpha_w, alpha_v, lambda_c, eta, args.num_tiles_per_dim, args.num_tilings, args.bias_unit)

    actor = BinaryACE(env.action_space.n, tc.total_num_tiles, alpha_a / tc.num_active_features)
    critic = BinaryTDC(tc.total_num_tiles, alpha_w / tc.num_active_features, alpha_v / tc.num_active_features, lambda_c)
    i = eval(args.interest_function)  # Create the interest function to use.
    mu = eval(args.behaviour_policy, {'np': np, 'env': env})  # Create the behaviour policy and give it access to numpy.
    transitions = experience_memmap[run_num]
    policies = np.zeros(num_policies, dtype=policy_dtype)
    gamma_t = 0.
    f_t = 0.
    rho_tm1 = 1.
    indices_t = tc.encode(transitions[0][0])
    for t, transition in enumerate(transitions):
        # Save the learned policy if it's a checkpoint timestep:
        if t % args.checkpoint_interval == 0:
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
        # Update the critic:
        delta_t = r_tp1 + gamma_tp1 * critic.estimate(indices_tp1) - critic.estimate(indices_t)
        critic.learn(delta_t, indices_t, gamma_t, indices_tp1, gamma_tp1, rho_t)
        # Update the actor:
        f_t = rho_tm1 * gamma_t * f_t + i_t
        m_t = (1 - eta) * i_t + eta * f_t
        actor.learn(indices_t, a_t, delta_t, m_t, rho_t)

        gamma_t = gamma_tp1
        indices_t = indices_tp1
        rho_tm1 = rho_t
    # Save the policy after the final timestep:
    policies[-1] = (t+1, np.copy(actor.theta))

    # Save the learned policies to the memmap:
    policies_memmap[config_num]['policies'][run_num] = policies


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='A script to run ACE (Actor-Critic with Emphatic weightings).', fromfile_prefix_chars='@', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experiment_name', type=str, default='experiment', help='The directory to read/write experiment files to/from')
    parser.add_argument('--checkpoint_interval', type=int, default=1000, help='The number of timesteps after which to save the learned policy.')
    parser.add_argument('--num_cpus', type=int, default=-1, help='The number of cpus to use (-1 for all).')
    parser.add_argument('--interest_function', type=str, default='lambda s, g=1: 1.', help='Interest function to use. Example: \'lambda s, g=1: 1. if g==0. else 0.\' (episodic interest function)')
    parser.add_argument('--behaviour_policy', type=str, default='lambda s: np.ones(env.action_space.n)/env.action_space.n', help='Policy to use. Default is uniform random. Another Example: \'lambda s: np.array([.9, .05, .05]) if s[1] < 0 else np.array([.05, .05, .9]) \' (energy pumping policy w/ 15 percent randomness)')
    parser.add_argument('--environment', type=str, default='MountainCar-v0', help='An OpenAI Gym environment string.')
    parser.add_argument('--gamma', '--discount_rate', type=float, default=.99, help='Discount rate.')
    parser.add_argument('-p', '--parameters', type=float, nargs=5, action='append', metavar=('alpha_a', 'alpha_w', 'alpha_v', 'lambda', 'eta'), help='Parameters to use. Can be specified multiple times to run multiple configurations in parallel.')
    parser.add_argument('--num_tiles_per_dim', type=int, nargs='+', default=[5, 5], help='The number of tiles per dimension to use in the tile coder.')
    parser.add_argument('--num_tilings', type=int, default=8, help='The number of tilings to use in the tile coder.')
    parser.add_argument('--bias_unit', type=int, choices=[0, 1], default=1, help='Whether or not to include a bias unit in the tile coder.')
    args = parser.parse_args()

    # Save the command line arguments in a format interpretable by argparse:
    experiment_path = Path(args.experiment_name)
    utils.save_args_to_file(args, experiment_path / Path(parser.prog).with_suffix('.args'))

    # Load the input data as a memmap to prevent a copy being loaded into memory in each sub-process:
    experience_memmap = np.lib.format.open_memmap(str(experiment_path / 'experience.npy'), mode='r')
    num_runs, num_timesteps = experience_memmap.shape

    # Create the tile coder to be used for all parameter settings:
    env = gym.make(args.environment).unwrapped  # Make a dummy env to get shape info.
    tc = TileCoder(np.array([env.observation_space.low, env.observation_space.high]).T, args.num_tiles_per_dim, args.num_tilings, args.bias_unit)

    # Create the memmapped array of learned policies that will be populated in parallel:
    num_policies = num_timesteps // args.checkpoint_interval + 1
    policy_dtype = np.dtype([
            ('timesteps', int),
            ('weights', float, (env.action_space.n, tc.total_num_tiles))
    ])
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
    configuration_dtype = np.dtype([
        ('parameters', parameters_dtype),
        ('policies', policy_dtype, (num_runs, num_policies))
    ])
    policies_memmap_path = str(experiment_path / 'policies.npy')
    if os.path.isfile(policies_memmap_path):
        policies_memmap = np.lib.format.open_memmap(policies_memmap_path, mode='r+')
    else:
        policies_memmap = np.lib.format.open_memmap(policies_memmap_path, shape=(len(args.parameters),), dtype=configuration_dtype, mode='w+')

    # Run ACE for each configuration in parallel:
    with tqdm_joblib(tqdm(total=num_runs * len(args.parameters))) as progress_bar:
        Parallel(n_jobs=args.num_cpus, verbose=0)(
            delayed(run_ace)(experience_memmap, policies_memmap, run_num, config_num, parameters)
            for config_num, parameters in enumerate(args.parameters)
            for run_num in range(num_runs)
        )
