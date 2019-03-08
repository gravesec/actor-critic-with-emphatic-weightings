# Add the src directory to the path, because python doesn't look in subfolders:
import sys, os
src_path = os.path.join(os.getcwd(), '.')
if src_path not in sys.path:
    sys.path.append(src_path)

import gym
import argparse
import numpy as np
from src import utils
from gym.utils import seeding
from src.function_approximation.tile_coder import TileCoder
from src.policy_types.discrete_policy import DiscretePolicy
from generate_data import num_actions, environment_name, behaviour_policy_name


if __name__ == '__main__':

    # Parse command line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('use_excursions', type=int, choices=[0, 1], help='whether to use excursions objective or not')
    parser.add_argument('bias_unit', type=int, choices=[0, 1, 2], help='0:no bias unit, 1:bias unit added to tile coder, 2:extra weights added to estimate value function for the learned policy with a reward of 1 everywhere.')
    parser.add_argument('alpha_u', type=float, help='alpha_u value')
    parser.add_argument('alpha_v', type=float, help='alpha_v value')
    parser.add_argument('alpha_w', type=float, help='alpha_w value')
    parser.add_argument('lamda_v', type=float, help='lamda value for critic')
    parser.add_argument('lamda_a', type=float, help='emphatic actor lamda value')
    parser.add_argument('run', type=str, help='the run that the policy was learned from')
    parser.add_argument('num_eval_runs', type=int, default=30, nargs='?', help='the number of times to evaluate the learned policies')
    parser.add_argument('max_timesteps', type=int, default=1000, nargs='?', help='the number of timesteps to evaluate the learned policies for')
    args = parser.parse_args()

    # Set up the function approximator:
    tc = TileCoder(min_values=[-1.2,-0.07], max_values=[0.6,0.07], num_tiles=[10,10], num_tilings=10, num_features=2048, bias_unit=(args.bias_unit == 1))

    # Scale the learning rates by the number of active features:
    args.alpha_u /= tc.num_active_features
    args.alpha_v /= tc.num_active_features
    args.alpha_w /= tc.num_active_features

    # Determine the algorithm name to load:
    algorithm_name = 'eac'

    # Initialize the environment and rngs:
    env = gym.make(environment_name).env  # Get the underlying environment to turn off the time limit.
    env_seeds = env.seed()
    rng, alg_seed = seeding.np_random()

    # Load the behaviour policy data to use for the excursions objective:
    if args.use_excursions:
        data_file_path = 'output/{}/behaviour_policies/{}/runs/'.format(environment_name, behaviour_policy_name)
        run_nums = os.listdir(data_file_path)
        data = []
        for run_num in run_nums:
            if run_num == args.run:
                # Skip the data that was trained on?
                continue
            data.append(np.load(data_file_path + run_num + '/data.npz')['transitions'])
        data = np.array(data)

    # Load the learned policies to evaluate:
    policies_file_path = 'output/{}/behaviour_policies/{}/algorithms/{}/bias_unit/{}/alpha_u/{}/alpha_v/{}/alpha_w/{}/lamda_v/{}/lamda_a/{}/runs/{}/policies.npz'.format(environment_name, behaviour_policy_name, algorithm_name, args.bias_unit, args.alpha_u, args.alpha_v, args.alpha_w, args.lamda_v, args.lamda_a, args.run)
    policies = np.load(policies_file_path)['policies']

    # Evaluate the policies:
    returns = np.empty((policies.shape[0], args.num_eval_runs))
    for policy_num, policy_weights in enumerate(policies):
        policy = DiscretePolicy(policy_weights)

        # Evaluate each policy several times:
        for eval_run_num in range(args.num_eval_runs):

            # Start an episode:
            s_t = env.reset()
            g = 0.

            if args.use_excursions:
                # randomly sample start state from behaviour policy data:
                row_index = rng.randint(0, data.shape[0])
                col_index = rng.randint(0, data.shape[1])
                env.state = data[row_index, col_index][0]
                s_t = env.state

            # Play out the episode:
            for t in range(args.max_timesteps):
                x_t = tc.features(s_t)
                a_t = rng.choice(env.action_space.n, p=policy.pi(x_t))
                s_tp1, r_tp1, terminal, _ = env.step(a_t)

                s_t = s_tp1
                g += r_tp1

                if terminal:
                    break

            # Store the returns:
            returns[policy_num, eval_run_num] = g

    # Save the returns to a file:
    returns_file_path = 'output/{}/behaviour_policies/{}/algorithms/{}/bias_unit/{}/alpha_u/{}/alpha_v/{}/alpha_w/{}/lamda_v/{}/lamda_a/{}/runs/{}/{}returns.npz'.format(environment_name, behaviour_policy_name, algorithm_name, args.bias_unit, args.alpha_u, args.alpha_v, args.alpha_w, args.lamda_v, args.lamda_a, args.run, 'excursions_' if args.use_excursions else '')
    utils.create_directory(returns_file_path)
    if os.path.isfile(returns_file_path):
        old_returns = np.load(returns_file_path)['returns']
        returns = np.append(old_returns, returns, axis=1)
    np.savez_compressed(returns_file_path, returns=returns)
