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
from src.function_approximation.tiny_counterexample_features import TinyCounterexampleFeatures
from src.policy_types.discrete_policy import DiscretePolicy
from generate_data import num_actions, environment_name, behaviour_policy_name, max_num_timesteps

from src.misc_envs.mdps import tiny_counterexample_env

if __name__ == '__main__':

    # Parse command line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('bias_unit', type=int, choices=[0,1], help='add a bias unit to the tile coder or not')
    parser.add_argument('alpha_u', type=float, help='alpha_u value')
    parser.add_argument('alpha_v', type=float, help='alpha_v value')
    parser.add_argument('alpha_w', type=float, help='alpha_w value')
    parser.add_argument('lamda', type=float, help='lamda value')
    parser.add_argument('lamda_a', type=float, help='emphatic actor lamda value')
    parser.add_argument('gamma', type=float, help='gamma value')
    parser.add_argument('run', type=str, help='the run that the policy was learned from')
    parser.add_argument('episode', type=str, help='the episode after which the policy was saved')
    parser.add_argument('num_eval_runs', type=int, default=5, nargs='?', help='the number of times to evaluate the learned policy')
    args = parser.parse_args()

    # Set up the function approximator:
    tc = TinyCounterexampleFeatures(bias_unit=args.bias_unit)

    # Scale the learning rates by the number of active features:
    args.alpha_u /= tc.num_active_features
    args.alpha_v /= tc.num_active_features
    args.alpha_w /= tc.num_active_features

    # Initialize the environment and rngs:
    env = tiny_counterexample_env().to_env()
    # env_seeds = env.seed() # TODO: Needed in deterministic env?
    rng, alg_seed = seeding.np_random()

    # Load the learned policy:
    algorithm_name = 'eac_bias_unit' if args.bias_unit else 'eac'
    policy_file_path = 'output/{}/behaviour_policies/{}/algorithms/{}/alpha_u/{}/alpha_v/{}/alpha_w/{}/lamda/{}/lamda_a/{}/runs/{}/episodes/{}/policy.npz'.format(environment_name, behaviour_policy_name, algorithm_name, args.alpha_u, args.alpha_v, args.alpha_w, args.lamda, args.lamda_a, args.run, args.episode)
    policy = DiscretePolicy(utils.load_policy_from_file(policy_file_path))

    # Evaluate the learned policy:
    returns = np.full((args.num_eval_runs), np.nan)
    for evaluation_run_num in range(args.num_eval_runs):

        # Start an episode:
        s_t = env._reset()
        g = 0.

        # Play out the episode:
        for t in range(max_num_timesteps): # Because the env is not registered.

            x_t = tc.features(s_t)
            a_t = rng.choice(env.action_space.n, p=policy.pi(x_t))
            s_tp1, r_tp1, terminal, _ = env._step(a_t)

            s_t = s_tp1
            g += r_tp1

            if terminal:
                break

        returns[evaluation_run_num] = g
    returns_file_path = 'output/{}/behaviour_policies/{}/algorithms/{}/alpha_u/{}/alpha_v/{}/alpha_w/{}/lamda/{}/lamda_a/{}/runs/{}/episodes/{}/returns.txt'.format(environment_name, behaviour_policy_name, algorithm_name, args.alpha_u, args.alpha_v, args.alpha_w, args.lamda, args.lamda_a, args.run, args.episode)
    utils.save_returns_to_file(returns, returns_file_path)
