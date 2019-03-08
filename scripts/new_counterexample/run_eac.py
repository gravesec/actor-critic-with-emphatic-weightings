# Add the src directory to the path, because python doesn't look in subfolders:
import sys, os
src_path = os.path.join(os.getcwd(), '.')
if src_path not in sys.path:
    sys.path.append(src_path)

import argparse
import numpy as np
from gym.utils import seeding
from src import utils
from src.algorithms.eac import EAC
from src.function_approximation.new_counterexample_features import NewCounterexampleFeatures
from generate_data import num_actions, environment_name, behaviour_policy_name, behavior_probs

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
    parser.add_argument('run', type=str, help='the data file to use')
    parser.add_argument('num_evaluation_points', type=int, default=10, nargs='?', help='the number of points in time during training to evaluate the learned policy')
    args = parser.parse_args()

    # Set up the function approximator:
    tc = NewCounterexampleFeatures(bias_unit=args.bias_unit)

    # Scale the learning rates by the number of active features:
    args.alpha_u /= tc.num_active_features
    args.alpha_v /= tc.num_active_features
    args.alpha_w /= tc.num_active_features

    # Set up the algorithm:
    algorithm_name = 'eac_bias_unit' if args.bias_unit else 'eac'
    agent = EAC(num_actions, tc.num_features, args.alpha_u, args.alpha_v, args.alpha_w, args.lamda, args.lamda_a)

    # Load data for the run:
    data_file_path = 'output/{}/behaviour_policies/{}/runs/{}/data.npz'.format(environment_name, behaviour_policy_name, args.run)
    episodes = utils.load_data_from_file(data_file_path)

    # Process the episodes:
    for episode_num, episode in enumerate(episodes):

        for t, transition in enumerate(episode):
            # Unpack the transition:
            s_t, a_t, r_tp1, s_tp1 = transition

            # Create feature vectors from observations:
            x_t = tc.features(s_t)
            x_tp1 = tc.features(s_tp1)

            # Compute rho:
            pi_t = agent.actor.policy.pi(x_t, a_t)
            b_t = behavior_probs[a_t]#1. / num_actions # don't assume equiprobable random behaviour policy
            rho_t = pi_t / b_t

            # Compute gammas:
            gamma_t = 0 if t == 0 else args.gamma   # Reset traces if it's the start of an episode.
            gamma_tp1 = 0 if t == len(episode) else args.gamma  # Terminal state value is zero.

            # Update the learner:
            agent.learn(x_t, gamma_t, a_t, r_tp1, x_tp1, gamma_tp1, rho_t)

        if (episode_num + 1) % (len(episodes) / args.num_evaluation_points) == 0.:
            # Save the learned policy for evaluation:
            policy_file_path = 'output/{}/behaviour_policies/{}/algorithms/{}/alpha_u/{}/alpha_v/{}/alpha_w/{}/lamda/{}/lamda_a/{}/runs/{}/episodes/{}/policy.npz'.format(environment_name, behaviour_policy_name, algorithm_name, args.alpha_u, args.alpha_v, args.alpha_w, args.lamda, args.lamda_a, args.run, episode_num + 1)
            utils.save_policy_to_file(agent.actor.policy.u, policy_file_path)
