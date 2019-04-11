import os
import json
import argparse
import numpy as np
from src.algorithms.eac import EAC
from src.function_approximation.tile_coder import TileCoder
from src.scripts.mountain_car.generate_experience import num_actions, environment_name, behaviour_policy_name

if __name__ == '__main__':

    # Parse command line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('bias_unit', type=int, choices=[0, 1, 2], help='0:no bias unit, 1:bias unit added to tile coder, 2:extra weights added to estimate value function for the learned policy with a reward of 1 everywhere.')
    parser.add_argument('alpha_u', type=float, help='alpha_u value')
    parser.add_argument('alpha_v', type=float, help='alpha_v value')
    parser.add_argument('alpha_w', type=float, help='alpha_w value')
    parser.add_argument('lamda_v', type=float, help='lamda value for critic')
    parser.add_argument('lamda_a', type=float, help='emphatic actor lamda value')
    parser.add_argument('run', type=str, help='the data file to use')
    parser.add_argument('checkpoint_interval', type=int, default=1000, nargs='?', help='the number of timesteps after which to save the learned policy')
    args = parser.parse_args()

    # Set up the function approximator:
    tc = TileCoder(min_values=[-1.2,-0.07], max_values=[0.6,0.07], num_tiles=[10,10], num_tilings=10, num_features=2048, bias_unit=(args.bias_unit == 1))

    # Scale the learning rates by the number of active features:
    args.alpha_u /= tc.num_active_features
    args.alpha_v /= tc.num_active_features
    args.alpha_w /= tc.num_active_features

    # Set up the algorithm:
    algorithm_name = 'eac'

    agent = EAC(num_actions, tc.num_features, args.alpha_u, args.alpha_v, args.alpha_w, args.lamda_v, args.lamda_a, args.bias_unit == 2)

    # Load data for the run:
    data_file_path = 'output/{}/behaviour_policies/{}/runs/{}/data.npz'.format(environment_name, behaviour_policy_name, args.run)
    npzfile = np.load(data_file_path)
    transitions = npzfile['transitions']

    num_policies = int(len(transitions) / args.checkpoint_interval)
    policies = np.empty((num_policies, *agent.actor.policy.u.shape))
    for t, transition in enumerate(transitions):
        s_t, gamma_t, a_t, s_tp1, r_tp1, gamma_tp1 = transition

        # Create feature vectors from observations:
        x_t = tc.features(s_t)
        x_tp1 = tc.features(s_tp1)

        # Compute rho:
        pi_t = agent.actor.policy.pi(x_t, a_t)
        b_t = 1. / num_actions  # assume equiprobable random behaviour policy
        rho_t = pi_t / b_t

        # Update the learner:
        agent.learn(x_t, gamma_t, a_t, r_tp1, x_tp1, gamma_tp1, rho_t)

        # If it's a checkpoint timestep,
        if t % args.checkpoint_interval == 0:
            # store the policy:
            policies[t // args.checkpoint_interval] = np.copy(agent.actor.policy.u)

    # Save the stored policies for later evaluation:
    policies_file_path = 'output/{}/behaviour_policies/{}/algorithms/{}/bias_unit/{}/alpha_u/{}/alpha_v/{}/alpha_w/{}/lamda_v/{}/lamda_a/{}/runs/{}/policies.npz'.format(environment_name, behaviour_policy_name, algorithm_name, args.bias_unit, args.alpha_u, args.alpha_v, args.alpha_w, args.lamda_v, args.lamda_a, args.run)
    utils.create_directory(policies_file_path)
    np.savez_compressed(policies_file_path, policies=policies)
