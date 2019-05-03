import os
import argparse
import numpy as np
from joblib import Parallel, delayed, dump, load
from src import utils
from src.algorithms.eac import EAC
from src.function_approximation.tile_coder import TileCoder
from src.scripts.mountain_car.generate_experience import num_actions, output_directory_name, environment_name, behaviour_policy_name, transition_dtype


algorithm_name = 'ACE'


def run_ace_tc(experience, policies, tc, num_timesteps, checkpoints, run_num, alpha_a_idx, alpha_a, lambda_a_idx, lambda_a, alpha_c_idx, alpha_c, lambda_c_idx, lambda_c, eta_idx, eta):

    # Configure the agent:
    learner = EAC(num_actions, tc.num_features, alpha_a, alpha_c, lambda_c, eta)

    # Process the experience:
    for t, transition in enumerate(experience[run_num]):
        pass


if __name__ == '__main__':

    # Parse command line arguments:
    parser = argparse.ArgumentParser(description='A script to run ACE (Actor-Critic with Emphatic weightings) in parallel.', fromfile_prefix_chars='@')
    parser.add_argument('--num_runs', type=int, help='Number of runs for each combination of parameters')
    parser.add_argument('--num_timesteps', type=int, help='Number of timesteps of experience for each run')
    parser.add_argument('--checkpoints', type=int, nargs='+', help='Number of timesteps after which to save the learned policy')
    parser.add_argument('--num_cpus', type=int, default=-1, help='The number of cpus to use')
    parser.add_argument('--alpha_a', '--actor_step_sizes', type=float, nargs='+', help='Step sizes for the actor')
    parser.add_argument('--lambda_a', '--actor_trace_decay_rates', type=float, nargs='+', default=0., help='Trace decay rates for the actor')
    parser.add_argument('--alpha_c', '--critic_step_sizes', type=float, nargs='+', help='Step sizes for the critic')
    parser.add_argument('--lambda_c', '--critic_trace_decay_rates', type=float, nargs='+', help='Trace decay rates for the critic')
    parser.add_argument('--eta', '--offpac_ace_tradeoff', type=float, nargs='+', help='Values for the parameter that interpolates between OffPAC (0) and ACE (1)')
    parser.add_argument('--num_tiles', type=int, default=8, help='The number of tiles to use in the tile coder')
    parser.add_argument('--num_tilings', type=int, default=8, help='The number of tilings to use in the tile coder')
    parser.add_argument('--num_features', type=int, default=1024, help='The number of features to use in the tile coder')
    parser.add_argument('--bias_unit', type=int, default=1, help='Whether or not to include a bias unit in the tile coder')
    parser.add_argument('--experiment_name', default='experiment1', help='The directory (within "{}") to write files to'.format(os.path.join(output_directory_name, environment_name)))
    parser.add_argument('--args_file', default='run_ace_parameter_sweep.args', help='Name of the file to store the command line arguments in')
    args = parser.parse_args()

    # Configure the input and output directories:
    input_directory = os.path.join(output_directory_name, environment_name, args.experiment_name, behaviour_policy_name)
    output_directory = os.path.join(input_directory, algorithm_name)
    os.makedirs(output_directory, exist_ok=True)

    # Write the command line arguments to a file:
    args_file_path = os.path.join(output_directory, args.args_file)
    utils.save_args_to_file(args, args_file_path)

    # Set up the tile coder:
    tc = TileCoder(min_values=[-1.2, -0.07], max_values=[0.6, 0.07], num_tiles=[args.num_tiles, args.num_tiles], num_tilings=args.num_tilings, num_features=args.num_features, bias_unit=args.bias_unit)

    # Create the memmapped array of learned policies to be populated in parallel:
    policies_file_path = os.path.join(output_directory, 'policies.npy')
    policies_shape = (args.num_runs, len(args.alpha_a), len(args.lambda_a), len(args.alpha_c), len(args.lambda_c), len(args.eta), len(args.checkpoints), num_actions, tc.num_features)
    policies = np.memmap(policies_file_path, shape=policies_shape, mode='w+')

    # Load the input data as a memmap to prevent multiple copies being loaded into memory:
    experience_file_path = os.path.join(input_directory, 'experience.npy')
    experience = np.memmap(experience_file_path, shape=(args.num_runs, args.num_timesteps), dtype=transition_dtype, mode='r')

    # Run ACE concurrently:
    Parallel(n_jobs=args.num_cpus, verbose=10)(delayed(run_ace_tc)(experience, policies, tc, args.num_timesteps, args.checkpoints, run_num, alpha_a_idx, alpha_a, lambda_a_idx, lambda_a, alpha_c_idx, alpha_c, lambda_c_idx, lambda_c, eta_idx, eta) for run_num in range(args.num_runs) for alpha_a_idx, alpha_a in enumerate(args.alpha_a) for lambda_a_idx, lambda_a in enumerate(args.lambda_a) for alpha_c_idx, alpha_c in enumerate(args.alpha_c) for lambda_c_idx, lambda_c in enumerate(args.lambda_c) for eta_idx, eta in enumerate(args.eta))




    # Old script for reference purposes:

    # agent = EAC(num_actions, tc.num_features, args.alpha_u, args.alpha_v, args.alpha_w, args.lamda_v, args.lamda_a, args.bias_unit == 2)
    #
    # # Load data for the run:
    # data_file_path = 'output/{}/behaviour_policies/{}/runs/{}/data.npz'.format(environment_name, behaviour_policy_name, args.run)
    # npzfile = np.load(data_file_path)
    # transitions = npzfile['transitions']
    #
    # num_policies = int(len(transitions) / args.checkpoint_interval)
    # policies = np.empty((num_policies, *agent.actor.policy.u.shape))
    # for t, transition in enumerate(transitions):
    #     s_t, gamma_t, a_t, s_tp1, r_tp1, gamma_tp1 = transition
    #
    #     # Create feature vectors from observations:
    #     x_t = tc.features(s_t)
    #     x_tp1 = tc.features(s_tp1)
    #
    #     # Compute rho:
    #     pi_t = agent.actor.policy.pi(x_t, a_t)
    #     b_t = 1. / num_actions  # assume equiprobable random behaviour policy
    #     rho_t = pi_t / b_t
    #
    #     # Update the learner:
    #     agent.learn(x_t, gamma_t, a_t, r_tp1, x_tp1, gamma_tp1, rho_t)
    #
    #     # If it's a checkpoint timestep,
    #     if t % args.checkpoint_interval == 0:
    #         # store the policy:
    #         policies[t // args.checkpoint_interval] = np.copy(agent.actor.policy.u)
    #
    # # Save the stored policies for later evaluation:
    # policies_file_path = 'output/{}/behaviour_policies/{}/algorithms/{}/bias_unit/{}/alpha_u/{}/alpha_v/{}/alpha_w/{}/lamda_v/{}/lamda_a/{}/runs/{}/policies.npz'.format(environment_name, behaviour_policy_name, algorithm_name, args.bias_unit, args.alpha_u, args.alpha_v, args.alpha_w, args.lamda_v, args.lamda_a, args.run)
    # utils.create_directory(policies_file_path)
    # np.savez_compressed(policies_file_path, policies=policies)
