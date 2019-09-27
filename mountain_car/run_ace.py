import os
import argparse
import numpy as np
from pathlib import Path
from itertools import product
from joblib import Parallel, delayed
from .generate_experience import transition_dtype, num_actions
from .tiles3 import tiles


class TileCoder:
    def __init__(self, min_values, max_values, num_tiles, num_tilings, num_features, bias_unit=False):
        self.num_tiles = np.array(num_tiles)
        self.scale_factor = self.num_tiles / (np.array(max_values) - np.array(min_values))
        self.num_tilings = num_tilings
        self.bias_unit = bias_unit
        self.num_features = num_features
        self.num_active_features = self.num_tilings + self.bias_unit

    def indices(self, observations):
        return np.array(tiles(int(self.num_features - self.bias_unit), self.num_tilings, list(np.array(observations) * self.scale_factor)), dtype=np.intp)

    def features(self, indices):
        features = np.zeros(self.num_features)
        features[indices] = 1.
        if self.bias_unit:
            features[self.num_features - 1] = 1.
        return features


class ACE:
    def __init__(self):
        pass

    def learn(self):
        pass


def run_ace(parameters):
    # Set up the tile coder:
    tc = TileCoder(min_values=[-1.2, -0.07], max_values=[0.6, 0.07], num_tiles=[args.num_tiles, args.num_tiles], num_tilings=args.num_tilings, num_features=args.num_features, bias_unit=args.bias_unit)
    pass


if __name__ == '__main__':

    # Parse command line arguments:
    parser = argparse.ArgumentParser(description='A script to run ACE (Actor-Critic with Emphatic weightings) in parallel.', fromfile_prefix_chars='@')
    parser.add_argument('--experiment_name', default='experiment', help='The directory to read/write experiment files to')
    parser.add_argument('--num_runs', type=int, default=5, help='The number of independent runs of experience')
    parser.add_argument('--num_timesteps', type=int, default=1000, help='The number of timesteps of experience per run')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='The number of timesteps after which to save the learned policy')
    parser.add_argument('--num_cpus', type=int, default=-1, help='The number of cpus to use (-1 for all)')
    parser.add_argument('--backend', type=str, choices=['loky', 'threading'], default='loky', help='The backend to use (\'loky\' for processes or \'threading\' for threads; always use \'loky\')')

    parser.add_argument('--iteration_type', type=str, choices=['corresponding', 'combinations'], default='corresponding', help='Whether to iterate over corresponding parameter settings (zip), or all possible combinations of parameter settings (product)')
    parser.add_argument('--alpha_a', '--actor_step_sizes', type=float, nargs='+', help='Step sizes for the actor')
    parser.add_argument('--lambda_a', '--actor_trace_decay_rates', type=float, nargs='+', default=[0.], help='Trace decay rates for the actor')
    parser.add_argument('--alpha_c', '--critic_step_sizes', type=float, nargs='+', help='Step sizes for the critic')
    parser.add_argument('--lambda_c', '--critic_trace_decay_rates', type=float, nargs='+', help='Trace decay rates for the critic')
    parser.add_argument('--eta', '--offpac_ace_tradeoff', type=float, nargs='+', default=[0.,1.], help='Values for the parameter that interpolates between OffPAC (0) and ACE (1)')
    parser.add_argument('--num_tiles', type=int, nargs='+', default=4, help='The number of tiles to use in the tile coder')
    parser.add_argument('--num_tilings', type=int, nargs='+', default=4, help='The number of tilings to use in the tile coder')
    parser.add_argument('--num_features', type=int, default=1024, help='The number of features to use in the tile coder')
    parser.add_argument('--bias_unit', type=int, nargs='+', default=1, help='Whether or not to include a bias unit in the tile coder')
    args = parser.parse_args()

    # Save the command line arguments in a format interpretable by argparse:
    experiment_path = Path(args.experiment_name)
    with open(experiment_path / Path(parser.prog).with_suffix('.args'), 'w') as args_file:
        for key, value in vars(args).items():
            if isinstance(value, list):
                value = '\n'.join(str(i) for i in value)
            args_file.write('--{}\n{}\n'.format(key, value))

    # Compute the number of policies that will be saved during training:
    num_policies = args.num_timesteps / args.checkpoint_interval

    if args.iteration_type == 'combinations':
        # Run ACE for all possible combinations of the given parameters:
        policies_shape = (len(args.alpha_a), len(args.lambda_a), len(args.alpha_c), len(args.lambda_c), len(args.eta), len(args.num_tiles), len(args.num_tilings), len(args.bias_unit), args.num_runs, num_policies, num_actions, args.num_features)
    else:
        # Run ACE for each set of parameters:
        parameters = [args.alpha_a, args.lambda_a, args.alpha_c, args.lambda_c, args.eta, args.num_tiles, args.num_tilings, args.bias_unit]
        lens = [len(p) for p in parameters]
        policies_shape = (max(lens), args.num_runs, num_policies, num_actions, args.num_features)

    # Create the memmapped array of learned policies to be populated in parallel:
    policies = np.memmap(experiment_path / 'policies.npy', shape=policies_shape, mode='w+')

    # Load the input data as a memmap to prevent multiple copies being loaded into memory:
    experience = np.memmap(experiment_path / 'experience.npy', shape=(args.num_runs, args.num_timesteps), dtype=transition_dtype, mode='r')

    # Run ACE in parallel:
    if args.iteration_type == 'combinations':
        Parallel(n_jobs=args.num_cpus, verbose=10)(
            delayed(run_ace)(
                policies, experience, args.num_timesteps, , run_num, alpha_a_idx, alpha_a, lambda_a_idx, lambda_a, alpha_c_idx, alpha_c, lambda_c_idx, lambda_c, eta_idx, eta) for run_num in range(args.num_runs) for alpha_a_idx, alpha_a in enumerate(args.alpha_a) for lambda_a_idx, lambda_a in enumerate(args.lambda_a) for alpha_c_idx, alpha_c in enumerate(args.alpha_c) for lambda_c_idx, lambda_c in enumerate(args.lambda_c) for eta_idx, eta in enumerate(args.eta))
    else:
        Parallel(n_jobs=args.num_cpus, verbose=10)(delayed(run_ace)(experience, policies, tc, args.num_timesteps, args.checkpoints, run_num, alpha_a_idx, alpha_a, lambda_a_idx, lambda_a, alpha_c_idx, alpha_c, lambda_c_idx, lambda_c, eta_idx, eta) for run_num in range(args.num_runs) for alpha_a_idx, alpha_a in enumerate(args.alpha_a) for lambda_a_idx, lambda_a in enumerate(args.lambda_a) for alpha_c_idx, alpha_c in enumerate(args.alpha_c) for lambda_c_idx, lambda_c in enumerate(args.lambda_c) for eta_idx, eta in enumerate(args.eta))

    del policies

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
