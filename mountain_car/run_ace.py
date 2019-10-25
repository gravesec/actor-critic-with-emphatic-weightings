import os
import argparse
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from mountain_car.ace import TileCoder, TOETD, ACE
from mountain_car.tdc import BinaryTDC
from mountain_car.generate_experience import num_actions, min_state_values, max_state_values


# TODO: Figure out how to do checkpointing (i.e. keep track of progress via a memmap so if the process gets killed it can pick up where it left off).
# TODO: Figure out how to append to a memmap in case we want to do more runs later on (we might get this without any extra work with checkpointing).


def run_ace(policies, experience, behaviour_policy, checkpoint_interval, num_features, interest_function, run_num, config_num, parameters):
    gamma, alpha_a, alpha_c, lambda_c, eta, num_tiles, num_tilings, bias_unit = parameters

    # Create the interest function to use:
    i = eval(interest_function)

    # Create the behaviour policy to query action probabilities from:
    mu = eval(behaviour_policy, {'np': np})  # Give the eval'd function access to numpy.

    # Create the agent:
    tc = TileCoder(min_state_values, max_state_values, [int(num_tiles), int(num_tiles)], int(num_tilings), num_features, int(bias_unit))
    actor = ACE(num_actions, num_features)
    critic = BinaryTDC(num_features, alpha_c, alpha_c / 10., lambda_c)

    # Get the run of experience to learn from:
    transitions = experience[run_num]

    # Learn from the experience:
    indices_tp1 = None  # Store features to prevent re-generating them.
    for t, transition in enumerate(transitions):

        # Save the learned policy if it's a checkpoint timestep:
        if t % checkpoint_interval == 0:
            policies[run_num, config_num, t // checkpoint_interval] = (gamma, alpha_a, alpha_c, lambda_c, eta, num_tiles, num_tilings, bias_unit, t, np.copy(actor.theta))

        s_t, a_t, r_tp1, s_tp1, terminal = transition

        # Transition-dependent discounting:
        gamma_tp1 = gamma if not terminal else 0

        # Get feature vectors for each state:
        indices_t = tc.indices(s_t) if indices_tp1 is None else indices_tp1
        indices_tp1 = tc.indices(s_tp1)

        # Get interest for the current state:
        i_t = i(t)

        # Compute importance sampling ratio for the policies:
        pi_t = actor.pi(indices_t)
        mu_t = mu(s_t)
        rho_t = pi_t[a_t] / mu_t[a_t]

        # Compute TD error:
        v_t = critic.estimate(indices_t)
        v_tp1 = critic.estimate(indices_tp1)
        delta_t = r_tp1 + gamma_tp1 * v_tp1 - v_t

        # Update actor:
        actor.learn(gamma_tp1, i_t, eta, alpha_a, rho_t, delta_t, indices_t, a_t)

        # Update critic:
        critic.learn(delta_t, indices_t, gamma, indices_tp1, gamma_tp1, rho_t)

    # Save the policy after the final timestep:
    policies[run_num, config_num, -1] = (gamma, alpha_a, alpha_c, lambda_c, eta, num_tiles, num_tilings, bias_unit, t+1, np.copy(actor.theta))


if __name__ == '__main__':

    # Parse command line arguments:
    parser = argparse.ArgumentParser(description='A script to run ACE (Actor-Critic with Emphatic weightings) in parallel.', fromfile_prefix_chars='@', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experiment_name', type=str, default='experiment', help='The directory to read/write experiment files to/from')
    parser.add_argument('--checkpoint_interval', type=int, default=100, help='The number of timesteps after which to save the learned policy.')
    parser.add_argument('--num_cpus', type=int, default=-1, help='The number of cpus to use (-1 for all).')
    parser.add_argument('--backend', type=str, choices=['loky', 'threading'], default='loky', help='The backend to use (\'loky\' for processes or \'threading\' for threads; always use \'loky\' because Python threading is terrible).')
    parser.add_argument('--interest_function', type=str, default='lambda s: 1.', help='Interest function to use. Example: \'lambda s: 1. if s==(-.5,.0) else 0.\' (episodic interest function)')
    parser.add_argument('--behaviour_policy', type=str, default='lambda s: np.array([1/3, 1/3, 1/3])', help='Policy used to generate data. Example: \'lambda s: np.array([.9, .05, .05]) if s[1] < 0 else np.array([.05, .05, .9]) \' (energy pumping policy w/ 15 percent randomness)')
    parser.add_argument('--num_features', type=int, default=512, help='The number of features to use in the tile coder.')
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

    # Load the input data as a memmap to prevent a copy being loaded into memory in each sub-process:
    experience = np.lib.format.open_memmap(str(experiment_path / 'experience.npy'), mode='r')
    num_runs, num_timesteps = experience.shape

    # Create the memmapped array of learned policies that will be populated in parallel:
    num_policies = num_timesteps // args.checkpoint_interval + 1
    policies_dtype = np.dtype(
        [
            ('gamma', float),
            ('alpha_a', float),
            ('alpha_c', float),
            ('lambda_c', float),
            ('eta', float),
            ('num_tiles', int),
            ('num_tilings', int),
            ('bias_unit', bool),
            ('timesteps', int),
            ('weights', float, (num_actions, args.num_features))
        ]
    )
    policies = np.lib.format.open_memmap(str(experiment_path / 'policies.npy'), shape=(num_runs, len(args.parameters), num_policies), dtype=policies_dtype, mode='w+')

    # Run ACE for each set of parameters in parallel:
    Parallel(n_jobs=args.num_cpus, verbose=10)(
        delayed(run_ace)(
            policies, experience,
            args.behaviour_policy, args.checkpoint_interval, args.num_features, args.interest_function,
            run_num, config_num, parameters
        )
        for config_num, parameters in enumerate(args.parameters)
        for run_num in range(num_runs)
    )

    # Close the memmap files:
    del experience
    del policies
