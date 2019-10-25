import os
import gym
import random
import argparse
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from src.algorithms.ace import ACE
from src.function_approximation.tile_coder import TileCoder

# TODO: implement our own mountain car environment (https://en.wikipedia.org/wiki/Mountain_car_problem) because openai's is slow, stateful, and has bizarre decisions built in like time limits and the inability to get properties of an environment without creating an instantiation of it.

num_actions = 3
min_state_values = [-1.2, -0.07]
max_state_values = [0.6, 0.07]


# This function is not used anywhere; I've been calling it from the Python console while debugging scripts.
def evaluate(policy):
    weights = policy['weights']
    env = gym.make('MountainCar-v0').env
    actor = ACE(weights.shape[0], weights.shape[1])
    actor.theta = weights
    tc = TileCoder(min_state_values, max_state_values, [int(policy['num_tiles']), int(policy['num_tiles'])], int(policy['num_tilings']), weights.shape[1], int(policy['bias_unit']))

    g_t = 0.
    s_t = env.reset()
    for t in range(1000):

        # Get feature vector for the current state:
        indices_t = tc.indices(s_t)

        # Select an action:
        a_t = np.random.choice(env.action_space.n, p=actor.pi(indices_t))

        # Take action a_t, observe next state s_tp1 and reward r_tp1:
        s_tp1, r_tp1, terminal, _ = env.step(a_t)

        # Add reward:
        g_t += r_tp1

        env.render()

        # If done, break the loop:
        if terminal:
            break
    return g_t


def evaluate_policy(performance, policies, max_timesteps, objective, evaluation_run_num, ace_run_num, config_num, policy_num, random_seed):

    # Initialize the environment:
    env = gym.make('MountainCar-v0').env  # Get the underlying environment object to bypass the built-in timestep limit.

    # Configure random state for the run:
    env.seed(random_seed)
    rng = env.np_random

    # Load the policy to evaluate:
    gamma, alpha_a, alpha_c, lambda_c, eta, num_tiles, num_tilings, bias_unit, t, policy = policies[ace_run_num, config_num, policy_num]
    agent = ACE(num_actions, policy.shape[1])
    agent.theta = policy

    tc = TileCoder(min_state_values, max_state_values, [int(num_tiles), int(num_tiles)], int(num_tilings), policy.shape[1], int(bias_unit))

    # Set up the environment:
    if objective == 'episodic':
        # Use the mountain_car start state:
        s_t = env.reset()
    elif objective == 'excursions':
        # Sample a state from the behaviour policy's state distribution:
        # TODO: sample a state from the generated_experience.npy file?
        raise NotImplementedError
    else:
        raise NotImplementedError

    # Evaluate the policy:
    g_t = 0.
    for t in range(max_timesteps):

        # Get feature vector for the current state:
        indices_t = tc.indices(s_t)

        # Select an action:
        a_t = rng.choice(env.action_space.n, p=agent.pi(indices_t))

        # Take action a_t, observe next state s_tp1 and reward r_tp1:
        s_tp1, r_tp1, terminal, _ = env.step(a_t)

        # Add reward:
        g_t += r_tp1

        # If done, break the loop:
        if terminal:
            break

    # Write the total rewards received to file:
    performance[evaluation_run_num, ace_run_num, config_num, policy_num] = g_t


if __name__ == '__main__':

    # Parse command line arguments:
    parser = argparse.ArgumentParser(description='A script to evaluate policies on the Mountain Car environment in parallel.', fromfile_prefix_chars='@', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experiment_name', type=str, default='experiment', help='The directory to read/write experiment files to/from')
    parser.add_argument('--num_evaluation_runs', type=int, default=5, help='The number of times to evaluate each policy')
    parser.add_argument('--max_timesteps', type=int, default=1000, help='The maximum number of timesteps allowed per run')
    parser.add_argument('--random_seed', type=int, default=1944801619, help='The master random seed to use')
    parser.add_argument('--num_cpus', type=int, default=-1, help='The number of cpus to use (-1 means all)')
    parser.add_argument('--backend', type=str, choices=['loky', 'threading'], default='loky', help='The backend to use (\'loky\' for processes or \'threading\' for threads; always use \'loky\' because Python threading is terrible).')
    parser.add_argument('--objective', type=str, choices=['excursions', 'alternative_life', 'episodic'], default='episodic', help='Determines the state distribution the starting state is sampled from (excursions: behaviour policy, alternative life: target policy, episodic: mountain car start state.)')
    args = parser.parse_args()

    # Generate the random seed for each run without replacement:
    random.seed(args.random_seed)
    random_seeds = random.sample(range(2**32), args.num_evaluation_runs)

    # Create the output directory:
    experiment_path = Path(args.experiment_name)
    os.makedirs(experiment_path, exist_ok=True)

    # Save the command line arguments in a format interpretable by argparse:
    with open(experiment_path / Path(parser.prog).with_suffix('.args'), 'w') as args_file:
        for key, value in vars(args).items():
            args_file.write('--{}\n{}\n'.format(key, value))

    # Load the learned policies to evaluate:
    policies = np.lib.format.open_memmap(str(experiment_path / 'policies.npy'), mode='r')
    num_ace_runs, num_configurations, num_policies = policies.shape

    # Create the memmapped array of results to be populated in parallel:
    performance = np.lib.format.open_memmap(str(experiment_path / '{}_performance.npy'.format(args.objective)), shape=(args.num_evaluation_runs, num_ace_runs, num_configurations, num_policies), dtype=float, mode='w+')

    # Evaluate the learned policies in parallel:
    Parallel(n_jobs=args.num_cpus, verbose=10, backend=args.backend)(
        delayed(evaluate_policy)(
            performance, policies, args.max_timesteps, args.objective,
            evaluation_run_num, ace_run_num, config_num, policy_num, random_seed
        )
        for evaluation_run_num, random_seed in enumerate(random_seeds)
        for ace_run_num in range(num_ace_runs)
        for config_num in range(num_configurations)
        for policy_num in range(num_policies)
    )

    # Close the memmap file:
    del policies
    del performance
