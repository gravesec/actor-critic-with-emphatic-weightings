import os
import gym
import gym_virtual_office
import random
import argparse
import numpy as np
from src import utils
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
from src.algorithms.ace import LinearACE
from src.algorithms.etd import LinearETD
from src.algorithms.tdrc import LinearTDRC


def generate_experience(experience, run_num, random_seed):
    # Initialize an environment:
    import gym_virtual_office  # Re-import the env in each subprocess or it sometimes isn't found during creation.
    env = gym.make(args.environment).unwrapped
    env.seed(random_seed)
    rng = env.np_random

    # Create the behaviour policy:
    mu = eval(args.behaviour_policy, {'np': np, 'env': env})  # Give the eval'd function access to some objects.

    # Generate the required timesteps of experience:
    o_t = env.reset()['image'].ravel()
    o_t = o_t / np.linalg.norm(o_t)  # normalize feature vector
    a_t = rng.choice(env.action_space.n, p=mu(o_t))
    for t in range(args.num_timesteps):
        # Take action a_t, observe next state o_tp1 and reward r_tp1:
        o_tp1, r_tp1, terminal, _ = env.step(a_t)
        o_tp1 = o_tp1['image'].ravel()
        o_tp1 = o_tp1 / np.linalg.norm(o_tp1)  # normalize feature vector

        # The agent is reset to a starting state after a terminal transition:
        if terminal:
            o_tp1 = env.reset()['image'].ravel()
            o_tp1 = o_tp1 / np.linalg.norm(o_tp1)  # normalize feature vector

        a_tp1 = rng.choice(env.action_space.n, p=mu(o_t))

        # Add the transition to the memmap:
        experience[run_num, t] = (o_t, a_t, r_tp1, o_tp1, a_tp1, terminal)

        # Update temporary variables:
        o_t = o_tp1
        a_t = a_tp1


def evaluate_policy(actor, env, rng):
    g = 0.
    o_t = env.reset()['image'].ravel()
    for t in range(args.max_timesteps):
        a_t = rng.choice(env.action_space.n, p=actor.pi(o_t))
        o_tp1, r_tp1, terminal, _ = env.step(a_t)
        o_tp1 = o_tp1['image'].ravel()
        o_t = o_tp1
        g += r_tp1
        if terminal:
            break
    return g


def run_ace(experience_memmap, policies_memmap, performance_memmap, run_num, config_num, parameters, random_seed):
    # If this run and configuration has already been done (i.e., previous run timed out), exit early:
    if np.count_nonzero(policies_memmap[config_num]['policies'][run_num]) != 0:
        return

    alpha_a, alpha_w, alpha_v, lambda_c, eta = parameters

    # If this is the first run with a set of parameters, save the parameters:
    if run_num == 0:
        policies_memmap[config_num]['parameters'] = (alpha_a, alpha_w, alpha_v, lambda_c, eta, args.gamma)
        performance_memmap[config_num]['parameters'] = (alpha_a, alpha_w, alpha_v, lambda_c, eta, args.gamma)

    # Create the environment to evaluate the learned policy in:
    import gym_virtual_office
    env = gym.make(args.environment).unwrapped
    env.seed(random_seed)
    rng = env.np_random

    # Create the agent:
    # Note: no need to divide learning rate because the feature vectors are already normalized.
    actor = LinearACE(env.action_space.n, dummy_obs.size, alpha_a)
    if args.critic == 'ETD':
        critic = LinearETD(dummy_obs.size, alpha_w, lambda_c)
    else:
        critic = LinearTDRC(dummy_obs.size, alpha_w, lambda_c)

    i = eval(args.interest_function)  # Create the interest function to use.
    mu = eval(args.behaviour_policy, {'np': np, 'env': env})  # Create the behaviour policy and give it access to numpy and the env.

    policies = np.zeros(num_policies, dtype=policies_dtype)
    performance = np.zeros(num_policies, dtype=results_dtype)

    np.seterr(divide='raise', over='raise', invalid='raise')
    try:
        transitions = experience_memmap[run_num]
        gamma_t = 0.
        f_t = 0.
        rho_tm1 = 1.
        for t, transition in enumerate(transitions):
            # Save and evaluate the learned policy if it's a checkpoint timestep:
            if t % args.checkpoint_interval == 0:
                performance[t // args.checkpoint_interval] = (t, [evaluate_policy(actor, env, rng) for _ in range(args.num_evaluation_runs)])
                policies[t // args.checkpoint_interval] = (t, np.copy(actor.theta))

            # Unpack the stored transition.
            o_t, a_t, r_tp1, o_tp1, a_tp1, terminal = transition
            gamma_tp1 = args.gamma if not terminal else 0  # Transition-dependent discounting.
            i_t = i(o_t, gamma_t)
            # Compute importance sampling ratio for the policy:
            pi_t = actor.pi(o_t)
            mu_t = mu(o_t)
            rho_t = pi_t[a_t] / mu_t[a_t]

            f_t = (1 - gamma_t) * i_t + rho_tm1 * gamma_t * f_t if args.normalize else i_t + rho_tm1 * gamma_t * f_t
            m_t = (1 - eta) * i_t + eta * f_t
            if args.critic == 'ETD':
                delta_t = r_tp1 + gamma_tp1 * critic.estimate(o_tp1) - critic.estimate(o_t)
                critic.learn(delta_t, o_t, gamma_t, i_t, rho_t, f_t)
                actor.learn(o_t, a_t, delta_t, m_t, rho_t)
            else:
                delta_t = r_tp1 + gamma_tp1 * critic.estimate(o_tp1) - critic.estimate(o_t)
                critic.learn(delta_t, o_t, gamma_t, o_tp1, gamma_tp1, rho_t)
                actor.learn(o_t, a_t, delta_t, m_t, rho_t)

            gamma_t = gamma_tp1
            rho_tm1 = rho_t

        # Save and evaluate the policy after the final timestep:
        policies[-1] = (t+1, np.copy(actor.theta))
        performance[-1] = (t+1, [evaluate_policy(actor, env, rng) for _ in range(args.num_evaluation_runs)])

        # Save the learned policies and their performance to the memmap:
        performance_memmap[config_num]['performance'][run_num] = performance
        policies_memmap[config_num]['policies'][run_num] = policies
    except (FloatingPointError, ValueError) as e:
        # Save NaN to indicate the weights overflowed and exit early:
        performance_memmap[config_num]['performance'][run_num] = np.full_like(performance, np.NaN)
        policies_memmap[config_num]['policies'][run_num] = np.full_like(policies, np.NaN)
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A script to run ACE (Actor-Critic with Emphatic weightings).', formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
    parser.add_argument('--output_dir', type=str, default='experiment', help='The directory to write experiment files to')
    parser.add_argument('--experience_file', type=str, default='experience.npy', help='The file to read experience from')
    parser.add_argument('--num_runs', type=int, default=5, help='The number of independent runs of experience to generate')
    parser.add_argument('--num_timesteps', type=int, default=20000, help='The number of timesteps of experience to generate per run')
    parser.add_argument('--random_seed', type=int, default=1944801619, help='The master random seed to use')
    parser.add_argument('--num_cpus', type=int, default=-1, help='The number of cpus to use (-1 for all).')
    parser.add_argument('--checkpoint_interval', type=int, default=1000, help='The number of timesteps after which to save the learned policy.')
    parser.add_argument('--num_evaluation_runs', type=int, default=10, help='The number of times to evaluate each policy')
    parser.add_argument('--max_timesteps', type=int, default=1000, help='The maximum number of timesteps allowed per policy evaluation')
    parser.add_argument('--critic', type=str, choices=['TDRC', 'ETD'], default='TDRC', help='Which critic to use.')
    parser.add_argument('--normalize', type=int, choices=[0, 1], default=0, help='Estimate the discounted follow-on distribution instead of the discounted follow-on visit counts.')
    parser.add_argument('--interest_function', type=str, default='lambda s, g=1: 1.', help='Interest function to use. Example: \'lambda s, g=1: 1. if g==0. else 0.\' (episodic interest function)')
    parser.add_argument('--behaviour_policy', type=str, default='lambda s: np.array([.2, .3, .3, .2])', help='Policy to use. Default is uniform random, slightly biased towards the \'south\' action.')
    parser.add_argument('--environment', type=str, default='VirtualOffice-v0', help='An OpenAI Gym environment string.')
    parser.add_argument('--gamma', '--discount_rate', type=float, default=.9, help='Discount rate.')
    parser.add_argument('-p', '--parameters', type=float, nargs=5, action='append', metavar=('alpha_a', 'alpha_w', 'alpha_v', 'lambda', 'eta'), help='Parameters to use. Can be specified multiple times to run multiple configurations in parallel.')
    args, unknown_args = parser.parse_known_args()

    # Generate the random seed for each run without replacement to prevent the birthday paradox:
    random.seed(args.random_seed)
    random_seeds = random.sample(range(2**32), args.num_runs)

    # Save the command line arguments in a format interpretable by argparse:
    output_dir = Path(args.output_dir)
    utils.save_args_to_file(args, output_dir / Path(parser.prog).with_suffix('.args'))

    # Make a dummy env to get action/observation shape info.
    dummy_env = gym.make(args.environment).unwrapped
    dummy_obs = dummy_env.reset()['image'].ravel()
    num_policies = args.num_timesteps // args.checkpoint_interval + 1

    # If the input file already exists:
    transition_dtype = np.dtype([
        ('s_t', float, dummy_obs.size),
        ('a_t', int),
        ('r_tp1', float),
        ('s_tp1', float, dummy_obs.size),
        ('a_tp1', int),
        ('terminal', bool)
    ])
    if os.path.isfile(output_dir / args.experience_file):
        # load it as a memmap to prevent a copy being loaded into memory in each sub-process:
        experience_memmap = np.lib.format.open_memmap(output_dir / args.experience_file, mode='r')
    else:
        # otherwise, create it and populate it in parallel:
        experience_memmap = np.lib.format.open_memmap(output_dir / args.experience_file, shape=(args.num_runs, args.num_timesteps), dtype=transition_dtype, mode='w+')
        with utils.tqdm_joblib(tqdm(total=args.num_runs)) as progress_bar:
            Parallel(n_jobs=args.num_cpus, verbose=0)(
                delayed(generate_experience)(experience_memmap, run_num, random_seed)
                for run_num, random_seed in enumerate(random_seeds)
            )

    # Create or load the file for storing learned policies:
    policies_memmap_path = str(output_dir / 'policies.npy')
    parameters_dtype = np.dtype([
        ('alpha_a', float),
        ('alpha_w', float),
        ('alpha_v', float),
        ('lambda', float),
        ('eta', float),
        ('gamma', float)
    ])
    policies_dtype = np.dtype([
        ('timesteps', int),
        ('weights', float, (dummy_env.action_space.n, dummy_obs.size))
    ])
    configuration_dtype = np.dtype([
        ('parameters', parameters_dtype),
        ('policies', policies_dtype, (args.num_runs, num_policies))
    ])
    # If the file for storing learned policies already exists:
    if os.path.isfile(policies_memmap_path):
        # load it as a memmap to prevent a copy being loaded into memory in each sub-process:
        policies_memmap = np.lib.format.open_memmap(policies_memmap_path, mode='r+')
    else:
        # otherwise, create it:
        policies_memmap = np.lib.format.open_memmap(policies_memmap_path, shape=(len(args.parameters),), dtype=configuration_dtype, mode='w+')

    # Create or load the file for storing policy performance results:
    performance_memmap_path = str(output_dir / 'episodic_performance.npy')
    results_dtype = np.dtype([
        ('timesteps', int),
        ('results', float, args.num_evaluation_runs)
    ])
    performance_dtype = np.dtype([
        ('parameters', parameters_dtype),
        ('performance', results_dtype, (args.num_runs, num_policies))
    ])
    # If the file for storing the performance results for the learned policies already exists:
    if os.path.isfile(performance_memmap_path):
        # load it as a memmap to prevent a copy being loaded into memory in each sub-process:
        performance_memmap = np.lib.format.open_memmap(performance_memmap_path, mode='r+')
    else:
        # otherwise, create it:
        performance_memmap = np.lib.format.open_memmap(performance_memmap_path, shape=(len(args.parameters),), dtype=performance_dtype, mode='w+')

    # Run ACE for each configuration in parallel:
    with utils.tqdm_joblib(tqdm(total=args.num_runs * len(args.parameters), smoothing=0)) as progress_bar:
        Parallel(n_jobs=args.num_cpus, verbose=0)(
            delayed(run_ace)(experience_memmap, policies_memmap, performance_memmap, run_num, config_num, parameters, random_seed)
            for config_num, parameters in enumerate(args.parameters)
            for run_num, random_seed in enumerate(random_seeds)
        )
