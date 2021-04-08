import os
import gym
import gym_puddle
import random
import argparse
import numpy as np
from src import utils
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
from src.algorithms.ace import BinaryACE
from src.algorithms.etd import BinaryETD
from src.algorithms.tdc import BinaryTDC, BinaryGQ
from src.algorithms.tdrc import BinaryTDRC
from src.algorithms.fhat import BinaryFHat
from src.function_approximation.tile_coder import TileCoder
from evaluate_policies import evaluate_policy


def run_ace(experience_memmap, policies_memmap, performance_memmap, run_num, config_num, parameters, random_seed, experience_memmap_test, num_test_eval, performance_memmap2):

    alpha_a, alpha_w, alpha_v, lambda_c, eta = parameters

    # If this is the first run with a set of parameters, save the parameters:
    if run_num == 0:
        performance_memmap2[config_num]['parameters'] = (alpha_a, alpha_w, alpha_v, lambda_c, eta, args.gamma, args.num_tiles_per_dim, args.num_tilings, args.bias_unit)

    # Create the environment to evaluate the learned policy in:
    import gym_puddle
    env = gym.make(args.environment).unwrapped
    env.seed(random_seed)
    rng = env.np_random

    actor = BinaryACE(env.action_space.n, tc.total_num_tiles, alpha_a / tc.num_active_features)

    performance = np.zeros((num_policies, args.num_evaluation_runs), dtype=float)
    performance_excursions = np.zeros((num_policies, num_test_eval), dtype=float)

    np.seterr(divide='raise', over='raise', invalid='raise')
    try:
        if np.isnan(performance_memmap[config_num]['results'][run_num][0,0]):
            raise ValueError("oops")

        if performance_memmap2[config_num]['results'][run_num][0,0] != 0.0:
            return

        transitions = experience_memmap[run_num]

        transitions_len = len(transitions)
        t = 0
        while t < transitions_len:
            # Save and evaluate the learned policy if it's a checkpoint timestep:
            actor.theta = policies_memmap[config_num]['policies'][run_num][t // args.checkpoint_interval][1]

            # #evaluate episodic, copy excursions
            # performance[t // args.checkpoint_interval] = [evaluate_policy(actor, tc, env, rng, args.max_timesteps) for _ in range(args.num_evaluation_runs)]
            # performance_excursions[t // args.checkpoint_interval] = performance_memmap[config_num]['results_excursions'][run_num][t // args.checkpoint_interval]

            # #copy episodic, evaluate excursions
            # performance[t // args.checkpoint_interval] = performance_memmap[config_num]['results'][run_num][t // args.checkpoint_interval]
            # perf_excursions = []
            # for sample in range(num_test_eval):
            #     env.state = experience_memmap_test[run_num][sample][0]
            #     perf_excursions.append(evaluate_policy(actor, tc, env, rng, args.max_timesteps,state=experience_memmap_test[run_num][sample][0]))
            # performance_excursions[t // args.checkpoint_interval] = perf_excursions

            #evaluate episodic, evaluate excursions
            performance[t // args.checkpoint_interval] = [evaluate_policy(actor, tc, env, rng, args.max_timesteps) for _ in range(args.num_evaluation_runs)]
            perf_excursions = []
            for sample in range(num_test_eval):
                env.state = experience_memmap_test[run_num][sample][0]
                perf_excursions.append(evaluate_policy(actor, tc, env, rng, args.max_timesteps,state=experience_memmap_test[run_num][sample][0]))
            performance_excursions[t // args.checkpoint_interval] = perf_excursions

            t += args.checkpoint_interval

        # Save and evaluate the policy after the final timestep:
        actor.theta = policies_memmap[config_num]['policies'][run_num][-1][1]

        # #evaluate episodic, copy excursions
        # performance[-1] = [evaluate_policy(actor, tc, env, rng, args.max_timesteps) for _ in range(args.num_evaluation_runs)]
        # performance_excursions[-1] = performance_memmap[config_num]['results_excursions'][run_num][-1]

        # #copy episodic, evaluate excursions
        # performance[-1] = performance_memmap[config_num]['results'][run_num][-1]
        # perf_excursions = []
        # for sample in range(num_test_eval):
        #     env.state = experience_memmap_test[run_num][sample][0]
        #     perf_excursions.append(evaluate_policy(actor, tc, env, rng, args.max_timesteps,state=experience_memmap_test[run_num][sample][0]))
        # performance_excursions[-1] = perf_excursions

        #evaluate episodic, evaluate excursions
        performance[-1] = [evaluate_policy(actor, tc, env, rng, args.max_timesteps) for _ in range(args.num_evaluation_runs)]
        perf_excursions = []
        for sample in range(num_test_eval):
            env.state = experience_memmap_test[run_num][sample][0]
            perf_excursions.append(evaluate_policy(actor, tc, env, rng, args.max_timesteps,state=experience_memmap_test[run_num][sample][0]))
        performance_excursions[-1] = perf_excursions

        # Save the learned policies and their performance to the memmap:
        performance_memmap2[config_num]['results'][run_num] = performance
        performance_memmap2[config_num]['results_excursions'][run_num] = performance_excursions
    except (FloatingPointError, ValueError) as e:
        # Save NaN to indicate the weights overflowed and exit early:
        print("Floating point error: ", parameters, run_num)
        performance_memmap2[config_num]['results'][run_num] = np.full_like(performance, np.NaN)
        performance_memmap2[config_num]['results_excursions'][run_num] = np.full_like(performance_excursions, np.NaN)
        return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='A script to run ACE (Actor-Critic with Emphatic weightings).', formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
    parser.add_argument('--output_dir', type=str, default='experiment', help='The directory to write experiment files to')
    parser.add_argument('--experience_file', type=str, default='experiment/experience.npy', help='The file to read experience from')
    parser.add_argument('--experience_file_test', type=str, default='experiment/experience_test.npy', help='The file to read experience from for evaluating the excursions objective')
    parser.add_argument('--num_cpus', type=int, default=-1, help='The number of cpus to use (-1 for all).')

    # Policy evaluation parameters:
    parser.add_argument('--checkpoint_interval', type=int, default=5000, help='The number of timesteps after which to save the learned policy.')
    parser.add_argument('--num_evaluation_runs', type=int, default=5, help='The number of times to evaluate each policy')
    parser.add_argument('--max_timesteps', type=int, default=1000, help='The maximum number of timesteps allowed per policy evaluation')
    parser.add_argument('--random_seed', type=int, default=1944801619, help='The master random seed to use')

    # Experiment parameters:
    parser.add_argument('--critic', type=str, choices=['ETD', 'TDRC'], default='TDRC', help='Which critic to use.')
    parser.add_argument('--direct_f', type=int, choices=[0, 1], default=0, help='Use a function approximator to estimate the emphatic weightings.')
    parser.add_argument('--all_actions', type=int, choices=[0, 1], default=0, help='Use all-actions updates instead of TD error-based updates.')
    parser.add_argument('--normalize', type=int, choices=[0, 1], default=0, help='Estimate the discounted follow-on distribution instead of the discounted follow-on visit counts.')
    parser.add_argument('--interest_function', type=str, default='lambda s, g=1: 1.', help='Interest function to use. Example: \'lambda s, g=1: 1. if g==0. else 0.\' (episodic interest function)')
    parser.add_argument('--behaviour_policy', type=str, default='lambda s: np.ones(env.action_space.n)/env.action_space.n', help='Policy to use. Default is uniform random. Another Example: \'lambda s: np.array([.9, .05, .05]) if s[1] < 0 else np.array([.05, .05, .9]) \' (energy pumping policy w/ 15 percent randomness)')
    parser.add_argument('--environment', type=str, default='MountainCar-v0', help='An OpenAI Gym environment string.')
    parser.add_argument('--gamma', '--discount_rate', type=float, default=.99, help='Discount rate.')
    parser.add_argument('-p', '--parameters', type=float, nargs=5, action='append', metavar=('alpha_a', 'alpha_w', 'alpha_v', 'lambda', 'eta'), help='Parameters to use. Can be specified multiple times to run multiple configurations in parallel.')
    parser.add_argument('--num_tiles_per_dim', type=int, nargs='+', default=[4, 4], help='The number of tiles per dimension to use in the tile coder.')
    parser.add_argument('--num_tilings', type=int, default=8, help='The number of tilings to use in the tile coder.')
    parser.add_argument('--bias_unit', type=int, choices=[0, 1], default=1, help='Whether or not to include a bias unit in the tile coder.')
    args, unknown_args = parser.parse_known_args()

    # Save the command line arguments in a format interpretable by argparse:
    output_dir = Path(args.output_dir)
    utils.save_args_to_file(args, output_dir / Path(parser.prog).with_suffix('.args'))

    # Load the input data as a memmap to prevent a copy being loaded into memory in each sub-process:
    experience_memmap = np.lib.format.open_memmap(args.experience_file, mode='r')
    num_runs, num_timesteps = experience_memmap.shape

    # Load the input data as a memmap to prevent a copy being loaded into memory in each sub-process:
    experience_memmap_test = np.lib.format.open_memmap(args.experience_file_test, mode='r')
    num_runs, num_test_eval = experience_memmap_test.shape

    # Generate the random seed for each run without replacement to prevent the birthday paradox:
    random.seed(args.random_seed)
    random_seeds = random.sample(range(2**32), num_runs)

    # Create the tile coder to be used for all parameter settings:
    dummy_env = gym.make(args.environment).unwrapped  # Make a dummy env to get shape info.
    tc = TileCoder(np.array([dummy_env.observation_space.low, dummy_env.observation_space.high]).T, args.num_tiles_per_dim, args.num_tilings, args.bias_unit)

    # Create the memmapped array of learned policies that will be populated in parallel:
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
    policy_dtype = np.dtype([
        ('timesteps', int),
        ('weights', float, (dummy_env.action_space.n, tc.total_num_tiles))
    ])
    num_policies = num_timesteps // args.checkpoint_interval + 1
    configuration_dtype = np.dtype([
        ('parameters', parameters_dtype),
        ('policies', policy_dtype, (num_runs, num_policies))
    ])
    policies_memmap_path = str(output_dir / 'policies.npy')
    if os.path.isfile(policies_memmap_path):
        policies_memmap = np.lib.format.open_memmap(policies_memmap_path, mode='r+')
    else:
        policies_memmap = np.lib.format.open_memmap(policies_memmap_path, shape=(len(args.parameters),), dtype=configuration_dtype, mode='w+')

    # Create the memmapped array of performance results for the learned policies:
    performance_dtype = np.dtype([
        ('parameters', parameters_dtype),
        ('results', float, (num_runs, num_policies, args.num_evaluation_runs)),
        ('results_excursions', float, (num_runs, num_policies, num_test_eval))
    ])
    performance_memmap_path = str(output_dir / 'performance.npy')
    if os.path.isfile(performance_memmap_path):
        performance_memmap = np.lib.format.open_memmap(performance_memmap_path, mode='r+')
    else:
        performance_memmap = np.lib.format.open_memmap(performance_memmap_path, shape=(len(args.parameters),), dtype=performance_dtype, mode='w+')


    # Create the memmapped array of performance results for the learned policies:
    performance_dtype2 = np.dtype([
        ('parameters', parameters_dtype),
        ('results', float, (num_runs, num_policies, args.num_evaluation_runs)),
        ('results_excursions', float, (num_runs, num_policies, num_test_eval))
    ])
    performance_memmap_path2 = str(output_dir / 'performance2.npy')
    if os.path.isfile(performance_memmap_path2):
        performance_memmap2 = np.lib.format.open_memmap(performance_memmap_path2, mode='r+')
    else:
        performance_memmap2 = np.lib.format.open_memmap(performance_memmap_path2, shape=(len(args.parameters),), dtype=performance_dtype2, mode='w+')

    # Run ACE for each configuration in parallel:
    with utils.tqdm_joblib(tqdm(total=num_runs * len(args.parameters), smoothing=0)) as progress_bar:
        Parallel(n_jobs=args.num_cpus, verbose=0)(
            delayed(run_ace)(experience_memmap, policies_memmap, performance_memmap, run_num, config_num, parameters, random_seed, experience_memmap_test, num_test_eval, performance_memmap2)
            for config_num, parameters in enumerate(args.parameters)
            for run_num, random_seed in enumerate(random_seeds)
        )
