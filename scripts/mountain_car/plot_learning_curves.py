# Add the src directory to the path, because python doesn't look in subfolders:
import sys, os
src_path = os.path.join(os.getcwd(), '.')
if src_path not in sys.path:
    sys.path.append(src_path)

import argparse
import numpy as np
from src import utils
import scipy.stats as st
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from generate_data import environment_name, behaviour_policy_name

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('use_excursions', type=int, choices=[0, 1], help='whether to use excursions objective or not')
    args = parser.parse_args()

    # Plot the highest return for each algorithm:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Returns')
    ax.set_ylim(-1000, 0)
    fig.suptitle('{}learning curves for {}'.format('excursions ' if args.use_excursions else '', environment_name), fontsize=12)
    ax.set_title('Data generated using {} policy'.format(behaviour_policy_name), fontsize=10)

    algorithms_directory = 'output/{}/behaviour_policies/{}/algorithms/'.format(environment_name, behaviour_policy_name)

    # For each algorithm:
    for algorithm_name in os.listdir(algorithms_directory):
        algorithm_directory = algorithms_directory + algorithm_name

        # Load the path for the highest-return parameters:
        highest_return_path = str(np.loadtxt(algorithm_directory + '/{}highest_return_path.txt'.format('excursions_' if args.use_excursions else ''), dtype=str))

        # Get the highest-return parameters:
        path_split = highest_return_path.split(os.sep)
        labels = path_split[1:-1:2]
        parameters = path_split[2::2]

        # Load in the results of each run:
        run_nums = os.listdir(algorithm_directory + highest_return_path)
        returns = []
        for run_num in run_nums:
            returns_path = algorithm_directory + highest_return_path + '/' + run_num + '/{}returns.npz'.format('excursions_' if args.use_excursions else '')
            if os.path.isfile(returns_path):
                run_returns = np.load(returns_path)['returns']
                returns.append(run_returns)
        returns = np.array(returns)

        # Calculate the mean over evaluation runs and runs:
        mean_eval_returns = np.mean(returns, axis=2)
        mean_returns = np.mean(mean_eval_returns, axis=0)
        std_returns = np.std(mean_eval_returns, axis=0, ddof=1)

        ax.errorbar(range(0,50000,1000), mean_returns, yerr=st.t.ppf(0.95, len(run_nums)-1)*std_returns/np.sqrt(len(run_nums)-1), label='{}'.format(algorithm_name))

    ax.legend()
    # plt.show()
    plt.savefig('{}_{}learning_curves.png'.format(environment_name, 'excursions_' if args.use_excursions else ''))
