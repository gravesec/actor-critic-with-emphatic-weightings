# Add the src directory to the path, because python doesn't look in subfolders:
import sys, os
src_path = os.path.join(os.getcwd(), '.')
if src_path not in sys.path:
    sys.path.append(src_path)

import argparse
import numpy as np
from generate_data import environment_name, behaviour_policy_name

if __name__ == '__main__':

    # Parse command line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('use_excursions', type=int, choices=[0, 1], help='whether to use excursions objective or not')
    args = parser.parse_args()

    algorithms_directory = 'output/{}/behaviour_policies/{}/algorithms/'.format(environment_name, behaviour_policy_name)

    # For each algorithm:
    for algorithm_name in os.listdir(algorithms_directory):
        algorithm_directory = algorithms_directory + algorithm_name

        # Find the best-performing parameter settings:
        current_max = -np.inf
        current_max_path = ['None']
        for current_directory, subdirectories, files in os.walk(algorithm_directory):

            if os.path.basename(current_directory) == 'runs':

                # Extract the path for the best performing parameters:
                path = current_directory.split(algorithm_directory)[1:]

                # Load in the results of each run:
                run_nums = os.listdir(current_directory)
                returns = []
                for run_num in run_nums:
                    returns_path = current_directory + '/' + run_num + '/{}returns.npz'.format('excursions_' if args.use_excursions else '')
                    if os.path.isfile(returns_path):
                        run_returns = np.load(returns_path)['returns']
                        returns.append(run_returns)
                returns = np.array(returns)

                # Calculate the average return:
                mean_returns = -np.inf if len(returns) is 0 else np.mean(returns, axis=None)

                # Store the maximum seen so far:
                if mean_returns > current_max:
                    current_max_path = path
                    current_max = mean_returns

        np.savetxt(algorithm_directory + '/{}highest_return_path.txt'.format('excursions_' if args.use_excursions else ''), current_max_path, fmt='%s')
