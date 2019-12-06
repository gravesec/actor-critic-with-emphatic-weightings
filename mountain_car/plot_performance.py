import argparse
import numpy as np
import matplotlib
# matplotlib.use('pdf')
from pathlib import Path
import matplotlib.pyplot as plt

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='A script to evaluate policies on the Mountain Car environment in parallel.', fromfile_prefix_chars='@', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experiment_name', type=str, default='experiment', help='The directory to read/write experiment files to/from')
    parser.add_argument('--objective', type=str, choices=['excursions', 'alternative_life', 'episodic'], default='episodic', help='Determines the state distribution the starting state is sampled from (excursions: behaviour policy, alternative life: target policy, episodic: mountain car start state.)')
    args = parser.parse_args()

    # Load the results of evaluating the learned policies:
    experiment_path = Path(args.experiment_name)
    performance = np.lib.format.open_memmap(str(experiment_path / '{}_performance.npy'.format(args.objective)), mode='r')
    num_evaluation_runs, num_ace_runs, num_configurations, num_policies = performance.shape

    mean_eval_performance = np.mean(performance, axis=0)  # Average results over evaluation runs.
    var_eval_performance = np.var(performance, axis=0)
    mean_performance = np.mean(mean_eval_performance, axis=0)  # Average results over ACE runs.
    se_performance = np.sqrt(np.sum(var_eval_performance / num_evaluation_runs, axis=0)) / num_ace_runs  # Combine estimates of performance correctly.

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for p in mean_performance:
        ax.plot(p)
    plt.title('Mountain car')
    plt.xlabel('Timesteps')
    plt.ylabel('Total Reward')
    plt.savefig('performance')