import argparse
import matplotlib
import numpy as np
# matplotlib.use('pdf')
from pathlib import Path
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


parameter_name_to_string = {
    'gamma': '$\\gamma$',
    'alpha_a': '$\\alpha_a$',
    'alpha_c': '$\\alpha_c$',
    'alpha_c2': '$\\alpha_{c2}$',
    'lambda_c': '$\\lambda_c$',
    'eta': '$\\eta$',
    'num_tiles': '$\\#_{tiles}$',
    'num_tilings': '$\\#_{tilings}$',
    'num_features': '$\\#_{features}$',
    'bias_unit': 'bias'
}


def create_configuration_label(configuration_num, configurations):
    configuration = list(configurations[configuration_num])
    parameter_names = list(configurations.dtype.names)
    label = []
    for parameter_index, parameter in enumerate(configuration[:-1]):
        parameter_name = parameter_names[parameter_index]
        parameter_values = configurations[parameter_name]
        if not np.all(parameter_values == parameter):
            label.append(':'.join([parameter_name_to_string[parameter_name], str(parameter)]))
    return ', '.join(label)


def create_common_parameters_string(configurations):
    parameter_names = list(configurations.dtype.names)
    common_params = []
    for parameter_index, parameter_name in enumerate(parameter_names[:-1]):
        if np.allclose(configurations[parameter_name], configurations[parameter_name][0]):
            common_params.append(':'.join([parameter_name_to_string[parameter_name], str(configurations[parameter_name][0])]))
    return ', '.join(common_params)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='A script to plot performance on the given environment.', fromfile_prefix_chars='@', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experiment_name', type=str, default='experiment', help='The directory to read/write experiment files to/from')
    parser.add_argument('--objective', type=str, choices=['excursions', 'alternative_life', 'episodic'], default='episodic', help='Determines the state distribution the starting state is sampled from (excursions: behaviour policy, alternative life: target policy, episodic: start state.)')
    parser.add_argument('--environment', type=str, choices=['MountainCar-v0', 'Acrobot-v1', 'PuddleWorld-v0'], default='MountainCar-v0', help='The environment to evaluate the learned policies on.')
    args = parser.parse_args()

    experiment_path = Path(args.experiment_name)

    # Load the learned policies to get parameter information:
    policies_memmap = np.lib.format.open_memmap(str(experiment_path / 'policies.npy'), mode='r')
    configurations = policies_memmap[0]

    # Load the results of evaluating the learned policies:
    performance = np.lib.format.open_memmap(str(experiment_path / '{}_performance.npy'.format(args.objective)), mode='r')
    num_evaluation_runs, num_ace_runs, num_configurations, num_policies = performance.shape

    mean_eval_performance = np.mean(performance, axis=0)  # Average results over evaluation runs.
    var_eval_performance = np.var(performance, axis=0)
    mean_performance = np.mean(mean_eval_performance, axis=0)  # Average results over ACE runs.
    sem_performance = np.sqrt(np.sum(var_eval_performance / num_evaluation_runs, axis=0)) / num_ace_runs  # Combine estimates of performance SEM correctly.
    # We're sampling polices learned by the algorithms, then sampling the performance of these sampled policies, so the standard errors must be combined appropriately (https://stats.stackexchange.com/questions/231027/combining-samples-based-off-mean-and-standard-error)

    fig, ax = plt.subplots()
    for configuration_num in range(num_configurations):
        policies = list(configurations[configuration_num])[-1]
        x = policies['timesteps']
        y = mean_performance[configuration_num]
        confidence_intervals = sem_performance * st.t.ppf((1.0 + 0.95) / 2, num_evaluation_runs - 1)
        ax.errorbar(x, y, yerr=[confidence_intervals[configuration_num], confidence_intervals[configuration_num]], label=create_configuration_label(configuration_num, configurations))

    plt.legend(loc="lower right")
    plt.suptitle(args.environment)
    plt.title(create_common_parameters_string(configurations), fontsize=9)
    plt.xlabel('Timesteps')
    plt.ylabel('Total Reward')
    plt.ylim(-1000, 0)
    plt.savefig('{}_performance'.format(args.objective))
