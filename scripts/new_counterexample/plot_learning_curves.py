# Add the src directory to the path, because python doesn't look in subfolders:
import sys, os
src_path = os.path.join(os.getcwd(), '.')
if src_path not in sys.path:
    sys.path.append(src_path)

import numpy as np
from src import utils
import scipy.stats as st
import matplotlib.pyplot as plt
from generate_data import environment_name, behaviour_policy_name

fig = plt.figure()
ax = fig.add_subplot(111)

# Parse command line arguments:

policy_returns = {}

# for each algorithm and setting of parameters,
algorithms_dir = 'output/{}/behaviour_policies/{}/algorithms/'.format(environment_name, behaviour_policy_name)
for cur, dirs, files in os.walk(algorithms_dir):
    if 'returns.txt' in files:
        evaluation_runs_returns = utils.load_returns_from_file(cur + '/returns.txt')
        path_split = tuple(cur.split('/'))
        settings = path_split[5:-4:2]
        run = path_split[-3]
        # print(cur, settings)
        if settings not in policy_returns:
            policy_returns[settings] = {}
        episode = int(path_split[-1])
        if episode not in policy_returns[settings]:
            policy_returns[settings][episode] = {}
        policy_returns[settings][episode][run] = np.mean(evaluation_runs_returns)

for settings in policy_returns.keys():
    for episode in policy_returns[settings].keys():
        episode_returns = np.fromiter(policy_returns[settings][episode].values(), dtype=np.float)
        policy_returns[settings][episode] = (np.mean(episode_returns), st.sem(episode_returns, axis=None, ddof=1))

# Ignoring bad performances
best_perfs = {}
best_params = {}
best_perfs_mean = {}
for key in sorted(policy_returns.keys()):
    if key[-1] not in best_perfs:
        best_perfs[key[-1]] = policy_returns[key]
        best_params[key[-1]] = key[:-1]
        best_perfs_mean[key[-1]] = np.mean([val[0] for val in policy_returns[key].values()])
    else:
        current_mean = np.mean([val[0] for val in policy_returns[key].values()])
        if current_mean > best_perfs_mean[key[-1]]:
            best_perfs[key[-1]] = policy_returns[key]
            best_params[key[-1]] = key[:-1]
            best_perfs_mean[key[-1]] = current_mean



ax.set_xlabel('Episodes')
ax.set_ylabel('Average Returns')
ax.set_ylim(-10, 0)
fig.suptitle('Learning curves for {}'.format(environment_name), fontsize=12)
ax.set_title('Data generated using {} policy'.format(behaviour_policy_name), fontsize=10)

for key in sorted(best_perfs.keys()):
    returns = best_perfs[key]
    x = [episode for episode in sorted(returns.keys())]
    y = np.array([returns[episode][0] for episode in x])
    y_interval = np.array([returns[episode][1] for episode in x])
    label = key + ', ' + str(best_params[key])
    print(label, y)
    ax.plot(x,y, label=label, linewidth=3)
    plt.fill_between(x, y + y_interval, y - y_interval, alpha=0.1)

# for key in sorted(policy_returns.keys()):
#     returns = policy_returns[key]
#     x = [episode for episode in sorted(returns.keys())]
#     y = [returns[episode][0] for episode in x]
#     label = key
#     if key[-1] == '0.0':# and key[-2] == '0.0':
#         ax.plot(x,y, label=label)

ax.legend()
# plt.savefig('{}_learning_curves.png'.format(environment_name))
plt.show()
