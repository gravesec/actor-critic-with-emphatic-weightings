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
        # print(cur)
        evaluation_runs_returns = utils.load_returns_from_file(cur + '/returns.txt')
        path_split = tuple(cur.split('/'))
        settings = path_split[5:-2:2]
        if settings not in policy_returns:
            policy_returns[settings] = {}
        episode = int(path_split[-1])
        policy_returns[settings][episode] = (np.mean(evaluation_runs_returns), st.sem(evaluation_runs_returns, axis=None, ddof=1))


ax.set_xlabel('Episodes')
ax.set_ylabel('Average Returns')
ax.set_ylim(0, 150)
fig.suptitle('Learning curves for {}'.format(environment_name), fontsize=12)
ax.set_title('Data generated using {} policy'.format(behaviour_policy_name), fontsize=10)


for key in sorted(policy_returns.keys()):
    returns = policy_returns[key]
    x = [episode for episode in sorted(returns.keys())]
    y = [returns[episode][0] for episode in x]
    label = key
    ax.plot(x,y, label=label)
    # label = str(key[:2])+','+str(key[5])
    # if key[-2] == '0.0':
    #     ax.plot(x,y, label=label)

ax.legend()
# plt.savefig('{}_learning_curves.png'.format(environment_name))
plt.show()
