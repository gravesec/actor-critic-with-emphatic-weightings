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

from src.policy_types.discrete_policy import DiscretePolicy
from src.function_approximation.long_counterexample_features import LongCounterexampleFeatures

from src.algorithms.oracle_critic import OracleCritic
from src.environments.mdps import long_counterexample_env
from generate_data import behavior_probs, middle_steps

import pickle

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=False)
linewidth = 3

fig = plt.figure()
ax = fig.add_subplot(111)

plotted_info = 'final'
last_episode = 10000


tc = LongCounterexampleFeatures(middle_steps=middle_steps, bias_unit=False)
num_states = 2* middle_steps + 1
features = [tc.features(s) for s in range(num_states)]
print(features)

# for plotting excursion objective j_mu
env = long_counterexample_env(middle_steps=middle_steps)
oracle = OracleCritic(env, tc, None)

mu = np.tile(np.array(behavior_probs), (num_states,1))
d_mu = oracle.steady_distribution(pi=mu)
# d_mu = np.array([1., 0., 0.]) #This is the episode start state dustribution!
# print(d_mu)

# Optimal j_mu if possible
v_star = np.array([2.]*(num_states - 1) + [0.])
j_mu_star = d_mu.dot(v_star)
j_mu_star = None


# Getting performance and pi and j_mu for each algorithm and setting of parameters,
policy_returns = {}
policies = {}
j_mu = {}
algorithms_dir = 'output/{}/behaviour_policies/{}/algorithms/'.format(environment_name, behaviour_policy_name)

try:
    with open(algorithms_dir + 'perfs_sensitivity.pkl', 'rb') as f:
        perfs = pickle.load(f)
except:
    for cur, dirs, files in os.walk(algorithms_dir):
        if 'extra_weights' in cur:
            continue
        if 'policy.npz' in files:
            path_split = tuple(cur.split('/'))

            episode = int(path_split[-1])
            if plotted_info == 'final' and episode != last_episode:
                continue

            if 'returns.txt' in files:
                evaluation_runs_returns = utils.load_returns_from_file(cur + '/returns.txt')
            else:
                evaluation_runs_returns = 0.

            settings = path_split[5:-4:2]
            run = path_split[-3]
            # print(cur, settings)
            if settings not in policy_returns:
                policy_returns[settings] = {}
                policies[settings] = {}
                j_mu[settings] = {}
            if episode not in policy_returns[settings]:
                policy_returns[settings][episode] = {}
                policies[settings][episode] = {}
                j_mu[settings][episode] = {}

            policy_returns[settings][episode][run] = np.mean(evaluation_runs_returns)

            policy_file_path = cur + '/policy.npz'
            current_policy = DiscretePolicy(utils.load_policy_from_file(policy_file_path))
            policies[settings][episode][run] = np.array([current_policy.pi(x) for x in features])

            try:
                current_v_pi = oracle.estimate(pi=policies[settings][episode][run])
            except:
                current_v_pi = np.zeros(num_states) - np.inf
            j_mu[settings][episode][run] = d_mu.dot(current_v_pi)

    # Averaging over runs
    for settings in policy_returns.keys():
        for episode in policy_returns[settings].keys():
            episode_returns = np.fromiter(policy_returns[settings][episode].values(), dtype=np.float)
            policy_returns[settings][episode] = (np.mean(episode_returns), st.sem(episode_returns, axis=None, ddof=1))

            episode_policies = np.array(list(policies[settings][episode].values()))
            policies[settings][episode] = (np.mean(episode_policies, axis=0), st.sem(episode_policies, axis=0, ddof=1))

            episode_objectives = np.fromiter(j_mu[settings][episode].values(), dtype=np.float)
            j_mu[settings][episode] = (np.mean(episode_objectives), st.sem(episode_objectives, axis=None, ddof=1))


    perfs = {}
    for key in sorted(policy_returns.keys()):
        if plotted_info == 'overall':
            # overall performance
            current_perf = (np.mean([val[0] for val in j_mu[key].values()]), 0.0)
            #TODO: std for overall
        else:
            # final performance
            current_perf = j_mu[key][last_episode]

        line_key = key[0] + ', lambda_a: ' + key[-1]
        if line_key not in perfs:
            perfs[line_key] = {}

        if key[1] not in perfs[line_key] or current_perf[0] > perfs[line_key][key[1]][0]:
            perfs[line_key][key[1]] = current_perf

    with open(algorithms_dir + 'perfs_sensitivity.pkl', 'wb') as f:
        pickle.dump(perfs, f)

for k, v in perfs.items():
    print(k, v)
# ------------------------ Plots -------------------------------------------

# Plotting j_mu
plt.xlabel('step-size', fontsize=20)
plt.ylabel('$J_\mu$', fontsize=20, rotation=0, labelpad=15)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
#plt.ylim(0.7, 1.27)
# plt.title('$\mu(A_0|.) = $ {}, number of episodes: {}'.format(behavior_probs[0], last_episode))
# fig.suptitle('Learning curves for {}'.format(environment_name), fontsize=12)
# ax.set_title('Data generated using {} policy'.format(behaviour_policy_name), fontsize=10)

if j_mu_star is not None:
    ax.axhline(j_mu_star, label='optimal', ls='dashed', color='grey', linewidth=linewidth)

for key in sorted(perfs.keys()):
    objectives = perfs[key]
    alpha_us = [alpha_u for alpha_u in sorted(objectives.keys(), key=float)]
    x = [float(alpha_u) for alpha_u in alpha_us]
    y = np.array([objectives[alpha_u][0] for alpha_u in alpha_us])
    y_interval = np.array([objectives[alpha_u][1] for alpha_u in alpha_us])
    label = str(key)#'lambda_a: ' + str(key)
    # if key[-1] == '1.0':# and key[1] == '0.001':
    plt.plot(x, y, label=label, linewidth=linewidth)
    plt.fill_between(x, y + y_interval, y - y_interval, alpha=0.1)
    plt.xticks(x)

plt.xscale('log')
plt.tight_layout()
# plt.legend()
plt.show()
# plt.savefig(str(behavior_probs[0]) + '_' + str(last_episode) + '.png')
