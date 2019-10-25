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

from src.policy_types.deterministic_policy import DeterministicPolicy
from src.policy_types.continuous_policy import ContinuousPolicy
from src.function_approximation.tiny_counterexample_features import TinyCounterexampleFeatures

# from src.algorithms.oracle_critic_dpg_ce import OracleCriticDPGCE
from src.algorithms.oracle_critic_stoch_dpg_ce import OracleCriticStochDPGCE
from src.misc_envs.dpg_ce_env import DPGCEEnv
from generate_data import behavior_params

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=False)
linewidth = 3

fig = plt.figure()
ax = fig.add_subplot(111)

# TODO: Maybe add some if/else to avoid computing unplotted info.

# What to plot
# plotted_info = 'returns'
# plotted_info = 'policies'
plotted_info = 'excursions'


# for plotting policies
state_ind = 1

tc = TinyCounterexampleFeatures(bias_unit=False)
num_states = 3
features = [tc.features(s) for s in range(num_states)]

# for plotting excursion objective j_mu
env = DPGCEEnv()
# oracle = OracleCriticDPGCE(tc, None)
oracle = OracleCriticStochDPGCE(tc, None)

mu = np.tile(np.array(behavior_params)[:,None], (num_states))
d_mu = oracle.steady_distribution(pi=mu)
# d_mu = np.array([1., 0., 0.]) #This is the episode start state dustribution!
print(d_mu)

# Optimal j_mu if possible
v_star = np.array([2., 2., 0.])
j_mu_star = d_mu.dot(v_star)


# Getting performance and pi and j_mu for each algorithm and setting of parameters,
policy_returns = {}
policies = {}
j_mu = {}
algorithms_dir = 'output/{}/behaviour_policies/{}/algorithms/'.format(environment_name, behaviour_policy_name)

for cur, dirs, files in os.walk(algorithms_dir):
    if 'policy.npz' in files:
        if 'returns.txt' in files:
            evaluation_runs_returns = utils.load_returns_from_file(cur + '/returns.txt')
        else:
            evaluation_runs_returns = 0.

        path_split = tuple(cur.split('/'))
        settings = path_split[5:-4:2]
        run = path_split[-3]
        # print(cur, settings)
        if settings not in policy_returns:
            policy_returns[settings] = {}
            policies[settings] = {}
            j_mu[settings] = {}
        episode = int(path_split[-1])
        if episode not in policy_returns[settings]:
            policy_returns[settings][episode] = {}
            policies[settings][episode] = {}
            j_mu[settings][episode] = {}

        policy_returns[settings][episode][run] = np.mean(evaluation_runs_returns)

        policy_file_path = cur + '/policy.npz'
        # current_policy = DeterministicPolicy(utils.load_policy_from_file(policy_file_path))

        if 'dpg' in settings[0]:
            current_policy = DeterministicPolicy(utils.load_policy_from_file(policy_file_path))
        else:
            current_policy = ContinuousPolicy(utils.load_policy_from_file(policy_file_path))
        policies[settings][episode][run] = np.array([current_policy.pi_params(x) for x in features])

        current_v_pi = oracle.estimate(pi=policies[settings][episode][run])
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

# Ignoring bad performances (e.g. algorithms with bad learning rates)
if plotted_info == 'returns':
    comparison_criteria = policy_returns
else:
    comparison_criteria = j_mu
best_perfs = {}
best_params = {}
best_perfs_mean = {}
best_policies = {}
for key in sorted(policy_returns.keys()):
    # overall performance
    # current_mean = np.mean([val[0] for val in comparison_criteria[key].values()])

    # final performance
    last_episode = sorted(comparison_criteria[key].keys())[-1]
    current_mean = comparison_criteria[key][last_episode][0]

    line_key = key[0] + ', lambda_a: ' + key[-1]
    if line_key not in best_perfs or current_mean > best_perfs_mean[line_key]:
        best_perfs[line_key] = comparison_criteria[key]
        best_params[line_key] = key[1:-1]
        best_perfs_mean[line_key] = current_mean
        best_policies[line_key] = policies[key]



# ------------------------ Plots -------------------------------------------

if plotted_info == 'returns':
    # Plotting performances
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Average Returns')
    ax.set_ylim(0, 2)
    fig.suptitle('Learning curves for {}'.format(environment_name), fontsize=12)
    ax.set_title('Data generated using {} policy'.format(behaviour_policy_name), fontsize=10)

    for key in sorted(best_perfs.keys()):
        perfs = best_perfs[key]
        x = [episode for episode in sorted(perfs.keys())]
        y = np.array([perfs[episode][0] for episode in x])
        y_interval = np.array([perfs[episode][1] for episode in x])
        # label = 'lambda_a: ' + key
        label = key
        print(label+ ', ' + str(best_params[key]), y)
        ax.plot(x,y, label=label, linewidth=linewidth)
        plt.fill_between(x, y + y_interval, y - y_interval, alpha=0.1)

    # for key in sorted(policy_returns.keys()):
    #     returns = policy_returns[key]
    #     x = [episode for episode in sorted(returns.keys())]
    #     y = [returns[episode][0] for episode in x]
    #     label = key
    #     if key[-1] == '0.5':# and key[1] == '0.001':
    #         ax.plot(x,y, label=label)

elif plotted_info == 'policies':
    # Plotting policies
    plt.xlabel('Episodes', fontsize=20)
    plt.ylabel('$\pi(S{})$'.format(state_ind), fontsize=16, rotation=0, labelpad=0)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    ax.set_ylim(-25, 25)
    # fig.suptitle('Probability of A{} at S{} for {}'.format(action_ind, state_ind, environment_name), fontsize=12)
    # ax.set_title('Data generated using {} policy'.format(behaviour_policy_name), fontsize=10)

    for key in sorted(best_policies.keys()):
        pi = best_policies[key]
        x = [episode for episode in sorted(pi.keys())]
        y = np.array([pi[episode][0][state_ind][0] for episode in x])
        # print('-')
        # print(y)
        y_interval = np.array([pi[episode][1][state_ind][0] for episode in x])
        # label = 'lambda_a: ' + key
        label = key
        print(label + ', ' + str(best_params[key]), y)
        ax.plot(x,y, label=label, linewidth=linewidth)
        plt.fill_between(x, y + y_interval, y - y_interval, alpha=0.1)

elif plotted_info == 'excursions':
    # Plotting j_mu
    plt.xlabel('Episodes', fontsize=20)
    plt.ylabel('$J_\mu$', fontsize=20, rotation=0, labelpad=15)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    # plt.ylim(0.6, 1.27)
    # fig.suptitle('Learning curves for {}'.format(environment_name), fontsize=12)
    # ax.set_title('Data generated using {} policy'.format(behaviour_policy_name), fontsize=10)

    if j_mu_star is not None:
        ax.axhline(j_mu_star, label='optimal', ls='dashed', color='grey', linewidth=linewidth)

    for key in sorted(best_perfs.keys()):
        perfs = best_perfs[key]
        x = [episode for episode in sorted(perfs.keys())]
        y = np.array([perfs[episode][0] for episode in x])
        y_interval = np.array([perfs[episode][1] for episode in x])
        # label = 'lambda_a: ' + key
        label = key
        print(label+ ', ' + str(best_params[key]), y)
        ax.plot(x,y, label=label, linewidth=linewidth)
        plt.fill_between(x, y + y_interval, y - y_interval, alpha=0.1)

    # for key in sorted(j_mu.keys()):
    #     objectives = j_mu[key]
    #     x = [episode for episode in sorted(objectives.keys())]
    #     y = np.array([objectives[episode][0] for episode in x])
    #     y_interval = np.array([objectives[episode][1] for episode in x])
    #     label = key[0] + ' step size: ' + key[1]
    #     if key[-1] == '0.0':# and key[1] == '0.001':
    #         ax.plot(x,y, label=label)
    #         plt.fill_between(x, y + y_interval, y - y_interval, alpha=0.1)

last_episode = 10000
step = int(last_episode/4)
plt.xticks(range(0,last_episode+1,step))
plt.tight_layout()
# plt.legend()
plt.show()
