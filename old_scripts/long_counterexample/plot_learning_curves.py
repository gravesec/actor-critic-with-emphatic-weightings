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

from matplotlib import rc

import pickle

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
plotted_info = 'policies'
# plotted_info = 'excursions'

# for plotting policies
state_ind = 9
action_ind = 0

tc = LongCounterexampleFeatures(middle_steps=middle_steps, bias_unit=False)
num_states = 2*middle_steps + 1
features = [tc.features(s) for s in range(num_states)]

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
    with open(algorithms_dir + 'best_policies.pkl', 'rb') as f:
        best_policies = pickle.load(f)
    with open(algorithms_dir + 'best_perfs.pkl', 'rb') as f:
        best_perfs = pickle.load(f)
    with open(algorithms_dir + 'best_params.pkl', 'rb') as f:
        best_params = pickle.load(f)
except:
    for cur, dirs, files in os.walk(algorithms_dir):
        if 'extra_weights' in cur:
            continue
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
            current_policy = DiscretePolicy(utils.load_policy_from_file(policy_file_path))
            policies[settings][episode][run] = np.array([current_policy.pi(x) for x in features])

            try:
                current_v_pi = oracle.estimate(pi=policies[settings][episode][run])
            except:
                current_v_pi = np.zeros(num_states) - np.inf
            j_mu[settings][episode][run] = d_mu.dot(current_v_pi)
            # if(settings[0]=='true_eac' and episode==2000):
            #     print(settings, policies[settings][episode][run][0,:])

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

    with open(algorithms_dir + 'best_policies.pkl', 'wb') as f:
        pickle.dump(best_policies, f)
    with open(algorithms_dir + 'best_perfs.pkl', 'wb') as f:
        pickle.dump(best_perfs, f)
    with open(algorithms_dir + 'best_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)

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
    plt.ylabel('$\pi(A{}|S{})$'.format(action_ind, state_ind), fontsize=16, rotation=0, labelpad=35)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    ax.set_ylim(0, 1.01)
    # fig.suptitle('Probability of A{} at S{} for {}'.format(action_ind, state_ind, environment_name), fontsize=12)
    # ax.set_title('Data generated using {} policy'.format(behaviour_policy_name), fontsize=10)

    for key in sorted(best_policies.keys()):
        pi = best_policies[key]
        x = [episode for episode in sorted(pi.keys())]
        y = np.array([pi[episode][0][state_ind,action_ind] for episode in x])
        y_interval = np.array([pi[episode][1][state_ind,action_ind] for episode in x])
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
        print(label+ ', ' + str(best_params[key]))
        print(y)
        ax.plot(x,y, label=label, linewidth=linewidth)
        plt.fill_between(x, y + y_interval, y - y_interval, alpha=0.1)

    #for key in sorted(j_mu.keys()):
        #objectives = j_mu[key]
        #x = [episode for episode in sorted(objectives.keys())]
        #y = np.array([objectives[episode][0] for episode in x])
        #y_interval = np.array([objectives[episode][1] for episode in x])
        #label = key[0] + ' step size: ' + key[1]
        #if key[-1] == '1.0':# and key[1] == '0.001':
            #ax.plot(x,y, label=label)
            #plt.fill_between(x, y + y_interval, y - y_interval, alpha=0.1)

plt.tight_layout()
last_episode = 10000
step = int(last_episode/4)
plt.xticks(range(0,last_episode+1,step))
# plt.legend()
plt.show()
