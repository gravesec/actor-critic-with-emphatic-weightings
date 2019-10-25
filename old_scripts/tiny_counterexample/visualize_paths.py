# Add the src directory to the path, because python doesn't look in subfolders:
import sys, os
src_path = os.path.join(os.getcwd(), '.')
if src_path not in sys.path:
    sys.path.append(src_path)

import numpy as np
from src import utils
import scipy.stats as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
from generate_data import environment_name, behaviour_policy_name

from src.policy_types.discrete_policy import DiscretePolicy
from src.function_approximation.tiny_counterexample_features import TinyCounterexampleFeatures

from src.algorithms.oracle_critic import OracleCritic
from src.misc_envs.mdps import tiny_counterexample_env
from generate_data import behavior_probs

import pickle


# for plotting policies
tc = TinyCounterexampleFeatures(bias_unit=False)
num_states = 3
features = [tc.features(s) for s in range(num_states)]

# for plotting excursion objective j_mu
env = tiny_counterexample_env()
oracle = OracleCritic(env, tc, None)

mu = np.tile(np.array(behavior_probs), (num_states,1))
d_mu = oracle.steady_distribution(pi=mu)
# print(d_mu)


# Getting performance and pi and j_mu for each algorithm and setting of parameters,
algorithms_dir = 'output/{}/behaviour_policies/{}/algorithms/'.format(environment_name, behaviour_policy_name)
try:
    with open(algorithms_dir + 'best_policies.pkl', 'rb') as f:
        best_policies = pickle.load(f)
except:
    policies = {}
    j_mu = {}

    for cur, dirs, files in os.walk(algorithms_dir):
        if 'policy.npz' in files:

            path_split = tuple(cur.split('/'))
            settings = path_split[5:-4:2]
            run = path_split[-3]
            # print(cur, settings)
            if settings not in policies:
                policies[settings] = {}
                j_mu[settings] = {}
            episode = int(path_split[-1])
            if episode not in policies[settings]:
                policies[settings][episode] = {}
                j_mu[settings][episode] = {}

            policy_file_path = cur + '/policy.npz'
            current_policy = DiscretePolicy(utils.load_policy_from_file(policy_file_path))
            policies[settings][episode][run] = np.array([current_policy.pi(x) for x in features])

            current_v_pi = oracle.estimate(pi=policies[settings][episode][run])
            j_mu[settings][episode][run] = d_mu.dot(current_v_pi)

    # Averaging over runs
    for settings in policies.keys():
        for episode in policies[settings].keys():
            episode_policies = np.array(list(policies[settings][episode].values()))
            policies[settings][episode] = (np.mean(episode_policies, axis=0), st.sem(episode_policies, axis=0, ddof=1))

            episode_objectives = np.fromiter(j_mu[settings][episode].values(), dtype=np.float)
            j_mu[settings][episode] = (np.mean(episode_objectives), st.sem(episode_objectives, axis=None, ddof=1))

    # Ignoring bad performances (e.g. algorithms with bad learning rates)
    comparison_criteria = j_mu

    best_perfs = {}
    best_params = {}
    best_perfs_mean = {}
    best_policies = {}
    for key in sorted(policies.keys()):
        # overall performance
        # current_mean = np.mean([val[0] for val in comparison_criteria[key].values()])

        # final performance
        last_episode = sorted(comparison_criteria[key].keys())[-1]
        current_mean = comparison_criteria[key][last_episode][0]

        if key[-1] not in best_perfs or current_mean > best_perfs_mean[key[-1]]:
            best_perfs[key[-1]] = comparison_criteria[key]
            best_params[key[-1]] = key[:-1]
            best_perfs_mean[key[-1]] = current_mean
            best_policies[key[-1]] = policies[key]

    with open(algorithms_dir + 'best_policies.pkl', 'wb') as f:
        pickle.dump(best_policies, f)

# ------------------------ Visualization -------------------------------------------

fig = plt.figure()
ax = Axes3D(fig)

mesh_p0 = np.arange(0., 1., 0.05)
mesh_p1 = np.arange(0., 1., 0.05)
mesh_p0, mesh_p1 = np.meshgrid(mesh_p0, mesh_p1)
mesh_z = np.zeros(mesh_p0.shape)

for i in range(mesh_z.shape[0]):
    for j in range(mesh_z.shape[1]):
        mesh_pi = np.array([mesh_p0[i,j],mesh_p1[i,j],mesh_p1[i,j]])
        mesh_pi = np.vstack(((mesh_pi),(1-mesh_pi))).T
        mesh_z[i][j] = d_mu.dot(oracle.estimate(pi=mesh_pi))

surf = ax.plot_surface(mesh_p0, mesh_p1, mesh_z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel('pi(a0|s0)')
ax.set_ylabel('pi(a0|s1)')
ax.set_zlabel('J_mu')

for key in sorted(best_policies.keys()):
        pi = best_policies[key]
        inds = [episode for episode in sorted(pi.keys())]
        x = np.array([pi[episode][0][0,0] for episode in inds])
        y = np.array([pi[episode][0][1,0] for episode in inds])
        z = [d_mu.dot(oracle.estimate(pi=pi[episode][0])) for episode in inds]
        label = 'lambda_a: ' + key
        # print(label + ', ' + str(best_params[key]), y)
        ax.plot(x,y,z, label=label, linewidth=2)

plt.legend()
plt.show()
