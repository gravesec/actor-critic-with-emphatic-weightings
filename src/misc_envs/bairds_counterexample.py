import numpy as np


class BairdsCounterexample:
    discount_rate = .99
    num_states = 7
    num_actions = 2
    dashed = 0
    solid = 1
    actions = np.array([dashed, solid])
    mu = np.array([6/7, 1/7])  # Behaviour policy.
    pi = np.array([0., 1.])  # Target policy.
    rho = pi / mu

    num_features = 8
    features = np.array([[2, 0, 0, 0, 0, 0, 0, 1],
                         [0, 2, 0, 0, 0, 0, 0, 1],
                         [0, 0, 2, 0, 0, 0, 0, 1],
                         [0, 0, 0, 2, 0, 0, 0, 1],
                         [0, 0, 0, 0, 2, 0, 0, 1],
                         [0, 0, 0, 0, 0, 2, 0, 1],
                         [0, 0, 0, 0, 0, 0, 1, 2]], dtype=float)
    d_mu = np.ones(num_states) / num_states
    D_mu = np.diag(d_mu)
    linear_projection = features.dot(np.linalg.pinv(features.T.dot(D_mu).dot(features))).dot(features.T.dot(D_mu))

    @staticmethod
    def reset():
        next_state = np.random.randint(0, BairdsCounterexample.num_states)  # uniform start-state distribution.
        return BairdsCounterexample.features[next_state]

    @staticmethod
    def step(x_t, action):
        reward = 0
        if action == BairdsCounterexample.solid:
            next_state = 6
        elif action == BairdsCounterexample.dashed:
            next_state = np.random.randint(0, BairdsCounterexample.num_states-1)
        return reward, BairdsCounterexample.features[next_state]

    @staticmethod
    def mspbe(agent):
        bellman_error = np.full(BairdsCounterexample.num_states, np.nan)
        for s_t in range(BairdsCounterexample.num_states):
            for s_tp1 in range(BairdsCounterexample.num_states):
                if s_tp1 == BairdsCounterexample.num_states - 1:
                    bellman_error[s_t] = BairdsCounterexample.discount_rate * agent.estimate(BairdsCounterexample.features[s_tp1]) - agent.estimate(BairdsCounterexample.features[s_t])
        projected_bellman_error = BairdsCounterexample.linear_projection.dot(bellman_error)
        return projected_bellman_error.T.dot(BairdsCounterexample.D_mu).dot(projected_bellman_error)
