import numpy as np


class LinearTOETD:
    """
    True Online Emphatic Temporal Difference learning algorithm by Ashique Rupam Mahmood.
    """
    def __init__(self, num_features, I, alpha):
        self.ep = np.zeros(num_features)
        self.theta = np.zeros(num_features)
        self.prevtheta = np.zeros(num_features)
        self.H = 0.
        self.M = alpha*I
        self.prevI = I
        self.prevgm = 0
        self.prevlm = 0

    def learn(self, phi, delta, rho, gm, lm, I, alpha):
        self.ep = rho*(self.prevgm*self.prevlm*self.ep + self.M*(1-rho*self.prevgm*self.prevlm*np.dot(self.ep, phi))*phi)
        Delta = delta*self.ep + np.dot(self.theta - self.prevtheta, phi)*(self.ep - rho*self.M*phi)
        self.prevtheta = self.theta.copy()
        self.theta += Delta
        self.H = rho*gm*(self.H + self.prevI)
        self.M = alpha*(I + (1-lm)*self.H)
        self.prevgm = gm
        self.prevlm = lm
        self.prevI = I

    def estimate(self, phi):
        return self.theta.dot(phi)


class BinaryTOETD:
    """
    Adapted from Ashique Rupam Mahmood's TOETD.
    """
    def __init__(self, num_features, I, alpha):
        self.ep = np.zeros(num_features)
        self.theta = np.zeros(num_features)
        self.prevtheta = np.zeros(num_features)
        self.H = 0.
        self.M = alpha*I
        self.prevI = I
        self.prevgm = 0
        self.prevlm = 0

    def learn(self, indices_t, delta, rho, gm, lm, I, alpha):
        ep_dot_phi = self.ep[indices_t].sum()
        self.ep *= rho*self.prevgm*self.prevlm
        self.ep[indices_t] += rho*self.M*(1-rho*self.prevgm*self.prevlm*ep_dot_phi)
        del_theta_dot_phi = (self.theta - self.prevtheta)[indices_t].sum()
        Delta = delta*self.ep + del_theta_dot_phi*self.ep
        Delta[indices_t] -= del_theta_dot_phi*rho*self.M
        self.prevtheta = self.theta.copy()
        self.theta += Delta
        self.H = rho*gm*(self.H + self.prevI)
        self.M = alpha*(I + (1-lm)*self.H)
        self.prevgm = gm
        self.prevlm = lm
        self.prevI = I

    def estimate(self, indices_t):
        return self.theta[indices_t].sum()
