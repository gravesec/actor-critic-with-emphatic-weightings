import numpy as np

class TOTD:

  def __init__(self, num_features, alpha, lamda):
      self.num_features = num_features

      self.alpha = alpha
      self.lamda = lamda

      self.e = np.zeros((self.num_features))
      self.w = np.zeros((self.num_features))
      self.v_old = 0

  def learn(self, delta_t, v_t, x_t, gamma_t, v_tp1, x_tp1, gamma_tp1, rho_t):
      self.e = gamma_t * self.lamda * self.e + (1 - self.alpha * gamma_t * self.lamda * self.e.dot(x_t)) * x_t
      self.w = self.w + self.alpha * (delta_t + v_t - self.v_old) * self.e - self.alpha * (v_t - self.v_old) * x_t
      self.v_old = v_tp1

  def estimate(self, x):
      return self.w.dot(x)