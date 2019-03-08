import numpy as np

class TD:

  def __init__(self, num_features, alpha, lamda):
      self.num_features = num_features

      self.alpha = alpha
      self.lamda = lamda

      self.e = np.zeros((self.num_features))
      self.w = np.zeros((self.num_features))

  def learn(self, delta_t, x_t, gamma_t):
      self.e = gamma_t * self.lamda * self.e + x_t
      self.w = self.w + self.alpha * delta_t * self.e
      
  def estimate(self, x):
      return self.w.dot(x)