import numpy as np
import unittest
from src.policy_types.discrete_policy import DiscretePolicy

class DiscretePolicyTestCase(unittest.TestCase):

	def test_init(self):
		num_actions = 2
		num_features = 3
		u = np.zeros((num_actions, num_features))

		policy = DiscretePolicy(u)

		self.assertEqual(policy.num_actions, num_actions)
		self.assertEqual(policy.num_features, num_features)
		np.testing.assert_almost_equal(policy.u, u)

	def test_probabilities_equal(self):
		# Equiprobable random policy:
		u = np.array([[0., 0., 0.], [0., 0., 0.]])
		policy = DiscretePolicy(u)
		# Arbitrary feature vector:
		x_t = np.array([0, 0, 1])

		pi = policy.pi(x_t)

		np.testing.assert_almost_equal(pi, np.array([0.5, 0.5]))

	def test_probabilities_not_equal(self):
		# 90% chance of taking action 1 when all features present:
		u = np.array([[0., 0., 0.], [np.log(9.)/3, np.log(9.)/3, np.log(9.)/3]])
		policy = DiscretePolicy(u)
		x_t = np.array([1, 1, 1])

		pi = policy.pi(x_t)

		np.testing.assert_almost_equal(pi, np.array([0.1, 0.9]))

	def test_probabilities_features_not_equal(self):
		# 95% chance of taking action 1 when last two features present:
		u = np.array([[0, np.log(5)/2, np.log(5)/2],[0, np.log(95)/2, np.log(95)/2]])
		policy = DiscretePolicy(u)
		x_t = np.array([0,1,1])

		pi = policy.pi(x_t)

		np.testing.assert_almost_equal(pi, np.array([0.05, 0.95]))

	def test_grad_log_probabilities_simple(self):
		# Equiprobable random policy:
		u = np.array([[0., 0., 0.],[0., 0., 0.]])
		policy = DiscretePolicy(u)

		# Only one feature eligible:
		x_t = np.array([0,1,0])
		# First action was taken:
		a_t = 0

		grad_log_pi = policy.grad_log_pi(x_t, a_t)

		np.testing.assert_almost_equal(grad_log_pi, np.array([[0., .5, 0.], [0., -.5, 0.]]))

if __name__ == '__main__':
	unittest.main()
