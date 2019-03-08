import numpy as np
import scipy.stats


class ContinuousPolicy:

    def __init__(self, u, learnable_sig=True):
        ''' Represents a Gaussian mean and std. '''
        self.num_features = u.shape[0]
        self.u = u
        self.sig_act = self.softplus#np.exp
        self.sig_act_grad = self.softplus_grad#np.exp
        self.learnable_sig = learnable_sig

    def pi(self, x_t, a_t=None):
        # print(self.u)
        out = self.u.dot(x_t)
        mu = out[0]
        sig = self.sig_act(out[1])
        if a_t is not None:
            return scipy.stats.norm.pdf(a_t, mu, sig)
        return np.random.normal(mu, sig)

    def pi_params(self, x_t):
        out = self.u.dot(x_t)
        mu = out[0]
        sig = self.sig_act(out[1])
        return [mu, sig]

    def grad_log_pi(self, x_t, a_t):

        out = self.u.dot(x_t)
        mu = out[0]
        sig = self.sig_act(out[1])
        grad_mu = x_t
        if self.learnable_sig:
            grad_sig = self.sig_act_grad(out[1]) * x_t
        else:
            grad_sig = 0.0 * x_t

        # print('--')
        # print(x_t, a_t)
        # print(mu, self.sig_act(sig))

        # pi_t in the IS ratio cancels with pi_t in the denominator of grad_log
        return np.vstack((grad_mu, grad_sig)) #/ scipy.stats.norm.pdf(a_t, mu, sig)

    # In case I can't use exp activation for sigma
    def softplus(self, z):
        return np.log(1.0+ np.exp(z))

    def softplus_grad(self, z):
        expz = np.exp(z)
        return expz/(1.0+expz)
