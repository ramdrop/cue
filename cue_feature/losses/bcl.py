import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class BayesianContrastiveLoss(nn.Module):

    def __init__(self, mp=0, mn=0, varPrior=1/32.0, kl_scale_factor=1e-6, distribution='gauss'):
        super(BayesianContrastiveLoss, self).__init__()

        self.mp = torch.tensor(mp)
        self.mn = torch.tensor(mn)
        self.varPrior = torch.tensor(varPrior)
        self.kl_scale_factor = kl_scale_factor
        self.distribution = distribution

    def forward(self, muA, muP, varA, varP, is_pos=True):

        muA, muP, varA, varP = muA.T, muP.T, varA.T, varP.T  # (D, N)

        # calculate nll
        if is_pos:
            nll, prob, mu, sigma = negative_loglikelihood(muA, muP, varA, varP, self.mp, is_pos)
        else:
            nll, prob, mu, sigma = negative_loglikelihood(muA, muP, varA, varP, self.mn, is_pos)

        # KL(anchor|| prior) + KL(positive|| prior) + KL(negative|| prior)
        if self.distribution == 'gauss':
            muPrior = torch.zeros_like(muA, requires_grad=False)
            varPrior = torch.ones_like(varA, requires_grad=False) * self.varPrior
            kl = kl_div_gauss(muA, varA, muPrior, varPrior) + kl_div_gauss(muP, varP, muPrior, varPrior)

        elif self.distribution == 'vMF':
            kl = kl_div_vMF(muA, varA)+kl_div_vMF(muP, varP)

        loss = nll + self.kl_scale_factor * kl
        return loss, prob, mu, sigma

    def __repr__(self):
        return self.__class__._Name__ + f'(mp={self.mp:.4f}, mn={self.mn:.4f})'


def negative_loglikelihood(muA, muP, varA, varP, mp, is_pos=True):

    mu = torch.sum(muA**2 - 2 * muA * muP + muP**2 + varA + varP, dim=0)  # (D, N)
    T1 = 4 * muA**2 * varA + 2 * varA**2
    T2 = 4 * muP**2 * varP + 2 * varP**2
    T3 = 4 * (varA * varP + muP**2 * varA + muA**2 * varP)
    T4 = -8 * (muA * muP * (varA + varP))
    sigma2 = torch.sum(T1 + T2 + T3 + T4, dim=0)
    if sigma2.min() < 0:
        return 0, 0, 0, 0
    sigma = sigma2**0.5
    probs = Normal(loc=mu, scale=sigma + 1e-8).cdf(mp)
    if not is_pos:
        probs = 1 - probs
    nll = -torch.log(probs + 1e-8)

    return nll.mean(), probs.mean(), mu.mean(), sigma.mean()


def kl_div_gauss(mu_q, var_q, mu_p, var_p):  # (D, N), (1, N)

    # N, D = mu_q.shape

    # kl diverence for isotropic gaussian
    # kl = 0.5 * ((var_q / var_p) * D + \
    #     1.0 / (var_p) * torch.sum(mu_p**2 + mu_q**2 - 2 * mu_p * mu_q, axis=1) - D + \
    #         D * (torch.log(var_p) - torch.log(var_q)))
    D, N = mu_q.shape

    kl = 0.5 * ((var_q / var_p) * D + 1.0 / (var_p) * torch.sum(mu_p**2 + mu_q**2 - 2 * mu_p * mu_q, axis=0) - D + D * (torch.log(var_p) - torch.log(var_q)))

    return kl.mean()


def kl_div_vMF(mu_q, var_q):
    N, D = mu_q.shape

    # we are estimating the variance and not kappa in the network.
    # They are propertional
    kappa_q = 1.0 / var_q
    kl = kappa_q - D * torch.log(2.0)

    return kl.mean()
