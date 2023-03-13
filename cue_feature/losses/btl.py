import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.distributions as d
from torch import linalg as LA
import torch.nn.functional as F

class BayesianTripletLossSampling(nn.Module):
    def __init__(self, margin=0, nb_samples=20, kl_scale_factor=1e-6, varPrior=1/32.0):
        super(BayesianTripletLossSampling, self).__init__()
        self.margin = margin
        self.varPrior = varPrior
        self.nb_samples = nb_samples
        self.kl_scale_factor = kl_scale_factor

    def forward(self, muA, muP, muN, varA, varP, varN): # [N,D]
        # TRIPLET LOSS =============== #
        da = d.Normal(muA, torch.sqrt(varA+ 1e-7) + 1e-7) # mean:[N,D], standard deviation:[N,1]
        dp = d.Normal(muP, torch.sqrt(varP+ 1e-7) + 1e-7)
        dn = d.Normal(muN, torch.sqrt(varN+ 1e-7) + 1e-7)
        emb_a = da.rsample((self.nb_samples,))  # [nb,N,D]
        emb_p = dp.rsample((self.nb_samples,))
        emb_n = dn.rsample((self.nb_samples,))
        nb, N, D = emb_a.shape
        emb_a = emb_a.reshape(-1, D)
        emb_p = emb_p.reshape(-1, D)
        emb_n = emb_n.reshape(-1, D)
        ap = LA.norm(emb_a - emb_p, dim=1)  # [nb*N, ]
        an = LA.norm(emb_a - emb_n, dim=1)
        loss_triplet = F.relu(ap - an + self.margin).mean()

        muA, muP, muN, varA, varP, varN = muA.T, muP.T, muN.T, varA.T, varP.T, varN.T
        # REGULARIZER ================ #
        muPrior = torch.zeros_like(muA, requires_grad=False)
        varPrior = torch.ones_like(varA, requires_grad=False) * self.varPrior

        loss_kl = (kl_div_gauss(muA, varA, muPrior, varPrior) + kl_div_gauss(muP, varP, muPrior, varPrior) + kl_div_gauss(muN, varN, muPrior, varPrior))

        return loss_triplet + self.kl_scale_factor * loss_kl

class BayesianTripletMultipleLoss(nn.Module):

    def __init__(self, margin=0, varPrior=1/32.0, kl_scale_factor=1e-6, distribution='gauss'):
        super(BayesianTripletLoss, self).__init__()

        self.margin = torch.tensor(margin)
        self.varPrior = torch.tensor(varPrior)
        self.kl_scale_factor = kl_scale_factor
        self.distribution = distribution

    def forward(self, muA, muP, muN, varA, varP, varN):  # x:(D, 1+1+neg_count)

        muA, muP, muN, varA, varP, varN = muA.T, muP.T, muN.T, varA.T, varP.T, varN.T

        # calculate nll
        nll, probs, mu, sigma = negative_loglikelihood(muA, muP, muN, varA, varP, varN, margin=self.margin)

        # KL(anchor|| prior) + KL(positive|| prior) + KL(negative|| prior)
        if self.distribution == 'gauss':
            muPrior = torch.zeros_like(muA, requires_grad=False)
            varPrior = torch.ones_like(varA, requires_grad=False) * self.varPrior

            kl = (kl_div_gauss(muA, varA, muPrior, varPrior) + \
                kl_div_gauss(muP, varP, muPrior, varPrior) + \
                kl_div_gauss(muN, varN, muPrior, varPrior))

        elif self.distribution == 'vMF':
            kl = (kl_div_vMF(muA, varA) + \
            kl_div_vMF(muP, varP) + \
            kl_div_vMF(muN, varN))

        return nll + self.kl_scale_factor * kl, probs, mu, sigma

    def __repr__(self):
        return self.__class__._Name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'

class BayesianTripletLoss(nn.Module):

    def __init__(self, margin=0, varPrior=1/32.0, kl_scale_factor=1e-6, distribution='gauss'):
        super(BayesianTripletLoss, self).__init__()

        self.margin = torch.tensor(margin)
        self.varPrior = torch.tensor(varPrior)
        self.kl_scale_factor = kl_scale_factor
        self.distribution = distribution

    def forward(self, muA, muP, muN, varA, varP, varN):  # x:(D, 1+1+neg_count)

        muA, muP, muN, varA, varP, varN = muA.T, muP.T, muN.T, varA.T, varP.T, varN.T

        # calculate nll
        nll, probs, mu, sigma = negative_loglikelihood(muA, muP, muN, varA, varP, varN, margin=self.margin)

        # KL(anchor|| prior) + KL(positive|| prior) + KL(negative|| prior)
        if self.distribution == 'gauss':
            muPrior = torch.zeros_like(muA, requires_grad=False)
            varPrior = torch.ones_like(varA, requires_grad=False) * self.varPrior

            kl = (kl_div_gauss(muA, varA, muPrior, varPrior) + \
                kl_div_gauss(muP, varP, muPrior, varPrior) + \
                kl_div_gauss(muN, varN, muPrior, varPrior))

        elif self.distribution == 'vMF':
            kl = (kl_div_vMF(muA, varA) + \
            kl_div_vMF(muP, varP) + \
            kl_div_vMF(muN, varN))

        return nll + self.kl_scale_factor * kl, probs, mu, sigma

    def __repr__(self):
        return self.__class__._Name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'


def negative_loglikelihood(muA, muP, muN, varA, varP, varN, margin=0.0):

    muA2 = muA**2  # ([D-1, N])   # N = S - 2
    muP2 = muP**2  # ([D-1, N])
    muN2 = muN**2  # ([D-1, N])
    varP2 = varP**2  # (1, N)
    varN2 = varN**2  # (1, N)

    mu = torch.sum(muP2 + varP - muN2 - varN - 2 * muA * (muP - muN), dim=0)  # sum of feature dimension ([1, 5])
    T1 = varP2 + 2 * muP2 * varP + 2 * (varA + muA2) * (varP + muP2) - 2 * muA2 * muP2 - 4 * muA * muP * varP  # ([2047, 5])
    T2 = varN2 + 2 * muN2 * varN + 2 * (varA + muA2) * (varN + muN2) - 2 * muA2 * muN2 - 4 * muA * muN * varN  # ([2047, 5])
    T3 = 4 * muP * muN * varA  # ([2047, 5])
    sigma2 = torch.sum(2 * T1 + 2 * T2 - 2 * T3, dim=0)  # sum of feature dimension ([1, 5])
    sigma = sigma2**0.5  # ([1, 5])

    # PyTorch uses a parametric CDF function that enables the gradient to flow back
    # try:
    probs = Normal(loc=mu, scale=sigma + 1e-8).cdf(margin)  # ([1, 5])
    # except:
    #     print(f'mu: min={mu.min()} max={mu.max()}')
    #     print(f'sigma: min={sigma.min()} sigma={mu.max()}')
    nll = -torch.log(probs + 1e-8)  # ([1, 5])

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
