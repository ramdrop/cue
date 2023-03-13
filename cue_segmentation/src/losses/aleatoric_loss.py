import torch
import torch.nn as nn
import torch.distributions as d
import torch.nn.functional as F
import gin


@gin.configurable
class AleatoricLoss(nn.Module):

    def __init__(self, is_log_sigma, nb_samples=10):
        super().__init__()
        self.is_log_sigma = is_log_sigma
        self.nb_samples = nb_samples
        self.ignore_index=255

    def forward(self, logits, sigma, target):
        if self.is_log_sigma:
            distribution = d.Normal(logits, torch.exp(sigma))
        else:
            distribution = d.Normal(logits, sigma + 1e-7)

        x_hat = distribution.rsample((self.nb_samples,))

        mc_expectation = F.softmax(x_hat, dim=2).mean(dim=0)
        log_probs = (mc_expectation + 1e-7).log()
        loss = F.nll_loss(log_probs, target, ignore_index=self.ignore_index)

        return loss

@gin.configurable
class AleatoricLossV2(nn.Module):

    def __init__(self, is_log_sigma, nb_samples=10):
        super().__init__()
        self.is_log_sigma = is_log_sigma
        self.nb_samples = nb_samples
        self.ignore_index=255

    def forward(self, logits, sigma, target):
        multi_logits = self.reparam_trick(logits, sigma, 1)
        if self.is_log_sigma:
            distribution = d.Normal(logits, torch.exp(sigma))
        else:
            distribution = d.Normal(logits, sigma)

        x_hat = distribution.rsample((self.nb_samples,))

        mc_expectation = F.softmax(x_hat, dim=2).mean(dim=0)
        log_probs = (mc_expectation + 1e-7).log()
        loss = F.nll_loss(log_probs, target, ignore_index=self.ignore_index)

        return loss

    def reparam_trick(self, emb_mu, emb_sigma2, max_t):
        """
        emb_mu:     ([N, 96])
        emb_sigma2:  ([N, 1])
        return:     ([m, N, 96])
        """
        emb_mu_ext = emb_mu[None].expand(max_t, *emb_mu.shape)                 # ([m, N, 96])
        emb_sigma = emb_sigma2 * 0.5                                           # ([N, 1])
        emb_sigma_ext = emb_sigma[None].expand(max_t, *emb_sigma.shape)        # ([m, N, 1])
        norm_v = torch.randn_like(emb_mu_ext)                                  # ([m, N, 96])
        emb_mu_sto = emb_mu_ext + norm_v * emb_sigma_ext

        return emb_mu_sto