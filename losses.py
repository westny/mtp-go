import torch
import torch.distributions as tdist
import torch.nn as nn
import numpy as np
from scipy.stats import gaussian_kde


def mask_reduce_losses(losses, mask):
    # Compute mean over vehicles, sum over time, masked
    # losses: (N,T)
    # mask: (N,T), \in {0,1}

    n_pred = torch.sum(mask, dim=0)  # (T,)
    t_losses = torch.sum(losses * mask, dim=0) / n_pred  # (T,)
    return torch.sum(t_losses)


def kde_likelihood(mu, sigma, pi, target, is_tril=False, dataset='rounD', n_samples=50):
    std = {'rounD': {'mean': torch.tensor([-2.8113458e-01, -5.9812081e-01]),
                     'std': torch.tensor([18.183529, 19.054676])},
           'highD': {'mean': torch.tensor([6.4364128e+01, -4.6085745e-01]),
                     'std': torch.tensor([9.4793625e+01, 3.8449910e+00])}}

    data_mu = std[dataset]['mean']
    data_std = std[dataset]['std']

    log_pdf_lower_bound = -20

    batch_size, seq_len, *_ = mu.shape

    # Create GMM
    pi_expanded = pi.unsqueeze(1).expand(-1, mu.shape[1], -1)
    mix = tdist.Categorical(logits=pi_expanded)
    L = sigma if is_tril else torch.linalg.cholesky(sigma)
    mvn = tdist.MultivariateNormal(mu, scale_tril=L)
    gmm = tdist.MixtureSameFamily(mix, mvn)

    sampled_traj = gmm.sample((n_samples,)).detach().cpu()
    target = target.detach().cpu()

    sampled_traj = (sampled_traj - data_mu) / data_std
    target = (target - data_mu) / data_std

    sampled_traj = sampled_traj.numpy()
    target = target.numpy()

    kde_ll = np.zeros((batch_size, seq_len))

    for bi in range(batch_size):
        for si in range(seq_len):
            try:
                kde = gaussian_kde(sampled_traj[:, bi, si].T)
                pdf = np.clip(kde.logpdf(target[bi, si].T), a_min=log_pdf_lower_bound, a_max=None)[0]
                kde_ll[bi, si] = pdf
            except np.linalg.LinAlgError:
                kde_ll[bi, si] = np.nan
    return kde_ll


class NLLMDNLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, sigma, pi, x, mask, is_tril=False):
        """
        N = batch size
        T = number of time steps
        k = number of states
        m = number of mixtures

        mu.shape (N, T, m, k)
        sigma.shape (N, T, m, k, k)
        pi.shape (N, m)
        x.shape (N, T, k)
        mask.shape (N, T)
        """

        pi_expanded = pi.unsqueeze(1).expand(-1, mu.shape[1], -1)
        mix = tdist.Categorical(logits=pi_expanded)  # Broadcast this over time dim.
        L = sigma if is_tril else torch.linalg.cholesky(sigma)
        mvn = tdist.MultivariateNormal(mu, scale_tril=L)
        gmm = tdist.MixtureSameFamily(mix, mvn)
        comp_loss = gmm.log_prob(x).neg()  # (N, T)
        return mask_reduce_losses(comp_loss, mask)


class EWTALoss(nn.Module):
    """Evolving Winner Takes All loss."""

    def __init__(self):
        super().__init__()
        self.huber = nn.HuberLoss(reduction='none')

    def forward(self, mu, x, mask, w=1):
        """
        N = batch size
        T = number of time steps
        k = number of states
        m = number of mixtures
        w = max number of winners

        mu.shape (N, T, m, k)
        x.shape (N, T, k)
        mask.shape (N, T)
        """
        x = x.unsqueeze(2).expand(-1, -1, mu.size(2), -1)  # (N, T, m, k)
        reg_loss = self.huber(mu, x).sum(dim=-1)  # (N, T, m)
        masked_loss = reg_loss * mask[..., None]  # (N, T, m)
        masked_time = masked_loss.sum(1)  # (N, m)
        vals, _ = torch.topk(masked_time, k=w, dim=-1, largest=False)
        loss = vals.mean()
        return loss


class ModeDist(nn.Module):
    """
    Calculates the summed pairwise distance between mixtures.
    """

    def __init__(self):
        super().__init__()
        self.pwd = nn.PairwiseDistance()

    def forward(self, pred_state, all_preds, mask):
        """
        N = batch size
        T = number of time steps
        k = number of states
        m = number of mixtures

        pred_state.shape (N, T k)
        all_preds.shape (N, T, m, k)
        mask.shape (N, T)
        """
        pwd_loss = self.pwd(pred_state.unsqueeze(2), all_preds)  # (N, T, m)
        return mask_reduce_losses(pwd_loss.sum(-1), mask)
