import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


def activation_fn(activation):
    if activation == nn.Softplus:
        return nn.Softplus()
    elif activation == nn.Sigmoid:
        return nn.Sigmoid()
    elif activation == nn.Tanh:
        return nn.Tanh()
    else:
        return None

class FCNet(nn.Module):
    def __init__(self, input_dim, layers, out_layers):
        super(FCNet, self).__init__()
        self.layers = nn.ModuleList()
        self.out_layers = nn.ModuleList()
        
        for layer_dim in layers:
            self.layers.append(nn.Linear(input_dim, layer_dim))
            input_dim = layer_dim

        for out_dim, activation in out_layers:
            self.out_layers.append(nn.Linear(input_dim, out_dim))
            if activation:
                self.out_layers.append(activation)
            input_dim = out_dim

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        for layer in self.out_layers:
            x = layer(x)
        return x

# def gaussianKL(mu_p, cov_p, mu_q, cov_q, mask=None):
#     if mu_p.dim() == 2:
#         mu_p = mu_p.unsqueeze(1)
#         cov_p = cov_p.unsqueeze(1)
#         mu_q = mu_q.unsqueeze(1)
#         cov_q = cov_q.unsqueeze(1)

#     diff_mu = mu_p - mu_q
#     KL = torch.log(1e-10 + cov_p) - torch.log(1e-10 + cov_q) - 1. + cov_q / (cov_p + 1e-10) + diff_mu ** 2 / (cov_p + 1e-10)
#     if mask is not None:
#         KL_masked = 0.5 * KL * mask.expand_as(KL)
#     else:
#         KL_masked = 0.5 * KL
#     return KL_masked.sum(dim=2).mean()
# def gaussianKL(mu_p, cov_p, mu_q, cov_q, mask=None):
#     if mu_p.dim() == 2:
#         mu_p = mu_p.unsqueeze(1)
#         cov_p = cov_p.unsqueeze(1)
#         mu_q = mu_q.unsqueeze(1)
#         cov_q = cov_q.unsqueeze(1)

#     diff_mu = mu_p - mu_q

#     # if torch.any(cov_p <= 0) or torch.any(cov_q <= 0):
#     #     raise ValueError("Covariance values must be positive")

#     KL = torch.log(1e-8 + cov_p) - torch.log(1e-8 + cov_q) - 1. + cov_q / (cov_p + 1e-8) + diff_mu ** 2 / (cov_p + 1e-8)
    
#     if mask is not None:
#         KL_masked = 0.5 * KL * mask.expand_as(KL)
#     else:
#         KL_masked = 0.5 * KL

#     return KL_masked.sum(dim=2).mean()
def gaussianKL(mu_p, cov_p, mu_q, cov_q, mask=None):
    epsilon = 1e-8  
    cov_p = cov_p + epsilon
    cov_q = cov_q + epsilon

    if mu_p.dim() == 2:
        mu_p = mu_p.unsqueeze(1)
        cov_p = cov_p.unsqueeze(1)
        mu_q = mu_q.unsqueeze(1)
        cov_q = cov_q.unsqueeze(1)

    dist_p = Normal(mu_p, torch.sqrt(cov_p))
    dist_q = Normal(mu_q, torch.sqrt(cov_q))
    kl = kl_divergence(dist_p, dist_q)

    if mask is not None:
        kl = kl * mask.expand_as(kl)

    return kl.sum(dim=2).mean()


def gaussianNLL(data, mu, cov, mask=None):
    nll = 0.5 * (torch.log(2 * torch.tensor(3.141592653589793)) + torch.log(1e-10 + cov) + (data - mu) ** 2 / (cov + 1e-10))
    if mask is not None:
        nll = nll * mask.expand_as(nll)
    return nll.sum(dim=-1).mean()

def recons_loss(cost, real, recons):
    if cost == 'l2':
        loss = ((real - recons) ** 2).sum(dim=-1)
        loss = 0.2 * torch.sqrt(1e-10 + loss.mean())
    elif cost == 'l2sq':
        loss = ((real - recons) ** 2).sum(dim=-1)
        loss = 0.05 * loss.mean()
    elif cost == 'l1':
        loss = torch.abs(real - recons).sum(dim=-1)
        loss = 0.02 * loss.mean()
    elif cost == 'cross_entropy':
        loss = - (real * torch.log(1e-10 + recons) + (1 - real) * torch.log(1e-10 + 1 - recons)).sum(dim=-1)
        loss = loss.mean()
    return loss

def save_plots(opts, x_gt_tr, x_gt_te, x_recons_tr, x_recons_te, train_nll, train_kl, validation_nll, validation_kl,
               train_x_loss, validation_x_loss, train_a_loss, validation_a_loss, train_r_loss, validation_r_loss,
               x_seq_sample, filename):
    # This function generates and saves the plot of the results as per the original implementation.
    # Here we'll skip the detailed implementation for brevity.
    pass
