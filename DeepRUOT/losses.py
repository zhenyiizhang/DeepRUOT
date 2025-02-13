# Adapt from MIOFlow
import os, math, numpy as np
import torch
import torch.nn as nn
# %% Adapt from MIOFlow
class MMD_loss(nn.Module):
    '''
    https://github.com/ZongxianLee/MMD_Loss.Pytorch/blob/master/mmd_loss.py
    '''
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

# %% Adapt from MIOFlow with modification
import ot
import torch.nn as nn
import torch
import numpy as np

class OT_loss1(nn.Module):
    _valid = 'emd sinkhorn sinkhorn_knopp_unbalanced'.split()

    def __init__(self, which='emd', use_cuda=True):
        if which not in self._valid:
            raise ValueError(f'{which} not known ({self._valid})')
        self.which = which
        self.use_cuda = use_cuda

    def __call__(self, source, target, mu, nu,  device=None, sigma=None, use_cuda=None):
        if use_cuda is None:
            use_cuda = self.use_cuda
        if not isinstance(mu, torch.Tensor):
            mu = torch.tensor(mu, dtype=torch.float32)
        if not isinstance(nu, torch.Tensor):
            nu = torch.tensor(nu, dtype=torch.float32)
        
        if use_cuda:
            #mu = mu.cuda()
            mu = mu.to(device)
            nu = nu.to(device)

        M = torch.cdist(source, target)**2

        if self.which == 'emd':
            pi = ot.emd(mu.detach().cpu().numpy(), nu.detach().cpu().numpy(), M.detach().cpu().numpy())
        elif self.which == 'sinkhorn':
            if sigma is None:
                raise ValueError('sigma must be provided for sinkhorn method')
            #pi = ot.sinkhorn(mu.detach().cpu().numpy(), nu.detach().cpu().numpy(), M.detach().cpu().numpy(), sigma.detach().cpu().numpy(), method='sinkhorn_log')
            #pi = ot.sinkhorn(mu, nu, M, sigma, method='sinkhorn_log')
            pi = ot.sinkhorn(mu, nu, M, sigma)
        elif self.which == 'sinkhorn_knopp_unbalanced':
            if sigma is None:
                raise ValueError('sigma must be provided for sinkhorn_knopp_unbalanced method')
            pi = ot.unbalanced.sinkhorn_knopp_unbalanced(mu.detach().cpu().numpy(), nu.detach().cpu().numpy(), M.detach().cpu().numpy(), sigma, sigma)
        else:
            raise ValueError(f'{self.which} not known ({self._valid})')

        if isinstance(pi, np.ndarray):
            pi = torch.tensor(pi, dtype=torch.float32)
        elif isinstance(pi, torch.Tensor):
            pi = pi.clone().detach()
        
        #pi = pi.cuda() if use_cuda else pi
        pi = pi.to(device) if use_cuda else pi
        M = M.to(pi.device)
        loss = torch.sum(pi * M)
        return loss

class OT_loss2(nn.Module):
    _valid = 'emd sinkhorn sinkhorn_knopp_unbalanced'.split()

    def __init__(self, which='sinkhorn', use_cuda=True):
        if which not in self._valid:
            raise ValueError(f'{which} not known ({self._valid})')
        self.which = which
        self.use_cuda = use_cuda

    def __call__(self, source, target, mu, nu, sigma=None, use_cuda=None):
        if use_cuda is None:
            use_cuda = self.use_cuda
        if not isinstance(mu, torch.Tensor):
            mu = torch.tensor(mu, dtype=torch.float32)
        if not isinstance(nu, torch.Tensor):
            nu = torch.tensor(nu, dtype=torch.float32)
        
        if use_cuda:
            mu = mu.cuda()
            nu = nu.cuda()

        M = torch.cdist(source, target)**2

        if self.which == 'emd':
            pi = ot.emd(mu.detach().cpu().numpy(), nu.detach().cpu().numpy(), M.detach().cpu().numpy())
        elif self.which == 'sinkhorn':
            if sigma is None:
                raise ValueError('sigma must be provided for sinkhorn method')
            #pi = ot.sinkhorn(mu.detach().cpu().numpy(), nu.detach().cpu().numpy(), M.detach().cpu().numpy(), sigma.detach().cpu().numpy(), method='sinkhorn_log')
            #pi = ot.sinkhorn(mu, nu, M, sigma, method='sinkhorn_log')
            pi = ot.sinkhorn(mu, nu, M, sigma)
        elif self.which == 'sinkhorn_knopp_unbalanced':
            if sigma is None:
                raise ValueError('sigma must be provided for sinkhorn_knopp_unbalanced method')
            pi = ot.unbalanced.sinkhorn_knopp_unbalanced(mu.detach().cpu().numpy(), nu.detach().cpu().numpy(), M.detach().cpu().numpy(), sigma, sigma)
        else:
            raise ValueError(f'{self.which} not known ({self._valid})')

        if isinstance(pi, np.ndarray):
            pi = torch.tensor(pi, dtype=torch.float32)
        elif isinstance(pi, torch.Tensor):
            pi = pi.clone().detach()
        
        pi = pi.cuda() if use_cuda else pi
        M = M.to(pi.device)
        loss = torch.sum(pi * M)
        return loss


# %%  Adapt from MIOFlow
import torch.nn as nn
import torch
class Density_loss(nn.Module):
    def __init__(self, hinge_value=0.01):
        self.hinge_value = hinge_value
        pass

    def __call__(self, source, target, groups = None, to_ignore = None, top_k = 5):
        if groups is not None:
            # for global loss
            c_dist = torch.stack([
                torch.cdist(source[i], target[i]) 
                # NOTE: check if this should be 1 indexed
                for i in range(1,len(groups))
                if groups[i] != to_ignore
            ])
        else:
            # for local loss
             c_dist = torch.stack([
                torch.cdist(source, target)                 
            ])
        values, _ = torch.topk(c_dist, top_k, dim=2, largest=False, sorted=False)
        values -= self.hinge_value
        values[values<0] = 0
        loss = torch.mean(values)
        return loss


class Local_density_loss(nn.Module):
    def __init__(self):
        pass

    def __call__(self, sources, targets, groups, to_ignore, top_k = 5):
        # print(source, target)
        # c_dist = torch.cdist(source, target) 
        c_dist = torch.stack([
            torch.cdist(sources[i], targets[i]) 
            # NOTE: check if should be from range 1 or not.
            for i in range(1, len(groups))
            if groups[i] != to_ignore
        ])
        vals, inds = torch.topk(c_dist, top_k, dim=2, largest=False, sorted=False)
        values = vals[inds[inds]]
        loss = torch.mean(values)
        return loss
