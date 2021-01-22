import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math


class VectorQuantizer(nn.Module):
    def __init__(self, z_num, z_dim, normalize=False, reduction='frame_mean'):
        super(VectorQuantizer, self).__init__()

        if normalize:
            self.target_norm = 1.0 # norm_scale * math.sqrt(z.size(2))
        else:
            self.target_norm = None

        self.embeddings = nn.Parameter( torch.randn(z_num, z_dim, requires_grad=True))

        self.embed_norm()

        self.z_num = z_num
        self.z_dim = z_dim
        self.normalize = normalize
        self.reduction = reduction
        self.quantize = True

    def embed_norm(self):
        if self.target_norm:
            with torch.no_grad():
                self.embeddings.mul_(
                    self.target_norm / self.embeddings.norm(dim=1, keepdim=True)
                )

    def encode(self, z, time_last=True):
        # Flatten
        if time_last:
            B,D,T = z.shape
            z = z.transpose(1,2).contiguous().view(-1, D)
        else:
            B,T,D = z.shape
            z = z.contiguous().view(-1, D)
        # Normalize
        if self.target_norm:
            z_norm = self.target_norm * z / z.norm(dim=1, keepdim=True)
            embeddings = self.target_norm * self.embeddings / self.embeddings.norm(dim=1, keepdim=True)
        else:
            z_norm = z
            embeddings = self.embeddings
        # Calculate distances
        distances = (torch.sum(z_norm.pow(2), dim=1, keepdim=True) 
                    + torch.sum(embeddings.pow(2), dim=1)
                    - 2 * torch.matmul(z_norm, embeddings.t()))            
        # # Quantization encode
        encoding_idx = torch.argmin(distances, dim=1)
        # Deflatten
        encoding_idx = encoding_idx.view(B, T)
        return encoding_idx


    def decode(self, z_id, time_last=True):
        # Flatten
        B,T = z_id.shape
        encoding_idx = z_id.flatten()
        # Normalize
        if self.target_norm:
            embeddings = self.target_norm * self.embeddings / self.embeddings.norm(dim=1, keepdim=True)
        else:
            embeddings = self.embeddings
        # Quantization decode
        z_vq = embeddings.index_select(dim=0, index=encoding_idx)
        # Deflatten
        z_vq = z_vq.view(B, T, -1)
        if time_last:
            z_vq = z_vq.transpose(1,2).contiguous()
        return z_vq


    def forward(self, z, time_last=True):
        if not self.quantize:
            tensor_0 = torch.tensor(0.0, dtype=torch.float, device=z.device)
            return z, tensor_0, tensor_0, tensor_0

        # Flatten
        if time_last:
            B,D,T = z.shape
            z = z.transpose(1,2).contiguous().view(-1, D)
        else:
            B,T,D = z.shape
            z = z.contiguous().view(-1, D)
        device = z.device

        # Normalize
        if self.target_norm:
            z_norm = self.target_norm * z / z.norm(dim=1, keepdim=True)
            self.embed_norm()
            embeddings = self.target_norm * self.embeddings / self.embeddings.norm(dim=1, keepdim=True)
        else:
            z_norm = z
            embeddings = self.embeddings

        # Calculate distances
        distances = (torch.sum(z_norm.pow(2), dim=1, keepdim=True) 
                    + torch.sum(embeddings.pow(2), dim=1)
                    - 2 * torch.matmul(z_norm, embeddings.t()))
            
        # Quantize
        encoding_idx = torch.argmin(distances, dim=1)
        z_vq = embeddings.index_select(dim=0, index=encoding_idx)

        # Calculate losses
        encodings = torch.zeros(encoding_idx.size(0), embeddings.size(0)+1, device=z.device)
        encodings.scatter_(1, encoding_idx.unsqueeze(1), 1)

        avg_probs = torch.sum(encodings[:,:-1], dim=0) / encodings.size(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        update_detail = {'entropy': perplexity.item()}

        z_qut_loss = F.mse_loss(z_vq, z_norm.detach(), reduction='none')
        z_enc_loss = F.mse_loss(z_vq.detach(), z_norm, reduction='none')
        if self.target_norm:
            z_enc_loss += F.mse_loss(z_norm, z, reduction='none')    # Normalization loss
        if self.reduction == 'sum':
            z_qut_loss = z_qut_loss.sum()
            z_enc_loss = z_enc_loss.sum()
        elif self.reduction == 'mean':
            z_qut_loss = z_qut_loss.mean()
            z_enc_loss = z_enc_loss.mean()
        elif self.reduction == 'batch_mean':
            z_qut_loss = z_qut_loss.sum() / B
            z_enc_loss = z_enc_loss.sum() / B
        elif self.reduction == 'frame_mean':
            z_qut_loss = z_qut_loss.sum() / (B*T)
            z_enc_loss = z_enc_loss.sum() / (B*T)
        elif self.reduction == 'none':
            z_qut_loss = z_qut_loss.view(B,T,D)
            z_enc_loss = z_enc_loss.view(B,T,D)
            if time_last:
                z_qut_loss = z_qut_loss.transpose(1,2)
                z_enc_loss = z_enc_loss.transpose(1,2)                    

        z_vq = z_norm + (z_vq-z_norm).detach()

        # Deflatten
        z_vq = z_vq.view(B, T, D)
        if time_last:
            z_vq = z_vq.transpose(1,2).contiguous()

        # Output
        return z_vq, z_qut_loss, z_enc_loss, update_detail


    def sparsity(self):
        sparsity = torch.mm(self.embeddings,self.embeddings.t())
        sparsity_target = torch.arange(sparsity.size(0),device=sparsity.device)
        sparsity = F.cross_entropy(sparsity,sparsity_target)
        return sparsity

    def extra_repr(self):
        s = '{z_num}, {z_dim}'
        if self.normalize is not False:
            s += ', normalize=True'
        return s.format(**self.__dict__)


class EMAVectorQuantizer(nn.Module):
    def __init__(self, z_num, z_dim, mu, threshold=1.0, reduction='frame_mean'):
        super(EMAVectorQuantizer, self).__init__()

        self.register_buffer('emb_init', torch.tensor(0).bool())
        self.register_buffer('emb_sum', torch.zeros(z_num, z_dim))
        self.register_buffer('emb_elem', torch.ones(z_num))        
        self.register_buffer('embeddings', torch.zeros(z_num, z_dim))

        self.mu = mu
        self.z_num = z_num
        self.z_dim = z_dim
        self.threshold = threshold
        self.reduction = reduction
        self.quantize = True
        self.update = True

    def _tile(self, z):
        Num, Dim = z.shape
        if Num < self.z_num:
            n_repeats = (self.z_num + Num - 1) // Num
            std = 0.01 / np.sqrt(Dim)
            z = z.repeat(n_repeats, 1)
            z = z + torch.randn_like(z) * std
        return z

    def init_emb(self, z):
        mu, z_dim, z_num = self.mu, self.z_dim, self.z_num
        self.emb_init = ~ self.emb_init
        # init k_w using random vectors from z
        _z = self._tile(z)
        _emb_rand = _z[torch.randperm(_z.shape[0])][:z_num]
        self.embeddings = _emb_rand
        assert self.embeddings.shape == (z_num, z_dim)
        self.emb_sum = self.embeddings.clone()
        self.emb_elem = torch.ones(z_num, device=self.embeddings.device)

    def update_emb(self, z, z_idx):
        mu, z_dim, z_num = self.mu, self.z_dim, self.z_num
        with torch.no_grad():
            # Calculate new centres
            z_onehot = torch.zeros(z_num, z.shape[0], device=z.device)  # z_num, N * L
            z_onehot.scatter_(0, z_idx.view(1, z.shape[0]), 1)

            _emb_sum = torch.matmul(z_onehot, z)  # z_num, w
            _emb_elem = z_onehot.sum(dim=-1)  # z_num
            _z = self._tile(z)
            _emb_rand = _z[torch.randperm(_z.shape[0])][:z_num]

            # Update centres
            old_embeddings = self.embeddings.clone()
            self.emb_sum = mu * self.emb_sum + (1. - mu) * _emb_sum  # w, z_num
            self.emb_elem = mu * self.emb_elem + (1. - mu) * _emb_elem  # z_num
            usage = (self.emb_elem.view(z_num, 1) >= self.threshold).float()
            self.embeddings = usage * (self.emb_sum.view(z_num, z_dim) / self.emb_elem.view(z_num, 1)) \
                     + (1 - usage) * _emb_rand
            _k_prob = _emb_elem / torch.sum(_emb_elem)  # z_onehot.mean(dim=-1)  # prob of each bin
            entropy = torch.exp(-torch.sum(_k_prob * torch.log(_k_prob + 1e-8)))  # entropy ie how diverse
            used_curr = (_emb_elem >= self.threshold).sum()
            usage = torch.sum(usage)
            dk = torch.norm(self.embeddings - old_embeddings) / np.sqrt(np.prod(old_embeddings.shape))

        return {    
            'entropy': entropy.item(),
            'used_curr': used_curr.item(),
            'usage': usage.item(),
            'diff_emb': dk.item()
        }


    def encode(self, z, time_last=True):
        # Flatten
        if time_last:
            B,D,T = z.shape
            z = z.transpose(1,2).contiguous().view(-1, D)
        else:
            B,T,D = z.shape
            z = z.contiguous().view(-1, D)
        # Calculate distances
        distances = (torch.sum(z.pow(2), dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.pow(2), dim=1)
                    - 2 * torch.matmul(z, self.embeddings.t()))           
        # # Quantization encode
        encoding_idx = torch.argmin(distances, dim=1)
        # Deflatten
        encoding_idx = encoding_idx.view(B, T)
        return encoding_idx


    def decode(self, z_id, time_last=True):
        # Flatten
        B,T = z_id.shape
        encoding_idx = z_id.flatten()
        # Quantization decode
        z_vq = self.embeddings.index_select(dim=0, index=encoding_idx)
        # Deflatten
        z_vq = z_vq.view(B, T, -1)
        if time_last:
            z_vq = z_vq.transpose(1,2).contiguous()
        return z_vq


    def forward(self, z, time_last=True):
        if not self.quantize:
            tensor_0 = torch.tensor(0.0, dtype=torch.float, device=z.device)
            return z, tensor_0, tensor_0, tensor_0

        # Flatten
        if time_last:
            B,D,T = z.shape
            z = z.transpose(1,2).contiguous().view(-1, D)
        else:
            B,T,D = z.shape
            z = z.contiguous().view(-1, D)

        # Init. embeddings
        if not self.emb_init and self.training:
            self.init_emb(z)

        with torch.no_grad():
            # Calculate distances
            distances = (torch.sum(z.pow(2), dim=1, keepdim=True) 
                        + torch.sum(self.embeddings.pow(2), dim=1)
                        - 2 * torch.matmul(z, self.embeddings.t()))
            # Quantize
            encoding_idx = torch.argmin(distances, dim=1)
            z_vq = self.embeddings.index_select(dim=0, index=encoding_idx)

        # Update codebook with EMA
        if self.update and self.training:
            update_detail = self.update_emb(z, encoding_idx)
        else:
            update_detail = dict()

        z_qut_loss = 0.0
        z_enc_loss = F.mse_loss(z_vq.detach(), z, reduction='none')
        if self.reduction == 'sum':
            z_enc_loss = z_enc_loss.sum()
        elif self.reduction == 'mean':
            z_enc_loss = z_enc_loss.mean()
        elif self.reduction == 'batch_mean':
            z_enc_loss = z_enc_loss.sum() / B
        elif self.reduction == 'frame_mean':
            z_enc_loss = z_enc_loss.sum() / (B*T)
        elif self.reduction == 'none':
            z_enc_loss = z_enc_loss.view(B,T,D)
            if time_last:
                z_enc_loss = z_enc_loss.transpose(1,2)                    

            z_vq = z + (z_vq-z).detach()

        # Deflatten
        z_vq = z_vq.view(B, T, D)
        if time_last:
            z_vq = z_vq.transpose(1,2).contiguous()

        # Output
        return z_vq, z_qut_loss, z_enc_loss, update_detail


    def sparsity(self):
        sparsity = torch.mm(self._embedding,self._embedding.t())
        sparsity_target = torch.arange(sparsity.size(0),device=sparsity.device)
        sparsity = F.cross_entropy(sparsity,sparsity_target)
        return sparsity

    def extra_repr(self):
        s = '{z_num}, {z_dim}, mu={mu}, threshold={threshold}'
        return s.format(**self.__dict__)


class Jitter(nn.Module):
    """
    Jitter implementation from [Chorowski et al., 2019].
    During training, each latent vector can replace either one or both of
    its neighbors. As in dropout, this prevents the model from
    relying on consistency across groups of tokens. Additionally,
    this regularization also promotes latent representation stability
    over time: a latent vector extracted at time step t must strive
    to also be useful at time steps t − 1 or t + 1.
    """

    def __init__(self, probability=0.12):
        super(Jitter, self).__init__()

        self._probability = probability

    def forward(self, quantized):
        if self._probability == 0.0 or not self.training:
            return quantized

        original_quantized = quantized.detach().clone()
        length = original_quantized.size(2)
        for i in range(length):
            """
            Each latent vector is replace with either of its neighbors with a certain probability
            (0.12 from the paper).
            """
            replace = [True, False][np.random.choice([1, 0], p=[self._probability, 1 - self._probability])]
            if replace:
                if i == 0:
                    neighbor_index = i + 1
                elif i == length - 1:
                    neighbor_index = i - 1
                else:
                    """
                    "We independently sample whether it is to
                    be replaced with the token right after
                    or before it."
                    """
                    neighbor_index = i + np.random.choice([-1, 1], p=[0.5, 0.5])
                quantized[:, :, i] = original_quantized[:, :, neighbor_index]

        return quantized

    def extra_repr(self):
        s = 'jitter_prob={_probability}'
        return s.format(**self.__dict__)
