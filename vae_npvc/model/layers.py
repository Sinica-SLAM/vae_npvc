import torch
import torch.nn as nn
import torch.nn.functional as F

import math

EPSILON = 1e-6
PI = math.pi
LOG_2PI = math.log( 2.0 * PI)


class Conditions(nn.Module):
    def __init__(self, cond_num, cond_dim, normalize=False):
        super(Conditions, self).__init__()
        self._embedding = nn.Embedding(   
            cond_num,
            cond_dim, 
            padding_idx=None, 
            max_norm=None, 
            norm_type=2.0, 
            scale_grad_by_freq=False, 
            sparse=False, 
            _weight=None
        )
        if normalize:
            self.target_norm = 1.0
        else:
            self.target_norm = None
        self.embed_norm()

        self.cond_num = cond_num
        self.cond_num = cond_num
        self.normalize = normalize

    def embed_norm(self):
        if self.target_norm:
            with torch.no_grad():
                self._embedding.weight.mul_(
                    self.target_norm / self._embedding.weight.norm(dim=1, keepdim=True)
                )

    def forward(self, input, pre_norm=True):
        if self.target_norm:
            if pre_norm:
                self.embed_norm()
            embedding = self.target_norm * self._embedding.weight / self._embedding.weight.norm(dim=1, keepdim=True)
            return F.embedding( input, embedding, 
                                padding_idx=None, 
                                max_norm=None, 
                                norm_type=2.0, 
                                scale_grad_by_freq=False, 
                                sparse=False)
        else:
            return self._embedding(input)

    def sparsity(self):
        sparsity = torch.mm(self._embedding.weight,self._embedding.weight.t())
        sparsity_target = torch.arange(sparsity.size(0),device=sparsity.device)
        sparsity = F.cross_entropy(sparsity,sparsity_target)
        return sparsity


class Conv1d_Layernorm_LRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(Conv1d_Layernorm_LRelu, self).__init__()

        padding = int((kernel_size*dilation - dilation)/2)
        self.conv = nn.Conv1d( 
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=True
        )
        self.layernorm = nn.GroupNorm( 
            1, 
            out_channels, 
            eps=1e-05, 
            affine=True
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.02)

        self.padding = (padding, padding)

    def forward(self, x):
        x = self.conv(x)
        x = self.layernorm(x)
        x = self.lrelu(x)

        return x


class DeConv1d_Layernorm_GLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(DeConv1d_Layernorm_GLU, self).__init__()

        padding = int((kernel_size*dilation - dilation)/2)
        self.deconv = nn.ConvTranspose1d( 
            in_channels=in_channels,
            out_channels=out_channels*2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=True
        )
        self.layernorm = nn.GroupNorm( 
            2,
            out_channels*2, 
            eps=1e-05, 
            affine=True
        )
        self.half_channel = out_channels

    def forward(self, x):
        x = self.deconv(x)
        x = self.layernorm(x)
        x_tanh = torch.tanh(x[:,:self.half_channel])
        x_sigmoid = torch.sigmoid(x[:,self.half_channel:])
        x = x_tanh * x_sigmoid

        return x


class Conv1d_Layernorm_LRelu_Residual(nn.Module):
    def __init__(self,
                 kernel_size=3,
                 channels=128,
                 layers=2,
                 dilation=1,              
                 use_causal_conv=False,
                 ):
        super(Conv1d_Layernorm_LRelu_Residual, self).__init__()

        self.use_causal_conv = use_causal_conv

        if not self.use_causal_conv:
            assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
            padding1 = (kernel_size - 1) // 2 * dilation
            padding2 = (kernel_size - 1) // 2
            self.total_padding = None
        else:
            padding1 = (kernel_size - 1) * dilation
            padding2 = (kernel_size - 1)
            self.total_padding = padding1 + padding2 * (layers - 1)

        stack = [
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding1, bias=True),
            nn.GroupNorm(num_groups=1, num_channels=channels, eps=1e-5, affine=True),
        ]
        for i in range(layers-1):
            stack += [
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv1d(channels, channels, kernel_size, padding=padding2, bias=True),
                nn.GroupNorm(num_groups=1, num_channels=channels, eps=1e-5, affine=True),
            ]

        self.stack = nn.Sequential(*stack)

        self.skip_layer = nn.Conv1d( channels, channels, 1, bias=True)


    def forward(self, c):
        """Calculate forward propagation.
        Args:
            c (Tensor): Input tensor (B, channels, T).
        Returns:
            Tensor: Output tensor (B, chennels, T).
        """
        if not self.use_causal_conv:
            return self.stack(c) + self.skip_layer(c)
        else:
            return self.stack(c)[...,:c.size(-1)] + self.skip_layer(c)


class DeConv1d_Layernorm_GLU_ResSkip(nn.Module):
    def __init__(self,
                 kernel_size=3,
                 in_channels=128,
                 cond_channels=128,
                 skip_channels=80,
                 dilation=0,
                 dropout=0.0, 
                 use_causal_conv=False,
                 ):
        super(DeConv1d_Layernorm_GLU_ResSkip, self).__init__()

        self.use_causal_conv = use_causal_conv
        self.dropout = dropout

        if not self.use_causal_conv:
            assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
            padding = (kernel_size - 1) // 2 * dilation
            self.conv_in = nn.ConvTranspose1d(in_channels, in_channels*2, kernel_size,
                                padding=padding, dilation=dilation, bias=True)
            self.norm_layer = nn.GroupNorm(num_groups=2, num_channels=in_channels*2, eps=1e-5, affine=True)

        else:
            padding = (kernel_size - 1) * dilation
            self.conv_in = nn.Conv1d(in_channels, in_channels*2, kernel_size,
                                padding=padding, dilation=dilation, bias=True)
            self.norm_layer = None

        if cond_channels is not None and cond_channels > 0:
            self.conv_cond = nn.Conv1d(cond_channels, in_channels*2, 1, bias=True)
        else:
            self.conv_cond = None
        self.res_skip_layers = nn.Conv1d(in_channels, in_channels+skip_channels, 1, bias=True)

        self.in_channels = in_channels


    def forward(self, x, c):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            c (Tensor): Local conditioning auxiliary tensor (B, cond_channels, T).
        Returns:
            Tensor: Output tensor for skip connection (B, in_channels, T).
        """

        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_res = self.conv_in(x)
        if self.conv_cond:
            x_c = self.conv_cond(c)
        else:
            x_c = 0.0

        if not self.use_causal_conv:
            x_res = self.norm_layer(x_res + x_c)
        else:
            x_res = x_res[..., :x.size(-1)] + x_c
        
        x_res_tanh = torch.tanh(x_res[:,:self.in_channels])
        x_res_sigmoid = torch.sigmoid(x_res[:,self.in_channels:])
        x_res = x_res_tanh * x_res_sigmoid

        x_res_skip = self.res_skip_layers(x_res)
        
        x = x_res_skip[:,:self.in_channels,:] + x
        x_skip = x_res_skip[:,self.in_channels:,:]

        return x, x_skip


def GaussianSampler(z_mu, z_lv):
    z = torch.randn_like(z_mu)
    z_std = torch.exp(0.5 * z_lv)
    z = z * z_std + z_mu
    return z


def GaussianKLD(mu1, lv1, mu2, lv2, dim=-1):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    mu_diff_sq = (mu1 - mu2).pow(2)
    element_kld = 0.5 * ((lv2 - lv1) + ( v1 + mu_diff_sq ) / ( v2 + EPSILON ) - 1.0 )
    return element_kld.sum(dim=dim)


def GaussianLogDensity(x, mu, log_var, dim=-1):
    var = torch.exp(log_var)
    mu_diff2_over_var = (x - mu).pow(2) / (var + EPSILON)
    log_prob = -0.5 * ( LOG_2PI + log_var + mu_diff2_over_var )
    return log_prob.sum(dim=dim)


def kl_loss(mu, lv):
    # Simplified from GaussianKLD
    return 0.5 * (torch.exp(lv) + mu.pow(2) - lv - 1.0).sum()

def skl_loss(mu1, lv1, mu2, lv2):
    # Symmetric GaussianKLD
    v1, v2 = torch.exp(lv1), torch.exp(lv2)
    return 0.5 * ( v2/v1 + v1/v2 - 2 + (mu1 - mu2).pow(2) / ( 1/v1 + 1/v2) ).sum()

def log_loss(x, mu, reduction='frame_mean'):
    # Simplified from GaussianLogDensity
    B, D, T = x.shape
    loss = 0.5 * (LOG_2PI + (x - mu).pow(2))
    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'batch_mean':
        return loss.sum() / B
    elif reduction == 'frame_mean':
        return loss.sum() / (B*T)
    elif reduction == 'none':
        return loss
