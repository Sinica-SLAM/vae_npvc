import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ( Conditions, log_loss, Conv1d_Layernorm_LRelu_Residual, DeConv1d_Layernorm_GLU_ResSkip)
from .layers_vq import ( VectorQuantizer, EMAVectorQuantizer, Jitter)


class Model(nn.Module):
    def __init__(self, arch):
        super(Model, self).__init__()

        self.encoder = Encoder(**arch['encoder'])
        self.decoder = Decoder(**arch['decoder'])

        self.use_ema = arch.get('use_ema', False)
        if self.use_ema:
            self.quantizer = EMAVectorQuantizer( 
                arch.get('z_num', 512),
                arch.get('z_dim', 128),
                arch.get('mu', 0.9),
                reduction='sum'
            )
        else:            
            self.quantizer = VectorQuantizer( 
                arch.get('z_num', 512),
                arch.get('z_dim', 128),
                normalize=arch.get('embed_norm', True),
                reduction='sum'
            )

        self.embeds = Conditions(
            arch.get('y_num', 10),
            arch.get('y_dim', 128),
            normalize=False
        )

        self.jitter = Jitter(probability=arch.get('jitter_p', 0.0))
        
        self.beta = arch.get('beta', 0.01)


    def encode(self, input):
        if isinstance(input, list) or isinstance(input, tuple):
            x = input[0]
        else:
            x = input
        z = self.encoder(x)
        z_idx = self.quantizer.encode(z)
        return z_idx


    def decode(self, input):
        z_idx, y_idx = input
        y = self.embeds(y_idx).transpose(1,2).contiguous()    # Size( N, y_dim, 1)
        z_vq = self.quantizer.decode(z_idx)
        xhat = self.decoder((z_vq, y))
        return xhat


    def infer(self, input):
        x, y_idx = input
        z_idx = self.encode(x)
        x_hat = self.decode((z_idx,y_idx))
        return x_hat


    def forward(self, input):
        # Preprocess
        x, y_idx = input    # ( Size( N, x_dim, nframes), Size( N, 1))
        y = self.embeds(y_idx).transpose(1,2).contiguous()    # Size( N, y_dim, 1)
        # Encode
        z = self.encoder(x)

        # Decode
        if self.training:
            z_vq, z_qut_loss, z_enc_loss, vq_detail = self.quantizer(z)
            z_vq = self.jitter(z_vq)

            xhat = self.decoder((z_vq, y))

            # Loss
            Batch, Dim, Time = x.shape
            mean_factor = Batch * Time

            z_qut_loss = z_qut_loss / mean_factor
            z_enc_loss = z_enc_loss / mean_factor
            
            x_loss = log_loss(xhat, x) / mean_factor

            loss = x_loss + z_qut_loss + self.beta * z_enc_loss
            
            losses = {'Total': loss.item(),
                      'VQ loss': z_enc_loss.item(),
                      'entropy': vq_detail['entropy'].item(),
                      'usage_batch': vq_detail['used_curr'].item(),
                      'usage': vq_detail['usage'].item(),
                      'diff_emb': vq_detail['dk'].item(),
                      'X like': x_loss.item()}

            return xhat, loss, losses

        else:
            z_vq = self.quantizer(z)
            xhat = self.decoder((z_vq,y))

            # Loss
            Batch, Dim, Time = x.shape
            mean_factor = Batch * Time

            z_loss = (z - z_vq).pow(2).sum() / mean_factor
            x_loss = log_loss(xhat, x) / mean_factor

            loss = x_loss + z_loss

            losses = {'Total': loss.item(),
                      'VQ loss': z_loss.item(),
                      'X like': x_loss.item()}

            return losses


    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                # logging.debug(f"Weight norm is removed from {m}.")
                # print(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


    def load_state_dict(self, state_dict):
        if not self.use_ema:
            warning_mseg =  'Embedding size mismatch for {}: '
            warning_mseg += 'copying a param with shape {} from checkpoint, '
            warning_mseg += 'resizing the param with shape {} in current model.'

            state_dict_shape, module_param_shape = state_dict['quantizer.embeddings'].shape, self.quantizer.embeddings.shape
            if state_dict_shape != module_param_shape:
                print(warning_mseg.format('model.quantizer', state_dict_shape, module_param_shape))
                self.quantizer = VectorQuantizer( 
                        state_dict_shape[0], state_dict_shape[1], 
                        normalize=self.quantizer.normalize, reduction=self.quantizer.reduction
                        )
        super(Model, self).load_state_dict(state_dict)


class Encoder(nn.Module):
    def __init__(self,
            in_channels=[513, 1024, 512, 256],
            out_channels=[1024, 512, 256, 128],
            downsample_scales=[1, 1, 1, 1],
            kernel_size=3,
            z_channels=128,
            dilation=True,
            stack_kernel_size=3,
            stack_layers=2,
            stacks=[3, 3, 3, 3],
            use_weight_norm=True,
            use_causal_conv=False,
        ):
        super(Encoder, self).__init__()

        # check hyper parameters is valid
        assert not use_causal_conv, "Not supported yet."

        # add initial layer
        layers = []

        for ( in_channel, out_channel, ds_scale, stack) in zip( in_channels, out_channels, downsample_scales, stacks):

            if ds_scale == 1:
                _kernel_size = kernel_size
                _padding = (kernel_size - 1) // 2
                _stride = 1
            else:
                _kernel_size = ds_scale * 2
                _padding = ds_scale // 2 + ds_scale % 2
                _stride = ds_scale

            layers += [
                nn.Conv1d(in_channel, out_channel, _kernel_size, stride=_stride, padding=_padding)
            ]

            # add residual stack
            for j in range(stack):
                layers += [
                    Conv1d_Layernorm_LRelu_Residual(
                        kernel_size=stack_kernel_size,
                        channels=out_channel,
                        layers=stack_layers,
                        dilation=2**j if dilation else 1,
                        use_causal_conv=use_causal_conv,
                    )
                ]

            layers += [nn.LeakyReLU(negative_slope=0.2),]

        # add final layer
        layers += [nn.Conv1d( out_channels[-1], z_channels, 1)]

        self.encode = nn.Sequential(*layers)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, input):
        """Calculate forward propagation.
        Args:
            input (Tensor): Input tensor (B, in_channels, T).
        Returns:
            Tensor: Output tensor (B, out_channels, T).
        """
        return self.encode(input)

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                # m.weight.data.normal_(0.0, 0.02)
                nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
                # logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)


class Decoder(nn.Module):
    def __init__(self,
            in_channels=[128, 256, 512, 1024],
            out_channels=[256, 512, 1024, 513],
            upsample_scales=[1, 1, 1, 1],
            cond_channels=128,
            skip_channels=80,
            final_channels=80,
            kernel_size=5,
            dilation=True,
            stack_kernel_size=3,
            stacks=[3, 3, 3, 3],
            use_weight_norm=True,
            use_causal_conv=False,
        ):
        super(Decoder, self).__init__()

        # check hyper parameters is valid
        assert not use_causal_conv, "Not supported yet."

        # add initial layer
        layers = nn.ModuleList()

        for ( in_channel, out_channel, us_scale, stack) in zip( in_channels, out_channels, upsample_scales, stacks):
            # add resampling layer
            if us_scale == 1:
                _kernel_size = kernel_size
                padding = (kernel_size - 1) // 2
                output_padding = 0
                stride = 1
            else:
                _kernel_size = us_scale * 2
                padding = us_scale // 2 + us_scale % 2
                output_padding = us_scale % 2
                stride = us_scale

            layers += [
                nn.ConvTranspose1d( 
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=_kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding
                )
            ]
            # add residual stack
            for j in range(stack):
                layers += [
                    DeConv1d_Layernorm_GLU_ResSkip(
                        kernel_size=stack_kernel_size,
                        in_channels=out_channel,
                        cond_channels=cond_channels,
                        skip_channels=skip_channels,
                        dilation=2**j if dilation else 1,
                        dropout=0.0,
                        use_causal_conv=use_causal_conv,
                    )
                ]             
            
        # add final layer
        final_layer = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d( skip_channels, skip_channels, 1),
                nn.ReLU(),
                nn.Conv1d( skip_channels, final_channels, 1),
            )

        self.layers = layers
        self.final_layer = final_layer

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, input):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            c (Tensor): Input tensor (B, cond_channels, 1).
        Returns:
            Tensor: Output tensor (B, out_channels, T).
        """
        # return self.decode(x)
        x, c = input
        x_out = 0.0
        c = c[:,:,:1]
        for layer in self.layers:
            if isinstance(layer, DeConv1d_Layernorm_GLU_ResSkip):
                x, x_skip = layer( x, c.repeat(1,1,x.size(2)))
                x_out += x_skip
            else:
                x = layer(x)
        x = x_out * math.sqrt(1.0 / len(self.layers))
        x = self.final_layer(x)
        return x

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                # m.weight.data.normal_(0.0, 0.02)
                nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
                # logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)
