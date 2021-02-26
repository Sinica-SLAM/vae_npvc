import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ( Conditions, log_loss, Conv1d_Layernorm_LRelu_Residual, DeConv1d_Layernorm_GLU_ResSkip)
from .layers_vq import ( VectorQuantizer, EMAVectorQuantizer, Jitter)
from .layers_gst import ( StyleTokenLayer)

class Model(nn.Module):
    def __init__(self, arch):
        super(Model, self).__init__()
        levels = arch.get('levels', 3)
        use_gst = arch.get('use_gst',True) if levels > 1 else False
        use_ema = arch.get('use_ema',True)
        y_num = arch.get('y_num', 10)
        y_dim = arch.get('y_dim', 128)

        encoders = list()
        quantizers = list()
        decoders = list()
        embeds = list()
        for i in range(levels):
            encoders.append(Encoder(**arch[f'encoder.{i}']))
            decoders.append(Decoder(**arch[f'decoder.{i}']))
            if use_gst and i == levels - 1:
                Quantizer = StyleTokenLayer
            elif use_ema:
                Quantizer = EMAVectorQuantizer
            else:
                Quantizer = VectorQuantizer
            quantizers.append(Quantizer(**arch[f'quantizer.{i}']))
            embeds.append(Conditions(y_num,y_dim,normalize=False))
            
        self.encoders = nn.ModuleList(encoders)
        self.quantizers = nn.ModuleList(quantizers)
        self.decoders = nn.ModuleList(decoders)
        self.embeds = nn.ModuleList(embeds)

        self.final_decoder = Decoder(**arch['final_decoder'])

        self.jitter = Jitter(probability=arch.get('jitter_p', 0.0))
        
        self.beta = arch.get('beta', 0.01)
        self.pooling_last = arch.get('pooling_last', True)
        self.upsample_last = arch.get('upsample_last', False)
        self.levels = levels
        self.use_gst = use_gst


    def encode(self, input):
        if isinstance(input, list) or isinstance(input, tuple):
            x = input[0]
        else:
            x = input
        zs = list()
        for i in range(self.levels):
            z_, x = self.encoders[i](x)
            # Pool last level
            if self.pooling_last and i == self.levels - 1:
                z_ = torch.mean(z_, dim=-1, keepdim=True)
            # Quantize
            if self.use_gst and i == self.levels - 1:
                z_ = self.quantizers[i](z_.squeeze(-1)).unsqueeze(-1)
            else:
                z_ = self.quantizers[i].encode(z_)
            zs.append(z_)
        
        return zs


    def decode(self, input, time=None):
        zs, ys = input
        time = time if time else max([z.size(-1) for z in zs])
        zs2 = list()
        for i in range(self.levels):
            y = self.embeds[i](ys[:,i:i+1]).transpose(1,2).contiguous()    # Size( N, y_dim, 1)
            if self.use_gst and i == self.levels - 1:
                z_vq = zs[i]
            else:
                z_vq = self.quantizers[i].decode(zs[i])
            if self.upsample_last:
                z = self.upsample(self.decoders[i]((z_vq, self.upsample(y, z_vq.size(-1)))), time)
            else:
                z = self.decoders[i]((self.upsample(z_vq, time), self.upsample(y, time)))
            zs2.append(z)
        z = torch.cat(zs2,dim=1)
        xhat = self.final_decoder((z, None))
        return xhat


    def infer(self, input):
        x, ys = input
        zs = self.encode(x)
        x_hat = self.decode((zs,ys))
        return x_hat

    def forward(self, input):
        # Preprocess
        x, y_idx = input    # ( Size( N, x_dim, nframes), Size( N, 1))
        # Init. lists
        z_vq_levels = list()
        z_qut_losses = list()
        z_enc_losses = list()
        vq_details = list()
        # Hierarchical encode & quantize
        x_ = x
        time = x.size(-1)
        for i in range(self.levels):
            z_, x_ = self.encoders[i](x_)
            # Pool last level
            if self.pooling_last and i == self.levels - 1:
                z_ = torch.mean(z_, dim=-1, keepdim=True)
            # Quantize
            if self.use_gst and i == self.levels - 1:
                z_vq = self.quantizers[i](z_.squeeze(-1)).unsqueeze(-1)
            else:
                z_vq, z_qut_loss, z_enc_loss, vq_detail = self.quantizers[i](z_)
                z_qut_losses.append(z_qut_loss)
                z_enc_losses.append(z_enc_loss)
                vq_detail['quanti_err'] = z_enc_loss.item()
                vq_details.append(vq_detail)
                z_vq = self.jitter(z_vq)
            # Decode
            y = self.embeds[i](y_idx[...,:1]).transpose(1,2).contiguous()    # Size( N, y_dim, 1)

            if self.upsample_last:
                z_vq = self.upsample(self.decoders[i]((z_vq, self.upsample(y, z_vq.size(-1)))), time)
            else:
                z_vq = self.decoders[i]((self.upsample(z_vq, time), self.upsample(y, time)))
            z_vq_levels.append(z_vq)
        # Final Decode
        z_vq = torch.cat(z_vq_levels,dim=1)
        xhat = self.final_decoder((z_vq, None))

        # Loss
        z_qut_loss = sum(z_qut_losses)
        z_enc_loss = sum(z_enc_losses)
        x_loss = log_loss(xhat, x)
        loss = x_loss + z_qut_loss + self.beta * z_enc_loss
        # Detail
        losses = {'Total': loss.item(),
                  'VQ loss': z_enc_loss.item(),
                  'X like': x_loss.item()}
        for i, vq_detail in enumerate(vq_details):
            losses.update({key+f'.{i}': val for key, val in vq_detail.items()})

        return xhat, loss, losses


    def upsample(self, z, target_len):
        """Upsample the last dim of input z"""
        z_len = z.size(-1)
        repeat = [1 for i in range(z.ndim)]
        repeat.append(target_len // z_len)
        z = z.unsqueeze(-1).repeat(*repeat)
        z = z.flatten(-2,-1)
        z_len = z.size(-1)
        if z_len >= target_len:
            z = z[...,:target_len]
        else:
            diff_len = target_len - z_len
            z = F.pad(z, (0, diff_len), 'replicate')
        return z


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
        if False:
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
            # Down-sampling or not
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

        self.encode = nn.Sequential(*layers)

        # add final layer
        self.z_proj = nn.Conv1d( out_channels[-1], z_channels, 1)

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
        input = self.encode(input)
        return self.z_proj(input), input

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
            c (Tensor): Input tensor (B, cond_channels, T).
        Returns:
            Tensor: Output tensor (B, out_channels, T).
        """
        # return self.decode(x)
        x, c = input
        x_out = 0.0
        for layer in self.layers:
            if isinstance(layer, DeConv1d_Layernorm_GLU_ResSkip):
                x, x_skip = layer( x, c)
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
