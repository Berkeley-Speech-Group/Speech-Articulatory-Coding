# -*- coding: utf-8 -*-

"""HiFi-GAN Modules.

This code is based on https://github.com/jik876/hifi-gan.

"""

import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F

from .block import HiFiGANResidualBlock as ResidualBlock
from .block import HiFiGANResidualFiLMBlock as ResidualFiLMBlock
from transformers import Wav2Vec2Model, HubertModel, WavLMModel
import torchaudio


        
class SoftClamp(torch.nn.Module):
    def __init__(self, temp=0.2):
        super().__init__()
        self.temp=0.2
        self.tanh =torch.nn.Tanh()
    
    def forward(self, x):
        return self.tanh(x*self.temp)/self.temp

class HiFiGANGenerator(torch.nn.Module):
    """HiFiGAN generator module."""

    def __init__(
        self,
        in_channels=14,
        out_channels=1,
        channels=512,
        kernel_size=7,
        input_layer_n=1,
        upsample_scales=(8, 8, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4),
        paddings=None,
        output_paddings=None,
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        use_additional_convs=True,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
        use_tanh=True,
        use_spk=True,
        spk_emb_size=64,
        pitch_offset=50,
        pitch_rescale=0.01,
        pitch_axis=12,
        **kwargs,
    ):
        """Initialize HiFiGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channels (int): Number of hidden representation channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            upsample_scales (list): List of upsampling scales.
            upsample_kernel_sizes (list): List of kernel sizes for upsampling layers.
            resblock_kernel_sizes (list): List of kernel sizes for residual blocks.
            resblock_dilations (list): List of dilation list for residual blocks.
            use_additional_convs (bool): Whether to use additional conv layers in residual blocks.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()

        self.use_spk = use_spk
        # check hyperparameters are valid
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        assert len(upsample_scales) == len(upsample_kernel_sizes),f"{upsample_scales} | {upsample_kernel_sizes}"
        assert len(resblock_dilations) == len(resblock_kernel_sizes)
        
        if paddings is None:
            paddings=[]
            for i in range(len(upsample_scales)):
                if upsample_scales[i]==1:
                    paddings.append(upsample_kernel_sizes[i]//2)
                else:
                    paddings.append(upsample_scales[i] // 2 + upsample_scales[i] % 2)
        else:
            new_paddings = []
            for i, s in enumerate(paddings):
                if s == "default":
                    new_paddings.append(upsample_scales[i] // 2 + upsample_scales[i] % 2)
                else:
                    print("not implemented")
                    exit()
            paddings = new_paddings
        if output_paddings is None:
            output_paddings = []
            for i, s in enumerate(upsample_scales):
                if s ==1:
                    output_paddings.append(0)
                else:
                    output_paddings.append(upsample_scales[i] % 2)
        else:
            new_output_paddings = []
            for i, s in enumerate(output_paddings):
                if s == "default":
                    new_output_paddings.append(upsample_scales[i] % 2)
                else:
                    print("not implemented")
                    exit()
            output_paddings = new_output_paddings
            
        # define modules
        self.num_upsamples = len(upsample_kernel_sizes)
        self.num_blocks = len(resblock_kernel_sizes)
        if input_layer_n==1:
            self.input_conv = torch.nn.Conv1d(
                in_channels,
                channels,
                kernel_size,
                1,
                padding=(kernel_size - 1) // 2,
            )
        else:
            self.input_conv = [torch.nn.Conv1d(
                                    in_channels,
                                    channels,
                                    kernel_size,
                                    1,
                                    padding=(kernel_size - 1) // 2,
                                )]
            for _ in range(1,input_layer_n):
                self.input_conv.append(torch.nn.Conv1d(
                                    channels,
                                    channels,
                                    kernel_size,
                                    1,
                                    padding=(kernel_size - 1) // 2,
                                ))
            self.input_conv = torch.nn.Sequential(*self.input_conv)
        self.upsamples = torch.nn.ModuleList()
        self.blocks = torch.nn.ModuleList()
        for i in range(len(upsample_kernel_sizes)):
            # assert upsample_kernel_sizes[i] == 2 * upsample_scales[i]
            self.upsamples += [
                torch.nn.Sequential(
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                    torch.nn.ConvTranspose1d(
                        channels // (2 ** i),
                        channels // (2 ** (i + 1)),
                        upsample_kernel_sizes[i],
                        upsample_scales[i],
                        padding=paddings[i],
                        output_padding=output_paddings[i],
                    ),
                )
            ]
            for j in range(len(resblock_kernel_sizes)):
                if not self.use_spk:
                    self.blocks += [
                        ResidualBlock(
                            kernel_size=resblock_kernel_sizes[j],
                            channels=channels // (2 ** (i + 1)),
                            dilations=resblock_dilations[j],
                            bias=bias,
                            use_additional_convs=use_additional_convs,
                            nonlinear_activation=nonlinear_activation,
                            nonlinear_activation_params=nonlinear_activation_params,
                        )
                    ]
                else:
                    self.blocks += [
                        ResidualFiLMBlock(
                            kernel_size=resblock_kernel_sizes[j],
                            channels=channels // (2 ** (i + 1)),
                            dilations=resblock_dilations[j],
                            bias=bias,
                            use_additional_convs=use_additional_convs,
                            nonlinear_activation=nonlinear_activation,
                            nonlinear_activation_params=nonlinear_activation_params,
                            spk_emb_size=spk_emb_size
                        )
                    ]
                
        if use_tanh:
            self.output_conv = torch.nn.Sequential(
                # NOTE(kan-bayashi): follow official implementation but why
                #   using different slope parameter here? (0.1 vs. 0.01)
                torch.nn.LeakyReLU(),
                torch.nn.Conv1d(
                    channels // (2 ** (i + 1)),
                    out_channels,
                    kernel_size,
                    1,
                    padding=(kernel_size - 1) // 2,
                ),
                torch.nn.Tanh(),
            )
        else:
            self.output_conv = torch.nn.Sequential(
                # NOTE(kan-bayashi): follow official implementation but why
                #   using different slope parameter here? (0.1 vs. 0.01)
                torch.nn.LeakyReLU(),
                torch.nn.Conv1d(
                    channels // (2 ** (i + 1)),
                    out_channels,
                    kernel_size,
                    1,
                    padding=(kernel_size - 1) // 2,
                ),
            )
    
        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

        self.pitch_offset=pitch_offset
        self.pitch_rescale=pitch_rescale
        self.pitch_axis=pitch_axis

    def forward(self, c, spk_emb=None, **kwargs):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, in_channels, T).

        Returns:
            Tensor: Output tensor (B, out_channels, T).

        """
        c[:,self.pitch_axis] = (c[:,self.pitch_axis]-self.pitch_offset)*self.pitch_rescale
        c = self.input_conv(c)
        # print('after input_conv', c.shape)
        for i in range(self.num_upsamples):
            c = self.upsamples[i](c)
            # print('after upsample %d' % i, c.shape)
            cs = 0.0  # initialize
            for j in range(self.num_blocks):
                if self.use_spk:
                    cs_ = self.blocks[i * self.num_blocks + j](c, spk_emb)
                else:
                    cs_ = self.blocks[i * self.num_blocks + j](c)
                cs += cs_
            # print('cs', cs.shape)
            c = cs / self.num_blocks  # (batch_size, some_channels, length)
        out = self.output_conv(c)  # (batch_size, 1, input_len*final_scale)
        
        return out

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows the official implementation manner.
        https://github.com/jik876/hifi-gan/blob/master/models.py

        """

        def _reset_parameters(m):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

