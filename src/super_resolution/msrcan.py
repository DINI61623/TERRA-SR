#!/usr/bin/env python3
"""
Multispectral Residual Channel Attention Network (MS-RCAN)
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

Implements:
1. ChannelAttention (Squeeze-and-Excitation covariance reduction)
2. RCAB (Residual Channel Attention Block)
3. ResidualGroup (Hierarchical RIR module)
4. MSRCAN (Multispectral Deep Residual Channel Attention Network with global residual skip)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Block (CAB)
    Learns cross-spectral interdependencies and dynamically reweights feature channels.
    """
    def __init__(self, num_features, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(num_features, max(num_features // reduction, 8), kernel_size=1, padding=0, bias=True),
            nn.PReLU(),
            nn.Conv2d(max(num_features // reduction, 8), num_features, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    """
    Residual Channel Attention Block (RCAB)
    Combines 3x3 convolutions, PReLU activation, Channel Attention, and identity shortcut.
    """
    def __init__(self, num_features, reduction=8):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=True),
            ChannelAttention(num_features, reduction=reduction)
        )

    def forward(self, x):
        return x + self.body(x)


class ResidualGroup(nn.Module):
    """
    Residual Group (RG) containing multiple RCAB blocks and a short skip connection.
    """
    def __init__(self, num_features, num_rcab=3, reduction=8):
        super(ResidualGroup, self).__init__()
        modules = [RCAB(num_features, reduction=reduction) for _ in range(num_rcab)]
        modules.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=True))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        return x + self.body(x)


class MSRCAN(nn.Module):
    """
    Multispectral Residual Channel Attention Network (MS-RCAN)
    
    Features:
    - 4-Band multispectral input (B02, B03, B04, B08)
    - Deep Residual in Residual (RIR) feature extraction backbone
    - Channel attention mechanism for inter-band spectral dependency modeling
    - Long and Short skip connections
    - Global residual learning: Output = Bilinear(Input) + SubpixelResidual
    """
    def __init__(self, in_channels=4, num_features=48, num_groups=4, num_rcab=3, reduction=8, upscale_factor=2):
        super(MSRCAN, self).__init__()
        self.in_channels = in_channels
        self.upscale_factor = upscale_factor
        
        # Head: Shallow Feature Extraction
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True),
            nn.PReLU()
        )
        
        # Body: Hierarchical Residual in Residual (RIR)
        self.rgs = nn.ModuleList([
            ResidualGroup(num_features, num_rcab=num_rcab, reduction=reduction) for _ in range(num_groups)
        ])
        self.conv_after_rir = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=True)
        
        # Tail: Sub-Pixel Convolution Upsampler + Reconstruction
        self.tail = nn.Sequential(
            nn.Conv2d(num_features, in_channels * (upscale_factor ** 2), kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, x):
        # Base bilinear interpolation for global residual connection
        base = F.interpolate(x, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        
        # Deep feature extraction
        h = self.head(x)
        res = h
        for rg in self.rgs:
            res = rg(res)
        res = self.conv_after_rir(res)
        res = res + h  # Long skip connection
        
        # High-frequency sub-pixel residual
        out_res = self.tail(res)
        
        # Final output combining base radiometry and learned high-frequency residual
        return base + out_res
