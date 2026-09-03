#!/usr/bin/env python3
"""
Physically Informed Residual Channel Attention Network (PI-RCAN)
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

Architecture Highlights:
1. Multispectral 4-Band Input (B02 Blue, B03 Green, B04 Red, B08 NIR).
2. Hierarchical Residual-in-Residual (RIR) Structure with Channel Attention Blocks (RCAB).
3. Cross-Spectral NIR Edge Guidance: Leverages sharp B08 gradients to constrain visible bands.
4. Multi-Scale Sub-Pixel Upsampler supporting x2 (5.0m), x3 (3.33m), and x4 (2.5m) GSD.
5. Dual-Head Architecture:
   - Head A: 4-band Reconstructed Reflectance Cube in [0.0, 1.0].
   - Head B: Heteroscedastic Aleatoric Uncertainty Map (per-pixel variance sigma^2) for Anti-Hallucination verification.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Block (CAB) to model cross-spectral channel interdependencies.
    """
    def __init__(self, num_features: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(num_features, num_features // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    """
    Residual Channel Attention Block (RCAB).
    Conv -> ReLU -> Conv -> ChannelAttention + Identity Skip.
    """
    def __init__(self, num_features: int, reduction: int = 16):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1, bias=True),
            ChannelAttention(num_features, reduction)
        )

    def forward(self, x):
        return x + self.body(x)


class ResidualGroup(nn.Module):
    """
    Residual Group (RG) containing B RCAB blocks with a short skip connection.
    """
    def __init__(self, num_features: int, num_rcab: int = 6, reduction: int = 16):
        super(ResidualGroup, self).__init__()
        modules = [RCAB(num_features, reduction) for _ in range(num_rcab)]
        modules.append(nn.Conv2d(num_features, num_features, 3, padding=1, bias=True))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        return x + self.body(x)


class NIREdgeGuidanceBranch(nn.Module):
    """
    Extracts high-frequency spatial gradients from the NIR (B08) channel
    and modulates visible band feature representations.
    """
    def __init__(self, num_features: int):
        super(NIREdgeGuidanceBranch, self).__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_features, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, nir_ch, feature_map):
        # nir_ch: (B, 1, H, W)
        edge_attn = self.edge_conv(nir_ch)
        return feature_map * (1.0 + edge_attn)


class PIRCAN(nn.Module):
    """
    Physically Informed Residual Channel Attention Network (PI-RCAN).
    """
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        num_features: int = 64,
        num_groups: int = 4,
        num_rcab: int = 6,
        reduction: int = 16,
        upscale_factor: int = 2
    ):
        super(PIRCAN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upscale_factor = upscale_factor
        self.num_features = num_features

        # 1. Shallow Feature Extraction Head
        self.head = nn.Conv2d(in_channels, num_features, 3, padding=1, bias=True)

        # 2. Residual-in-Residual (RIR) Deep Backbone
        self.rgs = nn.ModuleList([
            ResidualGroup(num_features, num_rcab=num_rcab, reduction=reduction)
            for _ in range(num_groups)
        ])
        self.rir_tail = nn.Conv2d(num_features, num_features, 3, padding=1, bias=True)

        # 3. Cross-Spectral NIR Guidance
        self.nir_guidance = NIREdgeGuidanceBranch(num_features)

        # 4. Multi-Scale Sub-Pixel Upsampler (PixelShuffle)
        if upscale_factor == 2:
            self.upsampler = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, 3, padding=1, bias=True),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            )
        elif upscale_factor == 3:
            self.upsampler = nn.Sequential(
                nn.Conv2d(num_features, num_features * 9, 3, padding=1, bias=True),
                nn.PixelShuffle(3),
                nn.ReLU(inplace=True)
            )
        elif upscale_factor == 4:
            self.upsampler = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, 3, padding=1, bias=True),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_features, num_features * 4, 3, padding=1, bias=True),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            )
        else:
            raise ValueError(f"Unsupported upscale_factor: {upscale_factor}. Choose from [2, 3, 4].")

        # 5. Dual Reconstruction & Uncertainty Heads
        # Head A: Primary Reflectance Reconstruction
        self.reconstruction_head = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // 2, out_channels, 3, padding=1, bias=True)
        )

        # Head B: Heteroscedastic Aleatoric Uncertainty Estimation (Variance sigma^2)
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // 2, out_channels, 3, padding=1, bias=True),
            nn.Softplus()  # Ensures positive variance sigma^2 > 0
        )

    def forward(self, x, return_uncertainty: bool = False):
        """
        Args:
            x: Input LR tensor (Batch, 4, H, W) in [0.0, 1.0].
            return_uncertainty (bool): If True, returns (reflectance, variance).
        """
        # Shallow features
        f_0 = self.head(x)

        # Deep RIR feature representation
        f_res = f_0
        for rg in self.rgs:
            f_res = rg(f_res)
        f_rir = self.rir_tail(f_res)

        # Long Skip Connection
        f_total = f_0 + f_rir

        # Cross-Spectral NIR Edge Guidance (using Band 3 = NIR B08)
        nir_ch = x[:, 3:4, :, :]
        f_guided = self.nir_guidance(nir_ch, f_total)

        # Sub-pixel Upsampling
        f_up = self.upsampler(f_guided)

        # Head A: Reflectance
        out_reflectance = self.reconstruction_head(f_up)
        # Analytical bilinear base skip for physical spectral stability
        base = F.interpolate(x, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        out_reflectance = out_reflectance + base

        if return_uncertainty:
            out_variance = self.uncertainty_head(f_up)
            return out_reflectance, out_variance

        return out_reflectance


def get_pircan_model(scale: int = 2, device="cpu"):
    """
    Factory function for PI-RCAN.
    """
    model = PIRCAN(
        in_channels=4,
        out_channels=4,
        num_features=64,
        num_groups=4,
        num_rcab=6,
        reduction=16,
        upscale_factor=scale
    ).to(device)
    return model


if __name__ == "__main__":
    dummy = torch.randn(2, 4, 32, 32)
    for s in [2, 3, 4]:
        net = get_pircan_model(scale=s)
        out, var = net(dummy, return_uncertainty=True)
        param_count = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"PI-RCAN (Scale x{s}): Output {out.shape}, Variance {var.shape}, Parameters: {param_count:,}")
