#!/usr/bin/env python3
"""
High-Frequency Residual Attention Network (HF-SRM) - Experiment 4E
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

Key Architectural Innovations:
1. Multi-Scale Dual-Path Feature Extraction:
   - 3x3 and 5x5 parallel receptive fields to capture both wide geographic context and sub-pixel lines.
2. Cross-Spectral Contrast Guidance (NIR-Guided Attention):
   - The NIR band (B08) and Red band (B04) possess the highest optical contrast in Earth observation.
   - Spatial gradient features from NIR/Red dynamically gate and sharpen Blue and Green channels.
3. Residual Dense Attention Blocks (RDAB):
   - Combines dense feature reuse with spatial & channel attention to prevent gradient vanishing and over-smoothing.
4. High-Frequency Reconstruction Tail:
   - PixelShuffle sub-pixel upscaling combined with direct Laplacian second-order residual refinement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleHead(nn.Module):
    """Dual-path multi-scale feature extractor."""
    def __init__(self, in_channels=4, num_features=48):
        super(MultiScaleHead, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, num_features // 2, kernel_size=3, padding=1, bias=True)
        self.conv5x5 = nn.Conv2d(in_channels, num_features // 2, kernel_size=5, padding=2, bias=True)
        self.fusion = nn.Conv2d(num_features, num_features, kernel_size=1, bias=True)
        self.act = nn.PReLU()

    def forward(self, x):
        f3 = self.conv3x3(x)
        f5 = self.conv5x5(x)
        out = self.fusion(torch.cat([f3, f5], dim=1))
        return self.act(out)


class SpatialChannelAttention(nn.Module):
    """Joint Spatial and Channel Attention Module."""
    def __init__(self, num_features, reduction=8):
        super(SpatialChannelAttention, self).__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1, bias=True)
        )
        self.sig_c = nn.Sigmoid()
        
        # Spatial Attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention branch
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        c_att = self.sig_c(avg_out + max_out)
        x_c = x * c_att
        
        # Spatial attention branch
        avg_s = torch.mean(x_c, dim=1, keepdim=True)
        max_s, _ = torch.max(x_c, dim=1, keepdim=True)
        s_att = self.spatial_conv(torch.cat([avg_s, max_s], dim=1))
        
        return x_c * s_att


class ResidualDenseAttentionBlock(nn.Module):
    """Residual Dense Block with Spatial-Channel Attention."""
    def __init__(self, num_features=48, growth_rate=24):
        super(ResidualDenseAttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, growth_rate, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(num_features + growth_rate, growth_rate, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(num_features + 2 * growth_rate, num_features, kernel_size=3, padding=1, bias=True)
        self.attention = SpatialChannelAttention(num_features)
        self.act = nn.PReLU()

    def forward(self, x):
        c1 = self.act(self.conv1(x))
        c2 = self.act(self.conv2(torch.cat([x, c1], dim=1)))
        c3 = self.conv3(torch.cat([x, c1, c2], dim=1))
        att = self.attention(c3)
        return x + att * 0.2  # Scaled residual shortcut for stable training


class HFSRM(nn.Module):
    """
    High-Frequency Multispectral Super-Resolution Network.
    
    Combines:
    - Multi-scale receptive head
    - Cascaded Residual Dense Attention blocks (RDAB)
    - Cross-spectral NIR guidance shortcut
    - Sub-pixel convolution upsampler
    - Global baseline interpolation skip connection
    """
    def __init__(self, in_channels=4, num_features=48, num_blocks=4, upscale_factor=2):
        super(HFSRM, self).__init__()
        self.in_channels = in_channels
        self.upscale_factor = upscale_factor
        
        # 1. Multi-scale Head
        self.head = MultiScaleHead(in_channels=in_channels, num_features=num_features)
        
        # 2. Body: Cascaded RDAB blocks
        self.blocks = nn.ModuleList([
            ResidualDenseAttentionBlock(num_features=num_features, growth_rate=24)
            for _ in range(num_blocks)
        ])
        self.fusion = nn.Conv2d(num_features * num_blocks, num_features, kernel_size=1, bias=True)
        self.conv_after_body = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=True)
        
        # 3. NIR/Red Contrast Guidance Branch (Bands 2 & 3: Red & NIR)
        self.nir_guidance = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(16, num_features, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )
        
        # 4. Tail: Sub-Pixel Reconstruction
        self.tail = nn.Sequential(
            nn.Conv2d(num_features, num_features * (upscale_factor ** 2), kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(upscale_factor),
            nn.PReLU(),
            nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1, bias=True)
        )
        
        # 5. Direct High-Frequency Edge Refinement Head
        self.edge_refine = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(16, in_channels, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, x):
        # Base interpolation for global radiometric anchoring
        base = F.interpolate(x, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        
        # NIR/Red spectral contrast guidance mask
        # Channels 2=Red, 3=NIR
        nir_red = x[:, 2:4, :, :]
        guidance_mask = self.nir_guidance(nir_red)
        
        # Deep feature extraction
        h = self.head(x)
        features = []
        cur = h
        for block in self.blocks:
            cur = block(cur)
            features.append(cur)
            
        # Dense multi-level feature fusion + NIR guidance modulation
        dense_feat = self.fusion(torch.cat(features, dim=1))
        body_out = self.conv_after_body(dense_feat * (1.0 + guidance_mask))
        body_out = body_out + h  # Long skip connection
        
        # Upsampling
        subpixel_res = self.tail(body_out)
        
        # Edge refinement
        refinement = self.edge_refine(subpixel_res)
        
        return base + subpixel_res + 0.1 * refinement


if __name__ == "__main__":
    model = HFSRM(in_channels=4, num_features=48, num_blocks=4, upscale_factor=2)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"HF-SRM Model initialized successfully.")
    print(f"Trainable parameters: {params:,}")
    
    x = torch.randn(2, 4, 64, 64)
    out = model(x)
    print(f"Forward test: Input {list(x.shape)} -> Output {list(out.shape)}")
