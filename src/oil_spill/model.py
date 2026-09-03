#!/usr/bin/env python3
"""
Downstream Satellite Oil-Spill Intelligence Module - Semantic Segmentation Network
For SIH 2026 - Problem Statement SIH26142

U-Net Semantic Segmentation Network for Marine Oil Slick Detection:
- Ingests 8-channel multispectral feature tensor:
  [B02, B03, B04, B08, NDWI, SOSI, FAI, Texture]
- Multi-scale skip connections preserve narrow sub-pixel slick ribbons.
- Predicts 3 classes:
  0: Clean Background Ocean Water
  1: Thin Oil Sheen (< 0.1 mm)
  2: Thick Emulsified Mousse (> 1.0 mm)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class OilSpillUNet(nn.Module):
    def __init__(self, in_channels=8, num_classes=3, base_features=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_features)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ConvBlock(base_features, base_features * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ConvBlock(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(base_features * 4, base_features * 8)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_features * 8, base_features * 4)
        
        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_features * 4, base_features * 2)
        
        self.up1 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_features * 2, base_features)
        
        # Classification Head
        self.head = nn.Conv2d(base_features, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool3(e3))
        
        # Decoder
        d3 = self.up3(b)
        # Handle odd spatial dimensions if any
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        logits = self.head(d1)
        return logits
