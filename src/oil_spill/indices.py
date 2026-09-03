#!/usr/bin/env python3
"""
Downstream Satellite Oil-Spill Intelligence Module - Spectral Indices Engine
For SIH 2026 - Problem Statement SIH26142

Computes physical remote sensing descriptors from enhanced 4-band multispectral data:
1. NDWI (Normalized Difference Water Index) - Water / Land segmentation
2. SOSI (Standardized Oil Spill Index) - Petroleum emulsion contrast
3. FAI (Floating Algae Index) - Biogenic slick vs petroleum discrimination
4. FI (Fluorescence Index) - Chlorophyll fluorescence rejection
5. Texture (Local Spatial Gradient Contrast) - Capillary wave damping analysis
"""

import numpy as np
import torch
import torch.nn.functional as F


def compute_ndwi(green, nir, eps=1e-7):
    """
    NDWI = (Green - NIR) / (Green + NIR)
    Positive for water bodies (> 0), negative for land and dense vegetation.
    """
    return (green - nir) / (green + nir + eps)


def compute_sosi(red, nir, eps=1e-7):
    """
    SOSI (Standardized Oil Spill Index) = (NIR - Red) / (NIR + Red)
    Clean seawater absorbs heavily in NIR (values near -0.8 to -0.9).
    Thick oil emulsions scatter in Red/NIR, shifting SOSI to positive / higher values.
    """
    return (nir - red) / (nir + red + eps)


def compute_fai(green, red, nir):
    """
    FAI (Floating Algae Index approximation for 4-band MSI):
    Measures baseline curvature across visible and NIR wavelengths.
    Algae blooms exhibit strong red-edge peak; mineral oils display monotonic slopes.
    """
    # Baseline interpolation between Green and NIR
    baseline = green + (nir - green) * (665.0 - 560.0) / (842.0 - 560.0)
    return nir - baseline


def compute_fluorescence_index(green, red, nir, eps=1e-7):
    """
    FI (Fluorescence Index):
    FI = (Red - Green) / (Red + Green)
    Useful for distinguishing suspended sediment and algae from thin oil sheens.
    """
    return (red - green) / (red + green + eps)


def compute_local_texture(gray_band, window_size=5):
    """
    Computes local standard deviation across a spatial window.
    Oil slicks damp high-frequency capillary waves, reducing spatial texture.
    """
    if isinstance(gray_band, np.ndarray):
        t = torch.tensor(gray_band, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    else:
        t = gray_band.unsqueeze(0).unsqueeze(0) if gray_band.ndim == 2 else gray_band
        
    pad = window_size // 2
    kernel = torch.ones(1, 1, window_size, window_size, dtype=torch.float32) / (window_size ** 2)
    mean = F.conv2d(t, kernel, padding=pad)
    mean_sq = F.conv2d(t ** 2, kernel, padding=pad)
    var = torch.clamp(mean_sq - mean ** 2, min=1e-7)
    std = torch.sqrt(var).squeeze().numpy() if isinstance(gray_band, np.ndarray) else torch.sqrt(var).squeeze()
    return std


def extract_multispectral_feature_cube(refl_cube):
    """
    Builds an 8-channel feature cube from a 4-band reflectance cube:
    Channels:
    0: Blue (B02)
    1: Green (B03)
    2: Red (B04)
    3: NIR (B08)
    4: NDWI (Water Index)
    5: SOSI (Oil Spill Index)
    6: FAI (Floating Algae Rejection)
    7: Texture (Wave Damping Contrast)
    
    Args:
        refl_cube (np.ndarray or torch.Tensor): Shape (4, H, W) in [0.0, 1.0]
    
    Returns:
        np.ndarray: Shape (8, H, W) feature cube
    """
    is_tensor = isinstance(refl_cube, torch.Tensor)
    if is_tensor:
        data = refl_cube.detach().cpu().numpy()
    else:
        data = np.array(refl_cube, dtype=np.float32)
        
    blue = data[0]
    green = data[1]
    red = data[2]
    nir = data[3]
    
    ndwi = compute_ndwi(green, nir)
    sosi = compute_sosi(red, nir)
    fai = compute_fai(green, red, nir)
    texture = compute_local_texture(nir, window_size=5)
    
    feature_cube = np.stack([
        blue,
        green,
        red,
        nir,
        ndwi,
        sosi,
        fai,
        texture
    ], axis=0)
    
    return feature_cube
