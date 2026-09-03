#!/usr/bin/env python3
"""
Downstream Satellite Oil-Spill Intelligence Module - Water Gating & Physics Preprocessing
For SIH 2026 - Problem Statement SIH26142

Performs robust marine physics gating:
1. NDWI Water Masking (NDWI > 0.0)
2. Cloud Masking (Bright surface reflection rejection)
3. Shoreline Buffer (Pure PyTorch morphological erosion to avoid coastal surf zone false positives)
"""

import numpy as np
import torch
import torch.nn.functional as F


def create_water_mask(ndwi_map, ndwi_threshold=0.0):
    """
    Identifies pure water pixels where NDWI exceeds threshold.
    """
    return ndwi_map > ndwi_threshold


def create_cloud_mask(rgb_img, threshold=0.45):
    """
    Identifies thick clouds and highly reflective structures (metals, concrete).
    Reflectance values exceeding 0.45 across all visible bands are masked as non-water.
    """
    mean_vis = np.mean(rgb_img[:3], axis=0)
    return mean_vis > threshold


def apply_shoreline_buffer(water_mask, buffer_pixels=5):
    """
    Applies morphological erosion on the binary water mask using PyTorch 
    to remove coastal surf zones and intertidal mudflats.
    """
    if buffer_pixels <= 0:
        return water_mask
        
    t = torch.tensor(water_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    kernel = torch.ones(1, 1, 3, 3, dtype=torch.float32)
    
    # Binary erosion: min filter via 1.0 - max_pool2d(1.0 - x)
    for _ in range(buffer_pixels):
        inverted = 1.0 - t
        pooled = F.max_pool2d(inverted, kernel_size=3, stride=1, padding=1)
        t = 1.0 - pooled
        
    eroded_water = (t.squeeze().numpy() > 0.5)
    return eroded_water


def compute_valid_marine_aoi(refl_cube, ndwi_threshold=0.0, buffer_pixels=3):
    """
    Generates a boolean mask of valid ocean/sea water suitable for oil spill detection.
    
    Args:
        refl_cube (np.ndarray): Shape (4, H, W)
        ndwi_threshold (float): Minimum NDWI value for water
        buffer_pixels (int): Erosion iterations for coastline buffer
        
    Returns:
        np.ndarray: 2D boolean mask of shape (H, W) where True = Valid Marine Water
    """
    green = refl_cube[1]
    nir = refl_cube[3]
    ndwi = (green - nir) / (green + nir + 1e-7)
    
    water_mask = create_water_mask(ndwi, ndwi_threshold=ndwi_threshold)
    cloud_mask = create_cloud_mask(refl_cube, threshold=0.45)
    
    # Pure water excluding clouds and land
    raw_marine = water_mask & (~cloud_mask)
    
    # Apply coastal exclusion buffer
    valid_marine = apply_shoreline_buffer(raw_marine, buffer_pixels=buffer_pixels)
    
    return valid_marine
