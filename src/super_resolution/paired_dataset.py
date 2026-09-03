#!/usr/bin/env python3
"""
Sentinel-2 Super-Resolution Mapping (SRM) - Paired Dataset Generator
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

Extracts paired geospatial LR/HR patches from coregistered Sentinel-2 (LR)
and PlanetScope (HR) datasets, validating data coverage and spatial alignment.
"""

import sys
import numpy as np
from pathlib import Path

try:
    import rasterio
    from affine import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def generate_paired_patches(s2_path, aligned_ref_path, lr_patch_size=32, upscale_factor=2, max_invalid_ratio=0.1):
    """
    Slices the coregistered Sentinel-2 (LR) and reference (HR) rasters into paired patches.
    
    Args:
        s2_path (Path): Path to Sentinel-2 stacked image.
        aligned_ref_path (Path): Path to aligned reference image.
        lr_patch_size (int): Size of the LR patches in pixels (default: 32).
        upscale_factor (int): Scaling factor (default: 2).
        max_invalid_ratio (float): Maximum allowed ratio of NoData/zero pixels in a patch (default: 0.1).
        
    Returns:
        list of dict: List containing paired patch tensors, geotransforms, and bounds.
    """
    if not HAS_RASTERIO:
        raise ImportError("The 'rasterio' library is required to generate paired patches.")
        
    s2_path = Path(s2_path)
    aligned_ref_path = Path(aligned_ref_path)
    
    hr_patch_size = lr_patch_size * upscale_factor
    
    with rasterio.open(s2_path) as s2_src:
        with rasterio.open(aligned_ref_path) as ref_src:
            
            # Read full arrays
            # Shapes: (4, H_lr, W_lr) and (4, H_hr, W_hr)
            s2_data = s2_src.read([1, 2, 3, 4])
            ref_data = ref_src.read([1, 2, 3, 4])
            
            s2_transform = s2_src.transform
            ref_transform = ref_src.transform
            
            _, lr_h, lr_w = s2_data.shape
            _, hr_h, hr_w = ref_data.shape
            
            # Verify grid scaling match
            if hr_h != lr_h * upscale_factor or hr_w != lr_w * upscale_factor:
                raise ValueError(
                    f"Aligned reference dimensions ({hr_w}x{hr_h}) do not match the expected "
                    f"upscaled Sentinel-2 dimensions ({lr_w * upscale_factor}x{lr_h * upscale_factor})."
                )
                
            pairs = []
            rejected_count = 0
            
            # Slide window over LR space
            for y in range(0, lr_h - lr_patch_size + 1, lr_patch_size):
                for x in range(0, lr_w - lr_patch_size + 1, lr_patch_size):
                    
                    # 1. Extract LR patch
                    lr_patch = s2_data[:, y : y + lr_patch_size, x : x + lr_patch_size]
                    
                    # 2. Extract HR patch
                    hy, hx = y * upscale_factor, x * upscale_factor
                    hr_patch = ref_data[:, hy : hy + hr_patch_size, hx : hx + hr_patch_size]
                    
                    # 3. Calculate geospatial bounds of the patch
                    # top-left coordinate of the LR patch
                    patch_left, patch_top = s2_transform * (x, y)
                    patch_right, patch_bottom = s2_transform * (x + lr_patch_size, y + lr_patch_size)
                    
                    # Create Affine transform for the LR patch
                    lr_patch_transform = Affine(s2_transform.a, 0.0, patch_left,
                                                0.0, s2_transform.e, patch_top)
                                                
                    # Create Affine transform for the HR patch
                    hr_patch_transform = Affine(ref_transform.a, 0.0, patch_left,
                                                0.0, ref_transform.e, patch_top)
                    
                    # 4. Quality check: Reject patches with too many NoData (0.0) values
                    # Digital number 0 is the typical NoData fill value
                    total_pixels_lr = lr_patch.size
                    invalid_pixels_lr = np.sum(lr_patch == 0.0)
                    invalid_ratio_lr = invalid_pixels_lr / total_pixels_lr
                    
                    total_pixels_hr = hr_patch.size
                    invalid_pixels_hr = np.sum(hr_patch == 0.0)
                    invalid_ratio_hr = invalid_pixels_hr / total_pixels_hr
                    
                    if invalid_ratio_lr > max_invalid_ratio or invalid_ratio_hr > max_invalid_ratio:
                        rejected_count += 1
                        continue
                        
                    pairs.append({
                        "lr_patch": lr_patch,
                        "hr_patch": hr_patch,
                        "geospatial": {
                            "lr_transform": lr_patch_transform,
                            "hr_transform": hr_patch_transform,
                            "bounds": {
                                "left": patch_left,
                                "bottom": patch_bottom,
                                "right": patch_right,
                                "top": patch_top
                            }
                        }
                    })
                    
            print(f"Generated {len(pairs)} valid patch pairs. Rejected {rejected_count} due to invalid coverage.")
            return pairs
