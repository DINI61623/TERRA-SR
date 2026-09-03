#!/usr/bin/env python3
"""
Sentinel-2 Super-Resolution Mapping (SRM) - Coregistration Pipeline
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

Reprojects high-resolution reference datasets (like PlanetScope) into the Sentinel-2 CRS
and aligns both grids to an exact sub-pixel intersection bounding box.
"""

import sys
import numpy as np
from pathlib import Path

try:
    import rasterio
    from rasterio.warp import transform_bounds, reproject, Resampling
    from affine import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def calculate_spatial_intersection(src_meta, dst_meta):
    """
    Checks if two raster metadata bounds overlap.
    Returns: (intersection_bounds, overlap_exists)
    """
    src_bounds = src_meta["bounds"]
    dst_bounds = dst_meta["bounds"]
    
    # Calculate overlap bounds
    left = max(src_bounds["left"], dst_bounds["left"])
    bottom = max(src_bounds["bottom"], dst_bounds["bottom"])
    right = min(src_bounds["right"], dst_bounds["right"])
    top = min(src_bounds["top"], dst_bounds["top"])
    
    overlap = left < right and bottom < top
    intersection = {
        "left": left,
        "bottom": bottom,
        "right": right,
        "top": top
    } if overlap else None
    
    return intersection, overlap


def align_datasets(ref_path, s2_path, output_aligned_ref_path, upscale_factor=2):
    """
    Spatially reprojects and aligns the high-resolution reference dataset to match
    the CRS and coordinate grid of the Sentinel-2 imagery.
    
    Creates a target grid where the pixel boundaries align perfectly with Sentinel-2.
    For upscale_factor=2, each 10m S2 pixel corresponds to exactly 2x2 5m reference pixels.
    For upscale_factor=4, each 10m S2 pixel corresponds to exactly 4x4 2.5m reference pixels.
    """
    if not HAS_RASTERIO:
        raise ImportError("The 'rasterio' and 'affine' libraries are required for coregistration.")
        
    ref_path = Path(ref_path)
    s2_path = Path(s2_path)
    output_aligned_ref_path = Path(output_aligned_ref_path)
    
    # 1. Open both source rasters
    with rasterio.open(ref_path) as ref_src:
        with rasterio.open(s2_path) as s2_src:
            
            ref_crs = ref_src.crs
            s2_crs = s2_src.crs
            
            # 2. Project reference bounds into Sentinel-2 CRS to compute intersection
            ref_bounds_in_s2_crs = transform_bounds(ref_crs, s2_crs, 
                                                    ref_src.bounds.left, ref_src.bounds.bottom,
                                                    ref_src.bounds.right, ref_src.bounds.top)
            
            # Intersection bounds in Sentinel-2 CRS
            left = max(ref_bounds_in_s2_crs[0], s2_src.bounds.left)
            bottom = max(ref_bounds_in_s2_crs[1], s2_src.bounds.bottom)
            right = min(ref_bounds_in_s2_crs[2], s2_src.bounds.right)
            top = min(ref_bounds_in_s2_crs[3], s2_src.bounds.top)
            
            # Validate spatial overlap
            if left >= right or bottom >= top:
                raise ValueError("No geographical overlap found between reference imagery and Sentinel-2 tile.")
                
            # 3. Calculate target Grid boundaries aligned to Sentinel-2 pixels
            s2_x0, s2_y0 = s2_src.transform.c, s2_src.transform.f
            s2_dx, s2_dy = s2_src.transform.a, s2_src.transform.e # dy is negative
            
            # Find the starting pixel coordinate indices in the S2 grid
            col_start = int(np.floor((left - s2_x0) / s2_dx))
            row_start = int(np.floor((top - s2_y0) / s2_dy)) # dy is negative, row index grows downward
            
            col_end = int(np.ceil((right - s2_x0) / s2_dx))
            row_end = int(np.ceil((bottom - s2_y0) / s2_dy))
            
            # Intersection dimensions in S2 pixels
            s2_width = col_end - col_start
            s2_height = row_end - row_start
            
            # Origin of aligned bounds in projected space
            aligned_left = s2_x0 + col_start * s2_dx
            aligned_top = s2_y0 + row_start * s2_dy
            
            # 4. Target HR parameters
            hr_res = 10.0 / upscale_factor
            dst_transform = Affine(hr_res, 0.0, aligned_left,
                                   0.0, -hr_res, aligned_top)
            
            dst_width = s2_width * upscale_factor
            dst_height = s2_height * upscale_factor
            
            print(f"Aligning Reference Grid:")
            print(f"  - Target CRS:      {s2_crs.to_string()}")
            print(f"  - Resolution:      {hr_res}m (upscale factor x{upscale_factor})")
            print(f"  - Grid Dimensions: {dst_width} x {dst_height} pixels")
            
            # Prepare destination profile
            profile = ref_src.profile.copy()
            profile.update({
                'crs': s2_crs,
                'transform': dst_transform,
                'width': dst_width,
                'height': dst_height,
                'count': 4, # Save Blue, Green, Red, NIR channels
                'dtype': 'float32',
                'driver': 'GTiff'
            })
            
            # Create aligned output file (does not modify raw files)
            output_aligned_ref_path.parent.mkdir(parents=True, exist_ok=True)
            
            with rasterio.open(output_aligned_ref_path, 'w', **profile) as dst:
                # Reproject each band
                for i in range(4):
                    source_band = rasterio.band(ref_src, i + 1)
                    dest_band = rasterio.band(dst, i + 1)
                    
                    # Target array allocation
                    destination_array = np.zeros((dst_height, dst_width), dtype=np.float32)
                    
                    reproject(
                        source=source_band,
                        destination=destination_array,
                        src_transform=ref_src.transform,
                        src_crs=ref_crs,
                        dst_transform=dst_transform,
                        dst_crs=s2_crs,
                        resampling=Resampling.bilinear
                    )
                    
                    dst.write(destination_array, i + 1)
                    
            print(f"Coregistered reference imagery saved to: {output_aligned_ref_path}")
            return output_aligned_ref_path
