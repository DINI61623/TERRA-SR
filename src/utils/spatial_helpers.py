#!/usr/bin/env python3
"""
Spatial and Patching Utilities for Satellite Imagery
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

This script provides utility functions to:
1. Compute STAC-compliant bounding boxes from point coordinates and radius.
2. Slice large-scale satellite images into small patches for training/inference.
3. Stitch processed patches back together to form a full-scale image.
"""

import numpy as np


def latlon_to_bbox(lat, lon, radius_deg=0.03):
    """
    Utility 1: Calculate bounding box from a center point and degree radius.
    Returns: [min_lon, min_lat, max_lon, max_lat]
    """
    min_lon = lon - radius_deg
    min_lat = lat - radius_deg
    max_lon = lon + radius_deg
    max_lat = lat + radius_deg
    return [min_lon, min_lat, max_lon, max_lat]


def slice_into_patches(image, patch_size=128, stride=128):
    """
    Utility 2: Slice a large 2D or 3D numpy image array into small patches.
    Args:
        image (numpy.ndarray): Input image array of shape (Channels, Height, Width) or (Height, Width)
        patch_size (int): Dimensions of the square patches to extract
        stride (int): Slicing window step stride (set equal to patch_size for non-overlapping patches)
    Returns:
        patches (list): List of extracted numpy patch arrays
        coordinates (list): List of tuples containing (y_start, x_start) pixel coordinates for each patch
    """
    # Standardize image shape to (Channels, Height, Width)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=0)
        
    channels, height, width = image.shape
    patches = []
    coordinates = []
    
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image[:, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            coordinates.append((y, x))
            
    print(f"Sliced image of shape {image.shape} into {len(patches)} patches of size {patch_size}x{patch_size}")
    return patches, coordinates


def stitch_patches(patches, coordinates, target_shape, patch_size=128, stride=128):
    """
    Utility 3: Stitch small processed patches back together to form a single large image.
    Handles overlapping regions using simple pixel averaging.
    Args:
        patches (list): List of numpy patch arrays of shape (Channels, patch_size, patch_size)
        coordinates (list): List of (y_start, x_start) coordinate positions for each patch
        target_shape (tuple): Output shape of the full-scale image as (Channels, Height, Width)
        patch_size (int): Dimensions of the patches
        stride (int): Sliding window step stride
    Returns:
        stitched_image (numpy.ndarray): Reassembled full-scale image array
    """
    channels, height, width = target_shape
    
    # Accumulator buffer for pixel values and counter buffer to record overlap count
    accumulator = np.zeros((channels, height, width), dtype=np.float32)
    counter = np.zeros((channels, height, width), dtype=np.float32)
    
    for patch, (y, x) in zip(patches, coordinates):
        accumulator[:, y:y+patch_size, x:x+patch_size] += patch
        counter[:, y:y+patch_size, x:x+patch_size] += 1.0
        
    # Prevent division by zero in unvisited pixels
    counter[counter == 0] = 1.0
    stitched_image = accumulator / counter
    
    print(f"Stitched {len(patches)} patches into image of shape {stitched_image.shape}")
    return stitched_image


if __name__ == "__main__":
    print("Testing Spatial and Patching Utilities...")
    
    # 1. Test coordinate to bbox utility
    bbox = latlon_to_bbox(12.85, 77.685, 0.03)
    print(f"Bounding Box calculation: {bbox}")
    
    # 2. Generate a mock large image array (shape: 3 bands, 512 height, 512 width)
    mock_large_image = np.ones((3, 512, 512), dtype=np.float32)
    # Fill with a gradient to visually represent spatial values
    for c in range(3):
        mock_large_image[c] = np.arange(512).reshape(1, 512) + np.arange(512).reshape(512, 1)
        
    # 3. Slice the large image into patches of 128x128 with stride 128 (no overlap)
    patches, coords = slice_into_patches(mock_large_image, patch_size=128, stride=128)
    
    # 4. Simulate a dummy super resolution process: multiply each patch by 2
    processed_patches = [patch * 2.0 for patch in patches]
    
    # 5. Stitch the processed patches back together
    stitched = stitch_patches(
        processed_patches, 
        coords, 
        target_shape=(3, 512, 512), 
        patch_size=128, 
        stride=128
    )
    
    # 6. Verify correct value reconstruction (value at index [0, 10, 10] should be doubled)
    original_val = mock_large_image[0, 10, 10]
    stitched_val = stitched[0, 10, 10]
    print(f"Verification: Original Val: {original_val} -> Stitched (SR Processed) Val: {stitched_val}")
    
    if np.isclose(original_val * 2.0, stitched_val):
        print("Success! Slicing and stitching utility test passed.")
    else:
        print("Error: Reconstructed values do not match.")
