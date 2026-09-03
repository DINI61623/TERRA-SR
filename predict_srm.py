#!/usr/bin/env python3
"""
Sentinel-2 Super Resolution Mapping (SRM) Inference Script
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

Upscales preprocessed Sentinel-2 10m bands to 5m resolution.
Supports both neural network inference (ESPCN) and standard geospatial upscaling (Bilinear) as a baseline.
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Optional geospatial libraries
try:
    import rasterio
    from rasterio.enums import Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# Try importing PyTorch
try:
    import torch
    from src.super_resolution.model import ESPCN
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


PROCESSED_DATA_PATH = Path("data/processed/s2_10m_stacked_roi.npy")
OUTPUT_DIR = Path("outputs")


def load_input_data():
    """
    Loads preprocessed stacked Sentinel-2 ROI bands.
    """
    if not PROCESSED_DATA_PATH.exists():
        print(f"Error: Preprocessed stacked image not found at {PROCESSED_DATA_PATH}")
        print("Please run 'python train_srm.py' first to stack and preprocess the bands.")
        sys.exit(1)
        
    data = np.load(PROCESSED_DATA_PATH)
    print(f"Loaded input image stack. Shape: {data.shape} (Channels, Height, Width)")
    return data


def run_espcn_inference(input_data, weights_path, upscale_factor=2):
    """
    Runs ESPCN neural network upscaling in PyTorch.
    """
    print(f"\n--- Running ESPCN Neural Network Upscaling (x{upscale_factor}) ---")
    
    # 4 channels input (RGB + NIR)
    model = ESPCN(in_channels=4, upscale_factor=upscale_factor)
    print(f"Loading weights from {weights_path}...")
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Convert input to PyTorch tensor and add batch dimension (1, C, H, W)
    input_tensor = torch.tensor(input_data).unsqueeze(0)
    
    with torch.no_grad():
        output_tensor = model(input_tensor)
        
    # Remove batch dimension and convert back to numpy
    output_data = output_tensor.squeeze(0).numpy()
    print("Inference completed successfully.")
    return output_data


def run_bilinear_baseline(input_data, upscale_factor=2):
    """
    Runs standard Bilinear Interpolation as a baseline upscaler.
    """
    print(f"\n--- Running Bilinear Interpolation Upscaling Baseline (x{upscale_factor}) ---")
    channels, h, w = input_data.shape
    new_h, new_w = h * upscale_factor, w * upscale_factor
    
    # Try importing scipy
    try:
        from scipy.interpolate import RegularGridInterpolator
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False
        
    output_data = np.zeros((channels, new_h, new_w), dtype=np.float32)
    
    if HAS_SCIPY:
        y_coords = np.linspace(0, h - 1, new_h)
        x_coords = np.linspace(0, w - 1, new_w)
        for c in range(channels):
            x = np.arange(w)
            y = np.arange(h)
            fn = RegularGridInterpolator((y, x), input_data[c], bounds_error=False, fill_value=None)
            pts = np.array(np.meshgrid(y_coords, x_coords, indexing='ij')).transpose(1, 2, 0)
            output_data[c] = fn(pts)
    else:
        print("[Notice] 'scipy' is not installed. Performing pure-NumPy bilinear interpolation...")
        y_indices = np.linspace(0, h - 1, new_h)
        x_indices = np.linspace(0, w - 1, new_w)
        
        y_low = np.floor(y_indices).astype(int)
        y_high = np.ceil(y_indices).astype(int)
        x_low = np.floor(x_indices).astype(int)
        x_high = np.ceil(x_indices).astype(int)
        
        y_diff = y_indices - y_low
        x_diff = x_indices - x_low
        
        for c in range(channels):
            for i in range(new_h):
                yl = y_low[i]
                yh = y_high[i]
                yd = y_diff[i]
                for j in range(new_w):
                    xl = x_low[j]
                    xh = x_high[j]
                    xd = x_diff[j]
                    
                    p00 = input_data[c, yl, xl]
                    p10 = input_data[c, yh, xl]
                    p01 = input_data[c, yl, xh]
                    p11 = input_data[c, yh, xh]
                    
                    val = (p00 * (1 - yd) * (1 - xd) +
                           p10 * yd * (1 - xd) +
                           p01 * (1 - yd) * xd +
                           p11 * yd * xd)
                    output_data[c, i, j] = val
                    
    print("Bilinear baseline interpolation completed successfully.")
    return output_data


def save_highres_output(output_data, upscale_factor=2, filename="s2_5m_upscaled.tiff"):
    """
    Saves high-resolution multi-spectral results to outputs/.
    """
    output_path = OUTPUT_DIR / filename
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    if HAS_RASTERIO:
        # Recreate profile profile mapping
        profile = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': None,
            'width': output_data.shape[2],
            'height': output_data.shape[1],
            'count': output_data.shape[0],
            # Note: in real inference we would read the input GeoTIFF geotransform and scale it:
            # new_transform = input_transform * Affine.scale(1 / upscale_factor)
            'crs': 'EPSG:32643',
            'transform': rasterio.transform.from_origin(700000, 1500000, 10 / upscale_factor, 10 / upscale_factor)
        }
        with rasterio.open(output_path, 'w', **profile) as dst:
            for i in range(output_data.shape[0]):
                dst.write(output_data[i], i + 1)
        print(f"Saved high-resolution GeoTIFF to: {output_path}")
    else:
        # Fallback to saving numpy file
        npy_path = output_path.with_suffix('.npy')
        np.save(npy_path, output_data)
        print(f"Saved high-resolution array to: {npy_path} (NumPy fallback).")
        
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Sentinel-2 Super Resolution Mapping Inference pipeline."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/espcn_srm_sentinel2.pth",
        help="Path to trained PyTorch weights file (default: models/espcn_srm_sentinel2.pth)"
    )
    parser.add_argument(
        "--upscale-factor",
        type=int,
        default=2,
        help="Upscaling factor (default: 2)"
    )
    
    args = parser.parse_args()
    
    # 1. Load low-resolution cropped stacked inputs
    lr_data = load_input_data()
    
    # 2. Check if PyTorch and weights are available for NN inference
    weights_path = Path(args.weights)
    use_espcn = HAS_TORCH and weights_path.exists()
    
    if use_espcn:
        # Run neural network upscaling
        hr_data = run_espcn_inference(lr_data, weights_path, args.upscale_factor)
        filename = f"s2_{10/args.upscale_factor:.1f}m_upscaled_espcn.tiff"
    else:
        if not HAS_TORCH:
            print("\n[Notice] PyTorch ('torch') is not installed.")
        elif not weights_path.exists():
            print(f"\n[Notice] Trained weights not found at {weights_path}.")
            
        print("Executing Bilinear Interpolation upscaling baseline instead.")
        hr_data = run_bilinear_baseline(lr_data, args.upscale_factor)
        filename = f"s2_{10/args.upscale_factor:.1f}m_upscaled_bilinear.tiff"
        
    # 3. Save the high-resolution output tiff
    save_highres_output(hr_data, args.upscale_factor, filename)
    
    # Print comparison dimensions
    print("\n==================================================")
    print("Upscaling Grid Comparison:")
    print("==================================================")
    print(f"Input Grid dimensions:  {lr_data.shape[2]} x {lr_data.shape[1]} (10.0m resolution)")
    print(f"Output Grid dimensions: {hr_data.shape[2]} x {hr_data.shape[1]} ({10/args.upscale_factor:.1f}m resolution)")
    print("==================================================")


if __name__ == "__main__":
    main()
