#!/usr/bin/env python3
"""
Sentinel-2 Super-Resolution Inference Runner
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

CLI tool to run model inference using our trained ESPCN checkpoint.
Usage:
  python scripts/run_inference.py --input <input_tiff> --output <output_tiff> --checkpoint <checkpoint_pth>
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np

# Add parent directory to sys.path to resolve src imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import pipeline components
from src.super_resolution.inference import ProductionInference

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


PRODUCT_ID = "S2B_MSIL2A_20260211T050839_N0512_R019_T43PGQ_20260211T085923"


def create_test_input_tiff(output_path):
    """
    Stitches a 1024x1024 4-band test GeoTIFF from raw JP2 bands for inference tests.
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required to build the test stack.")
        
    raw_dir = Path("data/raw/sentinel2")
    bands_paths = {
        "Blue": raw_dir / f"{PRODUCT_ID}_Blue_10m.jp2",
        "Green": raw_dir / f"{PRODUCT_ID}_Green_10m.jp2",
        "Red": raw_dir / f"{PRODUCT_ID}_Red_10m.jp2",
        "NIR": raw_dir / f"{PRODUCT_ID}_NIR_10m.jp2"
    }
    
    # Check if files exist
    for name, p in bands_paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing downloaded band JP2 file for test stack: {p}")
            
    # Read a center 1024x1024 window to keep it lightweight
    cx, cy = 10980 // 2, 10980 // 2
    window = rasterio.windows.Window(cx - 512, cy - 512, 1024, 1024)
    
    print(f"[Prep] Building 1024x1024 test stacked GeoTIFF from downloaded JP2 files...")
    
    # Read metadata template from Red band
    with rasterio.open(bands_paths["Red"]) as ref_src:
        profile = ref_src.profile.copy()
        new_transform = rasterio.windows.transform(window, ref_src.transform)
        profile.update({
            'driver': 'GTiff',
            'dtype': 'float32',
            'count': 4,
            'width': 1024,
            'height': 1024,
            'transform': new_transform
        })
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, 'w', **profile) as dst:
        for i, name in enumerate(["Blue", "Green", "Red", "NIR"]):
            with rasterio.open(bands_paths[name]) as src:
                # Divide by 10000.0 to convert to normalized reflectance 0.0 - 1.0
                data = src.read(1, window=window).astype(np.float32) / 10000.0
                dst.write(np.clip(data, 0.0, 1.0), i + 1)
                
    print(f"[Prep] Test stacked GeoTIFF written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run ESPCN super-resolution inference on Sentinel-2 multi-spectral imagery."
    )
    parser.add_argument("--input", type=str, default=None, help="Path to input 4-band GeoTIFF")
    parser.add_argument("--output", type=str, default=None, help="Path to save upscaled output GeoTIFF")
    parser.add_argument("--checkpoint", type=str, default="models/espcn_srm_synthetic.pth", help="Path to model weights checkpoint")
    parser.add_argument("--upscale-factor", type=int, default=2, help="Spatial upscale factor")
    parser.add_argument("--tile-size", type=int, default=256, help="Tile size for memory-safe processing")
    parser.add_argument("--overlap", type=int, default=16, help="Border overlap to eliminate seams")
    
    args = parser.parse_args()
    
    # 1. Verification of rasterio
    if not HAS_RASTERIO:
        print("Error: 'rasterio' is not installed in the active Python environment.")
        print("Please install it: pip install rasterio")
        sys.exit(1)
        
    # 2. Check checkpoint file
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Model checkpoint file not found at: {checkpoint_path}")
        print("Please run Experiment 2 training first to save checkpoint weights.")
        sys.exit(1)
        
    # 3. Handle Auto-Test if no files are provided
    input_path = args.input
    output_path = args.output
    
    if not input_path or not output_path:
        print("\n--- Running Automatic Test with Existing Sentinel-2 ROI ---")
        input_path = Path("data/processed/s2_10m_stacked_roi.tiff")
        output_path = Path("outputs/s2_5.0m_upscaled_espcn_inference.tiff")
        
        # Build test stack if not present
        if not input_path.exists():
            try:
                create_test_input_tiff(input_path)
            except Exception as e:
                print(f"Error creating test stack: {str(e)}")
                sys.exit(1)
    else:
        input_path = Path(input_path)
        output_path = Path(output_path)
        
    if not input_path.exists():
        print(f"Error: Input file not found at: {input_path}")
        sys.exit(1)
        
    # 4. Initialize and Run Tiled Inference
    infer = ProductionInference(
        checkpoint_path=checkpoint_path, 
        upscale_factor=args.upscale_factor
    )
    
    try:
        infer.run_tiled_inference(
            input_path=input_path, 
            output_path=output_path, 
            tile_size=args.tile_size, 
            overlap=args.overlap
        )
        
        # 5. Audit and Report Outputs parameters
        print("\n==================================================")
        print("Production Inference Performance Report:")
        print("==================================================")
        print("WARNING/LABEL: Synthetic 10m-to-5m SR baseline — NOT independent <4m ground-truth validation.")
        print("--------------------------------------------------")
        
        # Open the generated output and read parameters
        with rasterio.open(output_path) as dst:
            out_w = dst.width
            out_h = dst.height
            out_res = dst.res[0]
            out_crs = dst.crs.to_string() if dst.crs else "None"
            out_count = dst.count
            tags = dst.tags()
            
        with rasterio.open(input_path) as src:
            in_w = src.width
            in_h = src.height
            in_res = src.res[0]
            
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"  - Input Dimensions:   {in_w} x {in_h} pixels")
        print(f"  - Output Dimensions:  {out_w} x {out_h} pixels (scaling factor x{args.upscale_factor})")
        print(f"  - Input Resolution:   {in_res} m")
        print(f"  - Output Resolution:  {out_res} m")
        print(f"  - CRS:                {out_crs}")
        print(f"  - Output File Size:   {file_size_mb:.2f} MB")
        print(f"  - Spectral Channels:  {out_count} bands preserved (Blue, Green, Red, NIR)")
        print(f"  - Metadata tags:      {tags}")
        print("==================================================")
        
    except Exception as e:
        print(f"Error during inference execution: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
