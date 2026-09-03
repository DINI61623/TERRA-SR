#!/usr/bin/env python3
"""
Sentinel-2 Band Exploration and Pixel Inspector
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

Reads downloaded 10m bands, prints pixel statistics, and displays side-by-side
numerical data for a small center crop to help understand band structures and pixel values.
"""

import sys
from pathlib import Path
import numpy as np

# Load Pillow as a fallback image loader
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Try importing rasterio
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


DATA_DIR = Path("data/raw/sentinel2")
PRODUCT_ID = "S2B_MSIL2A_20260211T050839_N0512_R019_T43PGQ_20260211T085923"


def load_band_image(file_path):
    """
    Load a JP2 band file and return it as a numpy array, along with metadata.
    """
    if HAS_RASTERIO:
        with rasterio.open(file_path) as src:
            data = src.read(1)
            meta = {
                "driver": src.driver,
                "width": src.width,
                "height": src.height,
                "crs": src.crs.to_string() if src.crs else "None",
                "dtype": src.dtypes[0]
            }
            return data, meta
            
    elif HAS_PIL:
        # Pillow fallback
        with Image.open(file_path) as img:
            data = np.array(img)
            meta = {
                "driver": "PIL (JP2)",
                "width": img.width,
                "height": img.height,
                "crs": "Geospatial CRS unavailable (Pillow fallback)",
                "dtype": str(data.dtype)
            }
            return data, meta
            
    else:
        raise ImportError("Please install either 'rasterio' or 'pillow' to load JP2 images.")


def explore_bands():
    bands = {
        "Blue": DATA_DIR / f"{PRODUCT_ID}_Blue_10m.jp2",
        "Green": DATA_DIR / f"{PRODUCT_ID}_Green_10m.jp2",
        "Red": DATA_DIR / f"{PRODUCT_ID}_Red_10m.jp2",
        "NIR": DATA_DIR / f"{PRODUCT_ID}_NIR_10m.jp2"
    }

    # Verify files exist
    for name, path in bands.items():
        if not path.exists():
            print(f"Error: Spectral band file not found: {path}")
            print("Please ensure download_bands.py has completed successfully.")
            sys.exit(1)

    print("==================================================")
    print("Spectral Bands Exploration & Image Metadata")
    print("==================================================")
    
    loaded_bands = {}
    
    for name, path in bands.items():
        data, meta = load_band_image(path)
        loaded_bands[name] = data
        
        print(f"\nBand: {name} ({path.name})")
        print(f"  - Loader:      {meta['driver']}")
        print(f"  - Dimensions:  {meta['width']} x {meta['height']}")
        print(f"  - Data Type:   {meta['dtype']}")
        print(f"  - CRS:         {meta['crs']}")
        
        # Calculate statistics
        p_min = np.min(data)
        p_max = np.max(data)
        p_mean = np.mean(data)
        p_std = np.std(data)
        
        print(f"  - Min Value:   {p_min}")
        print(f"  - Max Value:   {p_max}")
        print(f"  - Mean Value:  {p_mean:.2f}")
        print(f"  - Std Dev:     {p_std:.2f}")

    # Inspect exact center pixels side-by-side
    print("\n==================================================")
    print("Pixel Inspection: 5x5 Center Grid Comparison")
    print("==================================================")
    print("Let's look at the raw Digital Numbers (DN) of the same 5x5 area")
    print("at the center of all four bands:")
    
    first_band_data = list(loaded_bands.values())[0]
    h, w = first_band_data.shape
    cy, cx = h // 2, w // 2
    
    print(f"\nCenter coordinate of image grid: y={cy}, x={cx}")
    
    # Slice a 5x5 patch from center
    center_slices = {}
    for name, data in loaded_bands.items():
        center_slices[name] = data[cy-2:cy+3, cx-2:cx+3]
        
    # Print side by side
    print("\n--- Raw Pixel Values (Digital Numbers, 0-10000 range) ---")
    for r in range(5):
        row_str = []
        for name in ["Blue", "Green", "Red", "NIR"]:
            val = center_slices[name][r, 2] # Print middle column of the row
            row_str.append(f"{name}: {val:4d}")
        print("  |  ".join(row_str))

    print("\n==================================================")
    print("Scientific Understanding of the Data")
    print("==================================================")
    print("1. DATA TYPE & VALUES:")
    print("   Sentinel-2 L2A images are stored as 16-bit unsigned integers (uint16).")
    print("   The values are Digital Numbers (DN) representing surface reflectance.")
    print("   reflectance = DN / 10000.0")
    print("   Example: A DN value of 2500 corresponds to 0.25 (or 25% light reflectance).")
    print("\n2. SPECTRAL REFLECTANCE:")
    print("   Notice the differences between the Red band and the NIR (Near-Infrared) band.")
    print("   In vegetated areas, chlorophyll absorbs red light strongly (low Red DN) but reflects")
    print("   Near-Infrared light very strongly (high NIR DN). This difference is the basis for")
    print("   vegetation indices like NDVI (Normalized Difference Vegetation Index):")
    print("   NDVI = (NIR - Red) / (NIR + Red)")


if __name__ == "__main__":
    explore_bands()
