#!/usr/bin/env python3
"""
Sentinel-2 Dataset Inspector and Visualizer
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

This script inspects metadata, validates georeferencing, crops center windows,
and saves true-color and false-color previews to outputs/.
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Import rasterio for GIS image handling
try:
    import rasterio
    from rasterio.windows import Window
except ImportError:
    print("Error: 'rasterio' is not installed in this environment.")
    print("Please run: pip install rasterio")
    sys.exit(1)


DATA_DIR = Path("data/raw/sentinel2")
OUTPUT_DIR = Path("outputs")
PRODUCT_ID = "S2B_MSIL2A_20260211T050839_N0512_R019_T43PGQ_20260211T085923"


def scale_band(band_data, max_val=4000):
    """
    Stretch and scale 16-bit satellite values (0-10000 DN) to 8-bit PNG values (0-255).
    Clips bright highlights at max_val to enhance local image contrast.
    """
    clipped = np.clip(band_data, 0, max_val)
    scaled = (clipped / max_val * 255.0).astype(np.uint8)
    return scaled


def main():
    # Make sure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Locate the downloaded bands
    bands_paths = {
        "Blue": DATA_DIR / f"{PRODUCT_ID}_Blue_10m.jp2",
        "Green": DATA_DIR / f"{PRODUCT_ID}_Green_10m.jp2",
        "Red": DATA_DIR / f"{PRODUCT_ID}_Red_10m.jp2",
        "NIR": DATA_DIR / f"{PRODUCT_ID}_NIR_10m.jp2"
    }

    # Verify that all files exist
    for name, path in bands_paths.items():
        if not path.exists():
            print(f"Error: Missing band file: {path}")
            print("Please ensure your download step ran successfully.")
            sys.exit(1)

    print("==================================================")
    print("Sentinel-2 Band Inspection and Metadata Verification")
    print("==================================================")

    band_metadata = {}
    
    # 2. Open each band and extract metadata
    for name, path in bands_paths.items():
        with rasterio.open(path) as src:
            # Read metadata
            width = src.width
            height = src.height
            crs = src.crs.to_string() if src.crs else "None"
            transform = src.transform
            res = src.res # Pixel resolution (meters per pixel)
            dtype = src.dtypes[0]
            nodata = src.nodata
            
            # Read full band values briefly to compute statistics
            data_full = src.read(1)
            p_min = np.min(data_full)
            p_max = np.max(data_full)
            p_mean = np.mean(data_full)
            
            band_metadata[name] = {
                "width": width,
                "height": height,
                "crs": crs,
                "transform": transform,
                "res": res,
                "dtype": dtype,
                "nodata": nodata
            }
            
            print(f"\nBand: {name} ({path.name})")
            print(f"  - Dimensions:  {width} x {height}")
            print(f"  - CRS:         {crs}")
            print(f"  - Transform:   {transform}")
            print(f"  - Resolution:  {res[0]}m x {res[1]}m")
            print(f"  - Data Type:   {dtype}")
            print(f"  - NoData:      {nodata}")
            print(f"  - Min DN:      {p_min}")
            print(f"  - Max DN:      {p_max}")
            print(f"  - Mean DN:     {p_mean:.2f}")

    # 3. Verify matching dimensions and georeferencing
    print("\n--------------------------------------------------")
    print("Verifying Georeferencing Alignment:")
    print("--------------------------------------------------")
    
    reference = band_metadata["Red"]
    aligned = True
    for name, meta in band_metadata.items():
        if meta["width"] != reference["width"] or meta["height"] != reference["height"]:
            print(f"  [X] Spatial dimension mismatch on Band {name}!")
            aligned = False
        if meta["crs"] != reference["crs"]:
            print(f"  [X] CRS coordinate system mismatch on Band {name}!")
            aligned = False
        if meta["transform"] != reference["transform"]:
            print(f"  [X] Affine transform georeferencing mismatch on Band {name}!")
            aligned = False
            
    if aligned:
        print("  [OK] All 10m bands are perfectly aligned spatially and georeferenced.")
    else:
        print("  [X] Warning: Some bands are misaligned. Verify source inputs.")

    # 4. Read only a small 512x512 window from the center of the image
    print("\n--------------------------------------------------")
    print("Extracting 512x512 Center Window...")
    print("--------------------------------------------------")
    
    # Calculate center pixel coordinates
    ref_w, ref_h = reference["width"], reference["height"]
    cx, cy = ref_w // 2, ref_h // 2
    
    # Create the reading window
    window = Window(cx - 256, cy - 256, 512, 512)
    print(f"Reading window: x_start={cx-256}, y_start={cy-256}, width=512, height=512")
    
    window_data = {}
    for name, path in bands_paths.items():
        with rasterio.open(path) as src:
            window_data[name] = src.read(1, window=window)

    # 5. Create True Color PNG (B04=B, B03=G, B02=R)
    # Scaling to 8-bit values for presentation
    r_8bit = scale_band(window_data["Blue"])   # B02
    g_8bit = scale_band(window_data["Green"])  # B03
    b_8bit = scale_band(window_data["Red"])    # B04
    
    # Stack into 3-channel RGB image
    true_color_img = np.stack([r_8bit, g_8bit, b_8bit], axis=-1)
    
    # Save True Color PNG
    tc_pil = Image.fromarray(true_color_img)
    tc_path = OUTPUT_DIR / "true_color_preview.png"
    tc_pil.save(tc_path)
    print(f"  [OK] True-Color Preview saved to: {tc_path}")

    # 7. Create NIR/Red false-color composite
    # Standard CIR: R=B08 (NIR), G=B04 (Red), B=B03 (Green)
    nir_8bit = scale_band(window_data["NIR"])    # B08
    red_8bit = scale_band(window_data["Red"])    # B04
    green_8bit = scale_band(window_data["Green"]) # B03
    
    false_color_img = np.stack([nir_8bit, red_8bit, green_8bit], axis=-1)
    
    # Save False Color PNG
    fc_pil = Image.fromarray(false_color_img)
    fc_path = OUTPUT_DIR / "false_color_preview.png"
    fc_pil.save(fc_path)
    print(f"  [OK] False-Color Preview saved to: {fc_path}")
    print("\nBand inspection process complete!")


if __name__ == "__main__":
    main()
