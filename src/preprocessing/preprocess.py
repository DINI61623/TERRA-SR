#!/usr/bin/env python3
"""
Sentinel-2 Image Preprocessing Pipeline
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

This script provides step-by-step workflows to:
1. Extract downloaded Sentinel-2 ZIP products (.SAFE formats)
2. Locate specific spectral bands (B02-Blue, B03-Green, B04-Red, B08-NIR)
3. Crop bands to a specific region of interest (latitude/longitude bounding box)
4. Normalize digital numbers (DN) to surface reflectance values (0.0 to 1.0)
5. Stack and save the processed bands as a multi-spectral GeoTIFF
"""

import os
import sys
import zipfile
import glob
import numpy as np
from pathlib import Path

# Optional geospatial libraries
try:
    import rasterio
    from rasterio.warp import transform_bounds
except ImportError:
    rasterio = None


def extract_zip(zip_path, extract_dir):
    """
    Step 1: Extract the raw downloaded Sentinel-2 zip archive.
    """
    print(f"[Step 1] Extracting {zip_path} to {extract_dir}...")
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found at {zip_path}")
        
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete.")
    
    # Locate the extracted .SAFE directory
    safe_dirs = list(extract_dir.glob("*.SAFE"))
    if not safe_dirs:
        raise FileNotFoundError("No .SAFE directory found after extracting zip.")
    return safe_dirs[0]


def locate_bands(safe_dir):
    """
    Step 2: Locate JPEG 2000 (JP2) files for 10m resolution bands (B02, B03, B04, B08).
    Sentinel-2 SAFE folders store images under:
    GRANULE/<Granule_ID>/IMG_DATA/R10m/
    """
    print(f"[Step 2] Locating spectral bands in {safe_dir}...")
    safe_dir = Path(safe_dir)
    
    # Search for bands in R10m (10-meter resolution) folder
    search_path = safe_dir / "GRANULE" / "*" / "IMG_DATA" / "R10m" / "*_B0[2348]_10m.jp2"
    band_files = glob.glob(str(search_path))
    
    # Fallback to older Sentinel-2 formats where all jp2 files are in one directory
    if not band_files:
        fallback_path = safe_dir / "GRANULE" / "*" / "IMG_DATA" / "*_B0[2348].jp2"
        band_files = glob.glob(str(fallback_path))
        
    bands = {}
    for f in band_files:
        if "B02" in f:
            bands["Blue"] = Path(f)
        elif "B03" in f:
            bands["Green"] = Path(f)
        elif "B04" in f:
            bands["Red"] = Path(f)
        elif "B08" in f:
            bands["NIR"] = Path(f)
            
    required_bands = ["Blue", "Green", "Red"]
    missing = [b for b in required_bands if b not in bands]
    if missing:
        raise FileNotFoundError(f"Missing required spectral bands: {missing}")
        
    print(f"Located bands: {list(bands.keys())}")
    return bands


def read_and_crop_band(band_path, bbox=None):
    """
    Step 3: Read band file and crop it to the given geographic bounding box.
    bbox format: [min_lon, min_lat, max_lon, max_lat]
    """
    if rasterio is None:
        print("[Warning] 'rasterio' is not installed. Loading mock/dummy band array.")
        # Return a mock array for demonstration if rasterio is missing
        return np.random.randint(0, 10000, size=(256, 256), dtype=np.uint16), None

    print(f"[Step 3] Reading and cropping band: {band_path.name}")
    with rasterio.open(band_path) as src:
        if bbox is None:
            # Read full image if no bounding box is provided
            return src.read(1), src.profile

        # Get bounds and transform bounding box coords to raster pixel coordinates
        from rasterio.windows import from_bounds
        # Sentinel-2 uses UTM coordinates (CRS). We need to project WGS84 lat/lon bounding box.
        left, bottom, right, top = bbox
        
        # Crop window based on coordinates bounds
        window = from_bounds(left, bottom, right, top, transform=src.transform)
        
        # Read the cropped window of data
        cropped_data = src.read(1, window=window)
        
        # Update raster metadata profile for the cropped output
        new_transform = rasterio.windows.transform(window, src.transform)
        profile = src.profile.copy()
        profile.update({
            'height': cropped_data.shape[0],
            'width': cropped_data.shape[1],
            'transform': new_transform
        })
        
        return cropped_data, profile


def normalize_band(data):
    """
    Step 4: Normalize Digital Numbers (DN) to surface reflectance values (0.0 to 1.0).
    Sentinel-2 DN values are typically scaled by 10,000 (QUANTIFICATION_VALUE).
    """
    print("[Step 4] Normalizing digital numbers to reflectance (0.0 - 1.0)...")
    # Cast to float32 and clip values to valid range
    reflectance = data.astype(np.float32) / 10000.0
    reflectance = np.clip(reflectance, 0.0, 1.0)
    return reflectance


def save_stacked_tiff(bands_data, profile, output_path):
    """
    Step 5: Stack bands and save as a multi-spectral GeoTIFF.
    bands_data format: list of numpy arrays [Red, Green, Blue, NIR]
    """
    print(f"[Step 5] Stacking {len(bands_data)} bands and saving to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stacked = np.stack(bands_data, axis=0) # Shape: (Bands, Height, Width)
    
    if rasterio is None:
        # Fallback to saving numpy array if rasterio is not installed
        npy_path = output_path.with_suffix('.npy')
        np.save(npy_path, stacked)
        print(f"Saved stacked array to {npy_path} (NumPy fallback).")
        return
        
    profile.update({
        'count': len(bands_data),
        'dtype': 'float32',
        'driver': 'GTiff'
    })
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        for i, band in enumerate(bands_data):
            dst.write(band, i + 1)
            
    print(f"Successfully saved multi-spectral GeoTIFF to {output_path}")


def run_mock_pipeline():
    """
    Simulates the preprocessing steps with generated arrays for testing.
    """
    print("\n--- Running Preprocessing Pipeline (Mock Mode) ---")
    
    # Generate mock 256x256 band inputs
    print("[Step 1-2] Mock: Simulating Sentinel-2 bands extraction...")
    mock_data = {
        "Red": np.random.randint(100, 8000, size=(256, 256), dtype=np.uint16),
        "Green": np.random.randint(100, 8000, size=(256, 256), dtype=np.uint16),
        "Blue": np.random.randint(100, 8000, size=(256, 256), dtype=np.uint16)
    }
    
    # Crop simulation
    print("[Step 3] Mock: Simulating cropping bounds...")
    cropped_red = mock_data["Red"][64:192, 64:192]
    cropped_green = mock_data["Green"][64:192, 64:192]
    cropped_blue = mock_data["Blue"][64:192, 64:192]
    
    # Normalize
    norm_red = normalize_band(cropped_red)
    norm_green = normalize_band(cropped_green)
    norm_blue = normalize_band(cropped_blue)
    
    # Stack and Save
    output_path = Path("data/processed/mock_s2_rgb_stacked.tiff")
    save_stacked_tiff(
        bands_data=[norm_red, norm_green, norm_blue],
        profile={},
        output_path=output_path
    )
    print("Mock preprocessing finished successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Sentinel-2 Satellite Imagery Preprocessing Pipeline."
    )
    parser.add_argument(
        "--zip-path",
        type=str,
        help="Path to the raw downloaded Sentinel-2 ZIP product"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="processed_srm_input.tiff",
        help="Output filename for stacked tiff (default: processed_srm_input.tiff)"
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=('MIN_LON', 'MIN_LAT', 'MAX_LON', 'MAX_LAT'),
        help="Bounding box coordinates to crop (e.g. 77.65 12.82 77.72 12.88)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run the pipeline in mock mode with generated data (requires no inputs)"
    )
    
    args = parser.parse_args()
    
    if args.mock or not args.zip_path:
        if not args.mock:
            print("No input zip file provided. Running in mock mode to verify script structure.")
        run_mock_pipeline()
        return

    try:
        # Run live pipeline
        raw_extract_dir = Path("data/raw/extracted")
        safe_dir = extract_zip(args.zip_path, raw_extract_dir)
        bands_paths = locate_bands(safe_dir)
        
        # Read, crop, and normalize each band
        processed_bands = []
        profile = None
        
        for name in ["Red", "Green", "Blue", "NIR"]:
            if name in bands_paths:
                band_data, band_profile = read_and_crop_band(bands_paths[name], args.bbox)
                normalized = normalize_band(band_data)
                processed_bands.append(normalized)
                
                # Keep the profile of the first cropped band to write outputs
                if profile is None and band_profile is not None:
                    profile = band_profile
                    
        # Save stacked result
        output_dir = Path("data/processed")
        output_path = output_dir / args.output_name
        save_stacked_tiff(processed_bands, profile, output_path)
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    main()
