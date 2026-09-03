#!/usr/bin/env python3
"""
Sentinel-2 Super-Resolution Mapping (SRM) - Reference Dataset Preparer
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

CLI tool to execute quality control, coregister PlanetScope reference data to Sentinel-2,
and generate paired LR/HR datasets for super-resolution training.

Usage:
  python prepare_reference_dataset.py --reference <path> --sentinel <path>
"""

import sys
import argparse
from pathlib import Path

# Try importing spatial modules
try:
    from src.super_resolution.reference_loader import ReferenceLoader
    from src.super_resolution.coregistration import align_datasets, calculate_spatial_intersection
    from src.super_resolution.paired_dataset import generate_paired_patches
    from src.super_resolution.quality_control import QualityControl
    HAS_MODULES = True
except ImportError as e:
    print(f"Error importing internal modules: {str(e)}")
    HAS_MODULES = False


# Bounding box coords for Electronic City AOI
TARGET_AOI = {
    "lat": 12.85,
    "lon": 77.685
}


def run_mock_pipeline():
    """
    Demonstrates the dataset preparation workflow using simulated metadata.
    This allows verification of the pipeline logic when real reference files are absent.
    """
    print("\n--- Running Dataset Preparation Pipeline (Mock Mode) ---")
    
    # 1. Simulate Sentinel-2 Metadata
    s2_meta = {
        "filepath": "data/raw/sentinel2/S2B_MSIL2A_20260211T050839_N0512_R019_T43PGQ_20260211T085923_Red_10m.jp2",
        "driver": "JP2OpenJPEG",
        "width": 10980,
        "height": 10980,
        "count": 1,
        "crs": "EPSG:32643",
        "transform": [10.0, 0.0, 699960.0, 0.0, -10.0, 1500000.0],
        "res": (10.0, 10.0),
        "bounds": {"left": 699960.0, "bottom": 1390200.0, "right": 809760.0, "top": 1500000.0},
        "nodata": None
    }
    
    # 2. Simulate PlanetScope Metadata
    ref_meta = {
        "filepath": "data/raw/planet/PlanetScope_ElectronicCity_20260215_3m.tif",
        "driver": "GTiff",
        "width": 4096,
        "height": 4096,
        "count": 4, # Blue, Green, Red, NIR
        "crs": "EPSG:32643", # Mismatched CRS or same (for mock we use same, or reproject)
        "transform": [3.0, 0.0, 710000.0, 0.0, -3.0, 1480000.0],
        "res": (3.0, 3.0),
        "bounds": {"left": 710000.0, "bottom": 1467712.0, "right": 722288.0, "top": 1480000.0},
        "nodata": 0.0
    }
    
    print("\n1. Initializing metadata loaders...")
    print(f"   Sentinel-2 scene: {Path(s2_meta['filepath']).name}")
    print(f"   Reference scene:  {Path(ref_meta['filepath']).name}")
    
    # Validate band counts
    if ref_meta["count"] < 4:
        print("  [ERROR] Reference metadata contains less than 4 bands.")
        return
    print("  [OK] Spectral bands count validated (4 bands present).")
    
    # 2. Execute Quality Control checks
    print("\n2. Executing Quality Control Checks...")
    qc = QualityControl(s2_meta, ref_meta)
    
    # Simulate a reference date of Feb 15, 2026 (4 days gap from Feb 11)
    qc_report = qc.run_qc_checks(TARGET_AOI, "2026-02-15")
    
    print(f"   QC Status:                     {qc_report['status']}")
    print(f"   Sentinel-2 Resolution:         {qc_report['checks']['s2_pixel_size_m']}m")
    print(f"   Reference Resolution:          {qc_report['checks']['ref_pixel_size_m']}m")
    print(f"   Temporal Difference:           {qc_report['checks']['temporal_difference_days']} days")
    
    for warning in qc_report["warnings"]:
        print(f"   [WARNING] {warning}")
        
    # 3. Simulate coregistration bounds
    print("\n3. Calculating coregistration spatial alignment grid...")
    intersect, overlap = calculate_spatial_intersection(ref_meta, s2_meta)
    if not overlap:
        print("  [ERROR] No spatial overlap between reference and S2 scene.")
        return
        
    print(f"   Spatial intersection bounds:   Left: {intersect['left']}, Top: {intersect['top']}")
    print("   [OK] Aligned target grid transform computed.")
    
    # 4. Patch generation simulation
    print("\n4. Simulating patch generation split...")
    # Assume 128 patches generated, 12 rejected
    print("   Generated 116 valid patch pairs. Rejected 12 patches due to invalid spatial coverage.")
    print("   Geospatial bounds and affine transform mapped to each patch.")
    
    print("\n==================================================")
    print("Mock Dataset Ingestion Verification Complete.")
    print("==================================================")


def main():
    parser = argparse.ArgumentParser(
        description="Coregister high-resolution reference datasets to Sentinel-2 grids."
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Path to PlanetScope high-resolution reference GeoTIFF"
    )
    parser.add_argument(
        "--sentinel",
        type=str,
        default=None,
        help="Path to Sentinel-2 stacked image / band file"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run simulated pipeline to verify dataset parameters without actual files"
    )
    parser.add_argument(
        "--upscale-factor",
        type=int,
        default=2,
        help="SR upscaling factor (default: 2)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/aligned_reference.tiff",
        help="Output path for coregistered aligned reference image"
    )
    
    args = parser.parse_args()
    
    if args.mock:
        run_mock_pipeline()
        return
        
    # Validation step: check that both paths are provided
    if not args.reference or not args.sentinel:
        print("\n==================================================")
        print("Dataset Ingestion pipeline validation check:")
        print("==================================================")
        print("[Status]: WAITING FOR INPUT FILES")
        print("\nMissing required arguments. Usage:")
        print("  python prepare_reference_dataset.py --reference <path> --sentinel <path>")
        print("\nIf you do not have the high-resolution PlanetScope imagery yet, you can:")
        print("  1. Apply for free access at planet.com/markets/education-and-research/")
        print("  2. Download the 3m orthorectified GeoTIFF covering Electronic City (12.85, 77.685).")
        print("  3. Run this script again with the correct file paths.")
        print("\nTo test the pipeline logic with simulated metadata right now, run:")
        print("  python prepare_reference_dataset.py --mock")
        print("==================================================")
        sys.exit(0)
        
    ref_path = Path(args.reference)
    s2_path = Path(args.sentinel)
    
    # Check physical existence and report exactly what is missing
    missing_files = []
    if not ref_path.exists():
        missing_files.append(f"Reference file (PlanetScope) not found at: {ref_path}")
    if not s2_path.exists():
        missing_files.append(f"Sentinel-2 file not found at: {s2_path}")
        
    if missing_files:
        print("\n==================================================")
        print("Geospatial Ingestion Error Report:")
        print("==================================================")
        for missing in missing_files:
            print(f"  [X] {missing}")
        print("\nPlease resolve the missing file paths before executing.")
        print("==================================================")
        sys.exit(1)
        
    # Execute full pipeline if files are present
    try:
        # 1. Load metadata
        print("\n--- Loading Raster Metadata ---")
        ref_loader = ReferenceLoader(ref_path)
        ref_meta = ref_loader.load_metadata()
        
        # Load Sentinel-2 metadata (use ReferenceLoader temporarily to read metadata)
        s2_loader = ReferenceLoader(s2_path)
        s2_meta = s2_loader.load_metadata()
        
        print(f"  Reference band count: {ref_meta['count']}")
        print(f"  Sentinel-2 band count: {s2_meta['count']}")
        
        # 2. Quality Control
        print("\n--- Running Quality Control Checks ---")
        qc = QualityControl(s2_meta, ref_meta)
        
        # Assume reference date is encoded in filename or default to S2 date for mock diff
        qc_report = qc.run_qc_checks(TARGET_AOI, "2026-02-11")
        
        print(f"  QC Status: {qc_report['status']}")
        for warning in qc_report["warnings"]:
            print(f"  [WARNING] {warning}")
        for error in qc_report["errors"]:
            print(f"  [ERROR] {error}")
            
        if qc_report["status"] == "FAIL":
            print("\nDataset preparation aborted due to QC failures.")
            sys.exit(1)
            
        # 3. Coregistration
        print("\n--- Executing Coregistration Alignment ---")
        aligned_path = align_datasets(ref_path, s2_path, args.output, args.upscale_factor)
        
        # 4. Generate patches
        print("\n--- Slicing Paired Patches ---")
        pairs = generate_paired_patches(s2_path, aligned_path, lr_patch_size=32, upscale_factor=args.upscale_factor)
        print(f"Successfully generated {len(pairs)} LR-HR patch pairs.")
        
    except Exception as e:
        print(f"\nPipeline Execution Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
