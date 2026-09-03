#!/usr/bin/env python3
"""
Sentinel-2 Super-Resolution Mapping (SRM) - Dataset Readiness Auditor
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

Audits spatial alignment, projections, band correspondence, and temporal offsets
between Sentinel-2 and high-resolution reference data (PlanetScope).
Outputs the final 'outputs/experiment4_dataset_readiness.json' state.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add src imports
try:
    from src.super_resolution.reference_loader import ReferenceLoader
    from src.super_resolution.quality_control import QualityControl
    from src.super_resolution.coregistration import calculate_spatial_intersection
    HAS_MODULES = True
except ImportError:
    HAS_MODULES = False


TARGET_AOI = {
    "lat": 12.85,
    "lon": 77.685,
    "name": "Bengaluru/Electronic City"
}

EXPECTED_CONTRACT = {
    "sensor": "PlanetScope (Dove-PS / SuperDove) or Airbus SPOT 6/7",
    "resolution": "Approximately 3m (PlanetScope) or 1.5m (SPOT)",
    "channels": 4,
    "bands_sequence": [
        "Band 1: Blue (mapping to S2 B02)",
        "Band 2: Green (mapping to S2 B03)",
        "Band 3: Red (mapping to S2 B04)",
        "Band 4: NIR (mapping to S2 B08)"
    ],
    "format": "GeoTIFF (.tif / .tiff)",
    "crs": "EPSG:32643 (UTM Zone 43N) matching Sentinel-2, or must be reprojected to it",
    "target_aoi": TARGET_AOI,
    "target_date": "11 February 2026 (or within a maximum 10-day temporal window of S2 pass)"
}


def write_waiting_status(output_path, reason):
    """
    Writes the WAITING status JSON.
    """
    data = {
        "status": "WAITING_FOR_REAL_HIGH_RES_REFERENCE",
        "ready_for_training": "NO",
        "reason": reason,
        "expected_data_contract": EXPECTED_CONTRACT
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"[Auditor] Written dataset readiness status to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Audit dataset readiness for Experiment 4.")
    parser.add_argument("--reference", type=str, default=None, help="Path to real PlanetScope GeoTIFF")
    parser.add_argument("--sentinel", type=str, default=None, help="Path to Sentinel-2 stacked image")
    parser.add_argument("--output", type=str, default="outputs/experiment4_dataset_readiness.json", help="Path to write JSON status")
    parser.add_argument("--ref-date", type=str, default="2026-02-15", help="Acquisition date of reference image (YYYY-MM-DD)")
    args = parser.parse_args()

    output_path = Path(args.output)

    # If parameters are not provided or files do not exist physically
    if not args.reference or not args.sentinel:
        reason = "Missing input arguments. Reference or Sentinel-2 file paths were not specified."
        write_waiting_status(output_path, reason)
        sys.exit(0)

    ref_file = Path(args.reference)
    s2_file = Path(args.sentinel)

    if not ref_file.exists() or not s2_file.exists():
        reason = f"Files do not exist on disk. Check paths: Reference exists={ref_file.exists()}, Sentinel exists={s2_file.exists()}"
        write_waiting_status(output_path, reason)
        sys.exit(0)

    if not HAS_MODULES:
        print("[Error] Failed to load internal geospatial python modules.")
        sys.exit(1)

    print("[Auditor] Running physical dataset readiness checks...")
    
    try:
        # Load metadata
        ref_loader = ReferenceLoader(ref_file)
        ref_meta = ref_loader.load_metadata()
        
        s2_loader = ReferenceLoader(s2_file)
        s2_meta = s2_loader.load_metadata()
        
        # Quality Control Audit
        qc = QualityControl(s2_meta, ref_meta)
        qc_report = qc.run_qc_checks(TARGET_AOI, args.ref_date)
        
        # Calculate intersection area
        intersect, overlap = calculate_spatial_intersection(ref_meta, s2_meta)
        
        # Slicing patch estimation:
        # Each LR patch is 32x32 pixels (320m x 320m). 
        # Calculate intersection width/height in S2 pixels.
        if overlap and intersect:
            import rasterio
            with rasterio.open(s2_file) as s2_src:
                s2_dx, s2_dy = s2_src.transform.a, s2_src.transform.e
                left, right = intersect["left"], intersect["right"]
                bottom, top = intersect["bottom"], intersect["top"]
                intersect_w_px = int(np.floor((right - left) / s2_dx))
                intersect_h_px = int(np.floor((top - bottom) / abs(s2_dy)))
                
                # Estimated non-overlapping 32x32 patches
                est_patches_x = intersect_w_px // 32
                est_patches_y = intersect_h_px // 32
                est_patches = est_patches_x * est_patches_y
        else:
            est_patches = 0

        # Construct final report
        ready = "YES" if (qc_report["status"] == "PASS" and overlap) else "NO"
        
        status_data = {
            "status": "PASS" if ready == "YES" else "WARN/FAIL",
            "ready_for_training": ready,
            "reason": "All checks passed successfully." if ready == "YES" else "Quality control or overlap audits generated warnings/errors.",
            "expected_data_contract": EXPECTED_CONTRACT,
            "audit_results": {
                "crs_match": qc_report["checks"].get("crs_match", False),
                "spatial_overlap_exists": qc_report["checks"].get("spatial_overlap_exists", False),
                "s2_resolution_m": qc_report["checks"].get("s2_pixel_size_m"),
                "ref_resolution_m": qc_report["checks"].get("ref_pixel_size_m"),
                "temporal_gap_days": qc_report["checks"].get("temporal_difference_days"),
                "estimated_usable_patches": est_patches,
                "warnings": qc_report["warnings"],
                "errors": qc_report["errors"]
            }
        }
        
        with open(output_path, "w") as f:
            json.dump(status_data, f, indent=4)
        print(f"[Auditor] Written dataset readiness status to: {output_path}")

    except Exception as e:
        print(f"[Error] Failed during verification pass: {str(e)}")
        write_waiting_status(output_path, f"Exception occurred during audit: {str(e)}")


if __name__ == "__main__":
    main()
