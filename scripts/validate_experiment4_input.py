#!/usr/bin/env python3
"""
validate_experiment4_input.py
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

Validates whether a candidate PlanetScope high-resolution reference GeoTIFF
and Sentinel-2 low-resolution image pair satisfy the Experiment 4 real-data contract.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

# Add workspace root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    import rasterio
    from rasterio.warp import transform_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def parse_date_from_filename(filename):
    """
    Tries to parse date from filename for Sentinel-2 or PlanetScope.
    """
    # Sentinel-2 format: e.g. S2B_MSIL2A_20260211T050839
    if "MSIL2A" in filename or "S2" in filename:
        try:
            parts = filename.split("_")
            for p in parts:
                if len(p) == 15 and "T" in p and p[0:8].isdigit():
                    return datetime.strptime(p[0:8], "%Y%m%d")
        except Exception:
            pass
            
    # PlanetScope format: e.g. 20260211_054815_64_254a
    # Check if first 8 characters are digits
    basename = Path(filename).name
    if len(basename) >= 8 and basename[0:8].isdigit():
        try:
            return datetime.strptime(basename[0:8], "%Y%m%d")
        except Exception:
            pass
            
    return None


def run_validation(ref_path_str, s2_path_str):
    results = {
        "status": "FAIL",
        "errors": [],
        "warnings": [],
        "checks": {}
    }
    
    # 1. File existence
    ref_path = Path(ref_path_str) if ref_path_str else None
    s2_path = Path(s2_path_str) if s2_path_str else None
    
    results["checks"]["reference_file_exists"] = ref_path is not None and ref_path.exists()
    results["checks"]["sentinel_file_exists"] = s2_path is not None and s2_path.exists()
    
    if not results["checks"]["reference_file_exists"]:
        results["errors"].append(f"Reference file does not exist on disk: {ref_path_str}")
    if not results["checks"]["sentinel_file_exists"]:
        results["errors"].append(f"Sentinel-2 file does not exist on disk: {s2_path_str}")
        
    if len(results["errors"]) > 0:
        results["status"] = "FAIL"
        return results
        
    # 2. Raster readability
    if not HAS_RASTERIO:
        results["errors"].append("Rasterio library not installed. Cannot inspect rasters.")
        results["status"] = "FAIL"
        return results
        
    try:
        ref_src = rasterio.open(ref_path)
        results["checks"]["reference_readable"] = True
    except Exception as e:
        results["checks"]["reference_readable"] = False
        results["errors"].append(f"Failed to open reference raster: {str(e)}")
        
    # Check Sentinel-2 readability. If it's a single band, we also check sibling bands.
    s2_bands_paths = {}
    is_s2_stacked = False
    try:
        s2_src = rasterio.open(s2_path)
        if s2_src.count >= 4:
            is_s2_stacked = True
            s2_bands_paths["stacked"] = s2_path
        else:
            # Single band passed (e.g. Red), try to find sibling bands
            filename = s2_path.name
            parent = s2_path.parent
            # Sibling bands mapping
            # Replace band name in filename
            for band in ["Blue", "Green", "Red", "NIR"]:
                if band in filename:
                    sib_name = filename.replace(band, band) # self check
                    # We will dynamically replace whatever band is present
                    s2_bands_paths[band] = s2_path # temporary
            
            # Let's find other bands by replacing the current band name
            current_band = None
            for b in ["Blue", "Green", "Red", "NIR"]:
                if f"_{b}_" in filename:
                    current_band = b
                    break
            
            if current_band:
                for b in ["Blue", "Green", "Red", "NIR"]:
                    sib_name = filename.replace(f"_{current_band}_", f"_{b}_")
                    s2_bands_paths[b] = parent / sib_name
            else:
                s2_bands_paths["single"] = s2_path
                
        results["checks"]["sentinel_readable"] = True
    except Exception as e:
        results["checks"]["sentinel_readable"] = False
        results["errors"].append(f"Failed to open Sentinel-2 raster: {str(e)}")
        
    if len(results["errors"]) > 0:
        results["status"] = "FAIL"
        return results

    # Close temporary readers
    ref_src.close()
    s2_src.close()
    
    # Verify all sibling bands exist if single band was passed
    if not is_s2_stacked and len(s2_bands_paths) > 1:
        for b, p in s2_bands_paths.items():
            if not p.exists():
                results["errors"].append(f"Missing required sibling Sentinel-2 band '{b}' at {p}")
                results["status"] = "FAIL"
                return results

    # Re-open rasters to run deep checks
    with rasterio.open(ref_path) as ref:
        # Determine S2 metadata source (stacked or first band)
        s2_main_path = s2_bands_paths["stacked"] if is_s2_stacked else s2_bands_paths.get("Red", s2_path)
        with rasterio.open(s2_main_path) as s2:
            
            # 3. CRS Alignment Check
            ref_crs = ref.crs
            s2_crs = s2.crs
            results["checks"]["reference_crs"] = ref_crs.to_string() if ref_crs else None
            results["checks"]["sentinel_crs"] = s2_crs.to_string() if s2_crs else None
            
            if ref_crs != s2_crs:
                results["warnings"].append(
                    f"CRS Mismatch: Reference is in {ref_crs.to_string() if ref_crs else 'None'} but Sentinel-2 is in "
                    f"{s2_crs.to_string() if s2_crs else 'None'}. Warp reprojection will be required."
                )
                crs_aligned = False
            else:
                crs_aligned = True
                
            results["checks"]["crs_aligned"] = crs_aligned

            # 4. Affine transform checks
            ref_transform = ref.transform
            s2_transform = s2.transform
            results["checks"]["reference_transform"] = list(ref_transform)
            results["checks"]["sentinel_transform"] = list(s2_transform)
            
            # 5. Dimensions
            results["checks"]["reference_dimensions"] = (ref.width, ref.height)
            results["checks"]["sentinel_dimensions"] = (s2.width, s2.height)
            
            # 6. Pixel resolution
            ref_res_x, ref_res_y = ref.res
            s2_res_x, s2_res_y = s2.res
            results["checks"]["reference_pixel_resolution"] = (ref_res_x, ref_res_y)
            results["checks"]["sentinel_pixel_resolution"] = (s2_res_x, s2_res_y)

            # 14. Resolution ratio check
            res_ratio = s2_res_x / ref_res_x
            results["checks"]["resolution_ratio"] = res_ratio
            if ref_res_x >= s2_res_x:
                results["errors"].append(
                    f"Invalid resolution hierarchy: Reference resolution ({ref_res_x}m) "
                    f"is not higher resolution than Sentinel-2 ({s2_res_x}m)."
                )
            elif not np.isclose(res_ratio, 2.0) and not np.isclose(res_ratio, 3.33) and not np.isclose(res_ratio, 4.0):
                results["warnings"].append(
                    f"Non-integer or atypical resolution ratio: Sentinel-2/Reference = {res_ratio:.2f}. "
                    f"Grid snapping and resampling are required."
                )

            # 7. Spatial overlap & 13. Geographic intersection
            # Project reference bounds into S2 CRS to test intersection
            try:
                ref_bounds_in_s2 = transform_bounds(
                    ref.crs, s2.crs,
                    ref.bounds.left, ref.bounds.bottom,
                    ref.bounds.right, ref.bounds.top
                )
                
                left = max(ref_bounds_in_s2[0], s2.bounds.left)
                bottom = max(ref_bounds_in_s2[1], s2.bounds.bottom)
                right = min(ref_bounds_in_s2[2], s2.bounds.right)
                top = min(ref_bounds_in_s2[3], s2.bounds.top)
                
                overlap_exists = left < right and bottom < top
                results["checks"]["spatial_overlap_exists"] = overlap_exists
                results["checks"]["intersection_bounds_in_s2"] = {
                    "left": left, "bottom": bottom, "right": right, "top": top
                } if overlap_exists else None
                
                if not overlap_exists:
                    results["errors"].append("No spatial overlap between Sentinel-2 and Reference raster.")
            except Exception as e:
                results["checks"]["spatial_overlap_exists"] = False
                results["errors"].append(f"Failed to calculate spatial overlap bounds: {str(e)}")

            # 8. Temporal metadata check
            ref_date = parse_date_from_filename(ref_path.name)
            s2_date = parse_date_from_filename(s2_path.name)
            
            results["checks"]["reference_date"] = ref_date.strftime("%Y-%m-%d") if ref_date else "Unknown"
            results["checks"]["sentinel_date"] = s2_date.strftime("%Y-%m-%d") if s2_date else "Unknown"
            
            if ref_date and s2_date:
                temporal_gap = abs((ref_date - s2_date).days)
                results["checks"]["temporal_gap_days"] = temporal_gap
                if temporal_gap > 30:
                    results["errors"].append(
                        f"Temporal gap too large: Sentinel-2 and Reference are separated by {temporal_gap} days (max allowed: 30)."
                    )
                elif temporal_gap > 5:
                    results["warnings"].append(
                        f"Temporal mismatch warning: Gap is {temporal_gap} days. Seasonal land cover changes may affect training."
                    )
            else:
                results["checks"]["temporal_gap_days"] = None
                results["warnings"].append("Could not parse acquisition dates from filenames to calculate temporal gap.")

            # 9. Band count
            ref_bands = ref.count
            results["checks"]["reference_band_count"] = ref_bands
            if ref_bands < 4:
                results["errors"].append(f"Reference GeoTIFF must contain at least 4 bands (RGB + NIR), found: {ref_bands}")

            # 10. Band correspondence & 11. NoData ratio & 12. Reflectance value ranges
            # Read a small spatial window from both rasters to check ranges and nodata
            # If they overlap, read a 128x128 sample block near the center of the overlap
            if overlap_exists:
                try:
                    # Convert overlap geographic center to pixel coordinate in both rasters
                    center_x = (left + right) / 2
                    center_y = (bottom + top) / 2
                    
                    # Read window from reference
                    ref_row, ref_col = ref.index(center_x, center_y)
                    ref_win = rasterio.windows.Window(ref_col - 64, ref_row - 64, 128, 128)
                    ref_sample = ref.read(list(range(1, min(ref_bands, 4) + 1)), window=ref_win, boundless=True, fill_value=0)
                    
                    # Read window from Sentinel-2
                    s2_row, s2_col = s2.index(center_x, center_y)
                    s2_win = rasterio.windows.Window(s2_col - 32, s2_row - 32, 64, 64)
                    
                    if is_s2_stacked:
                        s2_sample = s2.read([1, 2, 3, 4], window=s2_win, boundless=True, fill_value=0)
                    else:
                        # Read 1 band and mock the rest, or try to read from sibling bands
                        s2_sample_bands = []
                        for b in ["Blue", "Green", "Red", "NIR"]:
                            if b in s2_bands_paths:
                                with rasterio.open(s2_bands_paths[b]) as sib:
                                    s2_sample_bands.append(sib.read(1, window=s2_win, boundless=True, fill_value=0))
                            else:
                                s2_sample_bands.append(s2.read(1, window=s2_win, boundless=True, fill_value=0))
                        s2_sample = np.stack(s2_sample_bands, axis=0)

                    # Compute NoData fractions
                    ref_nodata_ratio = np.sum(ref_sample == 0.0) / ref_sample.size
                    s2_nodata_ratio = np.sum(s2_sample == 0.0) / s2_sample.size
                    
                    results["checks"]["reference_nodata_ratio_sample"] = ref_nodata_ratio
                    results["checks"]["sentinel_nodata_ratio_sample"] = s2_nodata_ratio
                    
                    if ref_nodata_ratio > 0.1:
                        results["warnings"].append(f"High NoData ratio in reference sample: {ref_nodata_ratio:.2%}")
                    if s2_nodata_ratio > 0.1:
                        results["warnings"].append(f"High NoData ratio in Sentinel-2 sample: {s2_nodata_ratio:.2%}")

                    # Check reflectance ranges
                    # Expected range: normalized reflectance 0.0 - 1.0 (or raw integer up to 10000)
                    ref_min, ref_max = ref_sample.min(), ref_sample.max()
                    s2_min, s2_max = s2_sample.min(), s2_sample.max()
                    
                    results["checks"]["reference_value_range"] = (float(ref_min), float(ref_max))
                    results["checks"]["sentinel_value_range"] = (float(s2_min), float(s2_max))
                    
                    # If maximum values are > 1.5, check if they are in raw integer scale (e.g. 0 - 10000)
                    if ref_max > 1.5:
                        if ref_max > 10000 or ref_min < 0:
                            results["errors"].append(f"Corrupted or unexpected value range in reference: [{ref_min}, {ref_max}]")
                        else:
                            results["warnings"].append(
                                f"Reference is in raw integer scale: [{ref_min}, {ref_max}]. Dividing by 10000.0 is required."
                            )
                    else:
                        if ref_min < -0.05 or ref_max > 1.2:
                            results["warnings"].append(
                                f"Atypical normalized reflectance range in reference: [{ref_min}, {ref_max}]. Clipping will be applied."
                            )
                            
                    if s2_max > 1.5:
                        if s2_max > 10000 or s2_min < 0:
                            results["errors"].append(f"Corrupted or unexpected value range in Sentinel-2: [{s2_min}, {s2_max}]")
                        else:
                            results["warnings"].append(
                                f"Sentinel-2 is in raw integer scale: [{s2_min}, {s2_max}]. Dividing by 10000.0 is required."
                            )
                except Exception as e:
                    results["errors"].append(f"Error reading sample pixel windows: {str(e)}")
            else:
                results["checks"]["reference_nodata_ratio_sample"] = None
                results["checks"]["sentinel_nodata_ratio_sample"] = None
                results["checks"]["reference_value_range"] = None
                results["checks"]["sentinel_value_range"] = None

    if len(results["errors"]) == 0:
        results["status"] = "READY_FOR_REAL_TRAINING"
    else:
        results["status"] = "NOT_READY_FOR_REAL_TRAINING"
        
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate Sentinel-2 and PlanetScope input files against Experiment 4 contract."
    )
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Path to the high-resolution PlanetScope reference scene (GeoTIFF)"
    )
    parser.add_argument(
        "--sentinel",
        type=str,
        required=True,
        help="Path to the low-resolution Sentinel-2 stacked file or a Red band jp2"
    )
    
    args = parser.parse_args()
    
    print(f"Validating real-data pair:")
    print(f"  - Reference: {args.reference}")
    print(f"  - Sentinel:  {args.sentinel}\n")
    
    report = run_validation(args.reference, args.sentinel)
    
    print("==================================================")
    print(f"Validation Result: {report['status']}")
    print("==================================================")
    
    if len(report["errors"]) > 0:
        print("\nErrors:")
        for err in report["errors"]:
            print(f"  [FAIL] {err}")
            
    if len(report["warnings"]) > 0:
        print("\nWarnings:")
        for warn in report["warnings"]:
            print(f"  [WARN] {warn}")
            
    print("\nDetailed Checks:")
    for k, v in report["checks"].items():
        print(f"  - {k}: {v}")
        
    print("==================================================")
    
    # Save validation status log
    readiness_path = Path("outputs/experiment4_real_training_readiness.json")
    readiness_path.parent.mkdir(exist_ok=True)
    
    # Write a copy to outputs
    # If the validation failed because reference doesn't exist, we must still write it
    # and mark the overall readiness as WAITING_FOR_REAL_PLANETSCOPE_DATA if the reference file is missing.
    ref_missing = not report["checks"].get("reference_file_exists", False)
    
    log_report = {
        "validation_timestamp": datetime.now().isoformat(),
        "reference_file": args.reference,
        "sentinel_file": args.sentinel,
        "checks": report["checks"],
        "errors": report["errors"],
        "warnings": report["warnings"],
        "status": "WAITING_FOR_REAL_PLANETSCOPE_DATA" if ref_missing else report["status"]
    }
    
    with open(readiness_path, "w") as jf:
        json.dump(log_report, jf, indent=4)
    print(f"Saved training readiness log to: {readiness_path}")


if __name__ == "__main__":
    main()
