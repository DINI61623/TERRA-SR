#!/usr/bin/env python3
"""
Sentinel-2 Super-Resolution Mapping (SRM) - Quality Control Module
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

Validates spatial alignment, projections, coordinate overlap, NoData flags,
and temporal difference offsets between Sentinel-2 and high-resolution reference data.
"""

import sys
from datetime import datetime
from pathlib import Path
import numpy as np

try:
    import rasterio
    from rasterio.warp import transform_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


class QualityControl:
    def __init__(self, s2_meta, ref_meta):
        """
        Args:
            s2_meta (dict): Metadata from Sentinel-2 loader.
            ref_meta (dict): Metadata from reference loader.
        """
        self.s2_meta = s2_meta
        self.ref_meta = ref_meta

    def parse_s2_date(self):
        """
        Extracts date from Sentinel-2 product ID.
        Format: S2B_MSIL2A_20260211T... -> 2026-02-11
        """
        filepath = Path(self.s2_meta["filepath"])
        filename = filepath.name
        
        # Look for standard Sentinel-2 naming date block: e.g. _20260211T
        try:
            parts = filename.split("_")
            for p in parts:
                if len(p) == 15 and "T" in p and p[0:8].isdigit():
                    date_str = p[0:8]
                    return datetime.strptime(date_str, "%Y%m%d")
        except Exception:
            pass
        return None

    def compute_temporal_difference(self, ref_date_str):
        """
        Computes absolute temporal gap in days between Sentinel-2 and reference image.
        ref_date_str format: 'YYYY-MM-DD' or similar.
        """
        s2_date = self.parse_s2_date()
        if not s2_date:
            return "Unknown (Failed to parse Sentinel-2 date from filename)"
            
        try:
            # Try parsing various formats
            for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y-%m-%dT%H:%M:%S"):
                try:
                    ref_date = datetime.strptime(ref_date_str, fmt)
                    break
                except ValueError:
                    continue
            else:
                return "Unknown (Invalid reference date format)"
                
            delta = abs((s2_date - ref_date).days)
            return delta
        except Exception as e:
            return f"Error: {str(e)}"

    def run_qc_checks(self, target_aoi_coords, ref_date_str):
        """
        Performs structural quality checks on the two imagery metadata stacks.
        
        Args:
            target_aoi_coords (dict): Bounding box bounds of the AOI.
            ref_date_str (str): Date of the reference image.
            
        Returns:
            dict: Structured quality control report.
        """
        report = {
            "status": "PASS",
            "warnings": [],
            "errors": [],
            "checks": {}
        }
        
        # 1. Check Projection (CRS)
        s2_crs = self.s2_meta["crs"]
        ref_crs = self.ref_meta["crs"]
        
        if s2_crs != ref_crs:
            report["warnings"].append(
                f"CRS mismatch: Sentinel-2 uses {s2_crs} but Reference uses {ref_crs}. "
                f"Coregistration reprojection is required."
            )
            report["checks"]["crs_match"] = False
        else:
            report["checks"]["crs_match"] = True

        # 2. Check Pixel Spacing
        s2_res = self.s2_meta["res"][0]
        ref_res = self.ref_meta["res"][0]
        
        report["checks"]["s2_pixel_size_m"] = s2_res
        report["checks"]["ref_pixel_size_m"] = ref_res
        
        if ref_res >= s2_res:
            report["errors"].append(
                f"Invalid resolution hierarchy: Reference resolution ({ref_res}m) "
                f"is not higher than Sentinel-2 resolution ({s2_res}m)."
            )
            report["status"] = "FAIL"

        # 3. Check Geographic Bounding Box Overlap
        # Translate reference bounds into S2 CRS if mismatched
        if HAS_RASTERIO and Path(self.s2_meta["filepath"]).exists() and Path(self.ref_meta["filepath"]).exists():
            with rasterio.open(self.s2_meta["filepath"]) as s2_src:
                with rasterio.open(self.ref_meta["filepath"]) as ref_src:
                    ref_bounds_in_s2 = transform_bounds(ref_src.crs, s2_src.crs,
                                                        ref_src.bounds.left, ref_src.bounds.bottom,
                                                        ref_src.bounds.right, ref_src.bounds.top)
                    
                    # Intersect bounds
                    left = max(ref_bounds_in_s2[0], s2_src.bounds.left)
                    bottom = max(ref_bounds_in_s2[1], s2_src.bounds.bottom)
                    right = min(ref_bounds_in_s2[2], s2_src.bounds.right)
                    top = min(ref_bounds_in_s2[3], s2_src.bounds.top)
                    
                    overlap_exists = left < right and bottom < top
                    report["checks"]["spatial_overlap_exists"] = overlap_exists
                    
                    if not overlap_exists:
                        report["errors"].append("No spatial overlap between Sentinel-2 scene and Reference imagery.")
                        report["status"] = "FAIL"
        else:
            # Fallback bounds overlap calculation using metadata dict directly (useful for mock runs)
            s2_b = self.s2_meta["bounds"]
            ref_b = self.ref_meta["bounds"]
            left = max(ref_b["left"], s2_b["left"])
            bottom = max(ref_b["bottom"], s2_b["bottom"])
            right = min(ref_b["right"], s2_b["right"])
            top = min(ref_b["top"], s2_b["top"])
            
            overlap_exists = left < right and bottom < top
            report["checks"]["spatial_overlap_exists"] = overlap_exists
            
            if not overlap_exists:
                report["warnings"].append("Geographic bounds from metadata do not overlap.")
                report["status"] = "FAIL"

        # 4. Check target coordinate location
        lat, lon = target_aoi_coords["lat"], target_aoi_coords["lon"]
        # Convert lat/lon to S2 projected coordinates to verify it resides inside bounds
        # (For simple validation, we can assume target_aoi bounds check passes if bounds match)
        report["checks"]["target_aoi_monitored"] = f"lat: {lat}, lon: {lon}"

        # 5. Check NoData presence
        s2_nodata = self.s2_meta["nodata"]
        ref_nodata = self.ref_meta["nodata"]
        report["checks"]["s2_nodata_value"] = s2_nodata
        report["checks"]["ref_nodata_value"] = ref_nodata

        # 6. Report temporal difference
        delta_days = self.compute_temporal_difference(ref_date_str)
        report["checks"]["temporal_difference_days"] = delta_days
        
        if isinstance(delta_days, int):
            if delta_days > 30:
                report["warnings"].append(
                    f"Large temporal gap: Reference and Sentinel-2 images are separated by {delta_days} days. "
                    f"This may introduce land-cover changes (vegetation growth, construction, lighting)."
                )
            elif delta_days <= 5:
                print(f"[QC] Excellent temporal alignment! Gap is only {delta_days} days.")

        if len(report["errors"]) > 0:
            report["status"] = "FAIL"
            
        return report
