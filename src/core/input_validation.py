#!/usr/bin/env python3
"""
TERRA-SR Centralized Satellite Input Validator & Product Identification Engine.
Inspects uploaded satellite rasters BEFORE inference and enforces strict geospatial,
spectral, radiometric, and format integrity contracts.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import numpy as np

try:
    import rasterio
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from affine import Affine
    HAS_AFFINE = True
except ImportError:
    HAS_AFFINE = False


class ValidationResult:
    """Structured result returned by SatelliteInputValidator."""
    def __init__(
        self,
        is_valid: bool,
        domain: str,
        checks: Dict[str, bool],
        reasons: List[str],
        metadata: Dict[str, Any]
    ):
        self.is_valid = is_valid
        self.status = "VALID" if is_valid else "INVALID"
        self.domain = domain
        self.checks = checks
        self.reasons = reasons
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "is_valid": self.is_valid,
            "domain": self.domain,
            "checks": self.checks,
            "reasons": self.reasons,
            "metadata": self.metadata,
            "summary_message": self.format_summary()
        }

    def format_summary(self) -> str:
        if self.is_valid:
            lines = [f"INPUT VALID ✅ ({self.metadata.get('product_type', 'Sentinel-2 Compatible')})"]
            lines.append(f"• Native GSD: {self.metadata.get('gsd', 10.0):.2f}m")
            lines.append(f"• Bands: {self.metadata.get('band_count', 4)} ({', '.join(self.metadata.get('band_names', []))})")
            lines.append(f"• CRS: {self.metadata.get('crs', 'EPSG:32643')}")
            lines.append(f"• Georeferencing: VALID (Affine Transformed)")
            return "\n".join(lines)
        else:
            lines = [f"INPUT REJECTED ❌ ({self.domain.upper()})"]
            for r in self.reasons:
                lines.append(f"• Reason: {r}")
            return "\n".join(lines)


class SatelliteInputValidator:
    """Centralized validator for Earth Observation raster datasets."""

    BAND_NAMES_S2_4BAND = ["B02 (Blue)", "B03 (Green)", "B04 (Red)", "B08 (NIR)"]

    @classmethod
    def inspect_raster_metadata(
        cls,
        source: Union[str, Path, np.ndarray],
        custom_affine: Optional[Any] = None,
        custom_crs: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extracts comprehensive metadata directly from raster header or array without user assumption.
        """
        meta = {
            "product_type": "Unknown Satellite Raster",
            "format": "NumPy Array",
            "acquisition_date": "2026-05-15 (Synthetic/Controlled Standard)",
            "dimensions": "0 × 0 px",
            "height": 0,
            "width": 0,
            "band_count": 0,
            "band_names": [],
            "crs": "EPSG:32643",
            "gsd": 10.0,
            "affine_transform": None,
            "bounds": {"min_x": 0.0, "min_y": 0.0, "max_x": 0.0, "max_y": 0.0},
            "nodata_value": None,
            "dtype": "float32",
            "data_range": [0.0, 1.0]
        }

        if isinstance(source, (str, Path)) and HAS_RASTERIO and Path(source).exists():
            path_obj = Path(source)
            meta["format"] = f"GeoTIFF ({path_obj.suffix})"
            try:
                with rasterio.open(path_obj) as src:
                    meta["height"] = src.height
                    meta["width"] = src.width
                    meta["dimensions"] = f"{src.width} × {src.height} px"
                    meta["band_count"] = src.count
                    meta["crs"] = str(src.crs) if src.crs else "NONE"
                    meta["affine_transform"] = list(src.transform) if src.transform else None
                    meta["nodata_value"] = src.nodata
                    meta["dtype"] = str(src.dtypes[0]) if src.dtypes else "float32"

                    if src.transform:
                        meta["gsd"] = abs(src.transform[0])
                        bounds = src.bounds
                        meta["bounds"] = {
                            "min_x": round(bounds.left, 2),
                            "min_y": round(bounds.bottom, 2),
                            "max_x": round(bounds.right, 2),
                            "max_y": round(bounds.top, 2)
                        }

                    # Heuristic product identification
                    if src.count == 4 and abs(meta["gsd"] - 10.0) <= 2.0:
                        meta["product_type"] = "Sentinel-2 L2A BOA 10m Multispectral (B02, B03, B04, B08)"
                        meta["band_names"] = cls.BAND_NAMES_S2_4BAND
                    elif src.count == 3:
                        meta["product_type"] = "Standard 3-Band RGB Optical Imagery"
                        meta["band_names"] = ["Red", "Green", "Blue"]
                    elif src.count >= 8:
                        meta["product_type"] = "Multispectral High-Band Scene (S2 12-Band / Landsat)"
                        meta["band_names"] = [f"Band_{i+1}" for i in range(src.count)]
                    else:
                        meta["product_type"] = f"Custom {src.count}-Band Raster"
                        meta["band_names"] = [f"Band_{i+1}" for i in range(src.count)]

                    # Read sample for range & date tag
                    tags = src.tags()
                    if "DATATAKE_1_DATATAKE_IDENTIFIER" in tags or "ACQUISITION_DATE" in tags:
                        meta["acquisition_date"] = tags.get("ACQUISITION_DATE", tags.get("DATATAKE_1_DATATAKE_IDENTIFIER"))
            except Exception as e:
                meta["error"] = str(e)

        elif isinstance(source, np.ndarray):
            meta["format"] = f"NumPy Tensor {source.shape}"
            meta["dtype"] = str(source.dtype)
            if source.ndim == 3:
                meta["band_count"] = source.shape[0]
                meta["height"] = source.shape[1]
                meta["width"] = source.shape[2]
            elif source.ndim == 2:
                meta["band_count"] = 1
                meta["height"] = source.shape[0]
                meta["width"] = source.shape[1]
            meta["dimensions"] = f"{meta['width']} × {meta['height']} px"

            if custom_crs:
                meta["crs"] = str(custom_crs)
            if custom_affine:
                meta["affine_transform"] = list(custom_affine)
                meta["gsd"] = abs(custom_affine[0])
                meta["bounds"] = {
                    "min_x": round(custom_affine[2], 2),
                    "max_x": round(custom_affine[2] + meta["width"] * custom_affine[0], 2),
                    "min_y": round(custom_affine[5] + meta["height"] * custom_affine[4], 2),
                    "max_y": round(custom_affine[5], 2)
                }

            if meta["band_count"] == 4:
                meta["product_type"] = "Sentinel-2 L2A BOA 10m Standardized Cube"
                meta["band_names"] = cls.BAND_NAMES_S2_4BAND
            else:
                meta["product_type"] = f"Multi-band Raster ({meta['band_count']} Channels)"
                meta["band_names"] = [f"Band_{i+1}" for i in range(meta["band_count"])]

            clean_src = np.nan_to_num(source, nan=0.0)
            meta["data_range"] = [float(np.min(clean_src)), float(np.max(clean_src))]

        return meta

    @classmethod
    def validate_for_domain(
        cls,
        data: Union[str, Path, np.ndarray],
        domain: str = "super_resolution",
        custom_affine: Optional[Any] = None,
        custom_crs: Optional[str] = None
    ) -> ValidationResult:
        """
        Validates raster data against the strict explicit contract of the requested domain.
        """
        domain = domain.lower().replace("-", "_")
        meta = cls.inspect_raster_metadata(data, custom_affine, custom_crs)
        
        checks = {}
        reasons = []

        # 1. Format & Spatial Dimensions
        dim_ok = meta["height"] >= 16 and meta["width"] >= 16
        checks["valid_dimensions"] = dim_ok
        if not dim_ok:
            reasons.append(f"Raster dimensions too small ({meta['dimensions']}); minimum required is 16 × 16 px.")

        # 2. Georeferencing & CRS
        crs_val = meta.get("crs", "")
        crs_ok = bool(crs_val and crs_val != "NONE" and ("EPSG" in crs_val or "UTM" in crs_val or "WGS" in crs_val))
        checks["georeferencing_present"] = crs_ok
        if not crs_ok:
            reasons.append("Geospatial Coordinate Reference System (CRS) is missing or unassigned. Raster must be georeferenced.")

        # 3. Affine Transform
        affine_ok = meta.get("affine_transform") is not None
        checks["affine_transform_present"] = affine_ok
        if not affine_ok:
            reasons.append("Affine geotransform matrix is missing. Projected coordinate mapping cannot be determined.")

        # 4. GSD Tolerance (Native Sentinel-2 is 10m, tolerance 2.0m - 30.0m for native & SR)
        gsd = meta.get("gsd", 10.0)
        gsd_ok = 2.0 <= gsd <= 30.0
        checks["native_gsd_compatible"] = gsd_ok
        if not gsd_ok:
            reasons.append(f"Ground Sampling Distance GSD ({gsd:.2f}m) outside supported Sentinel-2 range (2.0m - 30.0m).")

        # 5. Band Count & Contract per Domain
        bands = meta.get("band_count", 0)
        if domain in ["super_resolution", "urban"]:
            # Requires 4 bands (B02, B03, B04, B08)
            band_ok = bands >= 4
            checks["required_bands_present"] = band_ok
            if not band_ok:
                reasons.append(f"Requires 4-band Sentinel-2 compatible raster (B02 Blue, B03 Green, B04 Red, B08 NIR). Found {bands} bands.")
        elif domain == "water":
            # Requires Green (B03) and NIR (B08) -> at least 4 bands or multispectral 2-band
            band_ok = bands >= 2
            checks["required_bands_present"] = band_ok
            if not band_ok:
                reasons.append(f"Water Intelligence requires at least 2 bands for NDWI (Green B03, NIR B08). Found {bands} bands.")
        elif domain == "agriculture":
            # Requires Red (B04) and NIR (B08)
            band_ok = bands >= 2
            checks["required_bands_present"] = band_ok
            if not band_ok:
                reasons.append(f"Agriculture Intelligence requires Red (B04) and NIR (B08) for NDVI computation. Found {bands} bands.")
        elif domain == "oil_spill":
            # Requires 4-band optical
            band_ok = bands >= 4
            checks["required_bands_present"] = band_ok
            if not band_ok:
                reasons.append(f"Oil Spill UNet segmentation requires 4-band multispectral optical cube. Found {bands} bands.")
        elif domain == "disaster":
            band_ok = bands >= 2
            checks["required_bands_present"] = band_ok
            if not band_ok:
                reasons.append(f"Disaster & Change Intelligence requires at least 2 multispectral bands. Found {bands} bands.")
        else:
            band_ok = bands >= 1
            checks["required_bands_present"] = band_ok

        # 6. Numerical Integrity Check (NaNs, Infs, Reflectance range)
        num_ok = True
        if isinstance(data, np.ndarray):
            has_nan = bool(np.isnan(data).any())
            has_inf = bool(np.isinf(data).any())
            if has_nan:
                num_ok = False
                reasons.append("Corrupted numerical values detected: NaN (Not-a-Number) found in pixel matrix.")
            if has_inf:
                num_ok = False
                reasons.append("Corrupted numerical values detected: Infinite values found in pixel matrix.")
            
            # Check range
            min_v, max_v = float(np.nanmin(data)), float(np.nanmax(data))
            if min_v < -0.5 or max_v > 15000.0:
                num_ok = False
                reasons.append(f"Radiometric reflectance value range [{min_v:.2f}, {max_v:.2f}] outside physical bounds.")
        checks["radiometric_integrity"] = num_ok

        is_valid = all(checks.values()) and len(reasons) == 0
        return ValidationResult(
            is_valid=is_valid,
            domain=domain,
            checks=checks,
            reasons=reasons,
            metadata=meta
        )

    @classmethod
    def validate_temporal_pair(
        cls,
        pre_data: Union[str, Path, np.ndarray],
        post_data: Union[str, Path, np.ndarray],
        pre_meta: Optional[Dict[str, Any]] = None,
        post_meta: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validates temporal image pair (Before & After) for Flood/Change Intelligence.
        """
        p1 = cls.validate_for_domain(pre_data, "disaster")
        p2 = cls.validate_for_domain(post_data, "disaster")
        
        checks = {
            "pre_image_valid": p1.is_valid,
            "post_image_valid": p2.is_valid
        }
        reasons = []
        if not p1.is_valid:
            reasons.extend([f"Pre-Event Image: {r}" for r in p1.reasons])
        if not p2.is_valid:
            reasons.extend([f"Post-Event Image: {r}" for r in p2.reasons])

        m1 = p1.metadata
        m2 = p2.metadata

        # CRS compatibility
        crs_match = bool(m1.get("crs") and m2.get("crs") and str(m1.get("crs")) == str(m2.get("crs")))
        checks["crs_compatibility"] = crs_match
        if not crs_match:
            reasons.append(f"CRS Mismatch: Pre-event CRS is {m1.get('crs')}, Post-event CRS is {m2.get('crs')}. Projection must match.")

        # Spatial Extent & Dimensions
        dim_match = m1["height"] == m2["height"] and m1["width"] == m2["width"]
        checks["spatial_dimension_match"] = dim_match
        if not dim_match:
            reasons.append(f"Dimension mismatch: Pre is {m1['dimensions']}, Post is {m2['dimensions']}. Scenes must be co-registered.")

        is_valid = all(checks.values()) and len(reasons) == 0
        combined_meta = {
            "pre_event_metadata": m1,
            "post_event_metadata": m2,
            "co_registration_status": "ALIGNED" if dim_match and crs_match else "UNALIGNED"
        }
        return ValidationResult(
            is_valid=is_valid,
            domain="flood_temporal_differencing",
            checks=checks,
            reasons=reasons,
            metadata=combined_meta
        )
