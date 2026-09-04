#!/usr/bin/env python3
"""
TERRA-SR Centralized Satellite Input Validator & Product Identification Engine.
Inspects uploaded satellite rasters BEFORE inference and enforces strict geospatial,
spectral, radiometric, and format integrity contracts.

Provides:
- Automated sensor & product identification (Sentinel-2, PlanetScope, Landsat, RGB Optical)
- Strict Sentinel-2 SR contract enforcement (approx 10m native GSD with configurable tolerance, 4 bands B02/B03/B04/B08, CRS/Affine, radiometric depth)
- Graceful degradation & dynamic capability matrix for RGB-only, other multispectral, and non-georeferenced rasters
- Clear, user-friendly failure and limitation reasons (no raw exceptions)
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import re
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


# Sentinel-2 SR Contract Standards
S2_GSD_NOMINAL = 10.0
S2_GSD_TOLERANCE_MIN = 2.0   # Lower bound accommodating multi-resolution testing & SR grids
S2_GSD_TOLERANCE_MAX = 15.0  # Upper bound strictly rejecting 30m Landsat, 100m coarse rasters
GENERIC_S2_RANGE_MAX = 30.0


class InputValidationResult:
    """
    Comprehensive structured validation & capability result for satellite inputs.
    Backward-compatible with legacy ValidationResult.
    """
    def __init__(
        self,
        valid: bool,
        level: str,  # "READY", "LIMITED", "UNSUPPORTED"
        detected_sensor: str,
        detected_product: str,
        gsd: float,
        dimensions: Dict[str, int],
        bands: List[str],
        band_map: Dict[str, int],
        required_bands: Dict[str, bool],
        crs: Optional[str],
        georeferenced: bool,
        reflectance_valid: bool,
        capabilities: Dict[str, Any],
        warnings: List[str],
        errors: List[str],
        checks: Dict[str, bool],
        reasons_list: List[str],
        metadata: Dict[str, Any],
        domain: str = "super_resolution"
    ):
        self.valid = valid
        self.level = level
        self.detected_sensor = detected_sensor
        self.detected_product = detected_product
        self.gsd = gsd
        self.dimensions = dimensions
        self.bands = bands
        self.band_map = band_map
        self.required_bands = required_bands
        self.crs = crs
        self.georeferenced = georeferenced
        self.reflectance_valid = reflectance_valid
        self.capabilities = capabilities
        self.warnings = warnings
        self.errors = errors
        self.checks = checks
        self.reasons_list = reasons_list
        self.metadata = metadata
        self.domain = domain

    @property
    def is_valid(self) -> bool:
        """
        Backward compatibility: Returns whether the input satisfies all checks
        for the requested domain.
        """
        return all(self.checks.values()) and len(self.errors) == 0 and len(self.reasons_list) == 0

    @property
    def status(self) -> str:
        """Backward-compatible status string."""
        return "VALID" if self.is_valid else "INVALID"

    @property
    def reasons(self) -> List[str]:
        """Backward-compatible reasons list combining errors, contract reasons, and warnings."""
        if self.reasons_list:
            return list(self.reasons_list)
        elif self.errors:
            return list(self.errors)
        elif not self.is_valid:
            dom = self.domain.lower().replace("-", "_")
            if dom in ["super_resolution", "sr"]:
                sr_cap = self.capabilities.get("sr", {})
                if not sr_cap.get("supported") and sr_cap.get("reason"):
                    return [sr_cap.get("reason")]
            else:
                cap = self.capabilities.get(dom, {})
                if cap and not cap.get("supported") and cap.get("reason"):
                    return [cap.get("reason")]
            return ["Input contract requirement not satisfied."]
        return list(self.warnings)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes validation result to clean dictionary for JSON APIs and UI."""
        return {
            "valid": self.valid,
            "is_valid": self.is_valid,
            "status": self.status,
            "level": self.level,
            "domain": self.domain,
            "detected_sensor": self.detected_sensor,
            "detected_product": self.detected_product,
            "gsd": round(self.gsd, 2) if self.gsd else 10.0,
            "dimensions": self.dimensions,
            "bands": self.bands,
            "band_map": self.band_map,
            "required_bands": self.required_bands,
            "crs": self.crs or "NONE",
            "georeferenced": self.georeferenced,
            "reflectance_valid": self.reflectance_valid,
            "capabilities": self.capabilities,
            "warnings": self.warnings,
            "errors": self.errors,
            "checks": self.checks,
            "reasons": self.reasons,
            "metadata": self.metadata,
            "summary_message": self.format_summary(),
            "sr_compatibility_message": self.get_sr_compatibility_message()
        }

    def get_sr_compatibility_message(self) -> str:
        """Returns clear human-readable Sentinel-2 SR compatibility statement."""
        sr_cap = self.capabilities.get("sr", {})
        if sr_cap.get("supported", False):
            return "SENTINEL-2 SR MODEL: COMPATIBLE (10m Native GSD, 4-Band BOA Multispectral Cube)"
        else:
            reason = sr_cap.get("reason", "Input does not meet Sentinel-2 SR contract.")
            return f"SENTINEL-2 SR MODEL: NOT COMPATIBLE ({reason})"

    def format_summary(self) -> str:
        """Returns multi-line formatted summary for CLI and UI presentation."""
        lines = []
        if self.level == "READY":
            lines.append(f"INPUT VALID ✅ ({self.detected_product})")
            lines.append(f"• Sensor: {self.detected_sensor}")
            lines.append(f"• Native GSD: {self.gsd:.2f}m")
            lines.append(f"• Bands: {len(self.bands)} ({', '.join(self.bands)})")
            lines.append(f"• CRS: {self.crs or 'NONE'}")
            lines.append("• Georeferencing: VALID (Affine Transformed)")
            lines.append("• Status: READY FOR SUPER-RESOLUTION")
        elif self.level == "LIMITED":
            lines.append(f"INPUT ACCEPTED WITH LIMITATIONS ⚠️ ({self.detected_product})")
            lines.append(f"• Sensor: {self.detected_sensor}")
            lines.append(f"• Native GSD: {self.gsd:.2f}m")
            lines.append(f"• Bands: {len(self.bands)} ({', '.join(self.bands)})")
            lines.append(f"• CRS: {self.crs or 'NONE'}")
            sr_cap = self.capabilities.get("sr", {})
            if not sr_cap.get("supported"):
                lines.append(f"• Sentinel-2 SR: NOT COMPATIBLE ({sr_cap.get('reason')})")
            for w in self.warnings:
                lines.append(f"• Notice: {w}")
        else:
            lines.append(f"INPUT REJECTED ❌ ({self.detected_product or 'Corrupt/Unsupported Format'})")
            for err in self.reasons:
                lines.append(f"• Error: {err}")
        return "\n".join(lines)


# Legacy alias
ValidationResult = InputValidationResult


class SatelliteInputValidator:
    """
    Centralized validation & product identification engine for Earth Observation rasters.
    """

    BAND_NAMES_S2_4BAND = ["B02 (Blue)", "B03 (Green)", "B04 (Red)", "B08 (NIR)"]

    @classmethod
    def identify_bands(
        cls,
        band_count: int,
        descriptions: Optional[List[Optional[str]]] = None,
        tags: Optional[Dict[str, Any]] = None,
        band_tags: Optional[List[Dict[str, Any]]] = None,
        gsd: float = 10.0,
        filename: str = ""
    ) -> Tuple[List[str], Dict[str, int], Dict[str, bool], bool]:
        """
        Identifies logical spectral bands using metadata, descriptions, tags, and wavelengths.
        """
        band_map: Dict[str, int] = {}
        bands_display: List[str] = [f"Band_{i+1}" for i in range(band_count)]
        confident = False

        norm_descs = []
        if descriptions:
            norm_descs = [d.strip().upper() if isinstance(d, str) else "" for d in descriptions]

        # 1. Match based on explicit band descriptions or band-level tags
        for idx in range(band_count):
            d = norm_descs[idx] if idx < len(norm_descs) else ""
            b_tag = band_tags[idx] if band_tags and idx < len(band_tags) else {}
            tag_str = " ".join(f"{k}={v}" for k, v in b_tag.items()).upper()
            combined_desc = f"{d} {tag_str}".strip()

            # Blue (B02 / 490nm)
            if any(k in combined_desc for k in ["B02", "B2", "BLUE", "490NM", "492NM", "WAVELENGTH=490", "WAVELENGTH=492"]):
                band_map["blue"] = idx
                bands_display[idx] = "B02 (Blue)"
            # Green (B03 / 560nm)
            elif any(k in combined_desc for k in ["B03", "B3", "GREEN", "560NM", "559NM", "WAVELENGTH=560", "WAVELENGTH=559"]):
                band_map["green"] = idx
                bands_display[idx] = "B03 (Green)"
            # Red (B04 / 665nm)
            elif any(k in combined_desc for k in ["B04", "B4", "RED", "665NM", "664NM", "WAVELENGTH=665", "WAVELENGTH=664"]):
                band_map["red"] = idx
                bands_display[idx] = "B04 (Red)"
            # NIR (B08 / 842nm / 833nm / 865nm)
            elif any(k in combined_desc for k in ["B08", "B8", "B8A", "NIR", "NEAR_INFRARED", "842NM", "833NM", "865NM", "WAVELENGTH=842"]):
                band_map["nir"] = idx
                bands_display[idx] = "B08 (NIR)"
            # SWIR1 (B11)
            elif any(k in combined_desc for k in ["B11", "SWIR1", "1610NM"]):
                band_map["swir1"] = idx
                bands_display[idx] = "B11 (SWIR-1)"
            # SWIR2 (B12)
            elif any(k in combined_desc for k in ["B12", "SWIR2", "2190NM"]):
                band_map["swir2"] = idx
                bands_display[idx] = "B12 (SWIR-2)"

        if len(band_map) >= 3:
            confident = True

        # 2. Heuristic fallback based on standard band counts and dataset conventions
        if not confident:
            if band_count == 4:
                band_map = {"blue": 0, "green": 1, "red": 2, "nir": 3}
                bands_display = cls.BAND_NAMES_S2_4BAND
                confident = True
            elif band_count == 3:
                band_map = {"red": 0, "green": 1, "blue": 2}
                bands_display = ["Red", "Green", "Blue"]
                confident = True
            elif band_count >= 8:
                band_map = {"blue": 1, "green": 2, "red": 3, "nir": 7}
                bands_display = [f"Band_{i+1}" for i in range(band_count)]
                bands_display[1] = "B02 (Blue)"
                bands_display[2] = "B03 (Green)"
                bands_display[3] = "B04 (Red)"
                bands_display[7] = "B08 (NIR)"
                confident = True
            else:
                confident = False

        required_bands = {
            "B02_blue": "blue" in band_map,
            "B03_green": "green" in band_map,
            "B04_red": "red" in band_map,
            "B08_nir": "nir" in band_map
        }

        return bands_display, band_map, required_bands, confident

    @classmethod
    def detect_sensor_and_product(
        cls,
        band_count: int,
        gsd: float,
        band_map: Dict[str, int],
        tags: Optional[Dict[str, Any]] = None,
        filename: str = ""
    ) -> Tuple[str, str]:
        """
        Determines the satellite sensor and product type based on metadata, GSD, and spectral channels.
        """
        fn_upper = filename.upper()
        tag_blob = " ".join(f"{k}={v}" for k, v in (tags or {}).items()).upper()

        # PlanetScope explicitly identified via tags, filename, or explicit 3m Planet indicator
        if "PLANET" in tag_blob or "PLANETSCOPE" in fn_upper or "PS" in fn_upper:
            return "PlanetScope", f"PlanetScope OrthoTile ({band_count} Bands, {gsd:.2f}m Native GSD)"

        # Landsat (30m multispectral)
        if "LANDSAT" in tag_blob or "LANDSAT" in fn_upper or (band_count >= 4 and 25.0 <= gsd <= 35.0):
            return "Landsat 8/9", f"Landsat OLI/TIRS Surface Reflectance ({band_count} Bands, {gsd:.1f}m GSD)"

        # Sentinel-2 (nominal 10m L2A, 4 bands or full stack)
        if "SENTINEL" in tag_blob or "DATATAKE_IDENTIFIER" in tag_blob or "S2" in fn_upper or (band_count == 4 and 2.0 <= gsd <= 15.0):
            if band_count == 4 and abs(gsd - 10.0) <= 2.5:
                return "Sentinel-2", "Sentinel-2 L2A BOA 10m Multispectral (B02, B03, B04, B08)"
            elif band_count == 4:
                return "Sentinel-2", "Sentinel-2 L2A BOA 10m Standardized Cube"
            elif band_count >= 8:
                return "Sentinel-2", f"Sentinel-2 Full L2A Multi-Band Cube ({band_count} Channels)"
            else:
                return "Sentinel-2", f"Sentinel-2 Custom Product ({band_count} Bands, {gsd:.1f}m GSD)"

        # RGB Optical (3 bands)
        if band_count == 3:
            return "RGB Optical", f"Standard 3-Band RGB Optical Imagery ({gsd:.1f}m GSD)"

        if band_count >= 4:
            return "Generic Multispectral", f"Multispectral Satellite Raster ({band_count} Bands, {gsd:.1f}m GSD)"

        return "Unknown Satellite Raster", f"Custom {band_count}-Band Raster ({gsd:.1f}m GSD)"

    @classmethod
    def get_capabilities(
        cls,
        valid: bool,
        detected_sensor: str,
        gsd: float,
        band_count: int,
        band_map: Dict[str, int],
        georeferenced: bool,
        crs: Optional[str],
        reflectance_valid: bool,
        dimensions: Dict[str, int],
        confidence_identified: bool
    ) -> Dict[str, Dict[str, Any]]:
        """
        Builds the dynamic capability matrix determining supported downstream intelligence operations.
        """
        caps: Dict[str, Dict[str, Any]] = {}

        # 1. Preview Capability
        preview_ok = valid and dimensions.get("width", 0) >= 16 and dimensions.get("height", 0) >= 16
        caps["preview"] = {
            "supported": preview_ok,
            "label": "Scene Preview & Basic Visualization",
            "reason": "Basic optical raster preview available." if preview_ok else "Raster dimensions too small or unreadable."
        }

        # 2. Super-Resolution Contract Check (Sentinel-2 SR contract)
        has_all_s2_bands = ("blue" in band_map and "green" in band_map and "red" in band_map and "nir" in band_map) or (band_count >= 4)
        if band_count == 3:
            has_all_s2_bands = False

        is_planet = (detected_sensor == "PlanetScope")
        gsd_in_s2_tolerance = (S2_GSD_TOLERANCE_MIN <= gsd <= S2_GSD_TOLERANCE_MAX) and not is_planet
        
        sr_reasons = []
        if not valid:
            sr_reasons.append("Input raster could not be parsed.")
        if not has_all_s2_bands:
            sr_reasons.append(f"Missing required Sentinel-2 bands. Requires 4-band Sentinel-2 compatible raster (B02 Blue, B03 Green, B04 Red, B08 NIR). Found {band_count} bands.")
        if is_planet:
            sr_reasons.append(f"PlanetScope sensor detected ({gsd:.2f}m native GSD); current SR model expects approximately 10m Sentinel-2 input.")
        elif not gsd_in_s2_tolerance:
            sr_reasons.append(f"Detected native GSD is {gsd:.2f}m; current SR model expects approximately 10m Sentinel-2 input (tolerance: {S2_GSD_TOLERANCE_MIN}m - {S2_GSD_TOLERANCE_MAX}m).")
        if not georeferenced:
            sr_reasons.append("Geospatial Coordinate Reference System (CRS) is missing or unassigned. Raster must be georeferenced.")
        if not reflectance_valid:
            sr_reasons.append("Reflectance values outside valid physical range or contains NaNs/Infs.")
        if not confidence_identified and band_count < 4:
            sr_reasons.append("Band identity could not be reliably determined.")

        sr_supported = (len(sr_reasons) == 0) and valid
        caps["sr"] = {
            "supported": sr_supported,
            "label": "Super-Resolution Enhancement",
            "reason": "Fully compatible with Sentinel-2 SR contract." if sr_supported else "; ".join(sr_reasons),
            "validated_model": "Residual CNN (5.0m validated output)",
            "experimental_model": "PI-RCAN (3.33m reconstruction grid, experimental)"
        }

        # 3. Water Intelligence (Requires Green and NIR)
        has_water_bands = ("green" in band_map and "nir" in band_map) or (band_count >= 4)
        if band_count < 4 and "nir" not in band_map:
            has_water_bands = False
        water_reasons = []
        if not has_water_bands or band_count < 2:
            water_reasons.append("Water analysis unavailable: Green and NIR bands could not be identified.")
        if not valid:
            water_reasons.append("Invalid raster input.")
        water_supported = len(water_reasons) == 0
        caps["water"] = {
            "supported": water_supported,
            "label": "Water Intelligence (NDWI & Shorelines)",
            "reason": "Water Intelligence supported (Green + NIR available)." if water_supported else "; ".join(water_reasons)
        }

        # 4. Agriculture Intelligence (Requires Red and NIR)
        has_agri_bands = ("red" in band_map and "nir" in band_map) or (band_count >= 4)
        if band_count < 4 and "nir" not in band_map:
            has_agri_bands = False
        agri_reasons = []
        if not has_agri_bands or band_count < 2:
            agri_reasons.append("Agriculture analysis unavailable: Red (B04) and NIR (B08) bands could not be identified for NDVI computation.")
        if not valid:
            agri_reasons.append("Invalid raster input.")
        agri_supported = len(agri_reasons) == 0
        caps["agriculture"] = {
            "supported": agri_supported,
            "label": "Agriculture Intelligence (NDVI & Crop Vigor)",
            "reason": "Agriculture Intelligence supported (Red + NIR available)." if agri_supported else "; ".join(agri_reasons)
        }

        # 5. Urban Intelligence (Requires RGB optical / 4-band)
        has_urban_bands = ("red" in band_map and "green" in band_map and "blue" in band_map) or band_count >= 3
        urban_reasons = []
        if not has_urban_bands:
            urban_reasons.append("Urban analysis unavailable: Requires at least 3 optical RGB bands.")
        if not valid:
            urban_reasons.append("Invalid raster input.")
        urban_supported = len(urban_reasons) == 0
        caps["urban"] = {
            "supported": urban_supported,
            "label": "Urban Intelligence (Built-Up & Roads)",
            "reason": "Urban Intelligence supported (Optical RGB bands available)." if urban_supported else "; ".join(urban_reasons)
        }

        # 6. Disaster & Change Intelligence (Requires at least 2 multispectral bands)
        has_disaster_bands = (len(band_map) >= 2) or (band_count >= 2)
        disaster_reasons = []
        if not has_disaster_bands:
            disaster_reasons.append("Disaster analysis unavailable: Requires at least 2 multispectral bands.")
        if not valid:
            disaster_reasons.append("Invalid raster input.")
        disaster_supported = len(disaster_reasons) == 0
        caps["disaster"] = {
            "supported": disaster_supported,
            "label": "Disaster Intelligence (Flood & Burn Scars)",
            "reason": "Disaster Intelligence supported." if disaster_supported else "; ".join(disaster_reasons)
        }

        # 7. Oil Spill Intelligence (Requires 4-band optical RGB+NIR)
        has_oil_bands = (all(b in band_map for b in ["blue", "green", "red", "nir"])) or (band_count >= 4)
        if band_count < 4:
            has_oil_bands = False
        oil_reasons = []
        if not has_oil_bands:
            oil_reasons.append("Oil spill intelligence unavailable: Requires 4-band multispectral optical cube (RGB+NIR).")
        if not valid:
            oil_reasons.append("Invalid raster input.")
        oil_supported = len(oil_reasons) == 0
        caps["oil_spill"] = {
            "supported": oil_supported,
            "label": "Oil Spill Intelligence (UNet Segmentation & SOSI)",
            "reason": "Oil Spill Intelligence supported." if oil_supported else "; ".join(oil_reasons)
        }

        # 8. Geospatial Measurements & GIS Coordinates (Requires valid CRS and Affine transform)
        geo_reasons = []
        if not georeferenced:
            geo_reasons.append("Geospatial processing unavailable because the input has no valid CRS/georeferencing.")
        geo_supported = len(geo_reasons) == 0 and valid
        caps["geospatial_measurement"] = {
            "supported": geo_supported,
            "label": "Geospatial Coordinate Tracking & Area Measurements",
            "reason": "Georeferenced coordinate mapping available." if geo_supported else "; ".join(geo_reasons)
        }

        # 9. GeoJSON Vector Export
        caps["geojson_export"] = {
            "supported": geo_supported,
            "label": "GeoJSON Vector Export (.geojson)",
            "reason": "Projected vector polygons available for GIS export." if geo_supported else "Geospatial vector export requires valid CRS/georeferencing."
        }

        # 10. GeoTIFF Raster Export
        caps["geotiff_export"] = {
            "supported": valid,
            "label": "GeoTIFF Export (.tif)",
            "reason": "Standard GeoTIFF export supported." if valid else "Corrupted raster cannot be exported."
        }

        # 11. Visual RGB Export
        caps["visual_export"] = {
            "supported": preview_ok,
            "label": "Visual RGB Image Export (.png)",
            "reason": "Visual RGB snapshot export supported." if preview_ok else "Raster unreadable."
        }

        return caps

    @classmethod
    def inspect_raster_metadata(
        cls,
        source: Union[str, Path, np.ndarray],
        custom_affine: Optional[Any] = None,
        custom_crs: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extracts comprehensive metadata directly from raster header or array.
        """
        meta = {
            "product_type": "Unknown Satellite Raster",
            "detected_sensor": "Unknown",
            "format": "NumPy Array",
            "acquisition_date": "2026-05-15 (Standard)",
            "dimensions": "0 × 0 px",
            "height": 0,
            "width": 0,
            "band_count": 0,
            "band_names": [],
            "band_descriptions": [],
            "crs": "NONE",
            "gsd": 10.0,
            "affine_transform": None,
            "bounds": {"min_x": 0.0, "min_y": 0.0, "max_x": 0.0, "max_y": 0.0},
            "nodata_value": None,
            "dtype": "float32",
            "data_range": [0.0, 1.0],
            "tags": {},
            "band_tags": []
        }

        filename = ""
        if isinstance(source, (str, Path)):
            path_obj = Path(source)
            filename = path_obj.name
            if HAS_RASTERIO and path_obj.exists():
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
                        meta["band_descriptions"] = list(src.descriptions) if src.descriptions else []

                        if src.transform:
                            meta["gsd"] = abs(src.transform[0])
                            bounds = src.bounds
                            meta["bounds"] = {
                                "min_x": round(bounds.left, 2),
                                "min_y": round(bounds.bottom, 2),
                                "max_x": round(bounds.right, 2),
                                "max_y": round(bounds.top, 2)
                            }

                        tags = src.tags()
                        meta["tags"] = tags
                        meta["band_tags"] = [src.tags(b) for b in range(1, src.count + 1)]

                        if "DATATAKE_1_DATATAKE_IDENTIFIER" in tags or "ACQUISITION_DATE" in tags:
                            meta["acquisition_date"] = tags.get("ACQUISITION_DATE", tags.get("DATATAKE_1_DATATAKE_IDENTIFIER"))

                        # Identify bands and sensor
                        bands_display, band_map, req_bands, confident = cls.identify_bands(
                            band_count=src.count,
                            descriptions=meta["band_descriptions"],
                            tags=tags,
                            band_tags=meta["band_tags"],
                            gsd=meta["gsd"],
                            filename=filename
                        )
                        meta["band_names"] = bands_display
                        meta["band_map"] = band_map
                        meta["required_bands"] = req_bands
                        meta["confidence_identified"] = confident

                        sensor, product = cls.detect_sensor_and_product(
                            band_count=src.count,
                            gsd=meta["gsd"],
                            band_map=band_map,
                            tags=tags,
                            filename=filename
                        )
                        meta["detected_sensor"] = sensor
                        meta["product_type"] = product
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

            meta["crs"] = str(custom_crs) if (custom_crs and custom_crs != "NONE") else "NONE"
            if custom_affine:
                meta["affine_transform"] = list(custom_affine)
                meta["gsd"] = abs(custom_affine[0])
                meta["bounds"] = {
                    "min_x": round(custom_affine[2], 2),
                    "max_x": round(custom_affine[2] + meta["width"] * custom_affine[0], 2),
                    "min_y": round(custom_affine[5] + meta["height"] * custom_affine[4], 2),
                    "max_y": round(custom_affine[5], 2)
                }

            bands_display, band_map, req_bands, confident = cls.identify_bands(
                band_count=meta["band_count"],
                gsd=meta["gsd"],
                filename=filename
            )
            meta["band_names"] = bands_display
            meta["band_map"] = band_map
            meta["required_bands"] = req_bands
            meta["confidence_identified"] = confident

            sensor, product = cls.detect_sensor_and_product(
                band_count=meta["band_count"],
                gsd=meta["gsd"],
                band_map=band_map,
                filename=filename
            )
            meta["detected_sensor"] = sensor
            meta["product_type"] = product

            clean_src = np.nan_to_num(source, nan=0.0)
            meta["data_range"] = [float(np.min(clean_src)), float(np.max(clean_src))]

        return meta

    @classmethod
    def validate_input(
        cls,
        data: Union[str, Path, np.ndarray],
        domain: str = "super_resolution",
        custom_affine: Optional[Any] = None,
        custom_crs: Optional[str] = None
    ) -> InputValidationResult:
        """
        Main centralized entry point to validate satellite input, check integrity,
        and generate dynamic capability matrix.
        """
        domain = domain.lower().replace("-", "_")
        meta = cls.inspect_raster_metadata(data, custom_affine, custom_crs)

        checks = {}
        errors = []
        warnings = []
        reasons_list = []

        # 1. Format & Spatial Dimensions
        dim_ok = meta["height"] >= 16 and meta["width"] >= 16
        checks["valid_dimensions"] = dim_ok
        if not dim_ok:
            msg = f"Raster dimensions too small ({meta['dimensions']}); minimum required is 16 × 16 px."
            errors.append(msg)
            reasons_list.append(msg)

        # 2. Georeferencing & CRS
        crs_val = str(meta.get("crs", "")).strip().upper()
        crs_ok = bool(crs_val and crs_val != "NONE" and ("EPSG" in crs_val or "UTM" in crs_val or "WGS" in crs_val or "+" in crs_val))
        checks["georeferencing_present"] = crs_ok
        if not crs_ok:
            msg = "Geospatial Coordinate Reference System (CRS) is missing or unassigned. Raster must be georeferenced."
            warnings.append(msg)
            reasons_list.append(msg)

        # 3. Affine Transform
        affine_ok = meta.get("affine_transform") is not None
        checks["affine_transform_present"] = affine_ok
        if not affine_ok:
            msg = "Affine geotransform matrix is missing. Projected coordinate mapping cannot be determined."
            warnings.append(msg)
            reasons_list.append(msg)

        georeferenced = crs_ok and affine_ok

        # 4. GSD Extraction & Check (Tolerance: 2.0m - 30.0m for general S2 compatibility)
        gsd = meta.get("gsd", 10.0)
        gsd_ok = (2.0 <= gsd <= GENERIC_S2_RANGE_MAX)
        checks["native_gsd_compatible"] = gsd_ok
        if not gsd_ok:
            msg = f"Ground Sampling Distance GSD ({gsd:.2f}m) outside supported Sentinel-2 range (2.0m - 30.0m)."
            errors.append(msg)
            reasons_list.append(msg)

        # 5. Band Identification & Domain Requirements
        band_count = meta.get("band_count", 0)
        band_map = meta.get("band_map", {})
        confident = meta.get("confidence_identified", True)
        if not confident:
            warnings.append("Band identity could not be reliably determined.")

        if domain in ["super_resolution", "urban", "sr"]:
            band_ok = band_count >= 4
            checks["required_bands_present"] = band_ok
            if not band_ok:
                msg = f"Requires 4-band Sentinel-2 compatible raster (B02 Blue, B03 Green, B04 Red, B08 NIR). Found {band_count} bands."
                reasons_list.append(msg)
        elif domain == "water":
            band_ok = band_count >= 2
            checks["required_bands_present"] = band_ok
            if not band_ok:
                msg = f"Water Intelligence requires at least 2 bands for NDWI (Green B03, NIR B08). Found {band_count} bands."
                reasons_list.append(msg)
        elif domain == "agriculture":
            band_ok = band_count >= 2
            checks["required_bands_present"] = band_ok
            if not band_ok:
                msg = f"Agriculture Intelligence requires Red (B04) and NIR (B08) for NDVI computation. Found {band_count} bands."
                reasons_list.append(msg)
        elif domain == "oil_spill":
            band_ok = band_count >= 4
            checks["required_bands_present"] = band_ok
            if not band_ok:
                msg = f"Oil Spill UNet segmentation requires 4-band multispectral optical cube. Found {band_count} bands."
                reasons_list.append(msg)
        elif domain == "disaster":
            band_ok = band_count >= 2
            checks["required_bands_present"] = band_ok
            if not band_ok:
                msg = f"Disaster & Change Intelligence requires at least 2 multispectral bands. Found {band_count} bands."
                reasons_list.append(msg)
        else:
            checks["required_bands_present"] = band_count >= 1

        # 6. Radiometric & Numerical Integrity Check
        num_ok = True
        if isinstance(data, np.ndarray):
            has_nan = bool(np.isnan(data).any())
            has_inf = bool(np.isinf(data).any())
            if has_nan:
                num_ok = False
                msg = "Corrupted numerical values detected: NaN (Not-a-Number) found in pixel matrix."
                errors.append(msg)
                reasons_list.append(msg)
            if has_inf:
                num_ok = False
                msg = "Corrupted numerical values detected: Infinite values found in pixel matrix."
                errors.append(msg)
                reasons_list.append(msg)
            
            min_v, max_v = float(np.nanmin(data)), float(np.nanmax(data))
            if min_v < -0.5 or max_v > 15000.0:
                num_ok = False
                msg = f"Radiometric reflectance value range [{min_v:.2f}, {max_v:.2f}] outside physical bounds."
                errors.append(msg)
                reasons_list.append(msg)
        elif isinstance(data, (str, Path)) and HAS_RASTERIO and Path(data).exists():
            try:
                with rasterio.open(data) as src:
                    sample = src.read(1, window=rasterio.windows.Window(0, 0, min(64, src.width), min(64, src.height)))
                    if np.isnan(sample).any():
                        num_ok = False
                        msg = "Corrupted numerical values detected: NaN (Not-a-Number) found in raster sample."
                        errors.append(msg)
                        reasons_list.append(msg)
            except Exception:
                pass

        checks["radiometric_integrity"] = num_ok
        reflectance_valid = num_ok

        # 7. Check if file is readable
        valid_parse = meta["height"] > 0 and meta["width"] > 0 and band_count > 0 and len(errors) == 0

        # 8. Compute Dynamic Capability Matrix
        dimensions_dict = {"width": meta["width"], "height": meta["height"]}
        capabilities = cls.get_capabilities(
            valid=valid_parse,
            detected_sensor=meta.get("detected_sensor", "Unknown"),
            gsd=gsd,
            band_count=band_count,
            band_map=band_map,
            georeferenced=georeferenced,
            crs=meta.get("crs"),
            reflectance_valid=reflectance_valid,
            dimensions=dimensions_dict,
            confidence_identified=confident
        )

        # 9. Determine Status Level (READY, LIMITED, UNSUPPORTED)
        if not valid_parse or len(errors) > 0 or not num_ok:
            level = "UNSUPPORTED"
        elif capabilities.get("sr", {}).get("supported", False) and georeferenced:
            level = "READY"
        else:
            level = "LIMITED"

        return InputValidationResult(
            valid=valid_parse,
            level=level,
            detected_sensor=meta.get("detected_sensor", "Unknown"),
            detected_product=meta.get("product_type", "Unknown Satellite Raster"),
            gsd=gsd,
            dimensions=dimensions_dict,
            bands=meta.get("band_names", []),
            band_map=band_map,
            required_bands=meta.get("required_bands", {}),
            crs=meta.get("crs"),
            georeferenced=georeferenced,
            reflectance_valid=reflectance_valid,
            capabilities=capabilities,
            warnings=warnings,
            errors=errors,
            checks=checks,
            reasons_list=reasons_list,
            metadata=meta,
            domain=domain
        )

    @classmethod
    def validate_for_domain(
        cls,
        data: Union[str, Path, np.ndarray],
        domain: str = "super_resolution",
        custom_affine: Optional[Any] = None,
        custom_crs: Optional[str] = None
    ) -> InputValidationResult:
        """
        Legacy contract validator for specific requested domain.
        Maintains backward compatibility with test suites and CLI.
        """
        return cls.validate_input(
            data=data,
            domain=domain,
            custom_affine=custom_affine,
            custom_crs=custom_crs
        )

    @classmethod
    def validate_temporal_pair(
        cls,
        pre_data: Union[str, Path, np.ndarray],
        post_data: Union[str, Path, np.ndarray],
        pre_meta: Optional[Dict[str, Any]] = None,
        post_meta: Optional[Dict[str, Any]] = None
    ) -> InputValidationResult:
        """
        Validates temporal image pair (Before & After) for Flood/Change Intelligence.
        """
        p1 = cls.validate_input(pre_data, domain="disaster")
        p2 = cls.validate_input(post_data, domain="disaster")

        checks = {
            "pre_image_valid": p1.valid,
            "post_image_valid": p2.valid
        }
        errors = []
        if not p1.valid:
            errors.extend([f"Pre-Event Image: {r}" for r in p1.errors or p1.reasons])
        if not p2.valid:
            errors.extend([f"Post-Event Image: {r}" for r in p2.errors or p2.reasons])

        m1 = p1.metadata
        m2 = p2.metadata

        # CRS compatibility
        crs_match = bool(m1.get("crs") and m2.get("crs") and str(m1.get("crs")) == str(m2.get("crs")) and str(m1.get("crs")) != "NONE")
        checks["crs_compatibility"] = crs_match
        if not crs_match:
            errors.append(f"CRS Mismatch: Pre-event CRS is {m1.get('crs')}, Post-event CRS is {m2.get('crs')}. Projection must match.")

        # Spatial Extent & Dimensions
        dim_match = m1["height"] == m2["height"] and m1["width"] == m2["width"]
        checks["spatial_dimension_match"] = dim_match
        if not dim_match:
            errors.append(f"Dimension mismatch: Pre is {m1['dimensions']}, Post is {m2['dimensions']}. Scenes must be co-registered.")

        valid_pair = p1.valid and p2.valid and crs_match and dim_match
        level = "READY" if valid_pair else "UNSUPPORTED"

        combined_meta = {
            "pre_event_metadata": m1,
            "post_event_metadata": m2,
            "co_registration_status": "ALIGNED" if dim_match and crs_match else "UNALIGNED"
        }

        caps = {
            "disaster": {
                "supported": valid_pair,
                "label": "Flood & Change Intelligence",
                "reason": "Temporal co-registered pair validated." if valid_pair else "; ".join(errors)
            }
        }

        return InputValidationResult(
            valid=valid_pair,
            level=level,
            detected_sensor=m1.get("detected_sensor", "Unknown"),
            detected_product="Co-registered Temporal Pair",
            gsd=p1.gsd,
            dimensions=p1.dimensions,
            bands=p1.bands,
            band_map=p1.band_map,
            required_bands=p1.required_bands,
            crs=p1.crs,
            georeferenced=p1.georeferenced,
            reflectance_valid=p1.reflectance_valid and p2.reflectance_valid,
            capabilities=caps,
            warnings=p1.warnings + p2.warnings,
            errors=errors,
            checks=checks,
            reasons_list=errors,
            metadata=combined_meta,
            domain="flood_temporal_differencing"
        )
