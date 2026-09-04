#!/usr/bin/env python3
"""
TERRA-SR Comprehensive Input Validation & Capability Matrix Test Suite.

Tests at minimum:
1. Valid Sentinel-2 4-band 10m GeoTIFF / Array
2. RGB-only image
3. Missing B08 (NIR)
4. Missing CRS
5. Missing geotransform / Affine
6. Invalid GSD (e.g. 30m Landsat, 3m PlanetScope outside S2 SR tolerance)
7. NaN / Inf corrupted pixel matrices
8. Unsupported raster format / non-raster corrupted file
9. Other multispectral raster (PlanetScope, Landsat)
10. Band identification from metadata (descriptions, tags, wavelengths)
11. Capability matrix generation across diverse profiles
12. Dynamic UI capability state / API payload response
"""

import unittest
import json
import tempfile
import numpy as np
from pathlib import Path

from src.core.input_validation import (
    SatelliteInputValidator,
    InputValidationResult,
    S2_GSD_NOMINAL,
    S2_GSD_TOLERANCE_MIN,
    S2_GSD_TOLERANCE_MAX
)

try:
    import rasterio
    from rasterio.transform import from_origin
    from affine import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


class TestInputValidation(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.H, self.W = 64, 64
        # Standardized 4-band Sentinel-2 cube: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)
        self.s2_cube = np.random.uniform(0.05, 0.45, (4, self.H, self.W)).astype(np.float32)
        # S2 Affine transform: 10.0m GSD in UTM Zone 43N
        self.s2_affine = Affine.translation(750000.0, 1450000.0) * Affine.scale(10.0, -10.0)
        self.s2_crs = "EPSG:32643"

    # =========================================================================
    # Test 1: Valid Sentinel-2 4-band 10m Input
    # =========================================================================
    def test_1_valid_sentinel2_4band_10m(self):
        res = SatelliteInputValidator.validate_input(
            self.s2_cube,
            domain="super_resolution",
            custom_affine=self.s2_affine,
            custom_crs=self.s2_crs
        )
        self.assertTrue(res.valid)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.level, "READY")
        self.assertEqual(res.detected_sensor, "Sentinel-2")
        self.assertAlmostEqual(res.gsd, 10.0, delta=0.1)
        self.assertEqual(len(res.bands), 4)
        self.assertTrue(res.georeferenced)
        self.assertTrue(res.reflectance_valid)
        self.assertTrue(res.capabilities["sr"]["supported"])
        self.assertTrue(res.capabilities["water"]["supported"])
        self.assertTrue(res.capabilities["agriculture"]["supported"])
        self.assertTrue(res.capabilities["urban"]["supported"])
        self.assertTrue(res.capabilities["oil_spill"]["supported"])
        self.assertTrue(res.capabilities["disaster"]["supported"])
        self.assertTrue(res.capabilities["geospatial_measurement"]["supported"])
        self.assertTrue(res.capabilities["geojson_export"]["supported"])
        self.assertIn("READY FOR SUPER-RESOLUTION", res.format_summary())

    # =========================================================================
    # Test 2: RGB-Only Satellite Input
    # =========================================================================
    def test_2_rgb_only_image(self):
        rgb_cube = self.s2_cube[:3, :, :]  # Only 3 bands (R, G, B)
        res = SatelliteInputValidator.validate_input(
            rgb_cube,
            domain="super_resolution",
            custom_affine=self.s2_affine,
            custom_crs=self.s2_crs
        )
        self.assertTrue(res.valid, "Raster format is valid for preview")
        self.assertEqual(res.level, "LIMITED")
        self.assertEqual(res.detected_sensor, "RGB Optical")
        # SR must be disabled
        self.assertFalse(res.capabilities["sr"]["supported"])
        self.assertTrue(any("4-band" in r or "Sentinel-2" in r for r in res.reasons) or "4-band" in res.capabilities["sr"]["reason"])
        # Agriculture / NDVI requiring NIR must be disabled
        self.assertFalse(res.capabilities["agriculture"]["supported"])
        self.assertIn("Agriculture analysis unavailable", res.capabilities["agriculture"]["reason"])
        # Preview must be enabled
        self.assertTrue(res.capabilities["preview"]["supported"])
        # Urban RGB inspection enabled
        self.assertTrue(res.capabilities["urban"]["supported"])

    # =========================================================================
    # Test 3: Missing B08 (NIR) Band
    # =========================================================================
    def test_3_missing_b08_nir(self):
        # 3-band RGB + SWIR instead of NIR
        cube = np.random.uniform(0.1, 0.5, (3, self.H, self.W)).astype(np.float32)
        res = SatelliteInputValidator.validate_input(
            cube,
            domain="agriculture",
            custom_affine=self.s2_affine,
            custom_crs=self.s2_crs
        )
        self.assertFalse(res.required_bands.get("B08_nir", False))
        self.assertFalse(res.capabilities["agriculture"]["supported"])
        self.assertFalse(res.capabilities["water"]["supported"])
        self.assertFalse(res.capabilities["sr"]["supported"])

    # =========================================================================
    # Test 4: Missing CRS (Non-Georeferenced)
    # =========================================================================
    def test_4_missing_crs(self):
        res = SatelliteInputValidator.validate_input(
            self.s2_cube,
            domain="super_resolution",
            custom_affine=self.s2_affine,
            custom_crs=None
        )
        self.assertFalse(res.georeferenced)
        self.assertFalse(res.capabilities["sr"]["supported"])
        self.assertFalse(res.capabilities["geospatial_measurement"]["supported"])
        self.assertFalse(res.capabilities["geojson_export"]["supported"])
        self.assertIn("Geospatial processing unavailable", res.capabilities["geospatial_measurement"]["reason"])
        # Image preview is still allowed
        self.assertTrue(res.capabilities["preview"]["supported"])

    # =========================================================================
    # Test 5: Missing Geotransform / Affine
    # =========================================================================
    def test_5_missing_geotransform(self):
        res = SatelliteInputValidator.validate_input(
            self.s2_cube,
            domain="super_resolution",
            custom_affine=None,
            custom_crs=self.s2_crs
        )
        self.assertFalse(res.georeferenced)
        self.assertFalse(res.capabilities["sr"]["supported"])
        self.assertFalse(res.capabilities["geospatial_measurement"]["supported"])

    # =========================================================================
    # Test 6: Invalid GSD (Outside S2 Tolerance e.g. 30m Landsat or 100m Coarse)
    # =========================================================================
    def test_6_invalid_gsd_landsat_30m(self):
        # 30m Landsat GSD
        landsat_affine = Affine.translation(750000.0, 1450000.0) * Affine.scale(30.0, -30.0)
        res = SatelliteInputValidator.validate_input(
            self.s2_cube,
            domain="super_resolution",
            custom_affine=landsat_affine,
            custom_crs=self.s2_crs
        )
        self.assertEqual(res.level, "LIMITED")
        self.assertAlmostEqual(res.gsd, 30.0, delta=0.1)
        self.assertEqual(res.detected_sensor, "Landsat 8/9")
        self.assertFalse(res.capabilities["sr"]["supported"])
        self.assertIn("Detected native GSD is 30.00m; current SR model expects approximately 10m Sentinel-2 input", res.capabilities["sr"]["reason"])
        # Agriculture / NDVI is still supported on 30m multispectral
        self.assertTrue(res.capabilities["agriculture"]["supported"])
        self.assertTrue(res.capabilities["water"]["supported"])

    def test_6b_invalid_gsd_coarse_100m(self):
        # 100m coarse GSD
        coarse_affine = Affine.translation(750000.0, 1450000.0) * Affine.scale(100.0, -100.0)
        res = SatelliteInputValidator.validate_input(
            self.s2_cube,
            domain="super_resolution",
            custom_affine=coarse_affine,
            custom_crs=self.s2_crs
        )
        self.assertEqual(res.level, "UNSUPPORTED")
        self.assertFalse(res.is_valid)
        self.assertFalse(res.capabilities["sr"]["supported"])
        self.assertTrue(any("GSD" in r for r in res.reasons))

    # =========================================================================
    # Test 7: NaN / Inf Values Rejection
    # =========================================================================
    def test_7_nan_and_inf_values(self):
        corrupted_nan = self.s2_cube.copy()
        corrupted_nan[0, 5, 5] = np.nan
        res_nan = SatelliteInputValidator.validate_input(
            corrupted_nan,
            domain="super_resolution",
            custom_affine=self.s2_affine,
            custom_crs=self.s2_crs
        )
        self.assertEqual(res_nan.level, "UNSUPPORTED")
        self.assertFalse(res_nan.reflectance_valid)
        self.assertTrue(any("NaN" in err for err in res_nan.errors))

        corrupted_inf = self.s2_cube.copy()
        corrupted_inf[1, 10, 10] = np.inf
        res_inf = SatelliteInputValidator.validate_input(
            corrupted_inf,
            domain="super_resolution",
            custom_affine=self.s2_affine,
            custom_crs=self.s2_crs
        )
        self.assertEqual(res_inf.level, "UNSUPPORTED")
        self.assertFalse(res_inf.reflectance_valid)
        self.assertTrue(any("Infinite" in err for err in res_inf.errors))

    # =========================================================================
    # Test 8: Unsupported Raster Format / Corrupt File
    # =========================================================================
    def test_8_unsupported_raster_format(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"NOT_A_VALID_RASTER_FILE_CONTENT")
            corrupt_path = Path(f.name)

        try:
            res = SatelliteInputValidator.validate_input(corrupt_path, domain="super_resolution")
            self.assertEqual(res.level, "UNSUPPORTED")
            self.assertFalse(res.valid)
            self.assertFalse(res.is_valid)
        finally:
            if corrupt_path.exists():
                corrupt_path.unlink()

    # =========================================================================
    # Test 9: Other Multispectral Raster (PlanetScope & Landsat)
    # =========================================================================
    def test_9_other_multispectral_sensor_capabilities(self):
        # PlanetScope 3.0m GeoTIFF simulated with PlanetScope tags
        with tempfile.NamedTemporaryFile(suffix="_planetscope_ortho.tif", delete=False) as f:
            temp_ps_path = Path(f.name)

        try:
            with rasterio.open(
                temp_ps_path,
                "w",
                driver="GTiff",
                height=self.H,
                width=self.W,
                count=4,
                dtype="float32",
                crs="EPSG:32643",
                transform=Affine.translation(750000.0, 1450000.0) * Affine.scale(3.0, -3.0)
            ) as dst:
                dst.write(self.s2_cube)
                dst.update_tags(MISSION_ID="PlanetScope", SENSOR_ID="PS2")

            res = SatelliteInputValidator.validate_input(
                temp_ps_path,
                domain="super_resolution"
            )
            self.assertEqual(res.detected_sensor, "PlanetScope")
            self.assertFalse(res.capabilities["sr"]["supported"])
            # Downstream multi-spectral intelligence IS available
            self.assertTrue(res.capabilities["agriculture"]["supported"])
            self.assertTrue(res.capabilities["water"]["supported"])
            self.assertTrue(res.capabilities["urban"]["supported"])
            self.assertTrue(res.capabilities["geospatial_measurement"]["supported"])
            self.assertIn("SENTINEL-2 SR MODEL: NOT COMPATIBLE", res.get_sr_compatibility_message())
        finally:
            if temp_ps_path.exists():
                temp_ps_path.unlink()

    # =========================================================================
    # Test 10: Band Identification from Metadata (Descriptions, Tags, Wavelengths)
    # =========================================================================
    def test_10_band_identification_from_metadata(self):
        # Case A: Named descriptions
        descs = ["B02", "B03", "B04", "B08"]
        bands, b_map, reqs, conf = SatelliteInputValidator.identify_bands(
            band_count=4,
            descriptions=descs,
            gsd=10.0
        )
        self.assertTrue(conf)
        self.assertEqual(b_map["blue"], 0)
        self.assertEqual(b_map["green"], 1)
        self.assertEqual(b_map["red"], 2)
        self.assertEqual(b_map["nir"], 3)
        self.assertTrue(all(reqs.values()))

        # Case B: Wavelength metadata tags
        band_tags = [
            {"WAVELENGTH": "490nm"},
            {"WAVELENGTH": "560nm"},
            {"WAVELENGTH": "665nm"},
            {"WAVELENGTH": "842nm"}
        ]
        bands_wl, b_map_wl, reqs_wl, conf_wl = SatelliteInputValidator.identify_bands(
            band_count=4,
            band_tags=band_tags,
            gsd=10.0
        )
        self.assertTrue(conf_wl)
        self.assertEqual(b_map_wl["blue"], 0)
        self.assertEqual(b_map_wl["green"], 1)
        self.assertEqual(b_map_wl["red"], 2)
        self.assertEqual(b_map_wl["nir"], 3)

    # =========================================================================
    # Test 11: Comprehensive Dynamic Capability Matrix
    # =========================================================================
    def test_11_dynamic_capability_matrix(self):
        # Case A: Valid S2 10m
        caps_s2 = SatelliteInputValidator.get_capabilities(
            valid=True,
            detected_sensor="Sentinel-2",
            gsd=10.0,
            band_count=4,
            band_map={"blue": 0, "green": 1, "red": 2, "nir": 3},
            georeferenced=True,
            crs="EPSG:32643",
            reflectance_valid=True,
            dimensions={"width": 512, "height": 512},
            confidence_identified=True
        )
        self.assertTrue(caps_s2["sr"]["supported"])
        self.assertTrue(caps_s2["urban"]["supported"])
        self.assertTrue(caps_s2["agriculture"]["supported"])
        self.assertTrue(caps_s2["water"]["supported"])
        self.assertTrue(caps_s2["disaster"]["supported"])
        self.assertTrue(caps_s2["oil_spill"]["supported"])
        self.assertTrue(caps_s2["geospatial_measurement"]["supported"])

        # Case B: RGB-Only
        caps_rgb = SatelliteInputValidator.get_capabilities(
            valid=True,
            detected_sensor="RGB Optical",
            gsd=10.0,
            band_count=3,
            band_map={"red": 0, "green": 1, "blue": 2},
            georeferenced=True,
            crs="EPSG:32643",
            reflectance_valid=True,
            dimensions={"width": 512, "height": 512},
            confidence_identified=True
        )
        self.assertFalse(caps_rgb["sr"]["supported"])
        self.assertFalse(caps_rgb["agriculture"]["supported"])
        self.assertFalse(caps_rgb["water"]["supported"])
        self.assertTrue(caps_rgb["preview"]["supported"])
        self.assertTrue(caps_rgb["urban"]["supported"])

    # =========================================================================
    # Test 12: Dynamic UI Capability State / API Serialized Schema
    # =========================================================================
    def test_12_dynamic_ui_api_response_schema(self):
        res = SatelliteInputValidator.validate_input(
            self.s2_cube,
            domain="super_resolution",
            custom_affine=self.s2_affine,
            custom_crs=self.s2_crs
        )
        d = res.to_dict()
        required_keys = [
            "valid", "is_valid", "status", "level", "detected_sensor",
            "detected_product", "gsd", "dimensions", "bands", "band_map",
            "required_bands", "crs", "georeferenced", "reflectance_valid",
            "capabilities", "warnings", "errors", "summary_message",
            "sr_compatibility_message"
        ]
        for k in required_keys:
            self.assertIn(k, d, f"Missing required API key: {k}")

        json_str = json.dumps(d)
        self.assertTrue(len(json_str) > 0)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["level"], "READY")
        self.assertTrue(parsed["capabilities"]["sr"]["supported"])


if __name__ == "__main__":
    unittest.main()
