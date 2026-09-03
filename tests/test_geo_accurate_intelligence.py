#!/usr/bin/env python3
"""
TERRA-SR Geo-Accurate Intelligence & Strict Input Validation Test Suite.
Tests:
1. SatelliteInputValidator (valid S2, invalid RGB, missing B08, missing CRS, NaN/Inf rejection)
2. Georeferencing integrity & coordinate transformations
3. Water Intelligence (per-waterbody polygonization, area, perimeter, centroids, confidence, reference accuracy)
4. Flood Intelligence (temporal differencing, flood polygons, accuracy evaluation)
"""

import unittest
import numpy as np
from pathlib import Path

from src.core.input_validation import SatelliteInputValidator, ValidationResult
from src.core.georeference import (
    compute_polygon_metrics,
    transform_projected_to_latlon,
    verify_georeferencing_integrity
)
from src.intelligence.water import WaterIntelligenceModule
from src.intelligence.disaster import DisasterIntelligenceModule

try:
    from affine import Affine
    HAS_AFFINE = True
except ImportError:
    HAS_AFFINE = False


class TestGeoAccurateIntelligence(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Synthetic standardized 4-band Sentinel-2 cube (4, 128, 128)
        self.H, self.W = 128, 128
        self.sr_cube = np.random.uniform(0.05, 0.45, (4, self.H, self.W)).astype(np.float32)
        # Add high NIR for vegetation, high Green/low NIR for water
        self.sr_cube[1, 20:50, 20:50] = 0.35  # Green
        self.sr_cube[3, 20:50, 20:50] = 0.05  # NIR (Water body)
        
        self.affine = Affine.translation(750000.0, 1450000.0) * Affine.scale(3.33, -3.33) if HAS_AFFINE else None
        self.crs = "EPSG:32643"

    # ==========================================
    # 1. Input Validator Unit Tests
    # ==========================================
    def test_valid_sentinel2_input(self):
        res = SatelliteInputValidator.validate_for_domain(
            self.sr_cube,
            domain="super_resolution",
            custom_affine=self.affine,
            custom_crs=self.crs
        )
        self.assertTrue(res.is_valid)
        self.assertEqual(res.status, "VALID")
        self.assertIn("B08 (NIR)", res.metadata["band_names"])

    def test_invalid_rgb_only_rejection(self):
        rgb_cube = self.sr_cube[:3, :, :]  # Only 3 bands (Missing NIR)
        res = SatelliteInputValidator.validate_for_domain(
            rgb_cube,
            domain="super_resolution",
            custom_affine=self.affine,
            custom_crs=self.crs
        )
        self.assertFalse(res.is_valid)
        self.assertEqual(res.status, "INVALID")
        self.assertTrue(any("4-band" in r for r in res.reasons))

    def test_missing_crs_rejection(self):
        res = SatelliteInputValidator.validate_for_domain(
            self.sr_cube,
            domain="super_resolution",
            custom_affine=self.affine,
            custom_crs="NONE"
        )
        self.assertFalse(res.is_valid)
        self.assertTrue(any("CRS" in r for r in res.reasons))

    def test_nan_corrupted_data_rejection(self):
        corrupted = self.sr_cube.copy()
        corrupted[0, 10, 10] = np.nan
        res = SatelliteInputValidator.validate_for_domain(
            corrupted,
            domain="super_resolution",
            custom_affine=self.affine,
            custom_crs=self.crs
        )
        self.assertFalse(res.is_valid)
        self.assertTrue(any("NaN" in r for r in res.reasons))

    def test_gsd_out_of_bounds_rejection(self):
        weird_affine = Affine.translation(0, 0) * Affine.scale(100.0, -100.0) if HAS_AFFINE else None
        res = SatelliteInputValidator.validate_for_domain(
            self.sr_cube,
            domain="super_resolution",
            custom_affine=weird_affine,
            custom_crs=self.crs
        )
        self.assertFalse(res.is_valid)
        self.assertTrue(any("GSD" in r for r in res.reasons))

    # ==========================================
    # 2. Georeferencing & Geometry Tests
    # ==========================================
    def test_polygon_metrics_computation(self):
        coords = [
            [750000.0, 1450000.0],
            [750100.0, 1450000.0],
            [750100.0, 1450100.0],
            [750000.0, 1450100.0],
            [750000.0, 1450000.0]
        ]
        metrics = compute_polygon_metrics(coords, crs_str="EPSG:32643")
        self.assertAlmostEqual(metrics["area_m2"], 10000.0, delta=1.0)
        self.assertAlmostEqual(metrics["area_ha"], 1.0, delta=0.01)
        self.assertAlmostEqual(metrics["perimeter_m"], 400.0, delta=1.0)
        self.assertIn("lat", metrics["centroid_latlon"])
        self.assertIn("lon", metrics["centroid_latlon"])

    def test_georeferencing_integrity_audit(self):
        src_meta = {
            "crs": "EPSG:32643",
            "gsd": 10.0,
            "bounds": {"min_x": 750000.0, "max_x": 760000.0, "min_y": 1440000.0, "max_y": 1450000.0}
        }
        sr_meta = {
            "crs": "EPSG:32643",
            "gsd": 5.0,
            "bounds": {"min_x": 750000.0, "max_x": 760000.0, "min_y": 1440000.0, "max_y": 1450000.0}
        }
        audit = verify_georeferencing_integrity(src_meta, sr_meta, expected_scale_factor=2.0)
        self.assertEqual(audit["status"], "PASSED")
        self.assertTrue(audit["crs_preserved"])
        self.assertTrue(audit["spatial_extent_aligned"])

    # ==========================================
    # 3. Water Intelligence Geo-Accuracy Tests
    # ==========================================
    def test_water_polygon_extraction_and_confidence(self):
        mod = WaterIntelligenceModule()
        res = mod.process(
            sr_cube=self.sr_cube,
            gsd=3.33,
            affine_transform=self.affine,
            crs=self.crs
        )
        self.assertEqual(res["report"]["status"], "ANALYSIS_COMPLETE")
        geojson = res["geojson_full"]
        self.assertEqual(geojson["type"], "FeatureCollection")
        self.assertGreater(len(geojson["features"]), 0)
        
        first_feat = geojson["features"][0]
        self.assertIn("water_id", first_feat["properties"])
        self.assertIn("area_m2", first_feat["properties"])
        self.assertIn("centroid_lat", first_feat["properties"])
        self.assertIn(first_feat["properties"]["confidence"], ["HIGH", "MEDIUM", "LOW"])

    def test_water_accuracy_with_reference(self):
        mod = WaterIntelligenceModule()
        ref_mask = np.zeros((self.H, self.W), dtype=bool)
        ref_mask[20:50, 20:50] = True
        
        res = mod.process(
            sr_cube=self.sr_cube,
            gsd=3.33,
            affine_transform=self.affine,
            crs=self.crs,
            reference_mask=ref_mask
        )
        acc = res["report"]["accuracy_framework"]
        self.assertTrue(acc["reference_validated"])
        self.assertIsNotNone(acc["f1_score"])
        self.assertGreater(acc["f1_score"], 0.70)

    def test_water_accuracy_without_reference(self):
        mod = WaterIntelligenceModule()
        res = mod.process(
            sr_cube=self.sr_cube,
            gsd=3.33,
            affine_transform=self.affine,
            crs=self.crs,
            reference_mask=None
        )
        acc = res["report"]["accuracy_framework"]
        self.assertFalse(acc["reference_validated"])
        self.assertIsNone(acc["f1_score"])

    # ==========================================
    # 4. Flood Intelligence Tests
    # ==========================================
    def test_flood_differencing_mode(self):
        pre_cube = self.sr_cube.copy()
        pre_cube[1, 20:50, 20:50] = 0.15  # Pre-event dry
        pre_cube[3, 20:50, 20:50] = 0.35
        
        post_cube = self.sr_cube.copy()  # Post-event flooded
        
        mod = DisasterIntelligenceModule()
        res = mod.process(
            sr_cube=post_cube,
            pre_cube=pre_cube,
            gsd=3.33,
            affine_transform=self.affine,
            crs=self.crs
        )
        self.assertEqual(res["report"]["status"], "ANALYSIS_COMPLETE")
        self.assertIn("flood_inundation_area_ha", res["report"]["summary"])
        geojson = res["geojson_full"]
        self.assertEqual(geojson["type"], "FeatureCollection")
        if len(geojson["features"]) > 0:
            first = geojson["features"][0]
            self.assertIn("flood_id", first["properties"])
            self.assertIn("area_ha", first["properties"])


if __name__ == "__main__":
    unittest.main()
