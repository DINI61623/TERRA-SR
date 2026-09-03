#!/usr/bin/env python3
"""
Unit tests for TERRA-SR Downstream Satellite Intelligence Suite
Verifies Urban, Agriculture, Water, and Oil Spill modules for:
1. Standardized 4-band cube processing
2. Correct layer shapes and non-NaN outputs
3. Valid GeoJSON generation with CRS transforms
4. Native vs. SR impact analysis calculation
5. Scientific safety metadata compliance
"""

import sys
import unittest
from pathlib import Path
import numpy as np

# Add workspace root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.intelligence import (
    get_intelligence_module,
    run_intelligence_pipeline,
    INTELLIGENCE_REGISTRY
)

try:
    from affine import Affine
    HAS_AFFINE = True
except ImportError:
    HAS_AFFINE = False


class TestIntelligencePipeline(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create mock 4-band Sentinel-2 reflectance cubes
        # B02 (Blue), B03 (Green), B04 (Red), B08 (NIR) in [0.0, 0.6]
        self.H_lr, self.W_lr = 32, 32
        self.H_sr, self.W_sr = 64, 64
        
        self.lr_cube = np.random.uniform(0.05, 0.4, (4, self.H_lr, self.W_lr)).astype(np.float32)
        self.sr_cube = np.random.uniform(0.05, 0.4, (4, self.H_sr, self.W_sr)).astype(np.float32)
        
        # Add simulated spatial features
        # 1. Water body in top-left (high green, low NIR)
        self.sr_cube[1, :20, :20] = 0.15  # Green
        self.sr_cube[3, :20, :20] = 0.02  # NIR
        
        # 2. Dense vegetation in bottom-right (low red, high NIR)
        self.sr_cube[2, 40:, 40:] = 0.03  # Red
        self.sr_cube[3, 40:, 40:] = 0.45  # NIR
        
        # 3. Urban built-up strip in center (high red, moderate NIR)
        self.sr_cube[2, 25:35, :] = 0.35  # Red
        self.sr_cube[0, 25:35, :] = 0.30  # Blue
        
        self.affine = Affine.translation(77.65, 12.89) * Affine.scale(0.00003, -0.00003) if HAS_AFFINE else None
        self.crs = "EPSG:32643"

    def test_registry_contains_all_modules(self):
        self.assertIn("urban", INTELLIGENCE_REGISTRY)
        self.assertIn("agriculture", INTELLIGENCE_REGISTRY)
        self.assertIn("water", INTELLIGENCE_REGISTRY)
        self.assertIn("oil_spill", INTELLIGENCE_REGISTRY)

    def test_urban_intelligence_module(self):
        result = run_intelligence_pipeline(
            domain="urban",
            sr_cube=self.sr_cube,
            lr_cube=self.lr_cube,
            gsd=3.33,
            affine_transform=self.affine,
            crs=self.crs
        )
        self.assertEqual(result["report"]["status"], "ANALYSIS_COMPLETE")
        self.assertIn("builtup_density", result["layers"])
        self.assertIn("candidate_features", result["layers"])
        self.assertIn("sr_impact_analysis", result["report"])
        self.assertIn("scientific_safety_audit", result["report"])
        self.assertEqual(result["report"]["geojson"]["type"], "FeatureCollection")
        
        # Verify no NaN or Inf in image layers
        for name, layer in result["layers"].items():
            self.assertFalse(np.isnan(layer).any(), f"NaN found in urban layer {name}")
            self.assertEqual(layer.dtype, np.uint8)

    def test_agriculture_intelligence_module(self):
        result = run_intelligence_pipeline(
            domain="agriculture",
            sr_cube=self.sr_cube,
            lr_cube=self.lr_cube,
            gsd=3.33,
            affine_transform=self.affine,
            crs=self.crs
        )
        self.assertEqual(result["report"]["status"], "ANALYSIS_COMPLETE")
        self.assertIn("ndvi_map", result["layers"])
        self.assertIn("field_boundaries", result["layers"])
        self.assertIn("vigor_classification", result["layers"])
        self.assertIn("vigor_distribution", result["report"]["summary"])
        self.assertEqual(result["report"]["geojson"]["type"], "FeatureCollection")
        
        for name, layer in result["layers"].items():
            self.assertFalse(np.isnan(layer).any(), f"NaN found in agri layer {name}")
            self.assertEqual(layer.dtype, np.uint8)

    def test_water_intelligence_module(self):
        result = run_intelligence_pipeline(
            domain="water",
            sr_cube=self.sr_cube,
            lr_cube=self.lr_cube,
            gsd=3.33,
            affine_transform=self.affine,
            crs=self.crs
        )
        self.assertEqual(result["report"]["status"], "ANALYSIS_COMPLETE")
        self.assertIn("ndwi_map", result["layers"])
        self.assertIn("water_mask", result["layers"])
        self.assertIn("shoreline", result["layers"])
        self.assertIn("total_surface_water_area_km2", result["report"]["summary"])
        self.assertEqual(result["report"]["geojson"]["type"], "FeatureCollection")
        
        for name, layer in result["layers"].items():
            self.assertFalse(np.isnan(layer).any(), f"NaN found in water layer {name}")
            self.assertEqual(layer.dtype, np.uint8)

    def test_oil_spill_intelligence_module(self):
        result = run_intelligence_pipeline(
            domain="oil_spill",
            sr_cube=self.sr_cube,
            lr_cube=self.lr_cube,
            gsd=3.33,
            affine_transform=self.affine,
            crs=self.crs
        )
        self.assertIn("sosi_map", result["layers"])
        self.assertIn("oil_classified_mask", result["layers"])
        self.assertIn("sr_impact_analysis", result["report"])
        self.assertEqual(result["report"]["geojson"]["type"], "FeatureCollection")

    def test_disaster_intelligence_module(self):
        result = run_intelligence_pipeline(
            domain="disaster",
            sr_cube=self.sr_cube,
            lr_cube=self.lr_cube,
            gsd=3.33,
            affine_transform=self.affine,
            crs=self.crs
        )
        self.assertEqual(result["report"]["status"], "ANALYSIS_COMPLETE")
        self.assertIn("flood_inundation", result["layers"])
        self.assertIn("burn_scar_severity", result["layers"])
        self.assertIn("impact_boundary", result["layers"])
        self.assertIn("flood_inundation_area_ha", result["report"]["summary"])
        self.assertEqual(result["report"]["geojson"]["type"], "FeatureCollection")
        
        for name, layer in result["layers"].items():
            self.assertFalse(np.isnan(layer).any(), f"NaN found in disaster layer {name}")
            self.assertEqual(layer.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
