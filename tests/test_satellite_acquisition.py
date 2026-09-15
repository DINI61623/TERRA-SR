#!/usr/bin/env python3
"""
Unit Tests for TERRA-SR Satellite Data Acquisition Layer (Copernicus CDSE Integration)
Verifies:
1. AOI bounding box validation, coordinate normalization, and error handling
2. Geodesic area calculation in km² and hectares
3. Scene ranker scoring mathematics, tier assignment, and explainability rationale
4. Copernicus catalog STAC query parsing and cloud filtering
5. AOI retrieval fallback under unauthenticated / Demo Mode conditions
"""

import unittest
from pathlib import Path
import numpy as np

from src.satellite.validators import (
    validate_bbox,
    compute_bbox_area_km2,
    format_aoi_summary,
    extract_bbox_from_geometry,
    AOIValidationError
)
from src.satellite.scene_ranker import SceneRanker
from src.satellite.catalog import search_copernicus_catalog, calculate_aoi_overlap_percentage
from src.satellite.process import retrieve_aoi_raster, build_process_api_payload
from src.satellite.copernicus_auth import CopernicusAuthManager


class TestSatelliteAcquisition(unittest.TestCase):

    def setUp(self):
        # Bengaluru BBOX: [minLon, minLat, maxLon, maxLat]
        self.bengaluru_bbox = [77.65, 12.82, 77.72, 12.89]

    def test_validate_bbox_valid(self):
        valid = validate_bbox(self.bengaluru_bbox)
        self.assertEqual(len(valid), 4)
        self.assertAlmostEqual(valid[0], 77.65)
        self.assertAlmostEqual(valid[1], 12.82)
        self.assertAlmostEqual(valid[2], 77.72)
        self.assertAlmostEqual(valid[3], 12.89)

    def test_validate_bbox_inverted_order_raises(self):
        # Inverted longitude: maxLon < minLon
        with self.assertRaises(AOIValidationError):
            validate_bbox([77.72, 12.82, 77.65, 12.89])

        # Inverted latitude: maxLat < minLat
        with self.assertRaises(AOIValidationError):
            validate_bbox([77.65, 12.89, 77.72, 12.82])

    def test_validate_bbox_out_of_range(self):
        # Longitude > 180
        with self.assertRaises(AOIValidationError):
            validate_bbox([185.0, 12.82, 190.0, 12.89])

        # Latitude > 90
        with self.assertRaises(AOIValidationError):
            validate_bbox([77.65, 95.0, 77.72, 98.0])

    def test_compute_bbox_area(self):
        area_km2 = compute_bbox_area_km2(self.bengaluru_bbox)
        # Bounding box ~7.7km x 7.8km should be around 50-70 km²
        self.assertGreater(area_km2, 40.0)
        self.assertLess(area_km2, 80.0)

    def test_format_aoi_summary(self):
        summary = format_aoi_summary(self.bengaluru_bbox)
        self.assertEqual(summary["status"], "VALID")
        self.assertIn("area_km2", summary)
        self.assertIn("center", summary)
        self.assertEqual(summary["center"]["lat"], 12.855)
        self.assertEqual(summary["center"]["lon"], 77.685)
        self.assertEqual(summary["geometry"]["type"], "Polygon")

    def test_extract_bbox_from_geojson_polygon(self):
        geom = {
            "type": "Polygon",
            "coordinates": [[
                [77.65, 12.82],
                [77.72, 12.82],
                [77.72, 12.89],
                [77.65, 12.89],
                [77.65, 12.82]
            ]]
        }
        extracted = extract_bbox_from_geometry(geom)
        self.assertEqual(extracted, self.bengaluru_bbox)

    def test_scene_ranker_scoring(self):
        optimal_scene = {
            "scene_id": "S2B_OPTIMAL",
            "cloud_cover": 1.2,
            "aoi_coverage": 98.5,
            "datetime": "2026-02-15T05:08:39Z"
        }
        suboptimal_scene = {
            "scene_id": "S2B_CLOUDY",
            "cloud_cover": 45.0,
            "aoi_coverage": 60.0,
            "datetime": "2025-01-01T05:08:39Z"
        }

        score_opt, tier_opt, bullets_opt = SceneRanker.calculate_suitability_score(optimal_scene)
        score_sub, tier_sub, bullets_sub = SceneRanker.calculate_suitability_score(suboptimal_scene)

        self.assertGreater(score_opt, score_sub)
        self.assertEqual(tier_opt, "Optimal")
        self.assertEqual(tier_sub, "Suboptimal")
        self.assertTrue(any("cloud" in b.lower() for b in bullets_opt))

    def test_search_copernicus_catalog_and_ranking(self):
        res = search_copernicus_catalog(
            bbox=self.bengaluru_bbox,
            start_date="2026-01-01",
            end_date="2026-03-01",
            max_cloud_cover=15.0,
            min_aoi_coverage=80.0
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertGreater(res["total_found"], 0)
        self.assertIsNotNone(res["recommended_scene"])
        self.assertTrue(res["recommended_scene"]["is_recommended"])
        self.assertIn("suitability_score", res["recommended_scene"])

    def test_build_process_api_payload(self):
        payload = build_process_api_payload(
            bbox=self.bengaluru_bbox,
            start_date="2026-01-01",
            end_date="2026-03-01",
            width=512,
            height=512
        )
        self.assertIn("input", payload)
        self.assertIn("evalscript", payload)
        self.assertIn("B02", payload["evalscript"])
        self.assertIn("B08", payload["evalscript"])
        self.assertEqual(payload["output"]["width"], 512)

    def test_retrieve_aoi_raster_demo_fallback(self):
        test_out = Path("outputs") / "test_aoi_crop.tiff"
        res = retrieve_aoi_raster(
            bbox=self.bengaluru_bbox,
            start_date="2026-01-01",
            end_date="2026-03-01",
            output_path=test_out,
            force_demo=True
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["source"], "DEMO_MODE_SENTINEL2")
        self.assertTrue(test_out.exists())
        self.assertEqual(len(res["bands"]), 4)


if __name__ == "__main__":
    unittest.main()
