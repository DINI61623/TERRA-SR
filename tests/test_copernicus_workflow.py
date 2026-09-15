#!/usr/bin/env python3
"""
Unit tests for Copernicus Authentication, Scene Search, Ranking, and Error Handling.
All external network interactions are mocked to ensure deterministic, isolated execution.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

from src.satellite.copernicus_auth import CopernicusAuthManager
from src.satellite.catalog import search_copernicus_catalog
from src.satellite.scene_ranker import SceneRanker
from src.core.error_handler import create_error_response, create_success_response


class TestCopernicusWorkflow(unittest.TestCase):
    def setUp(self):
        CopernicusAuthManager.disconnect()

    def tearDown(self):
        CopernicusAuthManager.disconnect()

    def test_copernicus_status_disconnected_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            status = CopernicusAuthManager.get_status()
            self.assertFalse(status["connected"])
            self.assertIn(status["status"], ["DISCONNECTED", "UNCONFIGURED"])

    @patch("requests.post")
    def test_copernicus_connect_success_mock(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "access_token": "mock_valid_bearer_token_12345",
            "expires_in": 3600
        }
        mock_post.return_value = mock_resp

        with patch.dict(os.environ, {
            "COPERNICUS_CLIENT_ID": "mock-client-id",
            "COPERNICUS_CLIENT_SECRET": "mock-client-secret"
        }):
            res = CopernicusAuthManager.connect()
            self.assertEqual(res["status"], "CONNECTED")
            self.assertTrue(res["connected"])
            self.assertIn("Sentinel-2 L2A", res["access"])

            status = CopernicusAuthManager.get_status()
            self.assertTrue(status["connected"])
            self.assertEqual(status["status"], "CONNECTED")

            # Test Disconnect
            disc = CopernicusAuthManager.disconnect()
            self.assertEqual(disc["status"], "DISCONNECTED")
            self.assertFalse(disc["connected"])

    def test_scene_ranking_and_best_recommendation(self):
        candidate_scenes = [
            {
                "id": "S2_SCENE_CLOUDY",
                "product_id": "S2A_MSIL2A_20260115_CLOUDY",
                "cloud_cover": 45.0,
                "aoi_coverage": 90.0,
                "datetime": "2026-01-15T05:00:00Z"
            },
            {
                "id": "S2_SCENE_CLEAR_BEST",
                "product_id": "S2B_MSIL2A_20260211_CLEAR",
                "cloud_cover": 1.2,
                "aoi_coverage": 99.5,
                "datetime": "2026-02-11T05:00:00Z"
            },
            {
                "id": "S2_SCENE_MODERATE",
                "product_id": "S2A_MSIL2A_20260205_MODERATE",
                "cloud_cover": 12.0,
                "aoi_coverage": 85.0,
                "datetime": "2026-02-05T05:00:00Z"
            }
        ]

        ranked = SceneRanker.rank_scenes(candidate_scenes, target_date="2026-02-15")
        self.assertEqual(len(ranked), 3)
        self.assertEqual(ranked[0]["id"], "S2_SCENE_CLEAR_BEST")
        self.assertTrue(ranked[0]["recommended"])
        self.assertGreaterEqual(ranked[0]["suitability_score"], 85.0)
        self.assertEqual(ranked[0]["suitability_tier"], "Optimal")

    def test_structured_error_envelopes(self):
        err = create_error_response(
            stage="scene_retrieval",
            message="Copernicus token expired or credentials missing.",
            suggestion="Verify configuration or use Demo Mission.",
            retryable=True
        )
        self.assertEqual(err["status"], "ERROR")
        self.assertEqual(err["stage"], "scene_retrieval")
        self.assertTrue(err["retryable"])
        self.assertIn("Demo Mission", err["suggestion"])

        succ = create_success_response(
            stage="preprocessing",
            data={"bands": ["B02", "B03", "B04", "B08"], "gsd": 10.0}
        )
        self.assertEqual(succ["status"], "SUCCESS")
        self.assertEqual(succ["stage"], "preprocessing")
        self.assertEqual(len(succ["bands"]), 4)


if __name__ == "__main__":
    unittest.main()
