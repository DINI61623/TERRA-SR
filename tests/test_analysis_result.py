#!/usr/bin/env python3
"""
Unit tests for canonical AnalysisResult object
"""

import os
import json
import unittest
import tempfile
from pathlib import Path

from src.core.analysis_result import (
    AnalysisResult,
    MetricScores,
    UncertaintySummary,
    DifferenceAnalysis,
    IntelligenceSummary,
    EVIDENCE_OPERATIONAL_NO_REF,
    EVIDENCE_REFERENCE_VALIDATED
)


class TestAnalysisResult(unittest.TestCase):
    def setUp(self):
        self.result = AnalysisResult(
            mission_id="TEST_MISSION_001",
            source_type="COPERNICUS_CDSE",
            scene_id="S2B_TEST_SCENE",
            aoi_bbox=[77.58, 12.92, 77.68, 13.02],
            model_name="ResidualCNN",
            metrics=MetricScores(
                psnr=39.79,
                ssim=0.9541,
                sam=1.21,
                ergas=2.27,
                epi=0.9799,
                high_frequency_energy=99.2,
                ndvi_consistency=0.9982
            ),
            difference=DifferenceAnalysis(
                boundary_change_pct=14.2,
                high_frequency_change_pct=28.5,
                mean_absolute_diff=0.0142
            ),
            intelligence=IntelligenceSummary(
                domain="water",
                detected_features=["Shoreline", "Waterbody"],
                area_km2=1.45,
                feature_count=3
            ),
            evidence_level=EVIDENCE_OPERATIONAL_NO_REF
        )

    def test_dict_serialization_roundtrip(self):
        d = self.result.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["mission_id"], "TEST_MISSION_001")
        self.assertEqual(d["metrics"]["psnr"], 39.79)

        reconstructed = AnalysisResult.from_dict(d)
        self.assertEqual(reconstructed.mission_id, "TEST_MISSION_001")
        self.assertEqual(reconstructed.metrics.ssim, 0.9541)
        self.assertEqual(reconstructed.intelligence.domain, "water")

    def test_json_and_csv_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "metrics.json"
            csv_path = Path(tmpdir) / "metrics.csv"

            self.result.save_json(json_path)
            self.assertTrue(json_path.exists())

            self.result.save_csv(csv_path)
            self.assertTrue(csv_path.exists())

            with open(json_path, "r") as f:
                loaded = json.load(f)
                self.assertEqual(loaded["model_name"], "ResidualCNN")

            with open(csv_path, "r") as f:
                content = f.read()
                self.assertIn("TEST_MISSION_001", content)
                self.assertIn("PSNR (dB)", content)

    def test_reproducibility_config(self):
        cfg = self.result.get_reproducibility_config()
        self.assertEqual(cfg["mission_id"], "TEST_MISSION_001")
        self.assertIn("reproduction_command", cfg)
        self.assertIn("python cli.py", cfg["reproduction_command"])


if __name__ == "__main__":
    unittest.main()
