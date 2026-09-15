#!/usr/bin/env python3
"""
Unit tests for Research PDF Report, Mission Video MP4, Narrator, and Package Exporter.
"""

import os
import unittest
import tempfile
import zipfile
from pathlib import Path

from src.core.analysis_result import (
    AnalysisResult,
    MetricScores,
    DifferenceAnalysis,
    IntelligenceSummary,
    EVIDENCE_OPERATIONAL_NO_REF
)
from src.reporting.report_generator import ScientificReportGenerator
from src.reporting.video_generator import MissionVideoGenerator
from src.reporting.narrator import ScientificNarrator
from src.reporting.package_exporter import ResearchPackageExporter


class TestReportingAndVideo(unittest.TestCase):
    def setUp(self):
        self.result = AnalysisResult(
            mission_id="TEST_MISSION_002",
            source_type="DEMO_MISSION",
            scene_id="S2B_MSIL2A_TEST",
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
                area_km2=0.0977,
                perimeter_km=4.25,
                feature_count=2
            ),
            evidence_level=EVIDENCE_OPERATIONAL_NO_REF
        )

    def test_pdf_report_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "test_report.pdf"
            gen = ScientificReportGenerator(self.result)
            gen.generate(pdf_path)

            self.assertTrue(pdf_path.exists())
            self.assertGreater(pdf_path.stat().st_size, 1000)

            # Check PDF Magic Header
            with open(pdf_path, "rb") as f:
                header = f.read(5)
                self.assertEqual(header, b"%PDF-")

    def test_narration_script_modes(self):
        narrator = ScientificNarrator(self.result)
        for mode in ["Researcher", "Judge", "Mission", "Beginner"]:
            script = narrator.generate_script(mode)
            self.assertIsInstance(script, str)
            self.assertIn("TERRA-SR", script)
            self.assertIn("TEST_MISSION_002", script)

    def test_mission_replay_video_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mp4_path = Path(tmpdir) / "test_replay.mp4"
            # Fast test: 5 seconds total duration
            video_gen = MissionVideoGenerator(self.result, width=640, height=360, fps=10)
            video_gen.generate(mp4_path, total_duration_seconds=5)

            self.assertTrue(mp4_path.exists())
            self.assertGreater(mp4_path.stat().st_size, 5000)

    def test_package_exporter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ResearchPackageExporter(self.result, base_output_dir=tmpdir)
            zip_path = exporter.build_package()

            self.assertTrue(zip_path.exists())
            self.assertTrue(zipfile.is_zipfile(zip_path))

            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
                self.assertTrue(any("metrics.json" in n for n in names))
                self.assertTrue(any("metrics.csv" in n for n in names))
                self.assertTrue(any("experiment_config.json" in n for n in names))
                self.assertTrue(any("research_report.pdf" in n for n in names))
                self.assertTrue(any("narration_script.txt" in n for n in names))
                self.assertTrue(any("mission_replay.mp4" in n for n in names))
                self.assertTrue(any("README.txt" in n for n in names))


if __name__ == "__main__":
    unittest.main()
