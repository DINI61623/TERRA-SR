import os
import json
import zipfile
from pathlib import Path
import pytest
import numpy as np

from app.satellite_enhancer import generate_layer_assets, DEFAULT_INPUT_TIFF, OUTPUTS_DIR, STATIC_DIR
from src.core.analysis_result import AnalysisResult
from src.reporting.report_generator import ScientificReportGenerator
from src.reporting.video_generator import MissionVideoGenerator
from src.reporting.narrator import ScientificNarrator
from src.reporting.package_exporter import ResearchPackageExporter


class TestEndToEndMission:
    def test_full_mission_pipeline_and_deliverables(self):
        # 1. Execute full layer asset generation and canonical AnalysisResult construction
        res = generate_layer_assets(DEFAULT_INPUT_TIFF, model_name="ResidualCNN")
        assert res is not None
        assert "psnr" in res
        assert "mission_id" in res
        assert res["psnr"] > 35.0

        # Verify output files
        metrics_json = OUTPUTS_DIR / "metrics.json"
        metrics_csv = OUTPUTS_DIR / "metrics.csv"
        exp_config = OUTPUTS_DIR / "experiment_config.json"
        diff_map = OUTPUTS_DIR / "difference_map.png"
        enhanced_tiff = OUTPUTS_DIR / "s2_enhanced_residualcnn.tiff"

        assert metrics_json.exists(), "metrics.json must be generated"
        assert metrics_csv.exists(), "metrics.csv must be generated"
        assert exp_config.exists(), "experiment_config.json must be generated"
        assert diff_map.exists(), "difference_map.png must be generated"
        assert enhanced_tiff.exists(), "Enhanced GeoTIFF must be generated"

        # 2. Verify Canonical AnalysisResult Data Contract
        with open(metrics_json, "r") as f:
            data = json.load(f)
            analysis = AnalysisResult.from_dict(data)
            assert analysis.mission_id == res["mission_id"]
            assert analysis.metrics.psnr == res["psnr"]
            assert analysis.difference.mean_absolute_diff >= 0.0
            assert analysis.evidence_level == "OPERATIONAL / NO HIGH-RESOLUTION REFERENCE"

        # 3. Generate Scientific Research PDF Report
        report_pdf = OUTPUTS_DIR / "terra_sr_research_report.pdf"
        report_path = ScientificReportGenerator(analysis).generate(report_pdf)
        assert report_path.exists()
        assert report_path.stat().st_size > 1000
        with open(report_path, "rb") as f:
            assert f.read(5) == b"%PDF-"

        # 4. Generate Mission Replay Video (Quick 6-scene test)
        video_mp4 = OUTPUTS_DIR / "terra_sr_mission_replay.mp4"
        video_gen = MissionVideoGenerator(analysis, width=640, height=360, fps=10)
        v_path = video_gen.generate(video_mp4)
        assert v_path.exists()
        assert v_path.stat().st_size > 5000

        # 5. Generate Multi-Mode Scientific Narration Script
        script_txt = ScientificNarrator(analysis).generate_script(mode="Researcher")
        assert len(script_txt) > 200
        assert "TERRA-SR" in script_txt
        assert "Peak Signal-to-Noise Ratio" in script_txt or "PSNR" in script_txt

        # 6. Export Complete Research Package ZIP
        exporter = ResearchPackageExporter(analysis, base_output_dir=OUTPUTS_DIR)
        zip_path = exporter.build_package()
        assert zip_path.exists()
        assert zip_path.suffix == ".zip"

        # 7. Inspect ZIP contents
        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = zf.namelist()
            # Must contain original GeoTIFF, SR GeoTIFF, diff map, metrics, config, intelligence GeoJSONs, report PDF, narration, video, README
            assert any("README.txt" in n for n in namelist)
            assert any("original_10m.tif" in n for n in namelist)
            assert any("terra_sr_reconstruction.tif" in n for n in namelist)
            assert any("metrics.json" in n for n in namelist)
            assert any("research_report.pdf" in n for n in namelist)
            assert any("narration_script.txt" in n for n in namelist)
            assert any("mission_replay.mp4" in n for n in namelist)
            assert any("intelligence/" in n for n in namelist)

        print(f"\n[End-to-End Mission Verification] All 7 verification stages passed successfully! Package: {zip_path.name}")
