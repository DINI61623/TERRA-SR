#!/usr/bin/env python3
"""
Unit tests for Headless Batch CLI and GIS Map payload schema.
"""

import unittest
import subprocess
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


class TestCliAndGis(unittest.TestCase):

    def test_cli_execution_water_domain(self):
        output_dir = ROOT_DIR / "outputs" / "test_cli_out"
        cmd = [
            "python",
            "cli.py",
            "--input", "outputs/s2_5m_upscaled_bilinear.tiff",
            "--model", "Bilinear",
            "--domain", "water",
            "--output_dir", str(output_dir),
            "--quiet"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT_DIR))
        self.assertEqual(res.returncode, 0, f"CLI failed: {res.stderr}")
        
        summary_file = output_dir / "mission_execution_summary.json"
        self.assertTrue(summary_file.exists())
        
        with open(summary_file) as f:
            summary = json.load(f)
            self.assertEqual(summary["platform"], "TERRA-SR Geospatial Intelligence Suite")
            self.assertIn("water", summary["intelligence_domains"])
            
        geojson_file = output_dir / "water_vectors.geojson"
        self.assertTrue(geojson_file.exists())
        with open(geojson_file) as f:
            geo = json.load(f)
            self.assertEqual(geo["type"], "FeatureCollection")

    def test_cli_invalid_input_rejection(self):
        # Create dummy text file disguised as tif
        dummy_file = ROOT_DIR / "outputs" / "dummy_corrupt.tiff"
        with open(dummy_file, "w") as f:
            f.write("not a geotiff")
            
        cmd = [
            "python",
            "cli.py",
            "--input", str(dummy_file),
            "--model", "ResidualCNN",
            "--domain", "water",
            "--quiet"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT_DIR))
        self.assertNotEqual(res.returncode, 0, "CLI should reject corrupt non-geotiff file")
        
        if dummy_file.exists():
            dummy_file.unlink()


if __name__ == "__main__":
    unittest.main()
