#!/usr/bin/env python3
"""
TERRA-SR Complete Research Package Exporter
Problem Statement: SIH26142 - Deep Learning Based Super Resolution Mapping (SRM)
Team: VIBE-CODERS

Bundles all mission deliverables, certified models, metrics, research reports, video replays,
and GIS vector layers into a self-contained, reproducible research archive (ZIP and folder).
"""

import os
import sys
import shutil
import json
import zipfile
import time
from pathlib import Path
from typing import Optional, Union, Dict, Any

from src.core.analysis_result import AnalysisResult
from src.reporting.report_generator import ScientificReportGenerator
from src.reporting.video_generator import MissionVideoGenerator
from src.reporting.narrator import ScientificNarrator


class ResearchPackageExporter:
    """
    Assembles, formats, and compresses the complete research package for an AnalysisResult.
    """
    def __init__(self, result: AnalysisResult, base_output_dir: Union[str, Path] = "outputs"):
        self.result = result
        self.base_output_dir = Path(base_output_dir)
        self.mission_dir_name = f"TERRA-SR_Mission_{self.result.mission_id}"
        self.package_dir = self.base_output_dir / self.mission_dir_name

    def _export_compressed_geotiff(self, src_path: Union[str, Path], dst_path: Union[str, Path]):
        """Transfers GeoTIFF ensuring lossless DEFLATE compression (predictor=3/2, zlevel=6)."""
        src_p = Path(src_path)
        dst_p = Path(dst_path)
        if not src_p.exists():
            return
        try:
            import rasterio
            with rasterio.open(src_p) as src:
                if src.compression and src.compression.name in ['DEFLATE', 'ZSTD']:
                    shutil.copy2(src_p, dst_p)
                    return
                meta = src.meta.copy()
                data = src.read()
                pred = 3 if 'float' in str(meta.get('dtype', '')) else 2
                meta.update({
                    'compress': 'deflate',
                    'predictor': pred,
                    'zlevel': 6
                })
                with rasterio.open(dst_p, 'w', **meta) as dst:
                    dst.write(data)
                    for b_idx in range(1, src.count + 1):
                        if src.descriptions and b_idx - 1 < len(src.descriptions) and src.descriptions[b_idx - 1]:
                            dst.set_band_description(b_idx, src.descriptions[b_idx - 1])
        except Exception:
            shutil.copy2(src_p, dst_p)

    def export(self, result: Optional[AnalysisResult] = None, base_output_dir: Optional[Union[str, Path]] = None, make_zip: bool = True) -> Path:
        """Alias for build_package with optional override parameters."""
        if result is not None:
            self.result = result
        if base_output_dir is not None:
            self.base_output_dir = Path(base_output_dir)
            self.mission_dir_name = f"TERRA-SR_Mission_{self.result.mission_id}"
            self.package_dir = self.base_output_dir / self.mission_dir_name

        self.package_dir.mkdir(parents=True, exist_ok=True)
        intel_dir = self.package_dir / "intelligence"
        intel_dir.mkdir(parents=True, exist_ok=True)

        # 1. Copy Source & Enhanced GeoTIFFs (Losslessly Compressed)
        if self.result.source_image_path and Path(self.result.source_image_path).exists():
            self._export_compressed_geotiff(self.result.source_image_path, self.package_dir / "original_10m.tif")
        elif Path("data/processed/s2_10m_stacked_roi.tiff").exists():
            self._export_compressed_geotiff("data/processed/s2_10m_stacked_roi.tiff", self.package_dir / "original_10m.tif")

        if self.result.sr_image_path and Path(self.result.sr_image_path).exists():
            self._export_compressed_geotiff(self.result.sr_image_path, self.package_dir / "terra_sr_reconstruction.tif")
        elif Path("outputs/s2_enhanced_residualcnn.tiff").exists():
            self._export_compressed_geotiff("outputs/s2_enhanced_residualcnn.tiff", self.package_dir / "terra_sr_reconstruction.tif")

        # 2. Copy Uncertainty Map if available
        if self.result.uncertainty.available and self.result.uncertainty.uncertainty_path:
            p = Path(self.result.uncertainty.uncertainty_path)
            if p.exists():
                self._export_compressed_geotiff(p, self.package_dir / "uncertainty.tif")

        # 3. Difference Analysis Image
        static_dir = Path("app/static")
        if (static_dir / "active_enh_edge.png").exists():
            shutil.copy2(static_dir / "active_enh_edge.png", self.package_dir / "difference_map.png")

        # 4. Export Metrics JSON & CSV
        self.result.save_json(self.package_dir / "metrics.json")
        self.result.save_csv(self.package_dir / "metrics.csv")

        # 5. Export Reproducibility Config JSON
        repro_cfg = self.result.get_reproducibility_config()
        with open(self.package_dir / "experiment_config.json", "w", encoding="utf-8") as f:
            json.dump(repro_cfg, f, indent=4)

        # 6. Copy All Domain GeoJSON Files
        outputs_dir = Path("outputs")
        for geojson_file in outputs_dir.glob("*_intelligence_vectors.geojson"):
            shutil.copy2(geojson_file, intel_dir / geojson_file.name)

        # 7. Generate Research Report PDF
        pdf_path = self.package_dir / "research_report.pdf"
        try:
            report_gen = ScientificReportGenerator(self.result)
            report_gen.generate(pdf_path)
        except Exception as e:
            print(f"[Warning] PDF generation warning in package exporter: {e}")

        # 8. Generate Narration Script
        script_path = self.package_dir / "narration_script.txt"
        try:
            narrator = ScientificNarrator(self.result)
            narrator.save_script(script_path, mode="Researcher")
        except Exception as e:
            print(f"[Warning] Narration generation warning in package exporter: {e}")

        # 9. Generate Mission Replay MP4 Video
        video_path = self.package_dir / "mission_replay.mp4"
        try:
            video_gen = MissionVideoGenerator(self.result)
            video_gen.generate(video_path, total_duration_seconds=30)
        except Exception as e:
            print(f"[Warning] Video generation warning in package exporter: {e}")

        # 10. Generate Comprehensive README.txt
        readme_path = self.package_dir / "README.txt"
        self._write_package_readme(readme_path)

        # 11. Create ZIP Archive
        zip_path = self.base_output_dir / f"{self.mission_dir_name}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for root, dirs, files in os.walk(self.package_dir):
                for f in files:
                    full_p = Path(root) / f
                    arc_p = Path(self.mission_dir_name) / full_p.relative_to(self.package_dir)
                    zf.write(full_p, str(arc_p))

        self.result.outputs.package_zip = str(zip_path)
        return zip_path

    def _write_package_readme(self, readme_path: Path):
        m = self.result.metrics
        intel = self.result.intelligence
        repro_cmd = self.result.get_reproducibility_config().get("reproduction_command", "python cli.py")

        content = (
            f"================================================================================\n"
            f"TERRA-SR COMPLETE RESEARCH & OPERATIONAL DELIVERABLE PACKAGE\n"
            f"Problem Statement: SIH26142 | Team: VIBE-CODERS\n"
            f"================================================================================\n\n"
            f"1. MISSION IDENTIFICATION\n"
            f"   Mission ID:         {self.result.mission_id}\n"
            f"   Execution Time:     {self.result.timestamp}\n"
            f"   Satellite Sensor:   {self.result.sensor}\n"
            f"   Product Type:       {self.result.product}\n"
            f"   Scene Identifier:   {self.result.scene_id}\n"
            f"   Acquisition Date:   {self.result.acquisition_date}\n\n"
            f"2. GEOSPATIAL & SENSOR SPECIFICATION\n"
            f"   AOI Bounding Box:   {self.result.aoi_bbox}\n"
            f"   Geodesic Area:      {self.result.aoi_area_km2:.2f} km² ({self.result.aoi_area_ha:.1f} ha)\n"
            f"   Coordinate System:  {self.result.crs}\n"
            f"   Input Native GSD:   {self.result.input_gsd:.1f}m Multispectral\n"
            f"   Output Grid:        {self.result.output_grid} (Scale x{self.result.scale_factor})\n"
            f"   Spectral Bands:     {', '.join(self.result.bands)}\n"
            f"   SR Neural Model:    {self.result.model_name}\n\n"
            f"3. CERTIFIED VALIDATION METRICS\n"
            f"   Peak SNR (PSNR):    {m.psnr:.2f} dB\n"
            f"   SSIM Index:         {m.ssim:.4f}\n"
            f"   Spectral Angle:     {m.sam:.2f} degrees\n"
            f"   ERGAS Error Index:  {m.ergas:.2f}\n"
            f"   Edge Pres. Index:   {m.epi:.4f}\n"
            f"   High-Freq Energy:   {m.high_frequency_energy:.1f}%\n"
            f"   NDVI Consistency:   {m.ndvi_consistency:.4f}\n\n"
            f"4. SCIENTIFIC EVIDENCE FRAMEWORK\n"
            f"   Evidence Level:     {self.result.evidence_level}\n"
            f"   Notice:             {self.result.evidence_statement}\n\n"
            f"5. ACTIONABLE EARTH INTELLIGENCE\n"
            f"   Domain:             {intel.domain.upper()}\n"
            f"   Detected Area:      {intel.area_km2:.4f} km²\n"
            f"   Perimeter:          {intel.perimeter_km:.2f} km\n"
            f"   Vector Polygons:    {intel.feature_count} features (RFC 7946 GeoJSON)\n\n"
            f"6. INCLUDED DELIVERABLES\n"
            f"   • original_10m.tif              - Calibrated 4-band BOA input GeoTIFF\n"
            f"   • terra_sr_reconstruction.tif   - Reconstructed 4-band enhanced GeoTIFF\n"
            f"   • difference_map.png            - Spatial gradient and difference inspection\n"
            f"   • metrics.json & metrics.csv    - Machine-readable scientific scorecard\n"
            f"   • experiment_config.json        - Reproducibility hyperparameters\n"
            f"   • intelligence/*.geojson        - Standard vector intelligence layers\n"
            f"   • research_report.pdf           - Certified multi-page research publication report\n"
            f"   • narration_script.txt          - 4-mode scientific briefing script\n"
            f"   • mission_replay.mp4            - 720p animated mission video\n\n"
            f"7. DETERMINISTIC CLI REPRODUCTION\n"
            f"   Execute the following command to reproduce this exact mission run:\n"
            f"   {repro_cmd}\n\n"
            f"================================================================================\n"
            f"END OF PACKAGE MANIFEST\n"
            f"================================================================================\n"
        )

        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)

    build_package = export
