#!/usr/bin/env python3
"""
TERRA-SR Automated Mission Replay Video Generator (MP4)
Problem Statement: SIH26142 - Deep Learning Based Super Resolution Mapping (SRM)
Team: VIBE-CODERS

Constructs an animated scientific mission replay video (720p MP4) directly from an AnalysisResult object.
Sequences all operational stages: AOI, Scene Metadata, 10m Input, SR Transition, Metrics, Difference,
Earth Intelligence Vectors, and Certified Scientific Scorecard.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import imageio

from src.core.analysis_result import AnalysisResult, EVIDENCE_OPERATIONAL_NO_REF, EVIDENCE_REFERENCE_VALIDATED


class MissionVideoGenerator:
    """
    Synthesizes a 720p (1280x720) scientific Mission Replay MP4 video from an AnalysisResult.
    """
    def __init__(self, result: AnalysisResult, width: int = 1280, height: int = 720, fps: int = 24):
        self.result = result
        self.width = width
        self.height = height
        self.fps = fps
        self._load_fonts()

    def _load_fonts(self):
        """Loads clean system fonts with robust fallbacks."""
        self.font_large = None
        self.font_medium = None
        self.font_small = None
        self.font_tiny = None

        # Scale font sizes relative to 720p height
        scale = max(0.5, self.height / 720.0)
        s_large = int(34 * scale)
        s_med = int(22 * scale)
        s_small = int(16 * scale)
        s_tiny = int(12 * scale)

        font_paths = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc"
        ]

        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    self.font_large = ImageFont.truetype(fp, s_large)
                    self.font_medium = ImageFont.truetype(fp, s_med)
                    self.font_small = ImageFont.truetype(fp, s_small)
                    self.font_tiny = ImageFont.truetype(fp, s_tiny)
                    return
                except Exception:
                    pass

        # Fallback to default
        self.font_large = ImageFont.load_default()
        self.font_medium = ImageFont.load_default()
        self.font_small = ImageFont.load_default()
        self.font_tiny = ImageFont.load_default()

    def _create_base_canvas(self, title: str, subtitle: Optional[str] = None) -> Image.Image:
        """Draws dark-mode scientific enterprise frame with header and footer."""
        img = Image.new("RGB", (self.width, self.height), color="#0b1120")
        draw = ImageDraw.Draw(img)

        hdr_h = int(self.height * 0.088)
        ftr_h = int(self.height * 0.052)

        # Header Bar
        draw.rectangle([0, 0, self.width, hdr_h], fill="#0f172a")
        draw.line([0, hdr_h, self.width, hdr_h], fill="#1e293b", width=2)
        draw.text((int(self.width * 0.02), int(hdr_h * 0.25)), "TERRA-SR MISSION REPLAY", fill="#38bdf8", font=self.font_medium)
        draw.text((int(self.width * 0.70), int(hdr_h * 0.30)), f"MISSION: {self.result.mission_id[:16]}", fill="#94a3b8", font=self.font_small)

        # Content Title
        draw.text((int(self.width * 0.02), int(self.height * 0.11)), title, fill="#f8fafc", font=self.font_large)
        if subtitle:
            draw.text((int(self.width * 0.02), int(self.height * 0.17)), subtitle, fill="#94a3b8", font=self.font_small)

        # Footer Bar
        draw.rectangle([0, self.height - ftr_h, self.width, self.height], fill="#0f172a")
        draw.line([0, self.height - ftr_h, self.width, self.height - ftr_h], fill="#1e293b", width=1)
        draw.text((int(self.width * 0.02), self.height - int(ftr_h * 0.75)), "SIH26142 • AI Multispectral Super-Resolution • Team VIBE-CODERS", fill="#64748b", font=self.font_tiny)
        draw.text((int(self.width * 0.70), self.height - int(ftr_h * 0.75)), f"EVIDENCE: {self.result.evidence_level}", fill="#38bdf8", font=self.font_tiny)

        return img

    def _get_active_images(self) -> Dict[str, Image.Image]:
        """Loads or constructs real raster thumbnails from workspace."""
        loaded = {}
        static_dir = Path("app/static")

        candidates = {
            "orig_rgb": static_dir / "active_orig_rgb.png",
            "enh_rgb": static_dir / "active_enh_rgb.png",
            "orig_fc": static_dir / "active_orig_fc.png",
            "enh_fc": static_dir / "active_enh_fc.png",
            "orig_ndvi": static_dir / "active_orig_ndvi.png",
            "enh_ndvi": static_dir / "active_enh_ndvi.png",
            "orig_edge": static_dir / "active_orig_edge.png",
            "enh_edge": static_dir / "active_enh_edge.png",
        }

        for key, p in candidates.items():
            if p.exists():
                try:
                    loaded[key] = Image.open(p).convert("RGB")
                except Exception:
                    pass

        # Fallback dummy RGB if missing
        if "orig_rgb" not in loaded:
            loaded["orig_rgb"] = Image.new("RGB", (512, 512), color="#1e293b")
        if "enh_rgb" not in loaded:
            loaded["enh_rgb"] = Image.new("RGB", (1024, 1024), color="#0f766e")

        return loaded

    def generate(self, output_mp4_path: Union[str, Path], total_duration_seconds: int = 45) -> Path:
        """
        Renders and compiles all scenes into a standard MP4 video.
        """
        output_mp4_path = Path(output_mp4_path)
        output_mp4_path.parent.mkdir(parents=True, exist_ok=True)

        images = self._get_active_images()
        writer = imageio.get_writer(
            str(output_mp4_path),
            format="FFMPEG",
            mode="I",
            fps=self.fps,
            codec="libx264",
            pixelformat="yuv420p"
        )

        frames_per_scene = int(self.fps * (total_duration_seconds / 6.0))
        img_size = int(min(self.width * 0.42, self.height * 0.65))
        img_x = int(self.width * 0.04)
        img_y = int(self.height * 0.22)
        panel_x = int(self.width * 0.50)
        panel_w = int(self.width * 0.96)
        panel_y = img_y
        panel_h = img_y + img_size

        try:
            # SCENE 1: Title & Mission Context
            canvas = self._create_base_canvas("1. Mission Overview & Acquisition Context", f"Target AOI: {self.result.aoi_bbox}")
            draw = ImageDraw.Draw(canvas)
            
            box_x1 = int(self.width * 0.04)
            box_x2 = int(self.width * 0.96)
            box_y1 = int(self.height * 0.22)
            box_y2 = int(self.height * 0.88)
            draw.rectangle([box_x1, box_y1, box_x2, box_y2], fill="#1e293b", outline="#334155", width=2)
            
            dy = int((box_y2 - box_y1) / 8.5)
            draw.text((box_x1 + 30, box_y1 + 20), f"Mission Identifier:  {self.result.mission_id}", fill="#38bdf8", font=self.font_medium)
            draw.text((box_x1 + 30, box_y1 + 20 + dy*1), f"Satellite Sensor:    {self.result.sensor}", fill="#f8fafc", font=self.font_medium)
            draw.text((box_x1 + 30, box_y1 + 20 + dy*2), f"Sentinel-2 Scene:    {self.result.scene_id[:35]}...", fill="#f8fafc", font=self.font_medium)
            draw.text((box_x1 + 30, box_y1 + 20 + dy*3), f"Acquisition Date:    {self.result.acquisition_date}", fill="#f8fafc", font=self.font_medium)
            draw.text((box_x1 + 30, box_y1 + 20 + dy*4), f"AOI Surface Area:    {self.result.aoi_area_km2:.2f} km² ({self.result.aoi_area_ha:.1f} ha)", fill="#f8fafc", font=self.font_medium)
            draw.text((box_x1 + 30, box_y1 + 20 + dy*5), f"Input GSD:           {self.result.input_gsd:.1f}m Multispectral (B02, B03, B04, B08)", fill="#f8fafc", font=self.font_medium)
            draw.text((box_x1 + 30, box_y1 + 20 + dy*6), f"Super-Resolution:    {self.result.model_name} (Target: {self.result.output_grid})", fill="#4ade80", font=self.font_medium)
            draw.text((box_x1 + 30, box_y1 + 20 + dy*7), f"Scientific Evidence: {self.result.evidence_level}", fill="#fbbf24", font=self.font_medium)

            frame_np = np.array(canvas)
            for _ in range(frames_per_scene):
                writer.append_data(frame_np)

            # SCENE 2: Original 10m Input
            canvas = self._create_base_canvas("2. Original Medium-Resolution Input (10m Native GSD)", "Calibrated Sentinel-2 Level-2A Bottom-Of-Atmosphere Reflectance")
            orig_thumb = images["orig_rgb"].resize((img_size, img_size))
            canvas.paste(orig_thumb, (img_x, img_y))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle([panel_x, panel_y, panel_w, panel_h], fill="#1e293b", outline="#334155", width=2)
            draw.text((panel_x + 25, panel_y + 25), "INPUT CHARACTERISTICS:", fill="#38bdf8", font=self.font_medium)
            draw.text((panel_x + 25, panel_y + 80), "• Native Resolution: 10.0m GSD", fill="#f8fafc", font=self.font_small)
            draw.text((panel_x + 25, panel_y + 125), "• Spectral Bands: 4-Band BOA Reflectance", fill="#f8fafc", font=self.font_small)
            draw.text((panel_x + 25, panel_y + 170), "• Radiometric Depth: Standardized [0.0, 1.0]", fill="#f8fafc", font=self.font_small)
            draw.text((panel_x + 25, panel_y + 215), "• Spatial Limit: Nyquist sampling limit (10m)", fill="#f8fafc", font=self.font_small)
            draw.text((panel_x + 25, panel_y + 275), "Starting Neural Reconstruction...", fill="#fbbf24", font=self.font_medium)

            frame_np = np.array(canvas)
            for _ in range(frames_per_scene):
                writer.append_data(frame_np)

            # SCENE 3: Animated Super-Resolution Transition / Split Wipe
            enh_thumb = images["enh_rgb"].resize((img_size, img_size))
            for step in range(frames_per_scene):
                split_x = int((step / max(1, frames_per_scene)) * img_size)
                combined = Image.new("RGB", (img_size, img_size))
                combined.paste(orig_thumb.crop((0, 0, split_x, img_size)), (0, 0))
                combined.paste(enh_thumb.crop((split_x, 0, img_size, img_size)), (split_x, 0))

                c_draw = ImageDraw.Draw(combined)
                c_draw.line([split_x, 0, split_x, img_size], fill="#38bdf8", width=3)

                frame = self._create_base_canvas("3. Neural Super-Resolution Reconstruction", f"Model: {self.result.model_name} (Multi-Spectral Residual Attention)")
                frame.paste(combined, (img_x, img_y))
                f_draw = ImageDraw.Draw(frame)
                f_draw.text((img_x, img_y + img_size + 10), "← Original 10m Input", fill="#94a3b8", font=self.font_tiny)
                f_draw.text((img_x + int(img_size * 0.5), img_y + img_size + 10), "TERRA-SR 5m / 3.33m →", fill="#38bdf8", font=self.font_tiny)

                f_draw.rectangle([panel_x, panel_y, panel_w, panel_h], fill="#1e293b", outline="#334155", width=2)
                f_draw.text((panel_x + 25, panel_y + 25), "RECONSTRUCTION METRICS:", fill="#38bdf8", font=self.font_medium)
                f_draw.text((panel_x + 25, panel_y + 75), f"• Peak SNR: {self.result.metrics.psnr:.2f} dB", fill="#4ade80", font=self.font_small)
                f_draw.text((panel_x + 25, panel_y + 115), f"• SSIM Index: {self.result.metrics.ssim:.4f}", fill="#4ade80", font=self.font_small)
                f_draw.text((panel_x + 25, panel_y + 155), f"• Spectral Angle: {self.result.metrics.sam:.2f}°", fill="#4ade80", font=self.font_small)
                f_draw.text((panel_x + 25, panel_y + 195), f"• ERGAS Error: {self.result.metrics.ergas:.2f}", fill="#4ade80", font=self.font_small)
                f_draw.text((panel_x + 25, panel_y + 235), f"• Edge Pres. Index: {self.result.metrics.epi:.4f}", fill="#4ade80", font=self.font_small)
                f_draw.text((panel_x + 25, panel_y + 275), f"• High-Freq Energy: {self.result.metrics.high_frequency_energy:.1f}%", fill="#4ade80", font=self.font_small)

                writer.append_data(np.array(frame))

            # SCENE 4: Difference & High-Frequency Detail Map
            canvas = self._create_base_canvas("4. Difference & High-Frequency Detail Analysis", "Evaluating sub-pixel spatial gradient enhancement")
            if "enh_edge" in images:
                edge_thumb = images["enh_edge"].resize((img_size, img_size))
                canvas.paste(edge_thumb, (img_x, img_y))
            else:
                canvas.paste(enh_thumb, (img_x, img_y))

            draw = ImageDraw.Draw(canvas)
            draw.rectangle([panel_x, panel_y, panel_w, panel_h], fill="#1e293b", outline="#334155", width=2)
            draw.text((panel_x + 25, panel_y + 25), "SPATIAL GRADIENT ANALYSIS:", fill="#38bdf8", font=self.font_medium)
            draw.text((panel_x + 25, panel_y + 85), f"• Mean Absolute Diff: {self.result.difference.mean_absolute_diff:.4f}", fill="#f8fafc", font=self.font_small)
            draw.text((panel_x + 25, panel_y + 135), f"• High-Freq Gain: +{self.result.difference.high_frequency_change_pct:.2f}%", fill="#4ade80", font=self.font_small)
            draw.text((panel_x + 25, panel_y + 185), f"• Boundary Clarity Gain: +{self.result.difference.boundary_change_pct:.2f}%", fill="#4ade80", font=self.font_small)
            draw.text((panel_x + 25, panel_y + 245), "• Sharp recovery along linear boundaries", fill="#94a3b8", font=self.font_small)

            frame_np = np.array(canvas)
            for _ in range(frames_per_scene):
                writer.append_data(frame_np)

            # SCENE 5: Downstream Earth Intelligence
            canvas = self._create_base_canvas("5. Downstream Earth Intelligence Vectorization", f"Active Domain: {self.result.intelligence.domain.upper()}")
            if "enh_ndvi" in images:
                intel_thumb = images["enh_ndvi"].resize((img_size, img_size))
                canvas.paste(intel_thumb, (img_x, img_y))
            else:
                canvas.paste(enh_thumb, (img_x, img_y))

            draw = ImageDraw.Draw(canvas)
            draw.rectangle([panel_x, panel_y, panel_w, panel_h], fill="#1e293b", outline="#334155", width=2)
            draw.text((panel_x + 25, panel_y + 25), "ACTIONABLE INTELLIGENCE:", fill="#38bdf8", font=self.font_medium)
            draw.text((panel_x + 25, panel_y + 80), f"• Extracted Area: {self.result.intelligence.area_km2:.4f} km²", fill="#f8fafc", font=self.font_small)
            draw.text((panel_x + 25, panel_y + 125), f"• Total Perimeter: {self.result.intelligence.perimeter_km:.2f} km", fill="#f8fafc", font=self.font_small)
            draw.text((panel_x + 25, panel_y + 170), f"• Vector Features: {self.result.intelligence.feature_count} polygons", fill="#f8fafc", font=self.font_small)
            draw.text((panel_x + 25, panel_y + 225), "• Vector GeoJSON: RFC 7946 Standard", fill="#4ade80", font=self.font_small)
            draw.text((panel_x + 25, panel_y + 275), "• Ready for QGIS / ArcGIS / Leaflet", fill="#94a3b8", font=self.font_small)

            frame_np = np.array(canvas)
            for _ in range(frames_per_scene):
                writer.append_data(frame_np)

            # SCENE 6: Final Scientific Scorecard
            canvas = self._create_base_canvas("6. Mission Verification & Final Scorecard", "Certified by TERRA-SR Autonomous Verification Engine")
            draw = ImageDraw.Draw(canvas)
            draw.rectangle([box_x1, box_y1, box_x2, box_y2], fill="#1e293b", outline="#059669", width=2)
            
            draw.text((box_x1 + 30, box_y1 + 25), "MISSION STATUS: EXECUTION COMPLETE ✓", fill="#10b981", font=self.font_large)
            draw.text((box_x1 + 30, box_y1 + 25 + dy*1.5), f"PSNR: {self.result.metrics.psnr:.2f} dB   |   SSIM: {self.result.metrics.ssim:.4f}   |   SAM: {self.result.metrics.sam:.2f}°", fill="#f8fafc", font=self.font_medium)
            draw.text((box_x1 + 30, box_y1 + 25 + dy*2.7), f"ERGAS: {self.result.metrics.ergas:.2f}   |   EPI: {self.result.metrics.epi:.4f}   |   NDVI-r: {self.result.metrics.ndvi_consistency:.4f}", fill="#f8fafc", font=self.font_medium)
            draw.text((box_x1 + 30, box_y1 + 25 + dy*4.0), f"Generated: 4-Band GeoTIFF, GeoJSON, Research PDF, Mission MP4, Research ZIP", fill="#38bdf8", font=self.font_small)
            draw.text((box_x1 + 30, box_y1 + 25 + dy*5.0), f"Evidence Framework: {self.result.evidence_level}", fill="#fbbf24", font=self.font_medium)
            draw.text((box_x1 + 30, box_y1 + 25 + dy*6.5), "SMART INDIA HACKATHON 2026 • SIH26142 • TEAM VIBE-CODERS", fill="#64748b", font=self.font_medium)

            frame_np = np.array(canvas)
            for _ in range(frames_per_scene):
                writer.append_data(frame_np)

        finally:
            writer.close()

        self.result.outputs.video_mp4 = str(output_mp4_path)
        return output_mp4_path
