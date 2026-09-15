#!/usr/bin/env python3
"""
TERRA-SR Canonical Analysis Result Object & Unified Data Contract
Problem Statement: SIH26142 - Deep Learning Based Super Resolution Mapping (SRM)
Team: VIBE-CODERS

This module provides the single source of truth (AnalysisResult) consumed by:
1. Interactive Results Dashboard
2. Multi-Page Research PDF Report Generator
3. Mission Replay MP4 Video Generator
4. Scientific Voice Narrator & Script Engine
5. Complete Research Package Exporter (ZIP)
"""

import os
import json
import csv
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Union


# Standard Scientific Evidence Levels
EVIDENCE_REFERENCE_VALIDATED = "REFERENCE-VALIDATED"
EVIDENCE_OPERATIONAL_NO_REF = "OPERATIONAL / NO HIGH-RESOLUTION REFERENCE"
EVIDENCE_RESEARCH_HYPOTHESIS = "RESEARCH HYPOTHESIS"


@dataclass
class MetricScores:
    psnr: float = 0.0
    ssim: float = 0.0
    sam: float = 0.0              # Spectral Angle Mapper in degrees
    ergas: float = 0.0            # Relative dimensionless global error
    epi: float = 0.0              # Edge Preservation Index (0.0 to 1.0)
    high_frequency_energy: float = 0.0  # Percentage of high-frequency energy preserved
    ndvi_consistency: float = 0.0 # Pearson correlation of NDVI pre/post SR


@dataclass
class UncertaintySummary:
    available: bool = False
    mean: float = 0.0
    min: float = 0.0
    max: float = 0.0
    uncertainty_path: Optional[str] = None
    description: str = "Uncertainty unavailable for this model"


@dataclass
class DifferenceAnalysis:
    difference_image_path: Optional[str] = None
    boundary_change_pct: float = 0.0
    high_frequency_change_pct: float = 0.0
    mean_absolute_diff: float = 0.0
    spectral_difference_map_path: Optional[str] = None


@dataclass
class IntelligenceSummary:
    domain: str = "general"
    detected_features: List[str] = field(default_factory=list)
    area_km2: float = 0.0
    perimeter_km: float = 0.0
    feature_count: int = 0
    summary: Dict[str, Any] = field(default_factory=dict)
    geojson_path: Optional[str] = None
    geojson_paths: Dict[str, str] = field(default_factory=dict)


@dataclass
class OutputArtifacts:
    geotiff: Optional[str] = None
    geojson: Optional[str] = None
    metrics_json: Optional[str] = None
    metrics_csv: Optional[str] = None
    report_pdf: Optional[str] = None
    video_mp4: Optional[str] = None
    narration_script: Optional[str] = None
    narration_audio: Optional[str] = None
    package_zip: Optional[str] = None


@dataclass
class AnalysisResult:
    """
    Canonical single source of truth for an executed TERRA-SR mission run.
    """
    mission_id: str
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    source_type: str = "DEMO_MISSION"  # COPERNICUS_CDSE, LOCAL_UPLOAD, DEMO_MISSION
    sensor: str = "Sentinel-2 MSI"
    product: str = "Sentinel-2 L2A (BOA Surface Reflectance)"
    scene_id: str = "S2B_MSIL2A_20260211T050839_N0512_R019_T43PGQ"
    acquisition_date: str = "2026-02-11"

    # Geospatial Geometry & Grid
    aoi_geometry: Dict[str, Any] = field(default_factory=lambda: {"type": "Polygon", "coordinates": []})
    aoi_bbox: List[float] = field(default_factory=lambda: [77.58, 12.92, 77.68, 13.02])
    aoi_area_km2: float = 120.49
    aoi_area_ha: float = 12049.0
    crs: str = "EPSG:32643"

    # Resolution & Radiometry
    input_gsd: float = 10.0
    output_grid: str = "5.0m Target GSD"
    scale_factor: int = 2
    bands: List[str] = field(default_factory=lambda: ["B02 (Blue)", "B03 (Green)", "B04 (Red)", "B08 (NIR)"])

    # Model & Weights
    model_name: str = "ResidualCNN"
    model_checkpoint: str = "models/residual_srm_experiment3.pth"

    # Core File Paths
    source_image_path: str = "outputs/retrieved_aoi.tiff"
    sr_image_path: str = "outputs/s2_enhanced_residualcnn.tiff"

    # Analytics Sub-Structures
    metrics: MetricScores = field(default_factory=MetricScores)
    uncertainty: UncertaintySummary = field(default_factory=UncertaintySummary)
    difference: DifferenceAnalysis = field(default_factory=DifferenceAnalysis)
    intelligence: IntelligenceSummary = field(default_factory=IntelligenceSummary)
    outputs: OutputArtifacts = field(default_factory=OutputArtifacts)

    # Scientific Evidence Classification
    evidence_level: str = EVIDENCE_OPERATIONAL_NO_REF
    evidence_statement: str = (
        "No high-resolution reference was available for this scene; therefore reconstruction "
        "quality is presented as an operational/experimental result rather than directly validated accuracy."
    )

    # Software & Hardware Provenance
    software_version: str = "TERRA-SR v3.0 (SIH26142)"
    runtime_environment: Dict[str, Any] = field(default_factory=lambda: {
        "python_version": "3.13",
        "pytorch_version": "2.1+",
        "device": "cpu"
    })

    def to_dict(self) -> Dict[str, Any]:
        """Converts dataclass hierarchy to plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisResult":
        """Instantiates AnalysisResult from dictionary with nested dataclasses."""
        data_copy = dict(data)
        if isinstance(data_copy.get("metrics"), dict):
            data_copy["metrics"] = MetricScores(**data_copy["metrics"])
        if isinstance(data_copy.get("uncertainty"), dict):
            data_copy["uncertainty"] = UncertaintySummary(**data_copy["uncertainty"])
        if isinstance(data_copy.get("difference"), dict):
            data_copy["difference"] = DifferenceAnalysis(**data_copy["difference"])
        if isinstance(data_copy.get("intelligence"), dict):
            data_copy["intelligence"] = IntelligenceSummary(**data_copy["intelligence"])
        if isinstance(data_copy.get("outputs"), dict):
            data_copy["outputs"] = OutputArtifacts(**data_copy["outputs"])
        return cls(**data_copy)

    def save_json(self, output_path: Union[str, Path]) -> Path:
        """Saves canonical analysis result as structured JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)
        self.outputs.metrics_json = str(output_path)
        return output_path

    def save_csv(self, output_path: Union[str, Path]) -> Path:
        """Exports metrics and key attributes as standardized CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        rows = [
            ["Attribute", "Value"],
            ["Mission ID", self.mission_id],
            ["Timestamp", self.timestamp],
            ["Source Type", self.source_type],
            ["Sensor", self.sensor],
            ["Product", self.product],
            ["Scene ID", self.scene_id],
            ["Acquisition Date", self.acquisition_date],
            ["AOI Area (km2)", f"{self.aoi_area_km2:.2f}"],
            ["CRS", self.crs],
            ["Input GSD", f"{self.input_gsd:.1f}m"],
            ["Output Grid", self.output_grid],
            ["Scale Factor", f"x{self.scale_factor}"],
            ["Model Name", self.model_name],
            ["Evidence Level", self.evidence_level],
            ["PSNR (dB)", f"{self.metrics.psnr:.2f}"],
            ["SSIM", f"{self.metrics.ssim:.4f}"],
            ["Spectral Angle Mapper (deg)", f"{self.metrics.sam:.2f}"],
            ["ERGAS Error Index", f"{self.metrics.ergas:.2f}"],
            ["Edge Preservation Index (EPI)", f"{self.metrics.epi:.4f}"],
            ["High Frequency Energy Ratio (%)", f"{self.metrics.high_frequency_energy:.1f}"],
            ["NDVI Consistency", f"{self.metrics.ndvi_consistency:.4f}"],
            ["Mean Absolute Difference", f"{self.difference.mean_absolute_diff:.4f}"],
            ["High Frequency Change (%)", f"{self.difference.high_frequency_change_pct:.2f}"],
            ["Intelligence Domain", self.intelligence.domain],
            ["Intelligence Feature Count", str(self.intelligence.feature_count)],
            ["Intelligence Total Area (km2)", f"{self.intelligence.area_km2:.4f}"],
            ["Intelligence Perimeter (km)", f"{self.intelligence.perimeter_km:.2f}"]
        ]
        
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
            
        self.outputs.metrics_csv = str(output_path)
        return output_path

    def get_reproducibility_config(self) -> Dict[str, Any]:
        """Returns reproducible JSON configuration for this mission."""
        return {
            "mission_id": self.mission_id,
            "timestamp": self.timestamp,
            "provenance": {
                "source_type": self.source_type,
                "sensor": self.sensor,
                "product": self.product,
                "scene_id": self.scene_id,
                "acquisition_date": self.acquisition_date,
                "aoi_bbox": self.aoi_bbox,
                "crs": self.crs
            },
            "super_resolution": {
                "model_name": self.model_name,
                "checkpoint": self.model_checkpoint,
                "input_bands": self.bands,
                "input_gsd": self.input_gsd,
                "output_grid": self.output_grid,
                "scale_factor": self.scale_factor
            },
            "intelligence": {
                "domain": self.intelligence.domain,
                "detected_features": self.intelligence.detected_features
            },
            "evidence_level": self.evidence_level,
            "reproduction_command": (
                f"python cli.py --input {self.source_image_path} "
                f"--model {self.model_name} --domain {self.intelligence.domain} "
                f"--output_dir outputs/{self.mission_id}"
            )
        }
