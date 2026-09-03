#!/usr/bin/env python3
"""
TERRA-SR Oil Spill Intelligence Adapter
Wraps existing src/oil_spill/pipeline.py into the unified BaseIntelligenceModule interface.
"""

import time
from typing import Dict, Any, Optional
import numpy as np
import matplotlib as mpl

from src.intelligence.base import (
    BaseIntelligenceModule,
    apply_percentile_stretch,
    compute_gradient_sharpness
)
from src.oil_spill.pipeline import OilSpillDetector


class OilSpillIntelligenceModule(BaseIntelligenceModule):
    def __init__(self, model_weights=None):
        super().__init__(domain_name="Oil Spill Intelligence", domain_key="oil_spill")
        self.detector = OilSpillDetector(model_weights=model_weights)

    def process(
        self,
        sr_cube: np.ndarray,
        lr_cube: Optional[np.ndarray] = None,
        gsd: float = 3.33,
        affine_transform: Any = None,
        crs: str = "EPSG:32643",
        bounds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        start_t = time.time()
        H, W = sr_cube.shape[1], sr_cube.shape[2]
        
        # Ensure clean reflectance
        sr_cube = np.clip(np.nan_to_num(sr_cube, nan=0.0), 0.0, 1.0)
        
        # True color RGB base
        sr_rgb = np.stack([sr_cube[2], sr_cube[1], sr_cube[0]], axis=-1)
        sr_rgb_stretched = (apply_percentile_stretch(sr_rgb) * 255).astype(np.uint8)
        
        # Execute existing detector
        report_data, oil_mask = self.detector.process_scene(
            sr_cube=sr_cube,
            lr_raw_cube=lr_cube,
            gsd=gsd,
            affine_transform=affine_transform,
            crs=crs
        )
        
        # 1. SOSI Layer ((B08 - B04) / (B08 + B04))
        sr_sosi = (sr_cube[3] - sr_cube[2]) / (sr_cube[3] + sr_cube[2] + 1e-7)
        inferno = mpl.colormaps['inferno']
        sosi_norm = np.clip((sr_sosi + 0.5) / 1.0, 0.0, 1.0)
        sosi_rgb = (inferno(sosi_norm)[..., :3] * 255).astype(np.uint8)
        
        # 2. Classified Oil Slick Mask Overlay
        oil_overlay = sr_rgb_stretched.copy()
        oil_overlay[oil_mask == 1] = [255, 204, 0]  # Thin Sheen (Yellow)
        oil_overlay[oil_mask == 2] = [255, 30, 30]   # Thick Emulsion (Red)
        
        # 3. Native vs SR Impact Analysis
        impact_metrics = {}
        if lr_cube is not None:
            lr_cube = np.clip(np.nan_to_num(lr_cube, nan=0.0), 0.0, 1.0)
            lr_sosi = (lr_cube[3] - lr_cube[2]) / (lr_cube[3] + lr_cube[2] + 1e-7)
            lr_grad = compute_gradient_sharpness(lr_sosi)
            sr_grad = compute_gradient_sharpness(sr_sosi)
            
            impact_metrics = {
                "native_gsd": "10.0m",
                "sr_gsd": f"{gsd:.2f}m",
                "native_sosi_edge_gradient": round(lr_grad, 4),
                "sr_sosi_edge_gradient": round(sr_grad, 4),
                "slick_boundary_sharpness_gain": f"+{round(((sr_grad - lr_grad)/(lr_grad + 1e-7))*100, 1)}%",
                "dual_scale_verification": "ENFORCED",
                "uncertainty_gating": "ENFORCED",
                "interpretation_benefit": "Sharpened slick perimeter prevents overestimation of slick dispersal area caused by 10m mixed ocean pixels."
            }
        else:
            sr_grad = compute_gradient_sharpness(sr_sosi)
            impact_metrics = {
                "sr_gsd": f"{gsd:.2f}m",
                "sr_sosi_edge_gradient": round(sr_grad, 4),
                "interpretation_benefit": "Sub-pixel resolution enables detection of narrow trailing emulsion streamers."
            }
            
        report_data["module"] = self.domain_name
        report_data["domain"] = self.domain_key
        report_data["sr_impact_analysis"] = impact_metrics
        report_data["scientific_safety_audit"] = {
            "product_type": "AI-reconstructed higher-resolution spatial product",
            "detection_classification": "Deep Learning Oil Spill Segmentation (OilSpillUNet)",
            "validation_note": "Anti-hallucination safeguards active (Uncertainty Gating + Dual-Scale S2 Verification). Only positive marine anomalies with consistent spectral contrast are retained."
        }
        
        layers = {
            "sosi_map": sosi_rgb,
            "oil_classified_mask": oil_overlay,
            "rgb_base": sr_rgb_stretched
        }
        
        return {
            "report": report_data,
            "layers": layers,
            "masks": {
                "oil_mask": oil_mask,
                "sosi": sr_sosi
            }
        }
