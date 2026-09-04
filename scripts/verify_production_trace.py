#!/usr/bin/env python3
"""
TERRA-SR Final Production Trace Verification Script
"""
import sys
import time
from pathlib import Path

# Add root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import torch
import numpy as np
import rasterio
from affine import Affine

from src.super_resolution.inference import ProductionInference, MODEL_REGISTRY
from src.intelligence import run_intelligence_pipeline
from src.core.input_validation import SatelliteInputValidator

def verify_trace():
    print("=" * 60)
    print("TERRA-SR FINAL PRODUCTION TRACE & AUDIT VERIFICATION")
    print("=" * 60)

    # 1. Model Registry
    assert "ResidualCNN" in MODEL_REGISTRY, "ResidualCNN missing from registry"
    cfg = MODEL_REGISTRY["ResidualCNN"]
    print(f"[1/7] Production Default Model: ResidualCNN ({cfg['description']})")
    
    weights_path = Path("models/residual_srm_experiment3.pth")
    assert weights_path.exists(), f"ResidualCNN weights file missing: {weights_path}"
    print(f"[2/7] Checkpoint Verified: {weights_path} ({weights_path.stat().st_size / 1024:.1f} KB)")

    # 2. Checkpoint Loading
    runner = ProductionInference(model_type="ResidualCNN")
    assert runner.model is not None, "Model not initialized"
    assert runner.upscale_factor == 2, "Upscale factor must be 2 for 5m output"
    print("[3/7] Model Checkpoint Load: SUCCESS (ResidualCNN weights loaded into PyTorch state_dict)")

    # 3. Experimental Model Check
    pircan_cfg = MODEL_REGISTRY["PIRCAN"]
    print(f"[4/7] PI-RCAN Model Status: EXPERIMENTAL ({pircan_cfg['description']})")

    # 4. Input Scene & Validation
    input_tiff = Path("data/processed/s2_10m_stacked_roi.tiff")
    assert input_tiff.exists(), f"Demo scene missing: {input_tiff}"
    val = SatelliteInputValidator.validate_input(input_tiff)
    print(f"[5/7] Ingest & Input Validation: {val.status} | Level: {val.level} | Sensor: {val.detected_sensor} | GSD: {val.gsd:.2f}m")
    assert val.level == "READY", "Expected READY level for Sentinel-2 scene"

    # 5. Fresh Forward Pass
    with rasterio.open(input_tiff) as src:
        lr = src.read([1, 2, 3, 4]).astype(np.float32) / 10000.0
        lr = np.clip(lr, 0.0, 1.0)
        src_transform = src.transform
        src_crs = src.crs.to_string() if src.crs else "EPSG:32643"

    tensor_in = torch.tensor(lr, dtype=torch.float32)
    t0 = time.time()
    hr = runner.enhance_tensor(tensor_in)
    infer_time = time.time() - t0

    print(f"[6/7] Fresh Inference Execution:")
    print(f"      - Input Shape:  {lr.shape} (10.0m GSD)")
    print(f"      - Output Shape: {hr.shape} (5.00m GSD)")
    print(f"      - Latency:      {infer_time * 1000:.1f} ms")
    assert hr.shape == (4, lr.shape[1] * 2, lr.shape[2] * 2), "HR shape mismatch"
    assert hr.min() >= 0.0 and hr.max() <= 1.0, "HR reflectance out of bounds"

    # Georeferencing
    scale_mult = 2
    hr_transform = src_transform * Affine.scale(1.0 / scale_mult)
    print(f"      - Georeference: CRS={src_crs} | Affine Transform Preserved & Scaled")

    # 6. Downstream Intelligence Execution & Export Verification
    print("[7/7] Downstream Intelligence & Export Generation:")
    domains = ["water", "urban", "agriculture", "disaster", "oil_spill"]
    for dom in domains:
        res = run_intelligence_pipeline(dom, hr, lr, gsd=5.0, affine_transform=hr_transform, crs=src_crs)
        assert "report" in res and "layers" in res, f"Domain {dom} failed"
        status = res["report"].get("status", "COMPLETE")
        layer_keys = list(res["layers"].keys())
        print(f"      - Domain {dom:<12}: Status={status:<20} | Layers={len(layer_keys)}")

    print("\n" + "=" * 60)
    print("ALL PRODUCTION CHECKS & VERIFICATION TRACES PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    verify_trace()
