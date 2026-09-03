#!/usr/bin/env python3
"""
End-to-End System Integration Demo:
Sentinel-2 Super-Resolution (<4m GSD) -> Downstream Marine Oil Spill Intelligence
For SIH 2026 - Problem Statement SIH26142

Demonstrates the complete multi-stage pipeline:
1. Low-Resolution Sentinel-2 Ingestion (10m Native)
2. PI-RCAN Multi-Scale Spatial Enhancement (3.33m GSD) + Aleatoric Uncertainty Mapping
3. Downstream Water Gating, NDWI/SOSI/FAI Spectral Indices, and Slick Segmentation
4. Anti-Hallucination Dual-Scale Verification
5. Attributed Vector Polygon Extraction and GIS Incident Reporting
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add workspace root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from src.super_resolution.pircan import PIRCAN
from src.oil_spill.pipeline import OilSpillDetector
from src.oil_spill.indices import extract_multispectral_feature_cube

DATA_DIR = Path("data/raw/sentinel2")
PRODUCT_ID = "S2B_MSIL2A_20260211T050839_N0512_R019_T43PGQ_20260211T085923"
MODELS_DIR = Path("models/final_sr")
OUTPUTS_DIR = Path("outputs")

OUTPUTS_DIR.mkdir(exist_ok=True)


def load_scene_crop():
    bands_paths = {
        "Blue": DATA_DIR / f"{PRODUCT_ID}_Blue_10m.jp2",
        "Green": DATA_DIR / f"{PRODUCT_ID}_Green_10m.jp2",
        "Red": DATA_DIR / f"{PRODUCT_ID}_Red_10m.jp2",
        "NIR": DATA_DIR / f"{PRODUCT_ID}_NIR_10m.jp2"
    }
    import rasterio
    from rasterio.windows import Window
    # Crop a 256x256 water/coastal region
    window = Window(4500, 4500, 256, 256)
    stacked = []
    for name in ["Blue", "Green", "Red", "NIR"]:
        with rasterio.open(bands_paths[name]) as src:
            data = src.read(1, window=window)
            refl = np.clip(data.astype(np.float32) / 10000.0, 0.0, 1.0)
            stacked.append(refl)
    return np.stack(stacked, axis=0)


def main():
    print("=" * 80)
    print("EXECUTING END-TO-END DEMO: SENTINEL-2 SUPER-RESOLUTION -> OIL SPILL INTELLIGENCE")
    print("=" * 80)
    device = torch.device("cpu")

    # 1. Load LR Input (10m)
    lr_cube = load_scene_crop()
    # Simulate a realistic marine oil slick anomaly on water pixels for downstream testing
    # Emulsified oil elevated Red & NIR; thin sheen suppressed texture
    lr_cube_sim = lr_cube.copy()
    lr_cube_sim[1, 80:120, 90:170] = 0.06  # Green
    lr_cube_sim[2, 80:120, 90:170] = 0.11  # Red
    lr_cube_sim[3, 80:120, 90:170] = 0.16  # NIR
    
    print(f"[Stage 1] Ingested 10m Sentinel-2 Scene Crop: {lr_cube_sim.shape}")

    # 2. Run PI-RCAN Multi-Scale Super-Resolution (Scale x3 -> 3.33m GSD)
    model = PIRCAN(in_channels=4, out_channels=4, num_features=48, num_groups=3, num_rcab=4, reduction=8, upscale_factor=3).to(device)
    weights_path = MODELS_DIR / "best_model.pth"
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"[Stage 2] Loaded Super-Resolution Model: {weights_path}")
    model.eval()

    t_in = torch.tensor(lr_cube_sim, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        sr_cube, unc_map = model(t_in, return_uncertainty=True)
        sr_cube = torch.clamp(sr_cube, 0.0, 1.0).squeeze(0).cpu().numpy()
        unc_map = unc_map.squeeze(0).mean(dim=0).cpu().numpy()

    print(f"[Stage 2] Enhanced to {sr_cube.shape[1]}x{sr_cube.shape[2]} (~3.33m GSD) with Uncertainty Map.")

    # 3. Downstream Marine Oil Spill Intelligence
    detector = OilSpillDetector(device=device)
    report, mask = detector.process_scene(
        sr_cube=sr_cube,
        lr_raw_cube=lr_cube_sim,
        uncertainty_map=unc_map,
        gsd=3.33
    )

    print("\n" + "-" * 60)
    print("DOWNSTREAM OIL SPILL INCIDENT INTELLIGENCE REPORT:")
    print("-" * 60)
    print(f"Status               : {report['status']}")
    print(f"Target Resolution    : {report['spatial_resolution']}")
    print(f"Total Slick Area     : {report['summary']['total_slick_area_km2']} km²")
    print(f"Thin Sheen Area      : {report['summary']['thin_sheen_area_km2']} km²")
    print(f"Thick Emulsion Area  : {report['summary']['thick_emulsion_area_km2']} km²")
    print(f"Uncertainty Gating   : {report['anti_hallucination_audit']['uncertainty_gating']}")
    print(f"Dual-Scale Audit     : {report['anti_hallucination_audit']['dual_scale_verification']}")
    print("-" * 60)

    # Save JSON Report
    report_path = OUTPUTS_DIR / "oil_spill_incident_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"[Saved] Incident Intelligence Report saved to: {report_path}")

    # 4. Generate Visual Figure
    print("[Visualization] Generating End-to-End Pipeline Visualization Figure...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=150)
    
    def stretch(img_rgb):
        s = np.zeros_like(img_rgb)
        for c in range(3):
            low, high = np.percentile(img_rgb[..., c], 2), np.percentile(img_rgb[..., c], 98)
            s[..., c] = np.clip((img_rgb[..., c] - low) / (high - low + 1e-6), 0.0, 1.0)
        return s

    lr_rgb = stretch(np.stack([lr_cube_sim[2], lr_cube_sim[1], lr_cube_sim[0]], axis=-1))
    sr_rgb = stretch(np.stack([sr_cube[2], sr_cube[1], sr_cube[0]], axis=-1))
    sr_fc = stretch(np.stack([sr_cube[3], sr_cube[2], sr_cube[1]], axis=-1))
    
    # NDWI & SOSI
    green, red, nir = sr_cube[1], sr_cube[2], sr_cube[3]
    ndwi = (green - nir) / (green + nir + 1e-7)
    sosi = (nir - red) / (nir + red + 1e-7)

    # 1. Native LR
    axes[0, 0].imshow(lr_rgb)
    axes[0, 0].set_title("1. Native Sentinel-2 (10m)", fontweight='bold')
    axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])

    # 2. Enhanced SR RGB
    axes[0, 1].imshow(sr_rgb)
    axes[0, 1].set_title("2. PI-RCAN Enhanced (3.33m GSD)", fontweight='bold')
    axes[0, 1].set_xticks([]); axes[0, 1].set_yticks([])

    # 3. False Color NIR
    axes[0, 2].imshow(sr_fc)
    axes[0, 2].set_title("3. False Color NIR (3.33m)", fontweight='bold')
    axes[0, 2].set_xticks([]); axes[0, 2].set_yticks([])

    # 4. Aleatoric Uncertainty
    axes[0, 3].imshow(unc_map, cmap='plasma')
    axes[0, 3].set_title("4. Aleatoric Uncertainty σ²", fontweight='bold')
    axes[0, 3].set_xticks([]); axes[0, 3].set_yticks([])

    # 5. NDWI Water Gating
    axes[1, 0].imshow(ndwi, cmap='Blues')
    axes[1, 0].set_title("5. NDWI Water Masking", fontweight='bold')
    axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])

    # 6. SOSI Oil Spill Index
    axes[1, 1].imshow(sosi, cmap='inferno')
    axes[1, 1].set_title("6. SOSI Hydrocarbon Contrast", fontweight='bold')
    axes[1, 1].set_xticks([]); axes[1, 1].set_yticks([])

    # 7. Classified Slick Mask
    axes[1, 2].imshow(mask, cmap='tab10', vmin=0, vmax=9)
    axes[1, 2].set_title("7. Classified Oil Slick Mask", fontweight='bold')
    axes[1, 2].set_xticks([]); axes[1, 2].set_yticks([])

    # 8. Overlay
    overlay = sr_rgb.copy()
    overlay[mask == 1] = [1.0, 0.8, 0.0]  # Yellow for thin sheen
    overlay[mask == 2] = [1.0, 0.0, 0.0]  # Red for thick emulsion
    axes[1, 3].imshow(overlay)
    axes[1, 3].set_title("8. Verified Spill Attribution", fontweight='bold')
    axes[1, 3].set_xticks([]); axes[1, 3].set_yticks([])

    plt.suptitle("End-to-End System Pipeline: Sentinel-2 Super-Resolution (<4m) -> Downstream Marine Oil Spill Intelligence", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    fig_path = OUTPUTS_DIR / "oil_spill_intelligence_demo.png"
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"[Saved] End-to-End Pipeline Visualization saved to: {fig_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
