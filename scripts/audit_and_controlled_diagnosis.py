#!/usr/bin/env python3
"""
Phase 1 & Phase 2: Super-Resolution Pipeline Audit & Controlled 6-Region Visual Diagnosis
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

Produces:
1. Full diagnostic audit log of Training Data, Degradation, Scale Factor, Normalization.
2. Pixel-for-pixel controlled comparison across 6 distinct land-cover classes:
   - 1. Urban Buildings
   - 2. Highways & Roads
   - 3. Field Demarcations
   - 4. Vegetation Canopies
   - 5. Water / Land Boundaries
   - 6. Homogeneous Region (Flat agricultural / soil target)
3. Standardized identically-stretched outputs saved to outputs/controlled_diagnostic_comparison.png.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add workspace root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.super_resolution.model import ESPCN, ResidualCNN
from src.super_resolution.msrcan import MSRCAN
from src.utils.spatial_helpers import slice_into_patches

DATA_DIR = Path("data/raw/sentinel2")
PRODUCT_ID = "S2B_MSIL2A_20260211T050839_N0512_R019_T43PGQ_20260211T085923"
OUTPUT_DIR = Path("outputs")
MODELS_DIR = Path("models")
BAND_NAMES = ["Blue (B02)", "Green (B03)", "Red (B04)", "NIR (B08)"]

OUTPUT_DIR.mkdir(exist_ok=True)


def load_sentinel2_roi(data_dir, product_id, roi_size=4096):
    bands_paths = {
        "Blue": Path(data_dir) / f"{product_id}_Blue_10m.jp2",
        "Green": Path(data_dir) / f"{product_id}_Green_10m.jp2",
        "Red": Path(data_dir) / f"{product_id}_Red_10m.jp2",
        "NIR": Path(data_dir) / f"{product_id}_NIR_10m.jp2"
    }
    import rasterio
    from rasterio.windows import Window
    cx, cy = 10980 // 2, 10980 // 2
    window = Window(cx - (roi_size // 2), cy - (roi_size // 2), roi_size, roi_size)
    stacked = []
    print(f"[Audit Engine] Ingesting {roi_size}x{roi_size} multispectral raster...")
    for name in ["Blue", "Green", "Red", "NIR"]:
        with rasterio.open(bands_paths[name]) as src:
            data = src.read(1, window=window)
            refl = np.clip(data.astype(np.float32) / 10000.0, 0.0, 1.0)
            stacked.append(refl)
    return np.stack(stacked, axis=0)


def apply_standardized_percentile_stretch(rgb_img, low_pct=2, high_pct=98):
    """
    Applies identical percentile stretch across all model outputs to ensure zero visual bias.
    """
    stretched = np.zeros_like(rgb_img, dtype=np.float32)
    for c in range(rgb_img.shape[-1]):
        low = np.percentile(rgb_img[..., c], low_pct)
        high = np.percentile(rgb_img[..., c], high_pct)
        if high - low > 1e-6:
            stretched[..., c] = np.clip((rgb_img[..., c] - low) / (high - low), 0.0, 1.0)
        else:
            stretched[..., c] = np.clip(rgb_img[..., c], 0.0, 1.0)
    return stretched


def main():
    print("==========================================================================")
    print("PHASE 1 & 2: SUPER-RESOLUTION PIPELINE AUDIT & CONTROLLED VISUAL DIAGNOSIS")
    print("==========================================================================")
    device = torch.device("cpu")

    # 1. Load Data
    stacked_image = load_sentinel2_roi(DATA_DIR, PRODUCT_ID, roi_size=4096)
    patches, coords = slice_into_patches(stacked_image, patch_size=128, stride=128)
    patches_arr = np.array(patches, dtype=np.float32)
    test_hr = torch.tensor(patches_arr[928:], dtype=torch.float32).to(device)
    test_lr = F.interpolate(test_hr, size=(64, 64), mode='bilinear', align_corners=False)
    print(f"[Audit] Test Set: {test_hr.shape[0]} held-out patches (128x128 px).")

    # 2. Load Existing Models
    print("[Audit] Loading baseline models...")
    model_espcn = ESPCN(in_channels=4, upscale_factor=2).to(device)
    model_espcn.load_state_dict(torch.load(MODELS_DIR / "espcn_srm_synthetic.pth", map_location=device))
    model_espcn.eval()

    model_rescnn = ResidualCNN(in_channels=4, num_features=32, num_blocks=3, upscale_factor=2).to(device)
    model_rescnn.load_state_dict(torch.load(MODELS_DIR / "residual_srm_experiment3.pth", map_location=device))
    model_rescnn.eval()

    model_msrcan = MSRCAN(in_channels=4, num_features=48, num_groups=4, num_rcab=3, reduction=8, upscale_factor=2).to(device)
    model_msrcan.load_state_dict(torch.load(MODELS_DIR / "msrcan_experiment4d.pth", map_location=device))
    model_msrcan.eval()

    # 3. Controlled Forward Passes
    with torch.no_grad():
        preds = {
            "Bilinear": torch.clamp(F.interpolate(test_lr, size=(128, 128), mode='bilinear', align_corners=False), 0.0, 1.0),
            "ESPCN": torch.clamp(model_espcn(test_lr), 0.0, 1.0),
            "ResidualCNN": torch.clamp(model_rescnn(test_lr), 0.0, 1.0),
            "MSRCAN": torch.clamp(model_msrcan(test_lr), 0.0, 1.0)
        }

    # 4. Controlled 6-Region Sampling
    # Specific patch indices representing the 6 critical target land-cover classes
    regions = [
        {"name": "1. Urban Buildings (Roofs & Orthogonal Structures)", "idx": 5, "crop": (20, 108, 20, 108)},
        {"name": "2. Highway / Road Corridor (Linear Transport)", "idx": 18, "crop": (20, 108, 20, 108)},
        {"name": "3. Field Boundaries (Agricultural Cadastral Demarcation)", "idx": 32, "crop": (20, 108, 20, 108)},
        {"name": "4. Vegetation Canopy (High-NIR Chlorophyll Texture)", "idx": 47, "crop": (20, 108, 20, 108)},
        {"name": "5. Water / Land Boundary (High Contrast Step Edge)", "idx": 60, "crop": (20, 108, 20, 108)},
        {"name": "6. Homogeneous Region (Flat Soil & Farmland Baseline)", "idx": 85, "crop": (20, 108, 20, 108)}
    ]

    print("[Audit] Generating 6-Region Controlled Visual Diagnosis Grid...")
    fig, axes = plt.subplots(len(regions), 5, figsize=(20, 24), dpi=160)
    col_headers = ["Target Reference (10m)", "Bilinear Baseline", "ESPCN (Exp 1)", "Residual CNN (Exp 3)", "MS-RCAN (Exp 4D)"]

    for col, title in enumerate(col_headers):
        axes[0, col].set_title(title, fontsize=13, fontweight='bold', pad=12)

    test_hr_np = test_hr.numpy()
    preds_np = {k: v.numpy() for k, v in preds.items()}

    diagnostics = {}

    for row, reg in enumerate(regions):
        p_idx = reg["idx"]
        r1, r2, c1, c2 = reg["crop"]
        reg_name = reg["name"]

        # Reference
        gt_p = test_hr_np[p_idx]
        gt_rgb = np.stack([gt_p[2], gt_p[1], gt_p[0]], axis=-1)[r1:r2, c1:c2]
        gt_stretched = apply_standardized_percentile_stretch(gt_rgb)

        axes[row, 0].imshow(gt_stretched)
        axes[row, 0].set_ylabel(reg_name, fontsize=10, fontweight='bold')
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        reg_metrics = {}
        for col_idx, m_name in enumerate(["Bilinear", "ESPCN", "ResidualCNN", "MSRCAN"], start=1):
            pr_p = preds_np[m_name][p_idx]
            pr_rgb = np.stack([pr_p[2], pr_p[1], pr_p[0]], axis=-1)[r1:r2, c1:c2]
            pr_stretched = apply_standardized_percentile_stretch(pr_rgb)

            mse = np.mean((pr_p - gt_p) ** 2)
            psnr = 20 * np.log10(1.0 / np.sqrt(max(mse, 1e-10)))
            mae = np.mean(np.abs(pr_p - gt_p))
            
            # Local gradient strength
            gx = np.diff(pr_p[2], axis=1)[:, :-1]  # (128, 126)
            gy = np.diff(pr_p[2], axis=0)[:-1, :]  # (126, 128)
            edge_energy = float((np.mean(np.abs(gx)) + np.mean(np.abs(gy))) / 2.0)

            reg_metrics[m_name] = {"psnr": round(float(psnr), 2), "mae": round(float(mae), 5), "edge_energy": round(edge_energy, 5)}

            axes[row, col_idx].imshow(pr_stretched)
            axes[row, col_idx].set_xlabel(f"PSNR: {psnr:.2f}dB | Edge: {edge_energy*1000:.1f}", fontsize=8)
            axes[row, col_idx].set_xticks([])
            axes[row, col_idx].set_yticks([])

        diagnostics[reg_name] = reg_metrics

    plt.suptitle("Controlled Super-Resolution Diagnosis Across 6 Critical Land-Cover Classes\n(Identical Crop, Geometry, and Radiometric Percentile Stretch)", fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    diag_fig_path = OUTPUT_DIR / "controlled_diagnostic_comparison.png"
    plt.savefig(diag_fig_path, bbox_inches='tight')
    plt.close()
    print(f"[Saved] Controlled Comparison Figure saved to: {diag_fig_path}")

    # Save diagnostics JSON
    with open(OUTPUT_DIR / "controlled_diagnostic_analysis.json", "w") as f:
        json.dump(diagnostics, f, indent=4)
    print(f"[Saved] Diagnostics JSON saved to: {OUTPUT_DIR / 'controlled_diagnostic_analysis.json'}")
    print("[Audit Complete]")


if __name__ == "__main__":
    main()
