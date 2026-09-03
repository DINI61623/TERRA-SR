#!/usr/bin/env python3
"""
Fast Vectorized 6-Way Super-Resolution Benchmark Suite
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

Evaluates:
1. Bilinear Baseline
2. ESPCN (Exp 1)
3. Residual CNN (Exp 3)
4. MS-RCAN (Exp 4D)
5. HF-SRM (Exp 4E)
6. PI-RCAN (Exp 4F)

Computes full quantitative metrics and generates:
- outputs/experiment4f_results.json
- outputs/experiment4f_benchmark_comparison.png
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.super_resolution.model import ESPCN, ResidualCNN
from src.super_resolution.msrcan import MSRCAN
from src.super_resolution.hfsrm import HFSRM
from src.super_resolution.pircan import PIRCAN
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
    print(f"[Benchmark] Loading {roi_size}x{roi_size} multispectral ROI...")
    for name in ["Blue", "Green", "Red", "NIR"]:
        with rasterio.open(bands_paths[name]) as src:
            data = src.read(1, window=window)
            refl = np.clip(data.astype(np.float32) / 10000.0, 0.0, 1.0)
            stacked.append(refl)
    return np.stack(stacked, axis=0)


def main():
    print("==================================================================")
    print("EXECUTING VECTORIZED 6-WAY BENCHMARK EVALUATION")
    print("==================================================================")
    device = torch.device("cpu")

    # 1. Load Data
    stacked_image = load_sentinel2_roi(DATA_DIR, PRODUCT_ID, roi_size=4096)
    patches, coords = slice_into_patches(stacked_image, patch_size=128, stride=128)
    patches_arr = np.array(patches, dtype=np.float32)
    test_hr = torch.tensor(patches_arr[928:], dtype=torch.float32).to(device)
    test_lr = F.interpolate(test_hr, size=(64, 64), mode='bilinear', align_corners=False)
    print(f"[Benchmark] Test Dataset: {test_hr.shape[0]} held-out test patches.")

    # 2. Load Checkpoints
    print("[Benchmark] Loading all 5 trained model checkpoints...")
    model_espcn = ESPCN(in_channels=4, upscale_factor=2).to(device)
    model_espcn.load_state_dict(torch.load(MODELS_DIR / "espcn_srm_synthetic.pth", map_location=device))
    model_espcn.eval()

    model_rescnn = ResidualCNN(in_channels=4, num_features=32, num_blocks=3, upscale_factor=2).to(device)
    model_rescnn.load_state_dict(torch.load(MODELS_DIR / "residual_srm_experiment3.pth", map_location=device))
    model_rescnn.eval()

    model_msrcan = MSRCAN(in_channels=4, num_features=48, num_groups=4, num_rcab=3, reduction=8, upscale_factor=2).to(device)
    model_msrcan.load_state_dict(torch.load(MODELS_DIR / "msrcan_experiment4d.pth", map_location=device))
    model_msrcan.eval()

    model_hfsrm = HFSRM(in_channels=4, num_features=48, num_blocks=4, upscale_factor=2).to(device)
    if (MODELS_DIR / "hfsrm_experiment4e.pth").exists():
        model_hfsrm.load_state_dict(torch.load(MODELS_DIR / "hfsrm_experiment4e.pth", map_location=device))
    model_hfsrm.eval()

    model_pircan = PIRCAN(in_channels=4, out_channels=4, num_features=48, num_groups=3, num_rcab=4, reduction=8, upscale_factor=2).to(device)
    pircan_weights = MODELS_DIR / "pircan_experiment4f.pth"
    if pircan_weights.exists():
        model_pircan.load_state_dict(torch.load(pircan_weights, map_location=device))
    model_pircan.eval()

    # 3. Vectorized Forward Passes
    with torch.no_grad():
        preds = {
            "Bilinear": torch.clamp(F.interpolate(test_lr, size=(128, 128), mode='bilinear', align_corners=False), 0.0, 1.0),
            "ESPCN": torch.clamp(model_espcn(test_lr), 0.0, 1.0),
            "ResidualCNN": torch.clamp(model_rescnn(test_lr), 0.0, 1.0),
            "MSRCAN": torch.clamp(model_msrcan(test_lr), 0.0, 1.0),
            "HFSRM": torch.clamp(model_hfsrm(test_lr), 0.0, 1.0),
            "PIRCAN": torch.clamp(model_pircan(test_lr), 0.0, 1.0)
        }

    # 4. Metrics Computation
    results = {}
    for name, p in preds.items():
        mse = torch.mean((p - test_hr)**2, dim=(1, 2, 3))
        psnr_b = 20 * torch.log10(1.0 / torch.sqrt(torch.clamp(mse, min=1e-10)))
        mae_b = torch.mean(torch.abs(p - test_hr), dim=(1, 2, 3))
        rmse_b = torch.sqrt(mse)
        
        # SAM
        dot = torch.sum(p * test_hr, dim=1)
        norm_p = torch.norm(p, p=2, dim=1)
        norm_t = torch.norm(test_hr, p=2, dim=1)
        denom = torch.clamp(norm_p * norm_t, min=1e-7)
        sam_b = torch.rad2deg(torch.acos(torch.clamp(dot / denom, -0.9999, 0.9999)))

        # ERGAS
        N, C, H, W = p.shape
        ergas_list = []
        for i in range(N):
            e_sum = 0.0
            for c in range(C):
                r_c = torch.sqrt(torch.mean((p[i, c] - test_hr[i, c])**2))
                m_c = torch.mean(test_hr[i, c])
                if m_c > 0:
                    e_sum += (r_c / m_c)**2
            ergas_list.append(float(100.0 * 0.5 * np.sqrt(e_sum / C)))
        ergas_val = float(np.mean(ergas_list))

        # EPI (Sobel)
        k_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(4, 1, 1, 1)
        k_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(4, 1, 1, 1)
        gp = torch.sqrt(F.conv2d(p, k_x, padding=1, groups=4)**2 + F.conv2d(p, k_y, padding=1, groups=4)**2)
        gt = torch.sqrt(F.conv2d(test_hr, k_x, padding=1, groups=4)**2 + F.conv2d(test_hr, k_y, padding=1, groups=4)**2)
        epi_b = torch.sum(gp * gt, dim=(1, 2, 3)) / (torch.sqrt(torch.sum(gp**2, dim=(1, 2, 3)) * torch.sum(gt**2, dim=(1, 2, 3))) + 1e-7)

        # NDVI MAE
        ndvi_gt = (test_hr[:, 3] - test_hr[:, 2]) / (test_hr[:, 3] + test_hr[:, 2] + 1e-7)
        ndvi_pr = (p[:, 3] - p[:, 2]) / (p[:, 3] + p[:, 2] + 1e-7)
        ndvi_mae_b = torch.mean(torch.abs(ndvi_pr - ndvi_gt), dim=(1, 2))

        results[name] = {
            "psnr": float(torch.mean(psnr_b).item()),
            "mae": float(torch.mean(mae_b).item()),
            "rmse": float(torch.mean(rmse_b).item()),
            "sam_deg": float(torch.mean(sam_b).item()),
            "ergas": ergas_val,
            "epi": float(torch.mean(epi_b).item()),
            "ndvi_mae": float(torch.mean(ndvi_mae_b).item()),
            "hf_ratio": float(np.mean([np.std(p[i].numpy()) / (np.std(test_hr[i].numpy()) + 1e-7) for i in range(len(test_hr))]))
        }

    # Print Table
    print("\n" + "="*115)
    print("6-WAY COMPREHENSIVE BENCHMARK RESULTS (96 HELD-OUT TEST PATCHES)")
    print("="*115)
    header = f"{'Metric':<24} | {'Bilinear':<10} | {'ESPCN':<10} | {'ResCNN':<10} | {'MS-RCAN':<10} | {'HF-SRM':<10} | {'PI-RCAN (Exp 4F)':<16}"
    print(header)
    print("-" * 115)
    for m_label, m_key, fmt in [
        ("Peak SNR (PSNR dB)", "psnr", ".2f"),
        ("Mean Abs Error (MAE)", "mae", ".5f"),
        ("Root Mean Sq Err (RMSE)", "rmse", ".5f"),
        ("Spectral Angle SAM (°)", "sam_deg", ".2f"),
        ("ERGAS Error Index", "ergas", ".2f"),
        ("Edge Pres. Index (EPI)", "epi", ".4f"),
        ("High-Freq Energy Ratio", "hf_ratio", ".1%"),
        ("NDVI Mean Abs Error", "ndvi_mae", ".4f")
    ]:
        row = f"{m_label:<24} | "
        for m_name in ["Bilinear", "ESPCN", "ResidualCNN", "MSRCAN", "HFSRM", "PIRCAN"]:
            val = results[m_name][m_key]
            v_str = f"{val:{fmt}}"
            row += f"{v_str:<10} | "
        print(row)
    print("=" * 115)

    # Save JSON
    benchmark_json = {
        "experiment": "Experiment 4F: Physically Informed RCAN (PI-RCAN)",
        "models_evaluated": list(preds.keys()),
        "test_patches_count": len(test_hr),
        "metrics": results
    }
    with open(OUTPUT_DIR / "experiment4f_results.json", "w") as f:
        json.dump(benchmark_json, f, indent=4)
    print(f"[Saved] 6-Way Benchmark JSON saved to: {OUTPUT_DIR / 'experiment4f_results.json'}")

    # 5. Visual Figure
    print("[Visualization] Generating 6-Way Regional Zoom Comparison Figure...")
    fig, axes = plt.subplots(5, 7, figsize=(26, 19), dpi=160)
    col_titles = ["Target Reference (10m)", "Bilinear Baseline", "ESPCN (Exp 1)", "Residual CNN (Exp 3)", "MS-RCAN (Exp 4D)", "HF-SRM (Exp 4E)", "PI-RCAN (Exp 4F)"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight='bold', pad=12)

    def stretch(img_rgb):
        s = np.zeros_like(img_rgb)
        for c in range(3):
            low, high = np.percentile(img_rgb[..., c], 2), np.percentile(img_rgb[..., c], 98)
            s[..., c] = np.clip((img_rgb[..., c] - low) / (high - low + 1e-6), 0.0, 1.0)
        return s

    p_indices = [5, 18, 32, 47, 60]
    cat_names = ["1. Urban Buildings", "2. Highway Corridor", "3. Field Boundaries", "4. Vegetation Canopy", "5. Water / Canal Edge"]

    test_hr_np = test_hr.numpy()
    preds_np = {k: v.numpy() for k, v in preds.items()}

    for row, (idx, c_name) in enumerate(zip(p_indices, cat_names)):
        gt_rgb = stretch(np.stack([test_hr_np[idx, 2], test_hr_np[idx, 1], test_hr_np[idx, 0]], axis=-1))
        axes[row, 0].imshow(gt_rgb)
        axes[row, 0].set_ylabel(c_name, fontsize=11, fontweight='bold')
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])

        for c_idx, m_name in enumerate(["Bilinear", "ESPCN", "ResidualCNN", "MSRCAN", "HFSRM", "PIRCAN"], start=1):
            pr_rgb = stretch(np.stack([preds_np[m_name][idx, 2], preds_np[m_name][idx, 1], preds_np[m_name][idx, 0]], axis=-1))
            mse = np.mean((preds_np[m_name][idx] - test_hr_np[idx])**2)
            psnr = 20 * np.log10(1.0 / np.sqrt(max(mse, 1e-10)))
            axes[row, c_idx].imshow(pr_rgb)
            axes[row, c_idx].set_xlabel(f"PSNR: {psnr:.2f}dB", fontsize=9)
            axes[row, c_idx].set_xticks([]); axes[row, c_idx].set_yticks([])

    plt.suptitle("6-Way Super-Resolution Model Comparison: Spatial Detail & Regional Feature Zoom", fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    fig_path = OUTPUT_DIR / "experiment4f_benchmark_comparison.png"
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"[Saved] 6-Way Comparison figure saved to: {fig_path}")
    print("[Complete] Vectorized benchmark successfully executed!")


if __name__ == "__main__":
    main()
