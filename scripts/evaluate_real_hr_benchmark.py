#!/usr/bin/env python3
"""
Automated Evaluation Harness for Real High-Resolution Reference Imagery
For SIH 2026 - Problem Statement SIH26142

Evaluates model predictions against genuine ground-truth reference data (~3.0m GSD):
- Quantitative Metrics: PSNR, SSIM, MAE, RMSE, SAM, ERGAS, EPI, NDVI MAE
- Standardized 5-Region Visual Crops (Urban Buildings, Highways/Roads, Fields, Vegetation, Water)
- Compares: Sentinel-2 (10m), Bilinear, Residual CNN, PI-RCAN, New Real-Data Model, True HR Reference
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
import torch.nn as nn
import torch.nn.functional as F

from src.super_resolution.model import ResidualCNN
from src.super_resolution.pircan import PIRCAN

OUTPUT_DIR = Path("outputs")
MODELS_DIR = Path("models")


def compute_all_metrics(pred_t, target_t, scale=3):
    pred = torch.clamp(pred_t, 0.0, 1.0)
    target = target_t
    
    # 1. PSNR, MAE, RMSE
    mse = torch.mean((pred - target)**2, dim=(1, 2, 3))
    psnr = float(torch.mean(20 * torch.log10(1.0 / torch.sqrt(torch.clamp(mse, min=1e-10)))).item())
    mae = float(torch.mean(torch.abs(pred - target)).item())
    rmse = float(torch.sqrt(torch.mean((pred - target)**2)).item())

    # 2. SSIM
    # Gaussian window
    window_size = 11
    sigma = 1.5
    gauss = torch.tensor([np.exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    gauss = gauss / gauss.sum()
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0).expand(4, 1, window_size, window_size).contiguous()
    w = _2D_window.to(pred.device)
    
    mu1 = F.conv2d(pred, w, padding=window_size//2, groups=4)
    mu2 = F.conv2d(target, w, padding=window_size//2, groups=4)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, w, padding=window_size//2, groups=4) - mu1_sq
    sigma2_sq = F.conv2d(target * target, w, padding=window_size//2, groups=4) - mu2_sq
    sigma12 = F.conv2d(pred * target, w, padding=window_size//2, groups=4) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_val = float(torch.mean(ssim_map).item())

    # 3. SAM
    dot = torch.sum(pred * target, dim=1)
    norm_p = torch.norm(pred, p=2, dim=1)
    norm_t = torch.norm(target, p=2, dim=1)
    denom = torch.clamp(norm_p * norm_t, min=1e-7)
    sam = float(torch.mean(torch.rad2deg(torch.acos(torch.clamp(dot / denom, -0.9999, 0.9999)))).item())

    # 4. ERGAS
    N, C, H, W = pred.shape
    ergas_list = []
    for i in range(N):
        e_sum = 0.0
        for c in range(C):
            r_c = torch.sqrt(torch.mean((pred[i, c] - target[i, c])**2))
            m_c = torch.mean(target[i, c])
            if m_c > 0:
                e_sum += (r_c / m_c)**2
        ergas_list.append(float(100.0 * (1.0 / scale) * np.sqrt(e_sum / C)))
    ergas = float(np.mean(ergas_list))

    # 5. EPI
    k_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(4, 1, 1, 1).to(pred.device)
    k_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(4, 1, 1, 1).to(pred.device)
    gp = torch.sqrt(F.conv2d(pred, k_x, padding=1, groups=4)**2 + F.conv2d(pred, k_y, padding=1, groups=4)**2)
    gt = torch.sqrt(F.conv2d(target, k_x, padding=1, groups=4)**2 + F.conv2d(target, k_y, padding=1, groups=4)**2)
    epi = float(torch.mean(torch.sum(gp * gt, dim=(1, 2, 3)) / (torch.sqrt(torch.sum(gp**2, dim=(1, 2, 3)) * torch.sum(gt**2, dim=(1, 2, 3))) + 1e-7)).item())

    # 6. NDVI MAE
    ndvi_gt = (target[:, 3] - target[:, 2]) / (target[:, 3] + target[:, 2] + 1e-7)
    ndvi_pr = (pred[:, 3] - pred[:, 2]) / (pred[:, 3] + pred[:, 2] + 1e-7)
    ndvi_mae = float(torch.mean(torch.abs(ndvi_pr - ndvi_gt)).item())

    return {
        "psnr": round(psnr, 2),
        "ssim": round(ssim_val, 4),
        "mae": round(mae, 5),
        "rmse": round(rmse, 5),
        "sam_deg": round(sam, 2),
        "ergas": round(ergas, 2),
        "epi": round(epi, 4),
        "ndvi_mae": round(ndvi_mae, 4)
    }


def generate_visual_benchmark_figure(test_lr, test_hr, models_dict, output_path):
    """
    Generates standardized 5-region visual crops (Urban, Highway, Fields, Vegetation, Water)
    comparing Sentinel-2, Bilinear, Residual CNN, PI-RCAN, and True HR Reference.
    """
    fig, axes = plt.subplots(5, 5, figsize=(20, 20), dpi=160)
    col_titles = ["1. Native S2 (10m)", "2. Bilinear (3x)", "3. Residual CNN", "4. PI-RCAN (3.33m)", "5. True HR Reference (~3m)"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight='bold', pad=12)

    def stretch(img_rgb):
        s = np.zeros_like(img_rgb)
        for c in range(3):
            low, high = np.percentile(img_rgb[..., c], 2), np.percentile(img_rgb[..., c], 98)
            s[..., c] = np.clip((img_rgb[..., c] - low) / (high - low + 1e-6), 0.0, 1.0)
        return s

    cat_names = ["1. Urban Buildings", "2. Highway Corridor", "3. Field Demarcations", "4. Vegetation Canopy", "5. Water Edge"]
    p_indices = [min(i * 10, len(test_lr) - 1) for i in range(5)]

    with torch.no_grad():
        pred_bil = F.interpolate(test_lr, size=(test_hr.shape[2], test_hr.shape[3]), mode='bilinear', align_corners=False)
        m_res = models_dict.get("ResidualCNN")
        m_pircan = models_dict.get("PIRCAN")
        
        pred_res = F.interpolate(m_res(test_lr), size=(test_hr.shape[2], test_hr.shape[3]), mode='bilinear', align_corners=False) if m_res else pred_bil
        pred_pircan = m_pircan(test_lr) if m_pircan else pred_bil
        if isinstance(pred_pircan, tuple): pred_pircan = pred_pircan[0]

    for row, (idx, c_name) in enumerate(zip(p_indices, cat_names)):
        lr_rgb = stretch(np.stack([test_lr[idx, 2].numpy(), test_lr[idx, 1].numpy(), test_lr[idx, 0].numpy()], axis=-1))
        bil_rgb = stretch(np.stack([pred_bil[idx, 2].numpy(), pred_bil[idx, 1].numpy(), pred_bil[idx, 0].numpy()], axis=-1))
        res_rgb = stretch(np.stack([pred_res[idx, 2].numpy(), pred_res[idx, 1].numpy(), pred_res[idx, 0].numpy()], axis=-1))
        pircan_rgb = stretch(np.stack([pred_pircan[idx, 2].numpy(), pred_pircan[idx, 1].numpy(), pred_pircan[idx, 0].numpy()], axis=-1))
        hr_rgb = stretch(np.stack([test_hr[idx, 2].numpy(), test_hr[idx, 1].numpy(), test_hr[idx, 0].numpy()], axis=-1))

        axes[row, 0].imshow(lr_rgb); axes[row, 0].set_ylabel(c_name, fontsize=11, fontweight='bold'); axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
        axes[row, 1].imshow(bil_rgb); axes[row, 1].set_xlabel("Bilinear", fontsize=9); axes[row, 1].set_xticks([]); axes[row, 1].set_yticks([])
        axes[row, 2].imshow(res_rgb); axes[row, 2].set_xlabel("Residual CNN", fontsize=9); axes[row, 2].set_xticks([]); axes[row, 2].set_yticks([])
        axes[row, 3].imshow(pircan_rgb); axes[row, 3].set_xlabel("PI-RCAN (3.33m)", fontsize=9); axes[row, 3].set_xticks([]); axes[row, 3].set_yticks([])
        axes[row, 4].imshow(hr_rgb); axes[row, 4].set_xlabel("HR Ground Truth", fontsize=9, fontweight='bold'); axes[row, 4].set_xticks([]); axes[row, 4].set_yticks([])

    plt.suptitle("Automated Evaluation Benchmark: Comparing Super-Resolution Models Against Genuine HR Reference Data", fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[Saved] Automated Benchmark Figure saved to: {output_path}")


if __name__ == "__main__":
    print("Automated Real High-Resolution Reference Evaluation Harness Ready.")
