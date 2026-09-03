#!/usr/bin/env python3
"""
Model Failure and Frequency Analysis Script (Experiment 3 Audit)
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

Performs a rigorous pixel-level, gradient, frequency, local contrast, and
structural similarity audit of the existing Residual CNN model against
the Bilinear baseline, using the original 10m Sentinel-2 image as ground truth
in a synthetic 20m -> 10m validation framework.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
# Add workspace root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.super_resolution.model import ResidualCNN
from src.utils.spatial_helpers import slice_into_patches

# --- Custom PyTorch implementations of uniform_filter and gaussian_filter to replace SciPy ---

def uniform_filter(img, size=5):
    t = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    pad = size // 2
    t_padded = nn.functional.pad(t, (pad, pad, pad, pad), mode='reflect')
    avg = nn.functional.avg_pool2d(t_padded, kernel_size=size, stride=1)
    return avg.squeeze().numpy()


def gaussian_filter(img, sigma=1.5):
    size = int(2 * round(3 * sigma) + 1)
    x = np.arange(-size//2 + 1., size//2 + 1.)
    g = np.exp(-x**2 / (2. * sigma**2))
    g = g / g.sum()
    kernel = np.outer(g, g)
    
    t = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    k = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    pad = size // 2
    t_padded = nn.functional.pad(t, (pad, pad, pad, pad), mode='reflect')
    conv = nn.functional.conv2d(t_padded, k, stride=1)
    return conv.squeeze().numpy()

# Paths
INPUT_NPY_PATH = Path("data/processed/s2_10m_stacked_roi.npy")
MODEL_WEIGHTS_PATH = Path("models/residual_srm_experiment3.pth")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- Metrics Definitions ---

def calculate_mse(img1, img2):
    return np.mean((img1 - img2) ** 2)


def calculate_psnr(mse, max_val=1.0):
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def calculate_local_ssim_2d(img1, img2, sigma=1.5):
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2
    
    mu1 = gaussian_filter(img1, sigma=sigma)
    mu2 = gaussian_filter(img2, sigma=sigma)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = gaussian_filter(img1 ** 2, sigma=sigma) - mu1_sq
    sigma2_sq = gaussian_filter(img2 ** 2, sigma=sigma) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sigma=sigma) - mu1_mu2
    
    sigma1_sq = np.clip(sigma1_sq, 0.0, None)
    sigma2_sq = np.clip(sigma2_sq, 0.0, None)
    
    num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    
    ssim_map = num / den
    return ssim_map.mean(), ssim_map


def calculate_multispectral_ssim(img1, img2):
    channels = img1.shape[0]
    ssims = []
    ssim_maps = []
    for c in range(channels):
        val, map_2d = calculate_local_ssim_2d(img1[c], img2[c])
        ssims.append(val)
        ssim_maps.append(map_2d)
    return np.mean(ssims), np.stack(ssim_maps, axis=0)


def sobel_filter(img):
    padded = np.pad(img, 1, mode='edge')
    grad_x = (
        -padded[0:-2, 0:-2] + padded[0:-2, 2:]
        -2 * padded[1:-1, 0:-2] + 2 * padded[1:-1, 2:]
        -padded[2:, 0:-2] + padded[2:, 2:]
    )
    grad_y = (
        -padded[0:-2, 0:-2] - 2 * padded[0:-2, 1:-1] - padded[0:-2, 2:]
        +padded[2:, 0:-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:]
    )
    grad = np.sqrt(grad_x**2 + grad_y**2)
    return grad


def calculate_epi(grad_img, grad_ref):
    mean_img = grad_img.mean()
    mean_ref = grad_ref.mean()
    
    diff_img = grad_img - mean_img
    diff_ref = grad_ref - mean_ref
    
    num = np.sum(diff_img * diff_ref)
    den = np.sqrt(np.sum(diff_img ** 2) * np.sum(diff_ref ** 2))
    
    if den == 0:
        return 0.0
    return num / den


def calculate_local_contrast(img, size=5):
    mean = uniform_filter(img, size=size)
    mean_sq = uniform_filter(img**2, size=size)
    var = np.clip(mean_sq - mean**2, 0.0, None)
    return np.mean(np.sqrt(var))


def calculate_frequency_energy(img, high_freq_radius_ratio=0.15):
    """
    Computes FFT2 magnitude spectrum and returns total energy and high-frequency energy ratio.
    """
    h, w = img.shape
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    
    # Define circular high-frequency mask
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - cy)**2 + (x - cx)**2)
    
    radius = high_freq_radius_ratio * min(h, w)
    hf_mask = dist > radius
    
    total_energy = np.sum(magnitude)
    hf_energy = np.sum(magnitude[hf_mask])
    
    return float(total_energy), float(hf_energy), magnitude


def apply_percentile_stretch(rgb_img, low_pct=2, high_pct=98):
    stretched = np.zeros_like(rgb_img)
    for c in range(3):
        low = np.percentile(rgb_img[..., c], low_pct)
        high = np.percentile(rgb_img[..., c], high_pct)
        if high - low == 0:
            stretched[..., c] = rgb_img[..., c]
        else:
            stretched[..., c] = np.clip((rgb_img[..., c] - low) / (high - low), 0.0, 1.0)
    return stretched


def main():
    print("==================================================")
    print("Model Failure and Frequency Analysis Audit")
    print("==================================================")

    # 1. Load data & weights
    print("[1/5] Loading original 10m Sentinel-2 image...")
    target = np.load(INPUT_NPY_PATH)  # shape (4, 1024, 1024), float32, range [0, 1]
    print(f"  - Target shape: {target.shape}, range: [{target.min():.4f}, {target.max():.4f}]")
    
    print("[2/5] Initializing Model and executing upscaling...")
    device = torch.device("cpu")
    model = ResidualCNN(in_channels=4, num_features=32, num_blocks=3, upscale_factor=2)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.eval()

    # 2. Downsample and Upscale (Synthetic 20m -> 10m)
    target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)  # (1, 4, 1024, 1024)
    
    # Downsample to 20m (512x512)
    lr_tensor = nn.functional.interpolate(target_tensor, size=(512, 512), mode='bilinear', align_corners=False)
    
    # Upscale back to 10m using Bilinear
    bil_tensor = nn.functional.interpolate(lr_tensor, size=(1024, 1024), mode='bilinear', align_corners=False).squeeze(0)
    bil = np.clip(bil_tensor.numpy(), 0.0, 1.0)
    
    # Upscale back to 10m using Residual CNN
    with torch.no_grad():
        res_tensor = model(lr_tensor).squeeze(0)
        res_raw = res_tensor.numpy()
        res = np.clip(res_raw, 0.0, 1.0)

    # Note the raw unclipped means to detect scaling bias
    print(f"  - Original 10m Target Mean:       {[round(float(m), 4) for m in target.mean(axis=(1,2))]}")
    print(f"  - Residual CNN Output Raw Mean:   {[round(float(m), 4) for m in res_raw.mean(axis=(1,2))]}")
    print(f"  - Residual CNN Output Clipped Mean: {[round(float(m), 4) for m in res.mean(axis=(1,2))]}")

    # 3. Calculate Global Metrics
    print("[3/5] Computing global quantitative metrics...")
    global_stats = {}
    bands = ["Blue", "Green", "Red", "NIR"]
    
    for i, band in enumerate(bands):
        bil_mse = float(calculate_mse(bil[i], target[i]))
        res_mse = float(calculate_mse(res[i], target[i]))
        
        bil_psnr = float(calculate_psnr(bil_mse))
        res_psnr = float(calculate_psnr(res_mse))
        
        bil_ssim, _ = calculate_local_ssim_2d(bil[i], target[i])
        res_ssim, _ = calculate_local_ssim_2d(res[i], target[i])
        
        # FFT High Frequency Energy
        t_tot, t_hf, _ = calculate_frequency_energy(target[i])
        bil_tot, bil_hf, _ = calculate_frequency_energy(bil[i])
        res_tot, res_hf, _ = calculate_frequency_energy(res[i])
        
        # Local Contrast (Mean Standard Deviation)
        t_contrast = float(calculate_local_contrast(target[i]))
        bil_contrast = float(calculate_local_contrast(bil[i]))
        res_contrast = float(calculate_local_contrast(res[i]))
        
        # Gradient Sobel & EPI
        t_grad = sobel_filter(target[i])
        bil_grad = sobel_filter(bil[i])
        res_grad = sobel_filter(res[i])
        
        bil_epi = float(calculate_epi(bil_grad, t_grad))
        res_epi = float(calculate_epi(res_grad, t_grad))
        
        global_stats[band] = {
            "bilinear": {
                "mse": bil_mse,
                "psnr": bil_psnr,
                "ssim": float(bil_ssim),
                "epi": bil_epi,
                "local_contrast": bil_contrast,
                "hf_energy_ratio_to_target": bil_hf / t_hf if t_hf > 0 else 0.0
            },
            "residual_cnn": {
                "mse": res_mse,
                "psnr": res_psnr,
                "ssim": float(res_ssim),
                "epi": res_epi,
                "local_contrast": res_contrast,
                "hf_energy_ratio_to_target": res_hf / t_hf if t_hf > 0 else 0.0
            },
            "target": {
                "local_contrast": t_contrast,
                "hf_energy_ratio": t_hf / t_tot if t_tot > 0 else 0.0
            }
        }

    # Save frequency and metrics JSON
    with open(OUTPUT_DIR / "model_frequency_analysis.json", "w") as f:
        json.dump(global_stats, f, indent=4)
    print("  - Saved quantitative stats to: outputs/model_frequency_analysis.json")

    # 4. Regional Slicing and Verification
    print("[4/5] Running regional feature audit...")
    # S2 Crops (r_start, c_start)
    regions = {
        "Buildings": (350, 512),
        "Roads": (400, 480),
        "Field Boundaries": (600, 200),
        "Vegetation Boundaries": (200, 750),
        "Homogeneous Areas": (800, 800)
    }
    
    region_stats = {}
    crop_size = 128
    
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    for col_idx, (name, (r_s, c_s)) in enumerate(regions.items()):
        # Crop from Red band (channel 2)
        target_crop = target[:, r_s:r_s+crop_size, c_s:c_s+crop_size]
        bil_crop = bil[:, r_s:r_s+crop_size, c_s:c_s+crop_size]
        res_crop = res[:, r_s:r_s+crop_size, c_s:c_s+crop_size]
        
        # Calculate crops local metrics (averaged over 4 bands)
        c_bil_mse = np.mean([calculate_mse(bil_crop[c], target_crop[c]) for c in range(4)])
        c_res_mse = np.mean([calculate_mse(res_crop[c], target_crop[c]) for c in range(4)])
        
        c_bil_psnr = calculate_psnr(c_bil_mse)
        c_res_psnr = calculate_psnr(c_res_mse)
        
        c_bil_ssim = np.mean([calculate_local_ssim_2d(bil_crop[c], target_crop[c])[0] for c in range(4)])
        c_res_ssim = np.mean([calculate_local_ssim_2d(res_crop[c], target_crop[c])[0] for c in range(4)])
        
        c_t_grad = [sobel_filter(target_crop[c]) for c in range(4)]
        c_bil_grad = [sobel_filter(bil_crop[c]) for c in range(4)]
        c_res_grad = [sobel_filter(res_crop[c]) for c in range(4)]
        
        c_bil_epi = np.mean([calculate_epi(c_bil_grad[c], c_t_grad[c]) for c in range(4)])
        c_res_epi = np.mean([calculate_epi(c_res_grad[c], c_t_grad[c]) for c in range(4)])
        
        c_t_contrast = np.mean([calculate_local_contrast(target_crop[c]) for c in range(4)])
        c_bil_contrast = np.mean([calculate_local_contrast(bil_crop[c]) for c in range(4)])
        c_res_contrast = np.mean([calculate_local_contrast(res_crop[c]) for c in range(4)])
        
        c_t_hf = np.mean([calculate_frequency_energy(target_crop[c])[1] for c in range(4)])
        c_bil_hf = np.mean([calculate_frequency_energy(bil_crop[c])[1] for c in range(4)])
        c_res_hf = np.mean([calculate_frequency_energy(res_crop[c])[1] for c in range(4)])
        
        region_stats[name] = {
            "bilinear": {
                "psnr": float(c_bil_psnr),
                "ssim": float(c_bil_ssim),
                "epi": float(c_bil_epi),
                "local_contrast": float(c_bil_contrast),
                "hf_energy_ratio": float(c_bil_hf / c_t_hf) if c_t_hf > 0 else 0.0
            },
            "residual_cnn": {
                "psnr": float(c_res_psnr),
                "ssim": float(c_res_ssim),
                "epi": float(c_res_epi),
                "local_contrast": float(c_res_contrast),
                "hf_energy_ratio": float(c_res_hf / c_t_hf) if c_t_hf > 0 else 0.0
            }
        }
        
        # Plot RGB crops
        # Target
        t_rgb = np.stack([target_crop[2], target_crop[1], target_crop[0]], axis=-1)
        axes[0, col_idx].imshow(apply_percentile_stretch(t_rgb))
        axes[0, col_idx].axis('off')
        if col_idx == 0:
            axes[0, col_idx].set_ylabel("Target 10m", fontsize=12, labelpad=20, rotation=0, ha='right')
        axes[0, col_idx].set_title(f"{name}\nTarget", fontsize=10)
        
        # Bilinear
        b_rgb = np.stack([bil_crop[2], bil_crop[1], bil_crop[0]], axis=-1)
        axes[1, col_idx].imshow(apply_percentile_stretch(b_rgb))
        axes[1, col_idx].axis('off')
        axes[1, col_idx].text(5, 120, f"PSNR: {c_bil_psnr:.2f}\nSSIM: {c_bil_ssim:.3f}\nEPI: {c_bil_epi:.3f}", 
                              color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.6))
        if col_idx == 0:
            axes[1, col_idx].set_ylabel("Bilinear 10m", fontsize=12, labelpad=20, rotation=0, ha='right')
            
        # Residual CNN
        r_rgb = np.stack([res_crop[2], res_crop[1], res_crop[0]], axis=-1)
        axes[2, col_idx].imshow(apply_percentile_stretch(r_rgb))
        axes[2, col_idx].axis('off')
        axes[2, col_idx].text(5, 120, f"PSNR: {c_res_psnr:.2f}\nSSIM: {c_res_ssim:.3f}\nEPI: {c_res_epi:.3f}", 
                              color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.6))
        if col_idx == 0:
            axes[2, col_idx].set_ylabel("Residual CNN", fontsize=12, labelpad=20, rotation=0, ha='right')

    plt.suptitle("Experiment 3: Regional Slicing & Detail Audit Comparison", fontsize=14, y=0.98)
    plt.savefig(OUTPUT_DIR / "model_failure_regions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  - Saved regional audit plot to: outputs/model_failure_regions.png")

    # 5. Generate Full-Scene and FFT Plots
    print("[5/5] Generating FFT Frequency Analysis plot...")
    # Full scene RGB stretch
    t_rgb_full = np.stack([target[2], target[1], target[0]], axis=-1)
    b_rgb_full = np.stack([bil[2], bil[1], bil[0]], axis=-1)
    r_rgb_full = np.stack([res[2], res[1], res[0]], axis=-1)
    
    # FFT magnitude spectrum on Red channel (band 2)
    _, _, t_fft = calculate_frequency_energy(target[2])
    _, _, b_fft = calculate_frequency_energy(bil[2])
    _, _, r_fft = calculate_frequency_energy(res[2])
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: RGB Views
    axes[0, 0].imshow(apply_percentile_stretch(t_rgb_full))
    axes[0, 0].set_title("Ground Truth Target (10m)")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(apply_percentile_stretch(b_rgb_full))
    axes[0, 1].set_title("Bilinear Baseline (10m)")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(apply_percentile_stretch(r_rgb_full))
    axes[0, 2].set_title("Residual CNN Reconstruction (10m)")
    axes[0, 2].axis('off')
    
    # Row 2: FFT Frequency Spectra (Log scale)
    t_fft_log = np.log(1 + t_fft)
    axes[1, 0].imshow(t_fft_log, cmap='magma')
    axes[1, 0].set_title("Target FFT Frequency Spectrum")
    axes[1, 0].axis('off')
    
    b_fft_log = np.log(1 + b_fft)
    axes[1, 1].imshow(b_fft_log, cmap='magma')
    axes[1, 1].set_title("Bilinear FFT Frequency Spectrum\n(Low pass filter effect)")
    axes[1, 1].axis('off')
    
    r_fft_log = np.log(1 + r_fft)
    axes[1, 2].imshow(r_fft_log, cmap='magma')
    axes[1, 2].set_title("Residual CNN FFT Frequency Spectrum\n(High-frequency suppression)")
    axes[1, 2].axis('off')
    
    plt.suptitle("Experiment 3: Full-Scene & Frequency Spectrum Audit", fontsize=14, y=0.98)
    plt.savefig(OUTPUT_DIR / "model_failure_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  - Saved full-scene FFT analysis plot to: outputs/model_failure_analysis.png")
    print("==================================================")
    print("Audit Complete.")
    print("==================================================")


if __name__ == "__main__":
    main()
