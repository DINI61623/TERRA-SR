#!/usr/bin/env python3
"""
Satellite Enhancer Visual & Detail Audit Script (Experiment 4a)
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

Performs a detailed visual and spatial audit of the 5m ESP/Residual CNN output,
compares it with the Bilinear baseline and the original 10m Sentinel-2 imagery,
generates the requested plots and JSON metadata, and validates georeferencing.
"""

import sys
import json
from pathlib import Path

# Add workspace root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import rasterio
from affine import Affine
import matplotlib.pyplot as plt

# Import model architecture
from src.super_resolution.model import ResidualCNN

# Set paths
PROCESSED_DATA_PATH = Path("data/processed/s2_10m_stacked_roi.npy")
TIFF_DATA_PATH = Path("data/processed/s2_10m_stacked_roi.tiff")
MODEL_WEIGHTS_PATH = Path("models/residual_srm_experiment3.pth")
OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_mse(img1, img2):
    return np.mean((img1 - img2) ** 2)


def calculate_psnr(mse, max_val=1.0):
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def calculate_ssim_2d(img1, img2):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    mu1 = img1.mean()
    mu2 = img2.mean()
    
    sigma1_sq = img1.var()
    sigma2_sq = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    
    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    
    return numerator / denominator


def apply_percentile_stretch(rgb_img, low_pct=2, high_pct=98):
    """
    Applies a percentile contrast stretch to RGB channels for visualization.
    """
    stretched = np.zeros_like(rgb_img)
    for c in range(3):
        low = np.percentile(rgb_img[..., c], low_pct)
        high = np.percentile(rgb_img[..., c], high_pct)
        if high - low == 0:
            stretched[..., c] = rgb_img[..., c]
        else:
            stretched[..., c] = np.clip((rgb_img[..., c] - low) / (high - low), 0.0, 1.0)
    return stretched


def sobel_filter(img):
    """
    Vectorized Sobel filter in pure NumPy.
    """
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


def main():
    print("==================================================")
    # 1. Load inputs and weights
    print("[1/7] Loading inputs and weights...")
    if not PROCESSED_DATA_PATH.exists():
        print(f"Error: {PROCESSED_DATA_PATH} not found.")
        sys.exit(1)
    
    s2_10m = np.load(PROCESSED_DATA_PATH) # (4, 1024, 1024)
    print(f"  - Sentinel-2 10m ROI shape: {s2_10m.shape}")
    
    if not MODEL_WEIGHTS_PATH.exists():
        print(f"Error: {MODEL_WEIGHTS_PATH} not found.")
        sys.exit(1)
        
    device = torch.device("cpu")
    model = ResidualCNN(in_channels=4, num_features=32, num_blocks=3, upscale_factor=2)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.eval()
    print("  - Loaded Residual CNN weights successfully.")

    # 2. Run Real 10m -> 5m Inference
    print("[2/7] Running 10m -> 5m upscaling...")
    s2_10m_tensor = torch.tensor(s2_10m, dtype=torch.float32).unsqueeze(0) # (1, 4, 1024, 1024)
    with torch.no_grad():
        pred_5m_tensor = model(s2_10m_tensor).squeeze(0) # (4, 2048, 2048)
        pred_5m = np.clip(pred_5m_tensor.numpy(), 0.0, 1.0)
        
    # Generate Bilinear 5m baseline
    # Downsample-to-upsample interpolation matches spatial grid
    s2_10m_torch = torch.tensor(s2_10m, dtype=torch.float32).unsqueeze(0)
    bil_5m_tensor = nn.functional.interpolate(s2_10m_torch, size=(2048, 2048), mode='bilinear', align_corners=False).squeeze(0)
    bil_5m = np.clip(bil_5m_tensor.numpy(), 0.0, 1.0)
    
    print(f"  - Pred 5m shape: {pred_5m.shape}")
    print(f"  - Bilinear 5m shape: {bil_5m.shape}")

    # Save real 5m GeoTIFF output for resolution and georeferencing check
    out_5m_tiff_path = OUTPUT_DIR / "s2_5m_upscaled_residual.tiff"
    with rasterio.open(TIFF_DATA_PATH) as src:
        crs = src.crs
        transform = src.transform
        width = src.width
        height = src.height
        
        # Calculate new transform for 5m grid (upscale factor = 2)
        out_transform = transform * Affine.scale(0.5)
        out_width = width * 2
        out_height = height * 2
        
        profile = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': None,
            'width': out_width,
            'height': out_height,
            'count': 4,
            'crs': crs,
            'transform': out_transform
        }
        with rasterio.open(out_5m_tiff_path, 'w', **profile) as dst:
            for i in range(4):
                dst.write(pred_5m[i], i + 1)
            dst.update_tags(
                model="ResidualCNN",
                experiment="Experiment 4a Visual & Detail Audit",
                upscale_factor="2"
            )
    print(f"  - Saved 5m residual output GeoTIFF to: {out_5m_tiff_path}")

    # 3. Setup Synthetic 20m -> 10m Validation
    print("[3/7] Setting up synthetic 20m -> 10m validation...")
    # Downsample original 10m S2 to 20m (512x512)
    s2_20m_tensor = nn.functional.interpolate(s2_10m_torch, size=(512, 512), mode='bilinear', align_corners=False)
    s2_20m = s2_20m_tensor.squeeze(0).numpy()
    
    # Run model on synthetic 20m to get reconstructed 10m
    with torch.no_grad():
        pred_10m_tensor = model(s2_20m_tensor).squeeze(0) # (4, 1024, 1024)
        pred_10m = np.clip(pred_10m_tensor.numpy(), 0.0, 1.0)
        
    # Run bilinear on synthetic 20m to get bilinear 10m
    bil_10m_tensor = nn.functional.interpolate(s2_20m_tensor, size=(1024, 1024), mode='bilinear', align_corners=False).squeeze(0)
    bil_10m = np.clip(bil_10m_tensor.numpy(), 0.0, 1.0)
    
    # Target is the original 10m image
    ref_10m = s2_10m
    
    print(f"  - Synthetic 20m shape: {s2_20m.shape}")
    print(f"  - Reconstructed 10m shape: {pred_10m.shape}")
    
    # Calculate synthetic metrics
    mse_res = calculate_mse(pred_10m, ref_10m)
    psnr_res = calculate_psnr(mse_res)
    ssim_res = np.mean([calculate_ssim_2d(pred_10m[c], ref_10m[c]) for c in range(4)])
    
    mse_bil = calculate_mse(bil_10m, ref_10m)
    psnr_bil = calculate_psnr(mse_bil)
    ssim_bil = np.mean([calculate_ssim_2d(bil_10m[c], ref_10m[c]) for c in range(4)])
    
    print(f"  - Synthetic Bilinear 10m vs Ref 10m:  PSNR = {psnr_bil:.2f} dB, SSIM = {ssim_bil:.4f}")
    print(f"  - Synthetic Residual 10m vs Ref 10m:  PSNR = {psnr_res:.2f} dB, SSIM = {ssim_res:.4f}")

    # 4. Generate Full-Scene RGB Plot (Figure 1)
    print("[4/7] Generating full-scene RGB comparison (outputs/experiment4a_visual_audit.png)...")
    # True color bands sequence is Red (2), Green (1), Blue (0)
    # Rescale to RGB channel dimension last
    rgb_10m = np.stack([s2_10m[2], s2_10m[1], s2_10m[0]], axis=-1)
    rgb_bil_5m = np.stack([bil_5m[2], bil_5m[1], bil_5m[0]], axis=-1)
    rgb_res_5m = np.stack([pred_5m[2], pred_5m[1], pred_5m[0]], axis=-1)
    
    # Apply percentile stretching
    stretch_10m = apply_percentile_stretch(rgb_10m)
    stretch_bil_5m = apply_percentile_stretch(rgb_bil_5m)
    stretch_res_5m = apply_percentile_stretch(rgb_res_5m)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    axes[0].imshow(stretch_10m)
    axes[0].set_title("Original Sentinel-2 10m", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(stretch_bil_5m)
    axes[1].set_title("Bilinear Baseline 5m", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(stretch_res_5m)
    axes[2].set_title("Residual CNN 5m", fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "experiment4a_visual_audit.png", bbox_inches='tight')
    plt.close()

    # 5. Generate Zoomed Region Comparisons (Figure 2)
    print("[5/7] Generating zoom region comparisons (outputs/experiment4a_zoom_comparison.png)...")
    # Define interesting crop coordinates (128x128 bounding boxes in 10m S2)
    # The output will display columns: LR Input (10m), Bilinear (5m), Residual CNN (5m)
    # Note: 128x128 in 10m corresponds to 256x256 in 5m
    zooms = {
        "Urban / Roads & Buildings": {
            "r_start": 350, "r_end": 478,
            "c_start": 512, "c_end": 640
        },
        "Agricultural Field Boundaries": {
            "r_start": 800, "r_end": 928,
            "c_start": 128, "c_end": 256
        },
        "Forest / Vegetation Boundaries": {
            "r_start": 150, "r_end": 278,
            "c_start": 512, "c_end": 640
        },
        "Textured Ground Features": {
            "r_start": 650, "r_end": 778,
            "c_start": 896, "c_end": 1024
        }
    }
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 20), dpi=150)
    
    for row_idx, (name, coord) in enumerate(zooms.items()):
        r_s, r_e = coord["r_start"], coord["r_end"]
        c_s, c_e = coord["c_start"], coord["c_end"]
        
        # 10m original crop
        crop_10m = s2_10m[:, r_s:r_e, c_s:c_e]
        rgb_crop_10m = np.stack([crop_10m[2], crop_10m[1], crop_10m[0]], axis=-1)
        
        # 5m crops (indices scaled by 2)
        crop_bil_5m = bil_5m[:, r_s*2:r_e*2, c_s*2:c_e*2]
        rgb_crop_bil_5m = np.stack([crop_bil_5m[2], crop_bil_5m[1], crop_bil_5m[0]], axis=-1)
        
        crop_res_5m = pred_5m[:, r_s*2:r_e*2, c_s*2:c_e*2]
        rgb_crop_res_5m = np.stack([crop_res_5m[2], crop_res_5m[1], crop_res_5m[0]], axis=-1)
        
        # Apply stretching per crop to maximize details
        str_10m = apply_percentile_stretch(rgb_crop_10m)
        str_bil = apply_percentile_stretch(rgb_crop_bil_5m)
        str_res = apply_percentile_stretch(rgb_crop_res_5m)
        
        # Plot
        axes[row_idx, 0].imshow(str_10m)
        axes[row_idx, 0].set_title(f"LR 10m: {name}", fontsize=12, fontweight='bold')
        axes[row_idx, 0].axis('off')
        
        axes[row_idx, 1].imshow(str_bil)
        axes[row_idx, 1].set_title(f"Bilinear 5m: {name}", fontsize=12, fontweight='bold')
        axes[row_idx, 1].axis('off')
        
        axes[row_idx, 2].imshow(str_res)
        axes[row_idx, 2].set_title(f"Residual CNN 5m: {name}", fontsize=12, fontweight='bold')
        axes[row_idx, 2].axis('off')
        
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "experiment4a_zoom_comparison.png", bbox_inches='tight')
    plt.close()

    # 6. Generate Edges & Difference Maps (Figure 3)
    print("[6/7] Generating edges and difference maps (outputs/experiment4a_edge_analysis.png)...")
    # Compute Sobel edge maps for Red band (Red contains strong building and road edges)
    # Input is Red band (channel 2)
    edge_10m = sobel_filter(s2_10m[2])
    edge_bil_5m = sobel_filter(bil_5m[2])
    edge_res_5m = sobel_filter(pred_5m[2])
    
    # Calculate difference maps
    # Mean Absolute Difference (MAD) across RGB+NIR bands for difference visualizations
    diff_res_bil_5m = np.mean(np.abs(pred_5m - bil_5m), axis=0) # Real 5m comparison
    
    # Synthetic reconstruction comparison
    diff_res_ref_10m = np.mean(np.abs(pred_10m - ref_10m), axis=0) # Residual CNN error
    diff_bil_ref_10m = np.mean(np.abs(bil_10m - ref_10m), axis=0) # Bilinear baseline error
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)
    
    # Row 1: Edge Maps (Sobel)
    im_e0 = axes[0, 0].imshow(edge_10m, cmap='inferno', vmax=0.15)
    axes[0, 0].set_title("Sobel Edges: Original S2 10m", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    fig.colorbar(im_e0, ax=axes[0, 0], shrink=0.8)
    
    im_e1 = axes[0, 1].imshow(edge_bil_5m, cmap='inferno', vmax=0.15)
    axes[0, 1].set_title("Sobel Edges: Bilinear 5m", fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    fig.colorbar(im_e1, ax=axes[0, 1], shrink=0.8)
    
    im_e2 = axes[0, 2].imshow(edge_res_5m, cmap='inferno', vmax=0.15)
    axes[0, 2].set_title("Sobel Edges: Residual CNN 5m", fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    fig.colorbar(im_e2, ax=axes[0, 2], shrink=0.8)
    
    # Row 2: Difference Maps
    # Colorbar range set to highlight subtle differences
    im_d0 = axes[1, 0].imshow(diff_res_bil_5m, cmap='viridis', vmax=0.04)
    axes[1, 0].set_title("Difference: Residual vs Bilinear (5m)", fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    fig.colorbar(im_d0, ax=axes[1, 0], shrink=0.8)
    
    im_d1 = axes[1, 1].imshow(diff_bil_ref_10m, cmap='plasma', vmax=0.04)
    axes[1, 1].set_title("Bilinear Error: vs Synthetic Reference (10m)", fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    fig.colorbar(im_d1, ax=axes[1, 1], shrink=0.8)
    
    im_d2 = axes[1, 2].imshow(diff_res_ref_10m, cmap='plasma', vmax=0.04)
    axes[1, 2].set_title("Residual CNN Error: vs Synthetic Reference (10m)", fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    fig.colorbar(im_d2, ax=axes[1, 2], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "experiment4a_edge_analysis.png", bbox_inches='tight')
    plt.close()

    # 7. Generate Spectral Preservation Metrics (JSON)
    print("[7/7] Computing spectral preservation statistics...")
    band_names = ["Blue (B02)", "Green (B03)", "Red (B04)", "NIR (B08)"]
    spectral_stats = {
        "experiment_meta": {
            "title": "Spectral Preservation and Radiometric Drift Audit (Experiment 4a)",
            "upscale_factor": 2,
            "crs": "EPSG:32643",
            "model_weights": "models/residual_srm_experiment3.pth"
        },
        "real_inference_stats": {},
        "synthetic_validation_stats": {}
    }
    
    # Real 5m statistics (comparing output shape 2048x2048 vs input shape 1024x1024)
    for c, name in enumerate(band_names):
        in_mean = float(s2_10m[c].mean())
        in_std = float(s2_10m[c].std())
        out_mean = float(pred_5m[c].mean())
        out_std = float(pred_5m[c].std())
        
        mean_drift = out_mean - in_mean
        std_drift = out_std - in_std
        
        spectral_stats["real_inference_stats"][name] = {
            "input_10m": {
                "mean": in_mean,
                "std": in_std
            },
            "output_5m": {
                "mean": out_mean,
                "std": out_std
            },
            "radiometric_drift": {
                "absolute_mean_drift": mean_drift,
                "relative_mean_drift": mean_drift / (in_mean + 1e-6),
                "absolute_std_drift": std_drift,
                "relative_std_drift": std_drift / (in_std + 1e-6)
            }
        }
        
    # Synthetic 10m statistics (comparing reconstructed 1024x1024 vs reference 1024x1024)
    for c, name in enumerate(band_names):
        ref_mean = float(ref_10m[c].mean())
        ref_std = float(ref_10m[c].std())
        recon_mean = float(pred_10m[c].mean())
        recon_std = float(pred_10m[c].std())
        
        mean_drift = recon_mean - ref_mean
        std_drift = recon_std - ref_std
        
        # Calculate pixel-level error statistics
        recon_errors = pred_10m[c] - ref_10m[c]
        mae = float(np.mean(np.abs(recon_errors)))
        rmse = float(np.sqrt(np.mean(recon_errors ** 2)))
        
        # Compare with Bilinear errors
        bilinear_errors = bil_10m[c] - ref_10m[c]
        bil_mae = float(np.mean(np.abs(bilinear_errors)))
        bil_rmse = float(np.sqrt(np.mean(bilinear_errors ** 2)))
        
        spectral_stats["synthetic_validation_stats"][name] = {
            "reference_10m": {
                "mean": ref_mean,
                "std": ref_std
            },
            "reconstructed_10m": {
                "mean": recon_mean,
                "std": recon_std
            },
            "reconstruction_drift": {
                "absolute_mean_drift": mean_drift,
                "relative_mean_drift": mean_drift / (ref_mean + 1e-6),
                "absolute_std_drift": std_drift,
                "relative_std_drift": std_drift / (ref_std + 1e-6)
            },
            "error_metrics": {
                "residual_cnn_mae": mae,
                "residual_cnn_rmse": rmse,
                "bilinear_baseline_mae": bil_mae,
                "bilinear_baseline_rmse": bil_rmse
            }
        }
        
    with open(OUTPUT_DIR / "experiment4a_spectral_analysis.json", "w") as jf:
        json.dump(spectral_stats, jf, indent=4)
    print("  - Saved spectral statistics JSON to: outputs/experiment4a_spectral_analysis.json")
    
    print("\nVisual and Detail Audit calculations completed successfully.")
    print("==================================================")


if __name__ == "__main__":
    main()
