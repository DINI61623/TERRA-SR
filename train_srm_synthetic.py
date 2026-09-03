#!/usr/bin/env python3
"""
Synthetic Self-Supervised SRM Baseline Trainer (Experiment 2: Large Data & 100 Epochs)
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

Trains and evaluates an ESPCN model on synthetic LR-HR pairs generated from Sentinel-2 10m data.
Applies spatial train/val/test splits with a large buffer, computes per-band metrics,
saves checkpoints with the lowest validation loss, and outputs summaries.

EXPERIMENT DESCRIPTION:
"Synthetic 10m-to-5m SR baseline — NOT independent <4m ground-truth validation."
"""

import os
import sys
import json
import random
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Import ESPCN model architecture
from src.super_resolution.model import ESPCN
from src.utils.spatial_helpers import slice_into_patches


# --- Configuration and Reproducibility ---
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

DATA_DIR = Path("data/raw/sentinel2")
PRODUCT_ID = "S2B_MSIL2A_20260211T050839_N0512_R019_T43PGQ_20260211T085923"
OUTPUT_DIR = Path("outputs")
BAND_NAMES = ["Blue (B02)", "Green (B03)", "Red (B04)", "NIR (B08)"]


# --- Metrics Calculations ---
def calculate_mse(img1, img2):
    """
    Computes Mean Squared Error.
    """
    return np.mean((img1 - img2) ** 2)


def calculate_psnr(mse, max_val=1.0):
    """
    Computes Peak Signal-to-Noise Ratio (PSNR).
    Since inputs are normalized to 0.0 - 1.0, max_val defaults to 1.0.
    """
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def calculate_ssim_2d(img1, img2):
    """
    Computes Structural Similarity Index (SSIM) for a single 2D channel.
    """
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


def calculate_multispectral_ssim(img1, img2):
    """
    Computes mean per-band SSIM by averaging independent 2D SSIM calculations.
    Returns: (mean_ssim, list_of_band_ssims)
    """
    channels = img1.shape[0]
    ssims = []
    for c in range(channels):
        ssims.append(calculate_ssim_2d(img1[c], img2[c]))
    return np.mean(ssims), ssims


# --- Scale Helper for Previews ---
def scale_to_uint8(band_data):
    """
    Scale normalized 0-1 values to 8-bit uint8.
    """
    return (np.clip(band_data, 0.0, 1.0) * 255.0).astype(np.uint8)


def load_memory_safe_roi(data_dir, product_id, roi_size=4096):
    """
    Reads a center roi_size x roi_size region of interest from the JP2 files.
    This avoids loading the full 10980x10980 image into RAM.
    """
    bands_paths = {
        "Blue": Path(data_dir) / f"{product_id}_Blue_10m.jp2",
        "Green": Path(data_dir) / f"{product_id}_Green_10m.jp2",
        "Red": Path(data_dir) / f"{product_id}_Red_10m.jp2",
        "NIR": Path(data_dir) / f"{product_id}_NIR_10m.jp2"
    }
    
    for k, p in bands_paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing required band file: {p}")
            
    import rasterio
    from rasterio.windows import Window
    
    # Sentinel-2 size is 10980x10980
    cx, cy = 10980 // 2, 10980 // 2
    window = Window(cx - (roi_size // 2), cy - (roi_size // 2), roi_size, roi_size)
    
    stacked_bands = []
    print(f"Reading center {roi_size}x{roi_size} window from JP2 files...")
    for name in ["Blue", "Green", "Red", "NIR"]:
        with rasterio.open(bands_paths[name]) as src:
            data = src.read(1, window=window)
            # Normalize DN to reflectance [0, 1]
            reflectance = data.astype(np.float32) / 10000.0
            reflectance = np.clip(reflectance, 0.0, 1.0)
            stacked_bands.append(reflectance)
            
    return np.stack(stacked_bands, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Run Audited Synthetic Self-Supervised SRM training baseline (Experiment 2)."
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--visualize-only", action="store_true", help="Only generate visual comparisons using pre-trained weights")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("==================================================")
    print("Synthetic Self-Supervised SRM Pipeline - Experiment 2")
    print("==================================================")
    print("LABEL: Synthetic 10m-to-5m SR baseline — NOT independent <4m ground-truth validation.")
    
    # 1. Load memory-safe 4096x4096 ROI from JP2 bands
    stacked_image = load_memory_safe_roi(DATA_DIR, PRODUCT_ID, roi_size=4096)
    print(f"Loaded ROI image stack. Shape: {stacked_image.shape}")
    
    # 2. Slice into patches (128x128)
    # Slicing a 4096x4096 image into 128x128 non-overlapping patches yields a 32x32 grid = 1024 patches.
    patches, coords = slice_into_patches(stacked_image, patch_size=128, stride=128)
    patches_arr = np.array(patches, dtype=np.float32)
    print(f"Extracted unique patches: {patches_arr.shape[0]}")
    
    # 3. Spatial Patch Split with 2-Row Buffer Zone (Anti-Leakage)
    train_hr = torch.tensor(patches_arr[:768])
    val_hr = torch.tensor(patches_arr[832:928]) # Skip Rows 24-25
    test_hr = torch.tensor(patches_arr[928:])  # Rows 29-31
    
    # 4. Generate LR inputs by controlled bilinear downsampling (factor x2: 128x128 -> 64x64)
    train_lr = nn.functional.interpolate(train_hr, size=(64, 64), mode='bilinear', align_corners=False)
    val_lr = nn.functional.interpolate(val_hr, size=(64, 64), mode='bilinear', align_corners=False)
    test_lr = nn.functional.interpolate(test_hr, size=(64, 64), mode='bilinear', align_corners=False)
    
    print(f"\nSpatial Split (With 2560m Geographical Buffer Zone):")
    print(f"  - Train Split:      {train_lr.shape[0]} patches (Rows 0-23)")
    print(f"  - Buffer Zone:      64 patches (Rows 24-25 - Discarded)")
    print(f"  - Validation Split: {val_lr.shape[0]} patches (Rows 26-28)")
    print(f"  - Test Split:       {test_lr.shape[0]} patches (Rows 29-31)")
    
    # Pack into DataLoaders
    train_dataset = TensorDataset(train_lr, train_hr)
    val_dataset = TensorDataset(val_lr, val_hr)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 5. Initialize PyTorch model (4 input channels: RGB + NIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    model = ESPCN(in_channels=4, upscale_factor=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 6. Training and Validation loop with Checkpointing
    history = {"train_loss": [], "val_loss": []}
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "espcn_srm_synthetic.pth"
    
    best_val_loss = float('inf')
    
    if not args.visualize_only:
        print(f"\nStarting training loop for {args.epochs} epochs...")
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            for lr_batch, hr_batch in train_loader:
                lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
                
                optimizer.zero_grad()
                pred = model(lr_batch)
                loss = criterion(pred, hr_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * lr_batch.size(0)
                
            # Validation epoch loss
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for lr_val, hr_val in val_loader:
                    lr_val, hr_val = lr_val.to(device), hr_val.to(device)
                    pred_val = model(lr_val)
                    loss_val = criterion(pred_val, hr_val)
                    val_loss += loss_val.item() * lr_val.size(0)
                    
            epoch_train_loss = train_loss / len(train_dataset)
            epoch_val_loss = val_loss / len(val_dataset)
            history["train_loss"].append(epoch_train_loss)
            history["val_loss"].append(epoch_val_loss)
            
            # Checkpointing: Save model with lowest validation loss
            is_best = epoch_val_loss < best_val_loss
            if is_best:
                best_val_loss = epoch_val_loss
                torch.save(model.state_dict(), model_path)
                
            if (epoch + 1) % 10 == 0 or epoch == 0 or is_best:
                status_str = " (Best checkpoint saved)" if is_best else ""
                print(f"  Epoch [{epoch+1}/{args.epochs}] - Train Loss: {epoch_train_loss:.6f} - Val Loss: {epoch_val_loss:.6f}{status_str}")
        print(f"\nTraining completed. Loading best checkpoint model from: {model_path}")
    else:
        print(f"\n[Visualize Only Mode] Skipping training. Loading pre-trained checkpoint weights from: {model_path}")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 7. Evaluate on Test set
    print("\nEvaluating model against Bilinear Baseline on Test Set...")
    model.eval()
    
    # Compute bilinear baseline upscaled outputs for the test set
    test_bilinear = nn.functional.interpolate(test_lr, size=(128, 128), mode='bilinear', align_corners=False)
    
    with torch.no_grad():
        test_pred = model(test_lr.to(device)).cpu()
        
    # Convert tensors to numpy for metric calculations
    test_hr_np = test_hr.numpy()
    test_pred_np = test_pred.numpy()
    test_bilinear_np = test_bilinear.numpy()
    
    # Calculate metrics across all test patches
    model_mses, model_psnrs, model_ssims = [], [], []
    base_mses, base_psnrs, base_ssims = [], [], []
    
    model_band_mses = {c: [] for c in range(4)}
    model_band_psnrs = {c: [] for c in range(4)}
    model_band_ssims = {c: [] for c in range(4)}
    
    base_band_mses = {c: [] for c in range(4)}
    base_band_psnrs = {c: [] for c in range(4)}
    base_band_ssims = {c: [] for c in range(4)}
    
    num_test_patches = test_hr_np.shape[0]
    for i in range(num_test_patches):
        target = test_hr_np[i]
        
        # Model predictions (ESPCN) clipped to [0.0, 1.0] valid reflectance range for evaluation
        pred = np.clip(test_pred_np[i], 0.0, 1.0)
        
        # Bilinear baseline clipped to [0.0, 1.0] for fair evaluation
        bilinear = np.clip(test_bilinear_np[i], 0.0, 1.0)
        
        # --- Model Evaluation ---
        m_mse = calculate_mse(pred, target)
        m_psnr = calculate_psnr(m_mse)
        m_ssim, m_ssim_list = calculate_multispectral_ssim(pred, target)
        
        model_mses.append(m_mse)
        model_psnrs.append(m_psnr)
        model_ssims.append(m_ssim)
        
        # --- Bilinear Baseline Evaluation ---
        b_mse = calculate_mse(bilinear, target)
        b_psnr = calculate_psnr(b_mse)
        b_ssim, b_ssim_list = calculate_multispectral_ssim(bilinear, target)
        
        base_mses.append(b_mse)
        base_psnrs.append(b_psnr)
        base_ssims.append(b_ssim)
        
        # --- Per-Band Metric Logs ---
        for c in range(4):
            mc_mse = calculate_mse(pred[c], target[c])
            model_band_mses[c].append(mc_mse)
            model_band_psnrs[c].append(calculate_psnr(mc_mse))
            model_band_ssims[c].append(m_ssim_list[c])
            
            bc_mse = calculate_mse(bilinear[c], target[c])
            base_band_mses[c].append(bc_mse)
            base_band_psnrs[c].append(calculate_psnr(bc_mse))
            base_band_ssims[c].append(b_ssim_list[c])
            
    # Averaged Overall
    avg_model_mse = np.mean(model_mses)
    avg_model_psnr = np.mean(model_psnrs)
    avg_model_ssim = np.mean(model_ssims)
    
    avg_base_mse = np.mean(base_mses)
    avg_base_psnr = np.mean(base_psnrs)
    avg_base_ssim = np.mean(base_ssims)
    
    # Print Audited Comparisons
    print("\n==================================================")
    print("Quantitative Metric Results on Test Set (Average):")
    print("==================================================")
    print("Bilinear Baseline (Overall):")
    print(f"  - MSE:               {avg_base_mse:.6f}")
    print(f"  - PSNR:              {avg_base_psnr:.2f} dB")
    print(f"  - mean per-band SSIM:{avg_base_ssim:.4f}")
    print("\nTrained ESPCN Model (Overall Best Checkpoint):")
    print(f"  - MSE:               {avg_model_mse:.6f} (clipped)")
    print(f"  - PSNR:              {avg_model_psnr:.2f} dB")
    print(f"  - mean per-band SSIM:{avg_model_ssim:.4f}")
    
    print("\n--------------------------------------------------")
    print("Band-by-Band Performance Breakdown:")
    print("--------------------------------------------------")
    for c, band_name in enumerate(BAND_NAMES):
        print(f"\n* Band: {band_name}")
        print(f"  - Bilinear Baseline:")
        print(f"    MSE:  {np.mean(base_band_mses[c]):.6f} | PSNR: {np.mean(base_band_psnrs[c]):.2f} dB | SSIM: {np.mean(base_band_ssims[c]):.4f}")
        print(f"  - Trained ESPCN Model:")
        print(f"    MSE:  {np.mean(model_band_mses[c]):.6f} | PSNR: {np.mean(model_band_psnrs[c]):.2f} dB | SSIM: {np.mean(model_band_ssims[c]):.4f}")
    print("==================================================")

    # 8. Save side-by-side visualizations (Matplotlib 2x2 Grid)
    print("\nGenerating side-by-side comparison visualization...")
    import matplotlib.pyplot as plt
    
    patch_idx = 0
    
    lr_patch = test_lr[patch_idx].numpy()
    base_patch = np.clip(test_bilinear_np[patch_idx], 0.0, 1.0)
    pred_patch = np.clip(test_pred_np[patch_idx], 0.0, 1.0)
    hr_patch = test_hr_np[patch_idx]
    
    # We display standard True Color RGB composite: Red=B04 (index 2), Green=B03 (index 1), Blue=B02 (index 0)
    lr_rgb = np.stack([lr_patch[2], lr_patch[1], lr_patch[0]], axis=-1)
    base_rgb = np.stack([base_patch[2], base_patch[1], base_patch[0]], axis=-1)
    pred_rgb = np.stack([pred_patch[2], pred_patch[1], pred_patch[0]], axis=-1)
    hr_rgb = np.stack([hr_patch[2], hr_patch[1], hr_patch[0]], axis=-1)
    
    # Calculate 2% and 98% percentile bounds on the HR target image to ensure consistent contrast stretching
    low_bounds = [np.percentile(hr_rgb[..., c], 2) for c in range(3)]
    high_bounds = [np.percentile(hr_rgb[..., c], 98) for c in range(3)]
    
    def apply_percentile_stretch(img, low, high):
        stretched = np.zeros_like(img)
        for c in range(3):
            if high[c] - low[c] == 0:
                stretched[..., c] = img[..., c]
            else:
                stretched[..., c] = np.clip((img[..., c] - low[c]) / (high[c] - low[c]), 0.0, 1.0)
        return stretched
        
    lr_stretched = apply_percentile_stretch(lr_rgb, low_bounds, high_bounds)
    base_stretched = apply_percentile_stretch(base_rgb, low_bounds, high_bounds)
    pred_stretched = apply_percentile_stretch(pred_rgb, low_bounds, high_bounds)
    hr_stretched = apply_percentile_stretch(hr_rgb, low_bounds, high_bounds)
    
    # Nearest neighbor upscale the 64x64 LR patch to 128x128 for visualization alignment
    lr_rgb_up = np.repeat(np.repeat(lr_stretched, 2, axis=0), 2, axis=1)
    
    # Plot 2x2 comparison grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=100)
    
    axes[0, 0].imshow(lr_rgb_up)
    axes[0, 0].set_title("LR Input (20m)", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(base_stretched)
    axes[0, 1].set_title("Bilinear (10m)", fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(pred_stretched)
    axes[1, 0].set_title("ESPCN (10m)", fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(hr_stretched)
    axes[1, 1].set_title("HR Target (10m)", fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    comp_path = OUTPUT_DIR / "srm_synthetic_comparison.png"
    plt.savefig(comp_path, bbox_inches='tight', dpi=100)
    plt.close()
    print(f"Comparison preview saved to: {comp_path}")
    
    # Plot Zoomed Comparison (64x64 center crop of the 128x128 patches)
    # LR center is 32x32, upscaled to 64x64
    lr_zoom = lr_stretched[16:48, 16:48]
    lr_zoom_up = np.repeat(np.repeat(lr_zoom, 2, axis=0), 2, axis=1)
    
    base_zoom = base_stretched[32:96, 32:96]
    pred_zoom = pred_stretched[32:96, 32:96]
    hr_zoom = hr_stretched[32:96, 32:96]
    
    fig_zoom, axes_zoom = plt.subplots(2, 2, figsize=(12, 10), dpi=100)
    
    axes_zoom[0, 0].imshow(lr_zoom_up)
    axes_zoom[0, 0].set_title("LR Input (20m) [Zoomed]", fontsize=14, fontweight='bold')
    axes_zoom[0, 0].axis('off')
    
    axes_zoom[0, 1].imshow(base_zoom)
    axes_zoom[0, 1].set_title("Bilinear (10m) [Zoomed]", fontsize=14, fontweight='bold')
    axes_zoom[0, 1].axis('off')
    
    axes_zoom[1, 0].imshow(pred_zoom)
    axes_zoom[1, 0].set_title("ESPCN (10m) [Zoomed]", fontsize=14, fontweight='bold')
    axes_zoom[1, 0].axis('off')
    
    axes_zoom[1, 1].imshow(hr_zoom)
    axes_zoom[1, 1].set_title("HR Target (10m) [Zoomed]", fontsize=14, fontweight='bold')
    axes_zoom[1, 1].axis('off')
    
    plt.tight_layout()
    zoom_path = OUTPUT_DIR / "srm_synthetic_zoom_comparison.png"
    plt.savefig(zoom_path, bbox_inches='tight', dpi=100)
    plt.close()
    print(f"Zoomed comparison preview saved to: {zoom_path}")

    # 9. Record parameters and results log (JSON)
    # If in visualize-only mode, we keep existing history logs if available
    results_path = OUTPUT_DIR / "srm_synthetic_results.json"
    existing_history = history
    if args.visualize_only and results_path.exists():
        try:
            with open(results_path, "r") as rf:
                old_res = json.load(rf)
                existing_history = old_res.get("history", history)
        except Exception:
            pass
            
    results = {
        "experiment_label": "Synthetic 10m-to-5m SR baseline — NOT independent <4m ground-truth validation.",
        "parameters": {
            "model": "ESPCN",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "upscale_factor": 2,
            "total_unique_patches": len(patches),
            "train_patches": train_hr.shape[0],
            "val_patches": val_hr.shape[0],
            "test_patches": test_hr.shape[0],
            "spatial_split": "Rows 0-23: Train, Rows 24-25: Buffer (discarded), Rows 26-28: Val, Rows 29-31: Test",
            "spatial_buffer_meters": 2560.0
        },
        "history": existing_history,
        "metrics": {
            "bilinear_baseline": {
                "overall": {
                    "mse": float(avg_base_mse),
                    "psnr": float(avg_base_psnr),
                    "ssim": float(avg_base_ssim)
                },
                "per_band": {
                    BAND_NAMES[c]: {
                        "mse": float(np.mean(base_band_mses[c])),
                        "psnr": float(np.mean(base_band_psnrs[c])),
                        "ssim": float(np.mean(base_band_ssims[c]))
                    } for c in range(4)
                }
            },
            "trained_espcn": {
                "overall": {
                    "mse": float(avg_model_mse),
                    "psnr": float(avg_model_psnr),
                    "ssim": float(avg_model_ssim)
                },
                "per_band": {
                    BAND_NAMES[c]: {
                        "mse": float(np.mean(model_band_mses[c])),
                        "psnr": float(np.mean(model_band_psnrs[c])),
                        "ssim": float(np.mean(model_band_ssims[c]))
                    } for c in range(4)
                }
            }
        }
    }
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Experiment results log saved to: {results_path}")
    print("\nExperiment 2 execution complete!")


if __name__ == "__main__":
    main()
