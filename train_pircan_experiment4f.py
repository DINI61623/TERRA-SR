#!/usr/bin/env python3
"""
Experiment 4F: Physically Informed Residual Channel Attention Network (PI-RCAN)
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

Key Features:
1. Multi-Objective Physics & Structural Loss:
   L = L_Charbonnier + 0.25*L_Laplacian + 0.15*L_SAM + 0.10*L_NDVI + 0.05*L_NLL_Uncertainty
2. High-Frequency Edge-Weighted Patch Sampling to prioritize complex structures.
3. Cross-Spectral NIR Edge Guidance & Channel Attention.
4. Scale-Factor Alignment for <4m target GSD (Scale x2: 5m, Scale x3: 3.33m, Scale x4: 2.5m).
5. Comprehensive 6-Way Benchmark against all previous experiments.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add workspace root to Python path
sys.path.append(str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
MODELS_DIR.mkdir(exist_ok=True)


# -------------------------------------------------------------
# Loss Functions
# -------------------------------------------------------------
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


class SpectralAngleLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(SpectralAngleLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        dot = torch.sum(pred * target, dim=1)
        norm_p = torch.norm(pred, p=2, dim=1)
        norm_t = torch.norm(target, p=2, dim=1)
        denom = torch.clamp(norm_p * norm_t, min=self.eps)
        cos_theta = torch.clamp(dot / denom, -0.9999, 0.9999)  # Prevent NaN gradients
        return torch.mean(torch.acos(cos_theta))


class NDVIConsistencyLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(NDVIConsistencyLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        # Channel 3: NIR, Channel 2: Red
        ndvi_p = (pred[:, 3:4] - pred[:, 2:3]) / (pred[:, 3:4] + pred[:, 2:3] + self.eps)
        ndvi_t = (target[:, 3:4] - target[:, 2:3]) / (target[:, 3:4] + target[:, 2:3] + self.eps)
        return F.l1_loss(ndvi_p, ndvi_t)


class LaplacianSecondOrderLoss(nn.Module):
    def __init__(self):
        super(LaplacianSecondOrderLoss, self).__init__()
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kernel", kernel)

    def forward(self, pred, target):
        C = pred.shape[1]
        k = self.kernel.repeat(C, 1, 1, 1).to(pred.device)
        lap_p = F.conv2d(pred, k, padding=1, groups=C)
        lap_t = F.conv2d(target, k, padding=1, groups=C)
        return F.l1_loss(lap_p, lap_t)


class HeteroscedasticNLLLoss(nn.Module):
    def __init__(self, eps=1e-4):
        super(HeteroscedasticNLLLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target, variance):
        var = torch.clamp(variance, min=self.eps, max=10.0)
        loss = 0.5 * ((pred - target) ** 2 / var + torch.log(var))
        return torch.mean(loss)


# -------------------------------------------------------------
# Dataset with Edge-Energy Prioritization
# -------------------------------------------------------------
class HighFrequencySatelliteDataset(Dataset):
    def __init__(self, patches, scale=2, min_edge_energy=0.0):
        self.scale = scale
        self.hr_patches = []
        
        for p in patches:
            p_tensor = torch.tensor(p, dtype=torch.float32)
            # Edge energy across NIR and Red bands
            gx = torch.diff(p_tensor[2], dim=1)[:, :-1]
            gy = torch.diff(p_tensor[2], dim=0)[:-1, :]
            energy = float((torch.mean(torch.abs(gx)) + torch.mean(torch.abs(gy))) / 2.0)
            if energy >= min_edge_energy:
                self.hr_patches.append(p_tensor)
                
        print(f"[Dataset] Filtered {len(self.hr_patches)} high-frequency structural patches (scale x{scale}).")

    def __len__(self):
        return len(self.hr_patches)

    def __getitem__(self, idx):
        hr = self.hr_patches[idx]
        C, H, W = hr.shape
        lr_h, lr_w = H // self.scale, W // self.scale
        lr = F.interpolate(hr.unsqueeze(0), size=(lr_h, lr_w), mode='bilinear', align_corners=False).squeeze(0)
        return lr, hr


def load_data():
    bands_paths = {
        "Blue": DATA_DIR / f"{PRODUCT_ID}_Blue_10m.jp2",
        "Green": DATA_DIR / f"{PRODUCT_ID}_Green_10m.jp2",
        "Red": DATA_DIR / f"{PRODUCT_ID}_Red_10m.jp2",
        "NIR": DATA_DIR / f"{PRODUCT_ID}_NIR_10m.jp2"
    }
    import rasterio
    from rasterio.windows import Window
    roi_size = 4096
    cx, cy = 10980 // 2, 10980 // 2
    window = Window(cx - (roi_size // 2), cy - (roi_size // 2), roi_size, roi_size)
    stacked = []
    print(f"[Data Loader] Loading {roi_size}x{roi_size} Sentinel-2 L2A BOA Reflectance...")
    for name in ["Blue", "Green", "Red", "NIR"]:
        with rasterio.open(bands_paths[name]) as src:
            data = src.read(1, window=window)
            refl = np.clip(data.astype(np.float32) / 10000.0, 0.0, 1.0)
            stacked.append(refl)
    stacked_arr = np.stack(stacked, axis=0)
    patches, _ = slice_into_patches(stacked_arr, patch_size=128, stride=128)
    return np.array(patches, dtype=np.float32)


# -------------------------------------------------------------
# Training & Benchmarking Pipeline
# -------------------------------------------------------------
def train_and_benchmark(args):
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")

    print("\n" + "="*80)
    print(f"EXPERIMENT 4F: PI-RCAN TRAINING & 6-WAY BENCHMARK EVALUATION (SCALE x{args.scale})")
    print("="*80)

    # 1. Load Data
    all_patches = load_data()
    train_raw = all_patches[:716]
    val_raw = all_patches[716:870]
    test_raw = all_patches[870:]

    train_ds = HighFrequencySatelliteDataset(train_raw, scale=args.scale, min_edge_energy=0.003)
    val_ds = HighFrequencySatelliteDataset(val_raw, scale=args.scale, min_edge_energy=0.0)
    test_ds = HighFrequencySatelliteDataset(test_raw, scale=args.scale, min_edge_energy=0.0)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 2. Instantiate Model
    model = PIRCAN(
        in_channels=4,
        out_channels=4,
        num_features=48,
        num_groups=3,
        num_rcab=4,
        reduction=8,
        upscale_factor=args.scale
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] PI-RCAN Initialized: {param_count:,} trainable parameters.")

    # 3. Losses & Optimizer
    loss_rec = CharbonnierLoss()
    loss_sam = SpectralAngleLoss()
    loss_ndvi = NDVIConsistencyLoss()
    loss_lap = LaplacianSecondOrderLoss()
    loss_nll = HeteroscedasticNLLLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_psnr = -1.0
    checkpoint_path = MODELS_DIR / "pircan_experiment4f.pth"

    print("\n[Training Loop] Beginning Optimization...")
    start_train_t = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for lr_b, hr_b in train_loader:
            lr_b, hr_b = lr_b.to(device), hr_b.to(device)
            optimizer.zero_grad()
            
            pred_refl, pred_var = model(lr_b, return_uncertainty=True)
            
            l_c = loss_rec(pred_refl, hr_b)
            l_s = loss_sam(pred_refl, hr_b)
            l_n = loss_ndvi(pred_refl, hr_b)
            l_l = loss_lap(pred_refl, hr_b)
            l_u = loss_nll(pred_refl, hr_b, pred_var)
            
            total_loss = l_c + 0.25 * l_l + 0.15 * l_s + 0.10 * l_n + 0.05 * l_u
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += total_loss.item() * len(lr_b)

        train_loss /= len(train_ds)
        scheduler.step()

        # Validation
        model.eval()
        val_psnr = 0.0
        with torch.no_grad():
            for lr_v, hr_v in val_loader:
                lr_v, hr_v = lr_v.to(device), hr_v.to(device)
                pred_v = model(lr_v)
                mse = torch.mean((pred_v - hr_v)**2, dim=(1, 2, 3))
                psnr_b = 20 * torch.log10(1.0 / torch.sqrt(torch.clamp(mse, min=1e-10)))
                val_psnr += torch.sum(psnr_b).item()

        val_psnr /= len(val_ds)
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {train_loss:.5f} | Val PSNR: {val_psnr:.2f} dB")

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), checkpoint_path)

    print(f"\n[Training Complete in {time.time() - start_train_t:.1f}s]")
    print(f"[Checkpoint Saved] Best PI-RCAN model saved to: {checkpoint_path} (Val PSNR: {best_val_psnr:.2f} dB)")

    # 4. Comprehensive 6-Way Benchmark
    print("\n" + "="*95)
    print("EXECUTING 6-WAY BENCHMARK ON HELD-OUT TEST DATASET")
    print("="*95)
    
    test_hr_tensors = [test_ds[i][1].unsqueeze(0) for i in range(len(test_ds))]
    test_hr = torch.cat(test_hr_tensors, dim=0).to(device)
    test_lr = F.interpolate(test_hr, size=(128 // args.scale, 128 // args.scale), mode='bilinear', align_corners=False)

    # Load All Baselines
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

    model_pircan = model
    model_pircan.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model_pircan.eval()

    with torch.no_grad():
        preds = {
            "Bilinear": torch.clamp(F.interpolate(test_lr, size=(128, 128), mode='bilinear', align_corners=False), 0.0, 1.0),
            "ESPCN": torch.clamp(model_espcn(test_lr), 0.0, 1.0),
            "ResidualCNN": torch.clamp(model_rescnn(test_lr), 0.0, 1.0),
            "MSRCAN": torch.clamp(model_msrcan(test_lr), 0.0, 1.0),
            "HFSRM": torch.clamp(model_hfsrm(test_lr), 0.0, 1.0),
            "PIRCAN": torch.clamp(model_pircan(test_lr), 0.0, 1.0)
        }

    # Compute Metrics
    def calc_metrics(p, t):
        mse = torch.mean((p - t)**2, dim=(1, 2, 3))
        psnr = 20 * torch.log10(1.0 / torch.sqrt(torch.clamp(mse, min=1e-10)))
        mae = torch.mean(torch.abs(p - t), dim=(1, 2, 3))
        rmse = torch.sqrt(mse)
        
        # SAM
        dot = torch.sum(p * t, dim=1)
        norm_p = torch.norm(p, p=2, dim=1)
        norm_t = torch.norm(t, p=2, dim=1)
        denom = torch.clamp(norm_p * norm_t, min=1e-7)
        sam = torch.rad2deg(torch.acos(torch.clamp(dot / denom, -1.0, 1.0)))
        
        # NDVI
        ndvi_t = (t[:, 3] - t[:, 2]) / (t[:, 3] + t[:, 2] + 1e-7)
        ndvi_p = (p[:, 3] - p[:, 2]) / (p[:, 3] + p[:, 2] + 1e-7)
        ndvi_mae = torch.mean(torch.abs(ndvi_p - ndvi_t), dim=(1, 2))

        # EPI (Sobel)
        k_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(4, 1, 1, 1)
        k_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(4, 1, 1, 1)
        gp = torch.sqrt(F.conv2d(p, k_x, padding=1, groups=4)**2 + F.conv2d(p, k_y, padding=1, groups=4)**2)
        gt = torch.sqrt(F.conv2d(t, k_x, padding=1, groups=4)**2 + F.conv2d(t, k_y, padding=1, groups=4)**2)
        epi = torch.sum(gp * gt, dim=(1, 2, 3)) / (torch.sqrt(torch.sum(gp**2, dim=(1, 2, 3)) * torch.sum(gt**2, dim=(1, 2, 3))) + 1e-7)

        return {
            "psnr": float(torch.mean(psnr).item()),
            "mae": float(torch.mean(mae).item()),
            "rmse": float(torch.mean(rmse).item()),
            "sam_deg": float(torch.mean(sam).item()),
            "ndvi_mae": float(torch.mean(ndvi_mae).item()),
            "epi": float(torch.mean(epi).item())
        }

    results = {}
    for name, p_tensor in preds.items():
        results[name] = calc_metrics(p_tensor, test_hr)

    # Print Table
    header = f"{'Metric':<25} | {'Bilinear':<10} | {'ESPCN':<10} | {'ResCNN':<10} | {'MS-RCAN':<10} | {'HF-SRM':<10} | {'PI-RCAN (Exp 4F)':<16}"
    print(header)
    print("-" * 110)
    for m_label, m_key, fmt in [
        ("Peak SNR (PSNR dB)", "psnr", ".2f"),
        ("Mean Abs Error (MAE)", "mae", ".5f"),
        ("Root Mean Sq Err (RMSE)", "rmse", ".5f"),
        ("Spectral Angle SAM (°)", "sam_deg", ".2f"),
        ("Edge Pres. Index (EPI)", "epi", ".4f"),
        ("NDVI Mean Abs Error", "ndvi_mae", ".4f")
    ]:
        row = f"{m_label:<25} | "
        for m_name in ["Bilinear", "ESPCN", "ResidualCNN", "MSRCAN", "HFSRM", "PIRCAN"]:
            val = results[m_name][m_key]
            row += f"{val:{fmt}:<10} | "
        print(row)
    print("=" * 110)

    # Save JSON
    benchmark_json = {
        "experiment": f"Experiment 4F: Physically Informed RCAN (Scale x{args.scale})",
        "scale_factor": args.scale,
        "input_resolution": "10.0m GSD",
        "output_resolution": f"{10.0 / args.scale:.2f}m GSD",
        "parameters": param_count,
        "metrics": results
    }
    with open(OUTPUT_DIR / "experiment4f_results.json", "w") as f:
        json.dump(benchmark_json, f, indent=4)
    print(f"[Saved] 6-Way Benchmark JSON saved to: {OUTPUT_DIR / 'experiment4f_results.json'}")

    # 5. Visual Figure
    print("[Visualization] Generating 6-Way Regional Comparison Figure...")
    fig, axes = plt.subplots(5, 7, figsize=(24, 18), dpi=150)
    col_titles = ["Reference (10m)", "Bilinear", "ESPCN", "ResCNN", "MS-RCAN", "HF-SRM", "PI-RCAN (Exp 4F)"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight='bold', pad=10)

    def stretch(img_rgb):
        s = np.zeros_like(img_rgb)
        for c in range(3):
            low, high = np.percentile(img_rgb[..., c], 2), np.percentile(img_rgb[..., c], 98)
            s[..., c] = np.clip((img_rgb[..., c] - low) / (high - low + 1e-6), 0.0, 1.0)
        return s

    p_indices = [5, 18, 32, 47, 60]
    cat_names = ["Urban Buildings", "Highway Corridor", "Field Boundaries", "Vegetation Canopy", "Water / Canal Edge"]

    test_hr_np = test_hr.numpy()
    preds_np = {k: v.numpy() for k, v in preds.items()}

    for row, (idx, c_name) in enumerate(zip(p_indices, cat_names)):
        gt_rgb = stretch(np.stack([test_hr_np[idx, 2], test_hr_np[idx, 1], test_hr_np[idx, 0]], axis=-1))
        axes[row, 0].imshow(gt_rgb)
        axes[row, 0].set_ylabel(c_name, fontsize=10, fontweight='bold')
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])

        for c_idx, m_name in enumerate(["Bilinear", "ESPCN", "ResidualCNN", "MSRCAN", "HFSRM", "PIRCAN"], start=1):
            pr_rgb = stretch(np.stack([preds_np[m_name][idx, 2], preds_np[m_name][idx, 1], preds_np[m_name][idx, 0]], axis=-1))
            mse = np.mean((preds_np[m_name][idx] - test_hr_np[idx])**2)
            psnr = 20 * np.log10(1.0 / np.sqrt(max(mse, 1e-10)))
            axes[row, c_idx].imshow(pr_rgb)
            axes[row, c_idx].set_xlabel(f"{psnr:.2f}dB", fontsize=9)
            axes[row, c_idx].set_xticks([]); axes[row, c_idx].set_yticks([])

    plt.suptitle(f"6-Way Super-Resolution Model Comparison: Spatial Structure & Detail Fidelity (Scale x{args.scale})", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    fig_path = OUTPUT_DIR / "experiment4f_benchmark_comparison.png"
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"[Saved] Comparison figure saved to: {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--scale", type=int, default=2, help="Upscale factor (2, 3, or 4)")
    args = parser.parse_args()
    train_and_benchmark(args)
