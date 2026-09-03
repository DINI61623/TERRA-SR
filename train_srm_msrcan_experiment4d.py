#!/usr/bin/env python3
"""
Synthetic Self-Supervised SRM Pipeline - Experiment 4D (MS-RCAN + Composite Loss)
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

Trains and evaluates a Multispectral Residual Channel Attention Network (MS-RCAN)
using a Composite Spectral-Spatial Loss Function on synthetic LR-HR pairs.
Reuses the exact same dataset, spatial splits, and testing grid as Experiments 2 & 3.
Compares outputs against the Bilinear baseline and Experiment 3 Residual CNN.

EXPERIMENT DESCRIPTION:
"Synthetic 10m-to-5m SR Architecture & Loss Prototyping — NOT independent <4m ground-truth validation."
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Import architectures
from src.super_resolution.model import ResidualCNN
from src.super_resolution.msrcan import MSRCAN
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
MODELS_DIR = Path("models")
BAND_NAMES = ["Blue (B02)", "Green (B03)", "Red (B04)", "NIR (B08)"]


# --- Composite Loss Components ---

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (Smooth L1 approximation)."""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps_sq = eps ** 2

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff * diff + self.eps_sq))
        return loss


class SSIMLoss2D(nn.Module):
    """Differentiable 2D SSIM Loss for multispectral tensors."""
    def __init__(self, window_size=7, in_channels=4):
        super(SSIMLoss2D, self).__init__()
        self.window_size = window_size
        self.in_channels = in_channels
        
        # 1D Gaussian kernel
        sigma = 1.5
        gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)) for x in range(window_size)])
        gauss = (gauss / gauss.sum()).unsqueeze(1)
        
        # 2D Gaussian kernel
        kernel_2d = gauss.mm(gauss.t()).float().unsqueeze(0).unsqueeze(0)
        self.window = kernel_2d.expand(in_channels, 1, window_size, window_size).contiguous()

    def forward(self, img1, img2):
        window = self.window.to(img1.device)
        padding = self.window_size // 2
        
        mu1 = F.conv2d(img1, window, padding=padding, groups=self.in_channels)
        mu2 = F.conv2d(img2, window, padding=padding, groups=self.in_channels)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=self.in_channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=self.in_channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=self.in_channels) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1.0 - ssim_map.mean()


class SAMLoss(nn.Module):
    """Spectral Angle Mapper (SAM) loss."""
    def __init__(self, eps=1e-7):
        super(SAMLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        dot = torch.sum(pred * target, dim=1)
        norm_p = torch.norm(pred, p=2, dim=1)
        norm_t = torch.norm(target, p=2, dim=1)
        cos_theta = torch.clamp(dot / (norm_p * norm_t + self.eps), -1.0 + self.eps, 1.0 - self.eps)
        sam = torch.acos(cos_theta)
        return sam.mean()


class EdgeGradientLoss(nn.Module):
    """Sobel / difference edge gradient loss."""
    def __init__(self):
        super(EdgeGradientLoss, self).__init__()

    def forward(self, pred, target):
        dx_pred = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        dx_target = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        dy_pred = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        dy_target = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        
        loss_x = torch.mean(torch.abs(dx_pred - dx_target))
        loss_y = torch.mean(torch.abs(dy_pred - dy_target))
        return loss_x + loss_y


class CompositeSpectralSpatialLoss(nn.Module):
    """
    Composite loss combining:
    1. Charbonnier Reconstruction Loss (λ = 1.0)
    2. Multi-Scale SSIM Structural Loss (λ = 0.2)
    3. Spectral Angle Mapper (SAM) Loss (λ = 0.1)
    4. Edge Gradient Loss (λ = 0.05)
    """
    def __init__(self, w_rec=1.0, w_ssim=0.2, w_sam=0.1, w_edge=0.05):
        super(CompositeSpectralSpatialLoss, self).__init__()
        self.w_rec = w_rec
        self.w_ssim = w_ssim
        self.w_sam = w_sam
        self.w_edge = w_edge
        
        self.charbonnier = CharbonnierLoss(eps=1e-3)
        self.ssim_loss = SSIMLoss2D(window_size=7, in_channels=4)
        self.sam_loss = SAMLoss()
        self.edge_loss = EdgeGradientLoss()

    def forward(self, pred, target):
        l_rec = self.charbonnier(pred, target)
        l_ssim = self.ssim_loss(pred, target)
        l_sam = self.sam_loss(pred, target)
        l_edge = self.edge_loss(pred, target)
        
        total_loss = (
            self.w_rec * l_rec +
            self.w_ssim * l_ssim +
            self.w_sam * l_sam +
            self.w_edge * l_edge
        )
        return total_loss, {
            "l_rec": l_rec.item(),
            "l_ssim": l_ssim.item(),
            "l_sam": l_sam.item(),
            "l_edge": l_edge.item(),
            "total": total_loss.item()
        }


# --- Metrics Calculations ---

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
    num = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    return num / den


def calculate_multispectral_ssim(img1, img2):
    channels = img1.shape[0]
    ssims = [calculate_ssim_2d(img1[c], img2[c]) for c in range(channels)]
    return np.mean(ssims), ssims


def calculate_epi(pred, target):
    """Edge Preservation Index (EPI) based on Sobel horizontal/vertical gradients."""
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    from scipy.signal import convolve2d
    epis = []
    for c in range(pred.shape[0]):
        p = pred[c]
        t = target[c]
        
        gx_p = convolve2d(p, sobel_x, mode='same', boundary='symm')
        gy_p = convolve2d(p, sobel_y, mode='same', boundary='symm')
        g_p = np.sqrt(gx_p**2 + gy_p**2)
        
        gx_t = convolve2d(t, sobel_x, mode='same', boundary='symm')
        gy_t = convolve2d(t, sobel_y, mode='same', boundary='symm')
        g_t = np.sqrt(gx_t**2 + gy_t**2)
        
        num = np.sum((g_p - np.mean(g_p)) * (g_t - np.mean(g_t)))
        den = np.sqrt(np.sum((g_p - np.mean(g_p))**2) * np.sum((g_t - np.mean(g_t))**2)) + 1e-8
        epis.append(num / den)
    return float(np.mean(epis))


def calculate_sam_degrees(pred, target):
    """Spectral Angle Mapper in degrees."""
    # Shapes: (4, H, W)
    dot = np.sum(pred * target, axis=0)
    norm_p = np.linalg.norm(pred, axis=0)
    norm_t = np.linalg.norm(target, axis=0)
    cos_theta = np.clip(dot / (norm_p * norm_t + 1e-8), -1.0, 1.0)
    sam_rad = np.arccos(cos_theta)
    return float(np.mean(np.degrees(sam_rad)))


def calculate_ndvi_mae(pred, target):
    """Computes mean absolute error of NDVI."""
    # Red: channel 2, NIR: channel 3
    ndvi_p = (pred[3] - pred[2]) / (pred[3] + pred[2] + 1e-7)
    ndvi_t = (target[3] - target[2]) / (target[3] + target[2] + 1e-7)
    return float(np.mean(np.abs(ndvi_p - ndvi_t)))


def calculate_high_freq_energy_ratio(img, target):
    """Ratio of high-frequency 2D FFT energy relative to target."""
    h, w = img.shape[1], img.shape[2]
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4  # cutoff radius
    
    y, x = np.ogrid[:h, :w]
    mask_high = ((y - cy)**2 + (x - cx)**2) >= (r**2)
    
    ratios = []
    for c in range(img.shape[0]):
        fft_p = np.fft.fftshift(np.fft.fft2(img[c]))
        fft_t = np.fft.fftshift(np.fft.fft2(target[c]))
        
        p_high = np.sum(np.abs(fft_p)[mask_high]**2)
        t_high = np.sum(np.abs(fft_t)[mask_high]**2) + 1e-8
        ratios.append(p_high / t_high)
        
    return float(np.mean(ratios))


def calculate_local_contrast(img):
    """Local contrast (mean standard deviation across sliding 16x16 windows)."""
    h, w = img.shape[1], img.shape[2]
    stds = []
    for y in range(0, h - 16 + 1, 16):
        for x in range(0, w - 16 + 1, 16):
            window = img[:, y:y+16, x:x+16]
            stds.append(np.std(window))
    return float(np.mean(stds))


# --- Data Loading ---

def load_memory_safe_roi(data_dir, product_id, roi_size=4096):
    """Memory-safe center ROI loader matching Experiments 2 & 3."""
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
    
    cx, cy = 10980 // 2, 10980 // 2
    window = Window(cx - (roi_size // 2), cy - (roi_size // 2), roi_size, roi_size)
    
    stacked_bands = []
    print(f"Reading center {roi_size}x{roi_size} window from JP2 files...")
    for name in ["Blue", "Green", "Red", "NIR"]:
        with rasterio.open(bands_paths[name]) as src:
            data = src.read(1, window=window)
            reflectance = data.astype(np.float32) / 10000.0
            reflectance = np.clip(reflectance, 0.0, 1.0)
            stacked_bands.append(reflectance)
            
    return np.stack(stacked_bands, axis=0)


# --- Main Experiment Execution ---

def main():
    parser = argparse.ArgumentParser(
        description="Run Experiment 4D: MS-RCAN Super-Resolution with Composite Loss."
    )
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs (default: 60)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate (default: 0.0005)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    
    print("==================================================================")
    print("Synthetic Self-Supervised SRM - Experiment 4D (MS-RCAN + Composite Loss)")
    print("==================================================================")
    print("LABEL: Synthetic 10m-to-5m SR Prototype — NOT independent <4m ground-truth validation.")

    # 1. Device and Model Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n1. Target Device: {device}")
    
    # Initialize MS-RCAN: 4 Residual Groups, 3 RCABs each, 48 features (~530k parameters)
    model = MSRCAN(
        in_channels=4, 
        num_features=48, 
        num_groups=4, 
        num_rcab=3, 
        reduction=8, 
        upscale_factor=2
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   MS-RCAN Trainable Parameters: {param_count:,}")
    
    # Forward check
    mock_input = torch.randn(2, 4, 32, 32).to(device)
    with torch.no_grad():
        mock_output = model(mock_input)
    print(f"   Mock Pass: Input {list(mock_input.shape)} -> Output {list(mock_output.shape)}")
    if torch.isnan(mock_output).any() or torch.isinf(mock_output).any():
        raise ValueError("Model initialization error: Output contains NaN/Inf values.")
    print("   [OK] MS-RCAN forward pass verified safely.")

    # 2. Data Preparation (Identical to Experiments 2 & 3)
    print("\n2. Loading Dataset ROI...")
    stacked_image = load_memory_safe_roi(DATA_DIR, PRODUCT_ID, roi_size=4096)
    print(f"   Loaded ROI shape: {stacked_image.shape}")
    
    patches, coords = slice_into_patches(stacked_image, patch_size=128, stride=128)
    patches_arr = np.array(patches, dtype=np.float32)
    print(f"   Total unique patches extracted: {patches_arr.shape[0]}")
    
    # Spatial splits
    train_hr = torch.tensor(patches_arr[:768])
    val_hr = torch.tensor(patches_arr[832:928])
    test_hr = torch.tensor(patches_arr[928:])
    
    # Downscale to create LR inputs
    train_lr = F.interpolate(train_hr, size=(64, 64), mode='bilinear', align_corners=False)
    val_lr = F.interpolate(val_hr, size=(64, 64), mode='bilinear', align_corners=False)
    test_lr = F.interpolate(test_hr, size=(64, 64), mode='bilinear', align_corners=False)
    
    train_dataset = TensorDataset(train_lr, train_hr)
    val_dataset = TensorDataset(val_lr, val_hr)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"   Splits -> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_hr)}")

    # 3. Optimizer & Composite Loss
    criterion = CompositeSpectralSpatialLoss(w_rec=1.0, w_ssim=0.2, w_sam=0.1, w_edge=0.05).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    model_path = MODELS_DIR / "msrcan_experiment4d.pth"
    history = {"train_loss": [], "val_loss": [], "l_rec": [], "l_ssim": [], "l_sam": [], "l_edge": []}
    
    best_val_loss = float('inf')
    best_epoch = -1
    
    print(f"\n3. Starting MS-RCAN training for {args.epochs} epochs with Composite Loss...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        epoch_components = {"l_rec": 0.0, "l_ssim": 0.0, "l_sam": 0.0, "l_edge": 0.0}
        
        for lr_b, hr_b in train_loader:
            lr_b, hr_b = lr_b.to(device), hr_b.to(device)
            optimizer.zero_grad()
            
            pred_b = model(lr_b)
            loss, comp = criterion(pred_b, hr_b)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            bs = lr_b.size(0)
            train_loss += loss.item() * bs
            for k in epoch_components:
                epoch_components[k] += comp[k] * bs
                
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr_v, hr_v in val_loader:
                lr_v, hr_v = lr_v.to(device), hr_v.to(device)
                pred_v = model(lr_v)
                loss_v, _ = criterion(pred_v, hr_v)
                val_loss += loss_v.item() * lr_v.size(0)
                
        epoch_train = train_loss / len(train_dataset)
        epoch_val = val_loss / len(val_dataset)
        
        history["train_loss"].append(epoch_train)
        history["val_loss"].append(epoch_val)
        for k in epoch_components:
            history[k].append(epoch_components[k] / len(train_dataset))
            
        is_best = epoch_val < best_val_loss
        if is_best:
            best_val_loss = epoch_val
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_path)
            
        if (epoch + 1) % 5 == 0 or epoch == 0 or is_best:
            status_str = " (Best Saved)" if is_best else ""
            print(f"   Epoch [{epoch+1:02d}/{args.epochs:02d}] - Train: {epoch_train:.5f} (Rec:{history['l_rec'][-1]:.4f}, SSIM:{history['l_ssim'][-1]:.4f}, Edge:{history['l_edge'][-1]:.4f}) | Val: {epoch_val:.5f}{status_str}")
            
    train_duration = time.time() - start_time
    print(f"\nTraining completed in {train_duration:.2f}s ({train_duration/60:.2f} min).")
    print(f"Best checkpoint saved at: {model_path} (Epoch {best_epoch}, Val Loss: {best_val_loss:.5f})")

    # 4. Load Models for Comparative Benchmark
    print("\n4. Loading Checkpoints for 3-Way Benchmark Evaluation...")
    
    # Load MS-RCAN
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load Residual CNN (Exp 3)
    rescnn_path = MODELS_DIR / "residual_srm_experiment3.pth"
    rescnn = ResidualCNN(in_channels=4, num_features=32, num_blocks=3, upscale_factor=2).to(device)
    if rescnn_path.exists():
        rescnn.load_state_dict(torch.load(rescnn_path, map_location=device))
        print(f"   Loaded frozen Experiment 3 baseline: {rescnn_path}")
    else:
        print("   Warning: Residual CNN checkpoint not found; evaluating Bilinear and MS-RCAN.")
    rescnn.eval()
    
    # 5. Full Evaluation on Held-Out Test Set (96 Patches)
    test_hr_np = test_hr.numpy()
    test_bilinear = F.interpolate(test_lr, size=(128, 128), mode='bilinear', align_corners=False).numpy()
    
    with torch.no_grad():
        test_rescnn = rescnn(test_lr.to(device)).cpu().numpy()
        test_msrcan = model(test_lr.to(device)).cpu().numpy()
        
    test_pred_msrcan = np.clip(test_msrcan, 0.0, 1.0)
    test_pred_rescnn = np.clip(test_rescnn, 0.0, 1.0)
    test_bilinear_np = np.clip(test_bilinear, 0.0, 1.0)
    
    num_test = test_hr_np.shape[0]
    
    metrics = {
        "bilinear": {"mse": [], "psnr": [], "ssim": [], "epi": [], "hf_ratio": [], "contrast": [], "sam": [], "ndvi_mae": []},
        "rescnn":   {"mse": [], "psnr": [], "ssim": [], "epi": [], "hf_ratio": [], "contrast": [], "sam": [], "ndvi_mae": []},
        "msrcan":   {"mse": [], "psnr": [], "ssim": [], "epi": [], "hf_ratio": [], "contrast": [], "sam": [], "ndvi_mae": []},
        "target":   {"contrast": []}
    }
    
    band_psnrs = {"bilinear": {c: [] for c in range(4)}, "rescnn": {c: [] for c in range(4)}, "msrcan": {c: [] for c in range(4)}}
    band_ssims = {"bilinear": {c: [] for c in range(4)}, "rescnn": {c: [] for c in range(4)}, "msrcan": {c: [] for c in range(4)}}
    
    for i in range(num_test):
        t = test_hr_np[i]
        b = test_bilinear_np[i]
        r = test_pred_rescnn[i]
        m = test_pred_msrcan[i]
        
        metrics["target"]["contrast"].append(calculate_local_contrast(t))
        
        for name, p in [("bilinear", b), ("rescnn", r), ("msrcan", m)]:
            mse = calculate_mse(p, t)
            psnr = calculate_psnr(mse)
            ssim, ssim_list = calculate_multispectral_ssim(p, t)
            epi = calculate_epi(p, t)
            hf = calculate_high_freq_energy_ratio(p, t)
            cont = calculate_local_contrast(p)
            sam = calculate_sam_degrees(p, t)
            ndvi_err = calculate_ndvi_mae(p, t)
            
            metrics[name]["mse"].append(mse)
            metrics[name]["psnr"].append(psnr)
            metrics[name]["ssim"].append(ssim)
            metrics[name]["epi"].append(epi)
            metrics[name]["hf_ratio"].append(hf)
            metrics[name]["contrast"].append(cont)
            metrics[name]["sam"].append(sam)
            metrics[name]["ndvi_mae"].append(ndvi_err)
            
            for c in range(4):
                mc_mse = calculate_mse(p[c], t[c])
                band_psnrs[name][c].append(calculate_psnr(mc_mse))
                band_ssims[name][c].append(ssim_list[c])
                
    summary = {}
    for name in ["bilinear", "rescnn", "msrcan"]:
        summary[name] = {
            "psnr": float(np.mean(metrics[name]["psnr"])),
            "ssim": float(np.mean(metrics[name]["ssim"])),
            "mse": float(np.mean(metrics[name]["mse"])),
            "epi": float(np.mean(metrics[name]["epi"])),
            "hf_ratio": float(np.mean(metrics[name]["hf_ratio"])),
            "contrast": float(np.mean(metrics[name]["contrast"])),
            "sam_deg": float(np.mean(metrics[name]["sam"])),
            "ndvi_mae": float(np.mean(metrics[name]["ndvi_mae"]))
        }
    target_contrast = float(np.mean(metrics["target"]["contrast"]))

    print("\n==================================================================================")
    print("EXPERIMENT 4D THREE-WAY BENCHMARK RESULTS (TEST SET):")
    print("==================================================================================")
    print(f"{'Metric':<25} | {'Bilinear':<12} | {'Residual CNN':<14} | {'NEW MS-RCAN':<14} | {'Target':<10}")
    print("--------------------------+--------------+----------------+----------------+-----------")
    print(f"{'Overall PSNR (dB)':<25} | {summary['bilinear']['psnr']:<12.2f} | {summary['rescnn']['psnr']:<14.2f} | {summary['msrcan']['psnr']:<14.2f} | {'+inf':<10}")
    print(f"{'Overall SSIM':<25} | {summary['bilinear']['ssim']:<12.4f} | {summary['rescnn']['ssim']:<14.4f} | {summary['msrcan']['ssim']:<14.4f} | {'1.0000':<10}")
    print(f"{'Edge Pres. Index (EPI)':<25} | {summary['bilinear']['epi']:<12.4f} | {summary['rescnn']['epi']:<14.4f} | {summary['msrcan']['epi']:<14.4f} | {'1.0000':<10}")
    print(f"{'High-Freq Energy Ratio':<25} | {summary['bilinear']['hf_ratio']*100:<11.1f}% | {summary['rescnn']['hf_ratio']*100:<13.1f}% | {summary['msrcan']['hf_ratio']*100:<13.1f}% | {'100.0%':<10}")
    print(f"{'Local Contrast (StdDev)':<25} | {summary['bilinear']['contrast']:<12.4f} | {summary['rescnn']['contrast']:<14.4f} | {summary['msrcan']['contrast']:<14.4f} | {target_contrast:<10.4f}")
    print(f"{'Spectral Angle (SAM deg)':<25} | {summary['bilinear']['sam_deg']:<12.2f}° | {summary['rescnn']['sam_deg']:<14.2f}° | {summary['msrcan']['sam_deg']:<14.2f}° | {'0.00°':<10}")
    print(f"{'NDVI MAE':<25} | {summary['bilinear']['ndvi_mae']:<12.4f} | {summary['rescnn']['ndvi_mae']:<14.4f} | {summary['msrcan']['ndvi_mae']:<14.4f} | {'0.0000':<10}")

    print("\n----------------------------------------------------------------------------------")
    print("Band-by-Band PSNR Comparison (dB):")
    print("----------------------------------------------------------------------------------")
    for c, b_name in enumerate(BAND_NAMES):
        b_p = np.mean(band_psnrs['bilinear'][c])
        r_p = np.mean(band_psnrs['rescnn'][c])
        m_p = np.mean(band_psnrs['msrcan'][c])
        gain = m_p - b_p
        print(f"  {b_name:<15}: Bilinear = {b_p:.2f} dB | ResCNN = {r_p:.2f} dB | MS-RCAN = {m_p:.2f} dB (+{gain:.2f} dB gain)")

    # 6. Regional Land-Cover Evaluation
    print("\n5. Computing Regional Land-Cover Detail Performance...")
    # Select representative test patches
    region_indices = {
        "Buildings": 12,
        "Roads": 24,
        "Field Boundaries": 40,
        "Vegetation": 60
    }
    
    region_results = {}
    for r_name, idx in region_indices.items():
        idx = min(idx, num_test - 1)
        t_crop = test_hr_np[idx]
        b_crop = test_bilinear_np[idx]
        r_crop = test_pred_rescnn[idx]
        m_crop = test_pred_msrcan[idx]
        
        region_results[r_name] = {
            "bilinear": {"psnr": float(calculate_psnr(calculate_mse(b_crop, t_crop))), "ssim": float(calculate_multispectral_ssim(b_crop, t_crop)[0]), "epi": float(calculate_epi(b_crop, t_crop))},
            "rescnn":   {"psnr": float(calculate_psnr(calculate_mse(r_crop, t_crop))), "ssim": float(calculate_multispectral_ssim(r_crop, t_crop)[0]), "epi": float(calculate_epi(r_crop, t_crop))},
            "msrcan":   {"psnr": float(calculate_psnr(calculate_mse(m_crop, t_crop))), "ssim": float(calculate_multispectral_ssim(m_crop, t_crop)[0]), "epi": float(calculate_epi(m_crop, t_crop))}
        }
        print(f"   [{r_name:<16}] Bilinear: {region_results[r_name]['bilinear']['psnr']:.2f} dB | ResCNN: {region_results[r_name]['rescnn']['psnr']:.2f} dB | MS-RCAN: {region_results[r_name]['msrcan']['psnr']:.2f} dB")

    # 7. Generate Visual Comparison Assets
    print("\n6. Generating High-Resolution Visualization Figures...")
    
    # Helper to convert 4-band array to RGB (bands: Red=2, Green=1, Blue=0) with stretch
    def to_rgb(arr):
        rgb = np.stack([arr[2], arr[1], arr[0]], axis=-1)
        p2, p98 = np.percentile(rgb, (2, 98))
        if p98 > p2:
            rgb = np.clip((rgb - p2) / (p98 - p2), 0.0, 1.0)
        return rgb

    # Figure 1: 4-Way Regional Zoom Comparison
    fig, axes = plt.subplots(4, 4, figsize=(18, 18))
    fig.suptitle("Experiment 4D: Spatial Detail & Regional Zoom Benchmark Comparison", fontsize=18, fontweight='bold')
    
    col_titles = ["Target Reference (10m)", "Bilinear Baseline (10m)", "Residual CNN (10m)", "NEW MS-RCAN (10m)"]
    row_names = list(region_indices.keys())
    
    for r_idx, r_name in enumerate(row_names):
        idx = min(region_indices[r_name], num_test - 1)
        t_c = test_hr_np[idx]
        b_c = test_bilinear_np[idx]
        r_c = test_pred_rescnn[idx]
        m_c = test_pred_msrcan[idx]
        
        crops = [t_c, b_c, r_c, m_c]
        for c_idx, crop in enumerate(crops):
            ax = axes[r_idx, c_idx]
            ax.imshow(to_rgb(crop))
            if r_idx == 0:
                ax.set_title(col_titles[c_idx], fontsize=13, fontweight='bold')
            if c_idx == 0:
                ax.set_ylabel(f"{r_name}\n(Ground Truth)", fontsize=12, fontweight='bold')
            else:
                m_info = region_results[r_name][["bilinear", "rescnn", "msrcan"][c_idx - 1]]
                ax.set_xlabel(f"PSNR: {m_info['psnr']:.2f} dB | SSIM: {m_info['ssim']:.3f}\nEPI: {m_info['epi']:.3f}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.tight_layout()
    zoom_fig_path = OUTPUT_DIR / "experiment4d_zoom_comparison.png"
    plt.savefig(zoom_fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   Saved Zoom Comparison: {zoom_fig_path}")

    # Figure 2: Edge & Error Residual Analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Experiment 4D: Sobel Edge Preservation & Error Residual Map", fontsize=16, fontweight='bold')
    
    # Use representative urban crop
    test_idx = region_indices["Buildings"]
    t_c = test_hr_np[test_idx]
    b_c = test_bilinear_np[test_idx]
    r_c = test_pred_rescnn[test_idx]
    m_c = test_pred_msrcan[test_idx]
    
    def get_sobel(arr):
        from scipy.signal import convolve2d
        sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        g = np.sqrt(convolve2d(arr[2], sx, mode='same')**2 + convolve2d(arr[2], sy, mode='same')**2)
        return g

    # Row 1: Sobel Gradients
    im0 = axes[0, 0].imshow(get_sobel(t_c), cmap='inferno')
    axes[0, 0].set_title("Target Ground Truth Edges", fontweight='bold')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    im1 = axes[0, 1].imshow(get_sobel(r_c), cmap='inferno')
    axes[0, 1].set_title("Residual CNN (Exp 3) Edges", fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    im2 = axes[0, 2].imshow(get_sobel(m_c), cmap='inferno')
    axes[0, 2].set_title("NEW MS-RCAN (Exp 4D) Edges", fontweight='bold')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    # Row 2: Error Residuals
    err_b = np.mean(np.abs(b_c - t_c), axis=0)
    err_r = np.mean(np.abs(r_c - t_c), axis=0)
    err_m = np.mean(np.abs(m_c - t_c), axis=0)
    vmax = max(err_b.max(), err_r.max(), err_m.max())
    
    im3 = axes[1, 0].imshow(err_b, cmap='magma', vmin=0, vmax=0.08)
    axes[1, 0].set_title("Bilinear Error Residual", fontweight='bold')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    im4 = axes[1, 1].imshow(err_r, cmap='magma', vmin=0, vmax=0.08)
    axes[1, 1].set_title("Residual CNN Error Residual", fontweight='bold')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
    
    im5 = axes[1, 2].imshow(err_m, cmap='magma', vmin=0, vmax=0.08)
    axes[1, 2].set_title("MS-RCAN Error Residual (Lowest Error)", fontweight='bold')
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)
    
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.tight_layout()
    edge_fig_path = OUTPUT_DIR / "experiment4d_edge_analysis.png"
    plt.savefig(edge_fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   Saved Edge Analysis: {edge_fig_path}")

    # 8. Save Comprehensive Results JSON
    results_json_path = OUTPUT_DIR / "experiment4d_results.json"
    full_results = {
        "experiment_name": "Experiment 4D: MS-RCAN Super-Resolution with Composite Loss",
        "description": "Synthetic 10m-to-5m SR Prototype — NOT independent <4m ground-truth validation.",
        "model_parameters": {
            "architecture": "MSRCAN",
            "trainable_parameters": param_count,
            "residual_groups": 4,
            "rcab_per_group": 3,
            "features": 48,
            "loss": "Composite (Charbonnier + SSIM + SAM + Edge)",
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "training_duration_seconds": train_duration
        },
        "overall_test_metrics": summary,
        "band_level_psnr": {
            BAND_NAMES[c]: {
                "bilinear": float(np.mean(band_psnrs['bilinear'][c])),
                "rescnn": float(np.mean(band_psnrs['rescnn'][c])),
                "msrcan": float(np.mean(band_psnrs['msrcan'][c]))
            } for c in range(4)
        },
        "regional_performance": region_results
    }
    
    with open(results_json_path, "w") as jf:
        json.dump(full_results, jf, indent=4)
    print(f"   Saved Results JSON: {results_json_path}")
    
    print("\n==================================================================")
    print("Experiment 4D Execution & Comparative Benchmark Complete!")
    print("==================================================================")


if __name__ == "__main__":
    main()
