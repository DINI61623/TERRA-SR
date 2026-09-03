#!/usr/bin/env python3
"""
Sentinel-2 Super Resolution Mapping (SRM) Training Pipeline
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

Ties together preprocessing, patch slicing, PyTorch datasets, and the ESPCN model
to run a complete model training workflow (with full fallback simulations if PyTorch is missing).
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np

# Import our custom pipeline modules
from src.preprocessing.preprocess import normalize_band, save_stacked_tiff
from src.utils.spatial_helpers import slice_into_patches

# Try importing PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from src.super_resolution.model import ESPCN
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Try importing Pillow
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def load_raw_band(path):
    """
    Helper to load a band image path using Pillow as numpy array.
    """
    if not HAS_PIL:
        raise ImportError("Pillow ('pillow') is required to load the downloaded JP2 images.")
    with Image.open(path) as img:
        return np.array(img, dtype=np.uint16)


def preprocess_downloaded_scene(data_dir, product_id):
    """
    Locates the downloaded bands, crops a 1024x1024 region of interest (ROI)
    from the center (to avoid loading the full 10980x10980 image into memory during testing),
    normalizes the bands, and stacks them.
    """
    print(f"\n--- [Step 1/5] Preprocessing Downloaded Bands ---")
    
    bands_paths = {
        "Blue": Path(data_dir) / f"{product_id}_Blue_10m.jp2",
        "Green": Path(data_dir) / f"{product_id}_Green_10m.jp2",
        "Red": Path(data_dir) / f"{product_id}_Red_10m.jp2",
        "NIR": Path(data_dir) / f"{product_id}_NIR_10m.jp2"
    }

    # Verify download files exist
    for name, path in bands_paths.items():
        if not path.exists():
            print(f"Error: Required band file not found: {path}")
            print("Please ensure your download step succeeded.")
            sys.exit(1)

    print("Loading 1024x1024 Center Region of Interest (ROI) to save RAM...")
    processed_bands = []
    
    for name, path in bands_paths.items():
        raw_band = load_raw_band(path)
        h, w = raw_band.shape
        cy, cx = h // 2, w // 2
        
        # Crop 1024x1024 box around center
        cropped = raw_band[cy-512:cy+512, cx-512:cx+512]
        
        # Normalize (DN -> reflectance 0-1)
        normalized = normalize_band(cropped)
        processed_bands.append(normalized)
        print(f"  Processed {name} band: cropped shape {cropped.shape}, range [{np.min(normalized):.2f}, {np.max(normalized):.2f}]")

    # Stack bands (Red, Green, Blue, NIR)
    stacked = np.stack(processed_bands, axis=0) # shape: (4, 1024, 1024)
    print(f"Successfully stacked multi-spectral data. Shape: {stacked.shape}")
    return stacked


def prepare_training_patches(stacked_image, patch_size=128):
    """
    Slices the preprocessed stacked image into non-overlapping training patches.
    """
    print(f"\n--- [Step 2/5] Slicing Image into Training Patches ---")
    # Slicing with stride equal to patch_size (no overlap)
    patches, coords = slice_into_patches(stacked_image, patch_size=patch_size, stride=patch_size)
    
    # Convert patches list to a numpy array: shape (Num_Patches, Channels, Height, Width)
    patches_arr = np.array(patches, dtype=np.float32)
    print(f"Prepared training dataset: {patches_arr.shape[0]} patches of shape {patches_arr.shape[1:]}")
    return patches_arr


def build_pytorch_dataloader(patches, upscale_factor=2, batch_size=8):
    """
    Downsamples the patches to create low-res inputs and packages them into DataLoader.
    """
    print(f"\n--- [Step 3/5] Building PyTorch DataLoader ---")
    hr_targets = torch.tensor(patches)
    
    # Create Low-Res (LR) inputs by downsampling the High-Res (HR) targets
    lr_size = patches.shape[2] // upscale_factor
    
    # Downsample using bilinear interpolation
    lr_inputs = nn.functional.interpolate(
        hr_targets,
        size=(lr_size, lr_size),
        mode='bilinear',
        align_corners=False
    )
    
    print(f"HR Targets shape: {list(hr_targets.shape)}")
    print(f"LR Inputs shape:  {list(lr_inputs.shape)}")
    
    dataset = TensorDataset(lr_inputs, hr_targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train_model(dataloader, epochs=5, lr=0.001, upscale_factor=2):
    """
    Instantiates the ESPCN model and runs the optimization loop in PyTorch.
    """
    print(f"\n--- [Step 4/5] Initializing ESPCN Model ---")
    # 4 channels input (RGB + NIR), upscale_factor
    model = ESPCN(in_channels=4, upscale_factor=upscale_factor)
    print(model)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\n--- [Step 5/5] Running PyTorch Training Loop ---")
    print(f"Training for {epochs} epochs (learning rate = {lr})...")
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (lr_batch, hr_batch) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            hr_pred = model(lr_batch)
            
            # Calculate Mean Squared Error loss
            loss = criterion(hr_pred, hr_batch)
            
            # Backward pass & weight updates
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * lr_batch.size(0)
            
        avg_loss = epoch_loss / len(dataloader.dataset)
        print(f"  Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.6f}")
        
    print("\nTraining completed successfully!")
    
    # Save the trained model weights
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "espcn_srm_sentinel2.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved trained weights to {model_path}")


def simulate_numpy_training(patches, epochs=5, upscale_factor=2):
    """
    Simulates the forward pass and loss updates using NumPy when PyTorch is not installed.
    """
    print(f"\n--- [Step 3/5] Simulating Dataset Downsampling (NumPy) ---")
    # Downsample by simple average pooling slice
    lr_size = patches.shape[2] // upscale_factor
    lr_inputs = np.zeros((patches.shape[0], patches.shape[1], lr_size, lr_size), dtype=np.float32)
    
    for i in range(patches.shape[0]):
        for c in range(patches.shape[1]):
            # Simple 2x2 local block pooling
            lr_inputs[i, c] = patches[i, c].reshape(lr_size, upscale_factor, lr_size, upscale_factor).mean(axis=(1, 3))
            
    print(f"HR Target patch shape: {patches.shape[1:]}")
    print(f"LR Input patch shape:  {lr_inputs.shape[1:]}")
    print(f"Total training pairs:  {patches.shape[0]}")
    
    print(f"\n--- [Step 4/5] Simulating ESPCN Network Parameters ---")
    print("ESPCN Configuration:")
    print("  - Input Channels:  4 (Red, Green, Blue, NIR)")
    print(f"  - Upscale Factor:  x{upscale_factor}")
    print("  - Layers:          Conv2d(4->64) -> Tanh -> Conv2d(64->32) -> Tanh -> Conv2d(32->16) -> PixelShuffle")
    
    print(f"\n--- [Step 5/5] Simulating Training Epochs (NumPy) ---")
    # Simulate a decreasing loss curve
    loss = 0.125
    for epoch in range(epochs):
        # Fake learning step: reduce loss
        loss = loss * 0.75 + np.random.uniform(0.001, 0.005)
        print(f"  Epoch [{epoch+1}/{epochs}] - Simulated MSE Loss: {loss:.6f}")
        
    print("\nSimulated training completed successfully!")
    print("[Notice] Install PyTorch ('pip install torch torchvision') to execute actual training and write model files.")


def main():
    parser = argparse.ArgumentParser(
        description="Sentinel-2 Satellite Super Resolution Mapping (SRM) Training Pipeline."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/sentinel2",
        help="Directory where target JP2 bands are saved (default: data/raw/sentinel2)"
    )
    parser.add_argument(
        "--product-id",
        type=str,
        default="S2B_MSIL2A_20260211T050839_N0512_R019_T43PGQ_20260211T085923",
        help="Sentinel-2 product ID to process for training data"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=128,
        help="Training patch size (default: 128)"
    )
    parser.add_argument(
        "--upscale-factor",
        type=int,
        default=2,
        help="Super Resolution upscaling factor (default: 2)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs to train (default: 5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training (default: 8)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for optimizer (default: 0.001)"
    )

    args = parser.parse_args()

    # Step 1: Preprocess bands (Crop ROI, normalize, stack)
    stacked_image = preprocess_downloaded_scene(args.data_dir, args.product_id)
    
    # Save the processed tiff for visualization/reference later
    save_stacked_tiff(
        bands_data=[stacked_image[0], stacked_image[1], stacked_image[2], stacked_image[3]],
        profile={}, # Empty profile defaults to simple array write
        output_path=Path("data/processed") / "s2_10m_stacked_roi.tiff"
    )
    
    # Step 2: Slice stacked ROI into patches
    patches = prepare_training_patches(stacked_image, patch_size=args.patch_size)
    
    # Step 3-5: Training Loop (Live or Simulated)
    if HAS_TORCH:
        # Build dataloader with downsampled pairs
        dataloader = build_pytorch_dataloader(patches, args.upscale_factor, args.batch_size)
        # Train PyTorch model
        train_model(dataloader, epochs=args.epochs, lr=args.lr, upscale_factor=args.upscale_factor)
    else:
        # NumPy simulation fallback
        simulate_numpy_training(patches, epochs=args.epochs, upscale_factor=args.upscale_factor)


if __name__ == "__main__":
    main()
