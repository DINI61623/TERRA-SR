#!/usr/bin/env python3
"""
Test and Validation Script for Physical Degradation Pipeline (Revised Experiment 4c)
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

Verifies:
1. Module imports and class instantiation.
2. Configuration loading and parameter validation.
3. Target validation for missing parameters on enabled components.
4. Tensor shape transformations and spatial decimation scaling.
5. Numerical stability (no NaNs, Infs) and output range bounds.
6. Modular component toggling.
7. Evaluates the active config to output exact statuses:
   - DEGRADATION_READY
   - DEGRADATION_PARTIALLY_READY
   - DEGRADATION_BLOCKED
"""

import sys
import yaml
import torch
import numpy as np
from pathlib import Path

# Add workspace root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.super_resolution.degradation import PhysicalDegradationPipeline


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_degradation_tests():
    print("==================================================")
    print("Revised Physical Degradation Pipeline Audit Suite")
    print("==================================================")

    # 1. Verify Configuration Parsing
    config_path = Path("configs/experiment4_degradation.yaml")
    print(f"[Test 1/6] Parsing configuration from: {config_path}...")
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
        
    config = load_config(config_path)
    print("   - Successfully loaded YAML config.")
    
    # 2. Verify Required Parameters Detection on Enabled Components
    print("\n[Test 2/6] Verifying required sensor parameter validation (error checking)...")
    
    # Inject missing PSF sigmas into an enabled PSF configuration to verify it gets blocked
    bad_config = yaml.safe_load(open(config_path, "r").read()) # deep copy
    bad_config["psf"]["enabled"] = True
    bad_config["psf"]["band_sigmas"] = None
    
    try:
        pipeline = PhysicalDegradationPipeline(bad_config)
        print("   - [FAIL] Pipeline failed to raise ValueError when an enabled PSF has null band_sigmas!")
        sys.exit(1)
    except ValueError as e:
        print("   - [OK] Validation correctly caught missing enabled parameters.")
        print(f"     Expected error caught: {str(e).splitlines()[0]}")

    # 3. Verify execution under Injected/Mock Parameters
    print("\n[Test 3/6] Injecting mock sensor parameters for pipeline verification...")
    mock_config = yaml.safe_load(open(config_path, "r").read()) # deep copy
    
    # Enable all modules and inject valid mock parameters for testing math
    mock_config["psf"]["enabled"] = True
    mock_config["psf"]["band_sigmas"] = [1.045, 1.023, 1.045, 1.060]
    mock_config["spectral_matching"]["enabled"] = True
    mock_config["spectral_matching"]["slopes"] = [1.02, 0.98, 1.01, 0.95]
    mock_config["spectral_matching"]["intercepts"] = [-0.002, 0.005, -0.001, 0.01]
    mock_config["sensor_noise"]["enabled"] = True
    mock_config["sensor_noise"]["band_sigmas"] = [0.005, 0.004, 0.006, 0.008]
    
    try:
        pipeline = PhysicalDegradationPipeline(mock_config)
        print("   - [OK] Pipeline initialized successfully with mock parameters.")
    except Exception as e:
        print(f"   - [FAIL] Failed to initialize pipeline with mock parameters: {str(e)}")
        sys.exit(1)

    # 4. Verify shape transformations and spatial decimation
    print("\n[Test 4/6] Verifying tensor shape transformation...")
    upscale_factor = mock_config["spatial_decimation"]["upscale_factor"]
    
    # Mock high-resolution input (Batch=2, Channels=4, Height=128, Width=128) in raw DN range [0, 10000]
    hr_shape = (2, 4, 128, 128)
    hr_input = torch.randint(0, 10000, hr_shape, dtype=torch.float32)
    
    print(f"   - HR Input shape: {list(hr_input.shape)}")
    try:
        lr_output = pipeline(hr_input)
        print(f"   - LR Output shape: {list(lr_output.shape)}")
        
        expected_lr_shape = [2, 4, 128 // upscale_factor, 128 // upscale_factor]
        if list(lr_output.shape) != expected_lr_shape:
            raise ValueError(f"Output shape mismatch. Expected {expected_lr_shape}, got {list(lr_output.shape)}")
        print(f"   - [OK] Output shape matches target low-res grid (K={upscale_factor} decimation).")
    except Exception as e:
        print(f"   - [FAIL] Pipeline forward pass failed: {str(e)}")
        sys.exit(1)

    # 5. Verify numerical stability and output ranges
    print("\n[Test 5/6] Verifying numerical stability and output range bounds...")
    
    has_nan = torch.isnan(lr_output).any().item()
    has_inf = torch.isinf(lr_output).any().item()
    if has_nan or has_inf:
        print(f"   - [FAIL] Output contains NaNs ({has_nan}) or Infs ({has_inf})!")
        sys.exit(1)
    print("   - [OK] Output contains no NaNs or Infs.")
    
    min_val = lr_output.min().item()
    max_val = lr_output.max().item()
    print(f"   - Output value range: [{min_val:.4f}, {max_val:.4f}]")
    if min_val < -0.1 or max_val > 1.2:
        print(f"   - [FAIL] Output range is outside stable bounds [-0.1, 1.2]!")
        sys.exit(1)
    print("   - [OK] Output ranges are stable and bounded within physical expectations.")

    # 6. Verify modular toggling
    print("\n[Test 6/6] Verifying modular component toggling...")
    
    # Disable noise and check that range is strictly clamped to [0, 1]
    mock_config["sensor_noise"]["enabled"] = False
    mock_config["spectral_matching"]["enabled"] = False
    
    pipeline_no_noise = PhysicalDegradationPipeline(mock_config)
    lr_output_no_noise = pipeline_no_noise(hr_input)
    
    min_nn = lr_output_no_noise.min().item()
    max_nn = lr_output_no_noise.max().item()
    
    print(f"   - Range without noise: [{min_nn:.4f}, {max_nn:.4f}]")
    if min_nn < 0.0 or max_nn > 1.0:
        print(f"   - [FAIL] Range without noise exceeds physical [0.0, 1.0] limits!")
        sys.exit(1)
    print("   - [OK] Modular toggling verified (noise deactivated, clamping verified).")

    # 7. Evaluate the active configuration file and report exact statuses
    print("\n==================================================")
    print("ACTIVE CONFIGURATION STATUS REPORT:")
    print("==================================================")
    
    components = {
        "Spatial PSF Blurring": ("psf", ["band_sigmas"]),
        "Spectral Response Matching": ("spectral_matching", ["slopes", "intercepts"]),
        "Radiometric Normalization": ("radiometric_normalization", ["input_scale"]),
        "Sensor Noise Model": ("sensor_noise", ["band_sigmas"]),
        "Spatial Decimation": ("spatial_decimation", ["upscale_factor", "interpolation_mode"])
    }
    
    ready_count = 0
    disabled_count = 0
    blocked_count = 0
    
    for label, (cfg_key, params) in components.items():
        cfg = config.get(cfg_key, {})
        enabled = cfg.get("enabled", False)
        
        if not enabled:
            # Component is disabled because it is scene-specific or unknown
            print(f"{label:<30}: DISABLED (Scene-Specific/Unknown)")
            disabled_count += 1
        else:
            # Component is enabled, check if parameters are populated
            is_blocked = False
            for p in params:
                if cfg.get(p) is None:
                    is_blocked = True
            
            if is_blocked:
                print(f"{label:<30}: BLOCKED (Missing Parameters)")
                blocked_count += 1
            else:
                print(f"{label:<30}: READY")
                ready_count += 1
                
    print("--------------------------------------------------")
    
    # Status calculation
    if blocked_count > 0:
        status = "DEGRADATION_BLOCKED"
        print(f"FINAL STATUS: {status}")
        print("Reason: One or more enabled components are missing required parameters.")
    elif ready_count > 0 and disabled_count > 0:
        status = "DEGRADATION_PARTIALLY_READY"
        print(f"FINAL STATUS: {status}")
        print("Reason: All core/derived parameters are ready. Scene-specific parameters (noise, calibration) "
              "are disabled and must be dynamically supplied once coregistration is complete.")
    elif ready_count > 0 and disabled_count == 0:
        status = "DEGRADATION_READY"
        print(f"FINAL STATUS: {status}")
        print("Reason: All components are enabled and successfully configured.")
    else:
        status = "DEGRADATION_BLOCKED"
        print(f"FINAL STATUS: {status}")
        print("Reason: No components are enabled or configured.")
        
    print("==================================================")


if __name__ == "__main__":
    run_degradation_tests()
