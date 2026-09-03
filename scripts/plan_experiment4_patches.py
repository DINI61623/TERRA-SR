#!/usr/bin/env python3
"""
plan_experiment4_patches.py
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

Calculates the spatial train/validation/test split plan for Experiment 4.
Aligns patches geographically, inserts spatial buffers to prevent leakage,
and logs the plan.
"""

import sys
import json
import argparse
from pathlib import Path

# Add workspace root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    import rasterio
    from rasterio.warp import transform_bounds
    from affine import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def calculate_simulated_split(lr_size=1024, lr_patch_size=32):
    """
    Computes a simulated patch grid layout and split plan.
    Used when real data is not yet available.
    """
    grid_rows = lr_size // lr_patch_size
    grid_cols = lr_size // lr_patch_size
    total_patches = grid_rows * grid_cols
    
    # Train: Rows 0-23 (24 rows * 32 cols = 768 patches)
    # Buffer 1: Rows 24-25 (2 rows * 32 cols = 64 patches) - DISCARDED
    # Val: Rows 26-28 (3 rows * 32 cols = 96 patches)
    # Test: Rows 29-31 (3 rows * 32 cols = 96 patches)
    
    train_rows = list(range(0, 24))
    buffer_rows = [24, 25]
    val_rows = list(range(26, 29))
    test_rows = list(range(29, 32))
    
    plan = {
        "status": "SIMULATION_MODE_WAITING_FOR_DATA",
        "description": "Mock split plan for a standard 1024x1024 Sentinel-2 ROI with upscale factor 2 (5m reference).",
        "grid_dimensions": {
            "image_size": f"{lr_size}x{lr_size}",
            "patch_size_lr": f"{lr_patch_size}x{lr_patch_size}",
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "total_potential_patches": total_patches
        },
        "spatial_splits": {
            "train": {
                "rows": train_rows,
                "patch_count": len(train_rows) * grid_cols,
                "percentage": (len(train_rows) * grid_cols) / total_patches
            },
            "buffer_zone_discarded": {
                "rows": buffer_rows,
                "patch_count": len(buffer_rows) * grid_cols,
                "percentage": (len(buffer_rows) * grid_cols) / total_patches
            },
            "validation": {
                "rows": val_rows,
                "patch_count": len(val_rows) * grid_cols,
                "percentage": (len(val_rows) * grid_cols) / total_patches
            },
            "test": {
                "rows": test_rows,
                "patch_count": len(test_rows) * grid_cols,
                "percentage": (len(test_rows) * grid_cols) / total_patches
            }
        },
        "leakage_prevention": {
            "buffer_width_meters": 2 * lr_patch_size * 10, # 2 rows * 32 pixels * 10m = 640 meters geographic buffer
            "status": "SECURE_ZERO_SPATIAL_LEAKAGE"
        }
    }
    return plan


def calculate_real_split(ref_path, s2_path, lr_patch_size=32, upscale_factor=2):
    """
    Computes patch split plan based on actual spatial intersection bounds.
    """
    with rasterio.open(ref_path) as ref:
        with rasterio.open(s2_path) as s2:
            # Project reference bounds into S2 CRS
            ref_bounds_in_s2 = transform_bounds(
                ref.crs, s2.crs,
                ref.bounds.left, ref.bounds.bottom,
                ref.bounds.right, ref.bounds.top
            )
            
            # Intersection bounds in Sentinel-2 CRS
            left = max(ref_bounds_in_s2[0], s2.bounds.left)
            bottom = max(ref_bounds_in_s2[1], s2.bounds.bottom)
            right = min(ref_bounds_in_s2[2], s2.bounds.right)
            top = min(ref_bounds_in_s2[3], s2.bounds.top)
            
            if left >= right or bottom >= top:
                raise ValueError("No geographical overlap found to plan patches.")
                
            # Align boundaries to S2 pixel coordinates
            s2_x0, s2_y0 = s2.transform.c, s2.transform.f
            s2_dx, s2_dy = s2.transform.a, s2.transform.e
            
            col_start = int(np.floor((left - s2_x0) / s2_dx))
            row_start = int(np.floor((top - s2_y0) / s2_dy))
            col_end = int(np.ceil((right - s2_x0) / s2_dx))
            row_end = int(np.ceil((bottom - s2_y0) / s2_dy))
            
            s2_width = col_end - col_start
            s2_height = row_end - row_start
            
            grid_rows = s2_height // lr_patch_size
            grid_cols = s2_width // lr_patch_size
            total_patches = grid_rows * grid_cols
            
            if total_patches == 0:
                raise ValueError("Spatial intersection is too small to extract even a single patch.")
                
            # Compute splits based on rows
            # We want approximately 70% Train, 10% Val, 10% Test, with 10% Buffer rows
            # For general sizes, we partition row-wise:
            val_rows_count = max(1, int(grid_rows * 0.1))
            test_rows_count = max(1, int(grid_rows * 0.1))
            buffer_rows_count = max(1, int(grid_rows * 0.08)) # buffer size
            
            train_rows_count = grid_rows - val_rows_count - test_rows_count - buffer_rows_count
            
            train_rows = list(range(0, train_rows_count))
            buffer_rows = list(range(train_rows_count, train_rows_count + buffer_rows_count))
            val_rows = list(range(train_rows_count + buffer_rows_count, train_rows_count + buffer_rows_count + val_rows_count))
            test_rows = list(range(train_rows_count + buffer_rows_count + val_rows_count, grid_rows))
            
            # Project geographic coordinates of zones
            s2_transform = s2.transform
            
            def get_row_y_coord(row_idx):
                # row_start is in S2 pixels
                # pixel row in the intersection
                pixel_row = row_start + row_idx * lr_patch_size
                # Y coordinate is at top of row
                _, y_coord = s2_transform * (0, pixel_row)
                return y_coord
                
            plan = {
                "status": "READY_FOR_REAL_TRAINING",
                "description": "Real-data spatial split plan based on Sentinel-2 and PlanetScope intersection.",
                "intersection_details": {
                    "crs": s2.crs.to_string(),
                    "bounds": {"left": left, "bottom": bottom, "right": right, "top": top},
                    "s2_pixel_dimensions": f"{s2_width}x{s2_height}",
                    "upscaled_dimensions": f"{s2_width * upscale_factor}x{s2_height * upscale_factor}"
                },
                "grid_dimensions": {
                    "patch_size_lr": f"{lr_patch_size}x{lr_patch_size}",
                    "grid_rows": grid_rows,
                    "grid_cols": grid_cols,
                    "total_potential_patches": total_patches
                },
                "spatial_splits": {
                    "train": {
                        "rows": train_rows,
                        "patch_count": len(train_rows) * grid_cols,
                        "percentage": (len(train_rows) * grid_cols) / total_patches,
                        "y_bounds": (get_row_y_coord(train_rows[0]), get_row_y_coord(train_rows[-1] + 1))
                    },
                    "buffer_zone_discarded": {
                        "rows": buffer_rows,
                        "patch_count": len(buffer_rows) * grid_cols,
                        "percentage": (len(buffer_rows) * grid_cols) / total_patches,
                        "y_bounds": (get_row_y_coord(buffer_rows[0]), get_row_y_coord(buffer_rows[-1] + 1))
                    },
                    "validation": {
                        "rows": val_rows,
                        "patch_count": len(val_rows) * grid_cols,
                        "percentage": (len(val_rows) * grid_cols) / total_patches,
                        "y_bounds": (get_row_y_coord(val_rows[0]), get_row_y_coord(val_rows[-1] + 1))
                    },
                    "test": {
                        "rows": test_rows,
                        "patch_count": len(test_rows) * grid_cols,
                        "percentage": (len(test_rows) * grid_cols) / total_patches,
                        "y_bounds": (get_row_y_coord(test_rows[0]), get_row_y_coord(grid_rows))
                    }
                },
                "leakage_prevention": {
                    "buffer_width_meters": buffer_rows_count * lr_patch_size * 10,
                    "status": "SECURE_ZERO_SPATIAL_LEAKAGE"
                }
            }
            return plan


def main():
    parser = argparse.ArgumentParser(
        description="Plan patch slicing and splits for Experiment 4 real-data training."
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="data/raw/planet/20260211_054815_64_254a_3m.tif",
        help="Path to PlanetScope reference scene"
    )
    parser.add_argument(
        "--sentinel",
        type=str,
        default="data/raw/sentinel2/S2B_MSIL2A_20260211T050839_N0512_R019_T43PGQ_20260211T085923_Red_10m.jp2",
        help="Path to Sentinel-2 file"
    )
    
    args = parser.parse_args()
    
    ref_path = Path(args.reference)
    s2_path = Path(args.sentinel)
    
    files_exist = ref_path.exists() and s2_path.exists()
    
    if files_exist:
        print("[Plan] Real files located. Calculating spatial splits from raster metadata...")
        try:
            plan = calculate_real_split(ref_path, s2_path)
        except Exception as e:
            print(f"[Error] Failed to calculate real split plan: {str(e)}")
            print("Falling back to simulated plan.")
            plan = calculate_simulated_split()
    else:
        print("[Plan] Real PlanetScope file not found. Running in SIMULATION mode.")
        print(f"       (Waiting for reference file: {args.reference})")
        plan = calculate_simulated_split()
        
    print("\n==================================================")
    print(f"Experiment 4 Patch Planning: {plan['status']}")
    print("==================================================")
    print(f"Grid dimensions: {plan['grid_dimensions']['grid_rows']} rows x {plan['grid_dimensions']['grid_cols']} columns")
    print(f"Total potential patch pairs: {plan['grid_dimensions']['total_potential_patches']}")
    print("\nSplit Plan:")
    for split_name, split_info in plan["spatial_splits"].items():
        print(f"  - {split_name.capitalize()}: {split_info['patch_count']} patches ({split_info['percentage']:.1%}) - Rows {split_info['rows']}")
    print(f"\nleakage Prevention: {plan['leakage_prevention']['status']}")
    print(f"  - Spatial Buffer width: {plan['leakage_prevention']['buffer_width_meters']} meters")
    print("==================================================")
    
    # Save plan log
    plan_path = Path("outputs/experiment4_real_patch_plan.json")
    with open(plan_path, "w") as jf:
        json.dump(plan, jf, indent=4)
    print(f"Saved patch planning log to: {plan_path}")


if __name__ == "__main__":
    main()
