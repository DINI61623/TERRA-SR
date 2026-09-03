# Experiment 4 Real-Data Training Readiness Report

This report evaluates the readiness of the project infrastructure to begin training and validating the **Residual CNN** super-resolution model on real cross-sensor satellite data (Sentinel-2 to PlanetScope).

---

## 1. Readiness Status

### **STATUS**: `WAITING_FOR_REAL_PLANETSCOPE_DATA`

### **Summary**:
The training pipeline infrastructure is fully prepared, configured, and validated. However, the real high-resolution reference dataset (3m PlanetScope scene) is not yet present on the disk. Training cannot begin until the actual image file is supplied.

---

## 2. Infrastructure Validation Checklist

The input validator script `scripts/validate_experiment4_input.py` was executed with candidate file locations, yielding the following check matrix:

| Check | Target / Expected Contract | Actual Value | Status |
| :--- | :--- | :--- | :--- |
| **Sentinel-2 File** | `data/raw/sentinel2/S2B_MSIL2A_20260211T050839...` | File located on disk | **PASS** |
| **PlanetScope File** | `data/raw/planet/20260211_054815_64_254a_3m.tif` | **File missing from disk** | **FAIL** |
| **Raster Readability** | Openable via `rasterio` | N/A (Missing reference) | **FAIL** |
| **Geographic Intersection**| Overlaps Electronic City, Bengaluru AOI | N/A (Missing reference) | **FAIL** |
| **CRS Alignment** | UTM Zone 43N (`EPSG:32643`) | N/A (Missing reference) | **FAIL** |
| **Band Count** | At least 4 bands (RGB + NIR) | N/A (Missing reference) | **FAIL** |
| **Reflectance Range** | Normalized float `[0.0, 1.0]` | N/A (Missing reference) | **FAIL** |
| **NoData Ratio** | Less than 10% NoData values | N/A (Missing reference) | **FAIL** |
| **Temporal Matching** | Gap is $\le 30$ days (target $\le 5$ days) | N/A (Missing reference) | **FAIL** |

---

## 3. Training Config Summary

The configuration is saved in [`experiment4_real_training.yaml`](file:///c:/Users/urstr/New%20folder%20(3)/configs/experiment4_real_training.yaml):
*   **Model**: `ResidualCNN` (4 channels, 32 features, 3 blocks, x2 upscale).
*   **LR Patch Size**: 32x32 pixels (320m x 320m grid).
*   **Geographic Splits**: Spatial striping with a **640m buffer zone** (rows 24–25) to prevent geographic leakage.
*   **Checkpoint Destination**: [`models/experiment4_real/`](file:///c:/Users/urstr/New%20folder%20(3)/models/experiment4_real/) (*Never overwrites existing check-points*).

---

## 4. Steps to Start Training Once Data is Available

Once the 3m PlanetScope scene is downloaded and placed under `data/raw/planet/`, execute these three steps sequentially:

### **Step 1: Validate Inputs**
Run the validator script to ensure the imagery meets all contract parameters (CRS, bands, overlap, reflectance ranges):
```bash
python scripts/validate_experiment4_input.py \
    --reference data/raw/planet/20260211_054815_64_254a_3m.tif \
    --sentinel data/raw/sentinel2/S2B_MSIL2A_20260211T050839_N0512_R019_T43PGQ_20260211T085923_Red_10m.jp2
```
If the status changes to `READY_FOR_REAL_TRAINING`, proceed.

### **Step 2: Generate the Geographic Patch Plan**
Run the planning script to align grids, slice valid patches (excluding NoData areas), and define the non-leaking spatial splits:
```bash
python scripts/plan_experiment4_patches.py \
    --reference data/raw/planet/20260211_054815_64_254a_3m.tif \
    --sentinel data/raw/sentinel2/S2B_MSIL2A_20260211T050839_N0512_R019_T43PGQ_20260211T085923_Red_10m.jp2
```
Verify the number of valid patch pairs in `outputs/experiment4_real_patch_plan.json`.

### **Step 3: Coregister and Slice Dataset**
Use the coregistration and dataset slicer script to reproject, snap coordinates, and output training tensors:
```bash
python prepare_reference_dataset.py \
    --reference data/raw/planet/20260211_054815_64_254a_3m.tif \
    --sentinel data/raw/sentinel2/S2B_MSIL2A_20260211T050839_N0512_R019_T43PGQ_20260211T085923_Red_10m.jp2
```

### **Step 4: Launch Training**
Run the real-data training pipeline using the new configuration:
```bash
python train_srm.py --config configs/experiment4_real_training.yaml
```
*(Weights will be saved to `models/experiment4_real/best_model.pth`)*.
