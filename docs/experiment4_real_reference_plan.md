# Experiment 4: Real High-Resolution Super-Resolution Plan (Sentinel-2 to PlanetScope)

**Project Code**: SIH26142  
**Goal**: Build a cross-sensor Super-Resolution Mapping (SRM) training and validation pipeline from Sentinel-2 10m bands to an independent 3m PlanetScope high-resolution reference.

---

## 1. Real Data Contract & Specifications

To move from synthetic experiments to real-world super-resolution, the pipeline expects the following inputs:

### Low-Resolution (LR) Input: Sentinel-2 L2A
*   **Spatial Resolution**: 10 m/pixel.
*   **Spectral Bands**: Blue (B02), Green (B03), Red (B04), NIR (B08).
*   **Georeferencing**: CRS projected in local UTM (e.g., UTM Zone 43N - `EPSG:32643` for Bengaluru).

### High-Resolution (HR) Reference: PlanetScope (Dove / SuperDove)
*   **Spatial Resolution**: Approximately 3.0 m/pixel.
*   **Spectral Bands**: 4 bands mapped explicitly:
    *   Band 1 (Blue) $\rightarrow$ mapped to Sentinel-2 B02 (Blue)
    *   Band 2 (Green) $\rightarrow$ mapped to Sentinel-2 B03 (Green)
    *   Band 3 (Red) $\rightarrow$ mapped to Sentinel-2 B04 (Red)
    *   Band 4 (NIR) $\rightarrow$ mapped to Sentinel-2 B08 (NIR)
*   **File Format**: 4-band georeferenced GeoTIFF (`.tif` / `.tiff`).

---

## 2. Geospatial Coregistration & Grid Alignment
The high-resolution reference is projected and matched to the Sentinel-2 grid boundaries using the snapped grid method:
1.  **CRS Reprojection**: PlanetScope coordinate system is reprojected to match Sentinel-2 (`EPSG:32643`) using `rasterio.warp.reproject` and bilinear interpolation.
2.  **Grid Snapping**: The intersection origin of the reference bounds is snapped to the nearest integer Sentinel-2 grid boundary to ensure sub-pixel alignment:
    $$\text{Col}_{\text{start}} = \lfloor \frac{X_{\text{ref\_origin}} - X_{\text{S2\_origin}}}{dx_{\text{S2}}} \rfloor, \quad \text{Row}_{\text{start}} = \lfloor \frac{Y_{\text{ref\_origin}} - Y_{\text{S2\_origin}}}{dy_{\text{S2}}} \rfloor$$
3.  **No Direct PIL Resizing**: Alignment uses georeferenced transforms and `rasterio` windows to prevent coordinate drift.

---

## 3. Spectral and Temporal Alignment
*   **Spectral Matching**: Band sequence is strictly mapped. If the reference GeoTIFF contains fewer than 4 bands or does not match RGB+NIR, the pipeline will halt and report a band mismatch.
*   **Temporal Gap Checks**: Calculates the absolute difference between acquisition dates:
    $$\Delta t = |t_{\text{reference}} - t_{\text{sentinel}}| \text{ days}$$
    *   If $\Delta t \le 5$ days: Excellent.
    *   If $5 < \Delta t \le 30$ days: Warning issued (land-cover change risk).
    *   If $\Delta t > 30$ days: Rejected.

---

## 4. Critical Experiment Design (Geographic Patch Splitting)
To avoid spatial leakage (autocorrelation) between datasets, patch splitting is strictly structured by geographically separated rows/regions separated by a buffer zone:

```
+---------------------------------------------------------+
|                  TRAIN ZONE (Rows 0-23)                 |
+---------------------------------------------------------+
|            BUFFER ZONE (Rows 24-25) - DISCARDED         |
+---------------------------------------------------------+
|                VALIDATION ZONE (Rows 26-28)             |
+---------------------------------------------------------+
|            BUFFER ZONE (Rows 29-30) - DISCARDED         |
+---------------------------------------------------------+
|                  TEST ZONE (Rows 31-33)                 |
+---------------------------------------------------------+
```

*   **Buffer Width**: Minimum of 256 pixels (2560m geographic spacing) between Train, Val, and Test regions.
*   **Coordinate Bounds**: Every patch preserves its geotransform to map coordinates during model outputs.

---

## 5. Physical Degradation Model
We do **not** assume a simple bilinear downsampling relation between the high-resolution reference and Sentinel-2. Real sensor downsampling behaves according to:

1.  **Point Spread Function (PSF)**: Satellite optics scatter incoming light. A physical degradation model must convolve the high-resolution image with a spatial PSF kernel (e.g. Gaussian PSF matching the modulation transfer function of the sensor) before decimation.
2.  **Radiometric Normalization**: PlanetScope and Sentinel-2 observe targets through different atmospheric paths and calibrations. Reflectance values must be normalized (e.g., using linear regression or histogram matching over stable target land covers) to match their dynamic ranges before training.
3.  **Orthorectification Misalignment**: Residual 1–3m registration offsets remain even after snapped grid alignments. Training losses must incorporate structural indexes (like SSIM) or spatial shift alignment tolerance.

---

## 6. Current Dataset Readiness Status

### **READY_FOR_TRAINING: NO**

*   **Status Code**: `WAITING_FOR_REAL_HIGH_RES_REFERENCE`
*   **Reason**: The real high-resolution reference image file (e.g., PlanetScope 3m True-Color/NIR GeoTIFF covering the Electronic City AOI) is not present on the disk.
*   **Required Action**: Download the PlanetScope scene for **11 February 2026** (or closest cloud-free date within 10 days) and place it under `data/raw/planet/`.

---

## 7. Next Commands to Execute Once Reference is Available

1.  **Execute the Auditor**:
    ```bash
    python check_dataset_readiness.py --reference data/raw/planet/PlanetScope_ElectronicCity_20260211_3m.tif --sentinel data/raw/sentinel2/S2B_MSIL2A_20260211T050839_N0512_R019_T43PGQ_20260211T085923_Red_10m.jp2
    ```
2.  **Run Coregistration and Dataset Slicer**:
    ```bash
    python prepare_reference_dataset.py --reference data/raw/planet/PlanetScope_ElectronicCity_20260211_3m.tif --sentinel data/raw/sentinel2/S2B_MSIL2A_20260211T050839_N0512_R019_T43PGQ_20260211T085923_Red_10m.jp2
    ```
