# Geospatial Ingestion and Coregistration Pipeline: Sentinel-2 & PlanetScope

This document outlines the design and mathematical specifications of the cross-sensor super-resolution mapping dataset pipeline for project **SIH26142**.

---

## 1. Spatial Resolutions
*   **Low-Resolution (LR)**: Sentinel-2 Level-2A (10 m/pixel for bands B02, B03, B04, B08).
*   **High-Resolution (HR)**: PlanetScope Ortho Scene (3.0 m/pixel, resampled to exactly 5.0m or 2.5m for SRM grid alignment).

---

## 2. Coordinate Reference System (CRS) Handling
To align cross-sensor data, the reference imagery (PlanetScope) must be reprojected into the **Sentinel-2 local UTM coordinate system** (for our Bengaluru scene, this is **`EPSG:32643`** - UTM Zone 43N). Reprojection is handled using `rasterio.warp.reproject`.

---

## 3. Resampling Method
*   **Bilinear Resampling** is used during the reprojection phase. It interpolates pixel values smoothly, minimizing nearest-neighbor aliasing artifacts while preserving structural edges.
*   **Bicubic / Lanczos** resampling can be used as alternatives for higher-frequency edge sharpness, but bilinear is chosen as the default for radiometric stability.

---

## 4. Spectral Band Correspondence
The four selected Sentinel-2 bands correspond directly to the PlanetScope spectral bands:

| Spectral Band | Sentinel-2 L2A Band | PlanetScope Dove Band | Wavelength Range |
| :--- | :--- | :--- | :--- |
| **Blue** | B02 (10m) | Band 1 or 2 (approx. 3m) | 450 – 515 nm |
| **Green** | B03 (10m) | Band 2 or 3 (approx. 3m) | 510 – 590 nm |
| **Red** | B04 (10m) | Band 3 or 4 (approx. 3m) | 600 – 690 nm |
| **NIR** | B08 (10m) | Band 4 or 8 (approx. 3m) | 760 – 890 nm |

---

## 5. Spatial Grid Alignment
To perform Super-Resolution training, the high-resolution grid must align exactly with the Sentinel-2 grid without fractional pixel offsets.

Our coregistration script resolves this by:
1.  Projecting the reference bounds into `EPSG:32643`.
2.  Calculating the geographic intersection between the Sentinel-2 image and the reference image.
3.  Snapping the top-left origin coordinates of the intersection to the nearest integer Sentinel-2 pixel boundary:
    $$\text{Col}_{\text{start}} = \lfloor \frac{X_{\text{ref}} - X_{\text{S2}}}{dx_{\text{S2}}} \rfloor, \quad \text{Row}_{\text{start}} = \lfloor \frac{Y_{\text{ref}} - Y_{\text{S2}}}{dy_{\text{S2}}} \rfloor$$
4.  Setting the target grid resolution to exactly $\frac{10.0}{\text{upscale\_factor}}$ (e.g. 5.0m for factor 2, or 2.5m for factor 4).
This snapped transform ensures that each Sentinel-2 pixel maps directly to a discrete $K \times K$ block of sub-pixels in the reference grid, preventing interpolation smear during patch slicing.

---

## 6. Temporal Alignment
*   Satellite scenes are subject to rapid land-cover changes (construction, vegetation cycles, harvest) and atmospheric illumination shifts.
*   The pipeline evaluates the acquisition dates. A temporal gap of **$\le 5$ days** is considered excellent; gaps exceeding **30 days** generate a quality warning in the QC report.

---

## 7. Patch Generation
*   **LR Patch Size**: Default $32 \times 32$ pixels (Sentinel-2).
*   **HR Patch Size**: Default $32 \times K$ (where $K$ is the upscale factor). For $K=4$, the HR patch size is $128 \times 128$ pixels.
*   **Georeferencing Preservation**: Every single LR/HR patch retains its local Affine geotransform:
    $$\text{Patch Transform} = \text{Affine}(dx, 0, X_{\text{patch\_origin}}, 0, dy, Y_{\text{patch\_origin}})$$
    This preserves the geographic coordinates of every sub-grid for mapping outputs.

---

## 8. Quality Control Rules
Patches are automatically audited and filtered according to the following rules:
1.  **NoData Check**: If a patch contains more than $10\%$ NoData values (pixels with zero reflectance or value matching `src.nodata`), the patch is rejected.
2.  **Resolution Hierarchy Check**: The reference pixel size must be strictly smaller than the Sentinel-2 pixel size (Reference < 10m).
3.  **Intersection Check**: If the spatial overlap between reference and Sentinel-2 bounds is empty, dataset generation is aborted.

---

## 9. Inconsistency & Uncertainty Considerations

> [!IMPORTANT]
> **Sentinel-2 and PlanetScope are separate instruments with distinct physical characteristics.**

1.  **Spectral Response Function (SRF) Discrepancies**:
    Even though bands cover the same general range, their sensor sensitivities differ. Direct comparison of raw Digital Numbers (DN) is scientifically invalid. Inputs must be normalized to **Surface Reflectance (L2A equivalent)**, and radiometric normalization (e.g., histogram matching or linear scaling) should be applied before loss calculations.
2.  **Orthorectification and Coregistration Offsets**:
    Even after snapped grid alignment, sub-pixel shifts of 1–3 meters may remain due to differing elevation models used during processing. This spatial misalignment can inflate MSE loss during supervised training.
3.  **Viewing Geometry and Atmosphere**:
    Varying satellite view angles (off-nadir angles) and atmospheric conditions can alter the apparent reflectance of ground features, adding training noise.
