# Experiment 4: Real-Data Training and Quality Control Contract

This contract defines the strict data validation rules, technical specifications, and quality acceptance criteria for training and validating the Sentinel-2 (10m) to PlanetScope (3m) Super-Resolution Mapping (SRM) model for project **SIH26142**.

---

## 1. Input Data Specifications

### Low-Resolution (LR) Input: Sentinel-2 L2A
*   **Source**: Copernicus Data Space Ecosystem (CDSE) / Sentinel-2 Level-2A Bottom-of-Atmosphere (BOA) Surface Reflectance.
*   **Bands Required**: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR).
*   **Native Spatial Resolution**: 10.0 meters per pixel.
*   **Bit Depth / Range**: Floating-point reflectance normalized in the range `[0.0, 1.0]` (converted from uint16 by dividing by 10000.0).

### High-Resolution (HR) Reference: PlanetScope
*   **Source**: Planet Insights Platform / PlanetScope Dove Classic, Dove-R, or SuperDove.
*   **Bands Required**: Blue, Green, Red, Near-Infrared (NIR).
*   **Native Spatial Resolution**: ~3.0 meters orthorectified (resampled from native 3.7–4.1m).
*   **Bit Depth / Range**: Floating-point reflectance normalized in the range `[0.0, 1.0]` (converted from uint16 by dividing by 10000.0).
*   **Acceptable Product Types**: 
    *   `PlanetScope Ortho Scene` (Surface Reflectance / `SR` asset class) - **Required**.
    *   *Note*: Analytic Radiance (`analytic`) products are not accepted directly without atmospheric calibration. True-color RGB JPEGs or lossy basemaps are strictly prohibited.

---

## 2. Spectral Band Correspondence

The 4-channel input tensor and reference tensor must map exactly as follows:

| Tensor Channel | Spectral Band | Sentinel-2 L2A Band | PlanetScope Band Sequence | Wavelength Range |
| :--- | :--- | :--- | :--- | :--- |
| **Channel 0** | **Blue** | B02 (490 nm) | Band 1 (Classic/Dove-R) or Band 2 (SuperDove) | 450 – 515 nm |
| **Channel 1** | **Green** | B03 (560 nm) | Band 2 (Classic/Dove-R) or Band 3 (SuperDove) | 510 – 590 nm |
| **Channel 2** | **Red** | B04 (665 nm) | Band 3 (Classic/Dove-R) or Band 4 (SuperDove) | 600 – 690 nm |
| **Channel 3** | **NIR** | B08 (842 nm) | Band 4 (Classic/Dove-R) or Band 8 (SuperDove) | 760 – 890 nm |

---

## 3. Geospatial & Grid Requirements

*   **Coordinate Reference System (CRS)**: Must be projected in the local UTM coordinate system matching Sentinel-2. For the Electronic City, Bengaluru AOI, this is strictly **`EPSG:32643` (UTM Zone 43N)**. If the reference is in another CRS (e.g., WGS84 - `EPSG:4326`), coregistration must warp it using bilinear resampling to `EPSG:32643`.
*   **Geotransform Transform Alignment**: Snapped grid mapping must be applied to prevent coordinate drift. The top-left corner origin of the coregistered HR image must align with a discrete pixel boundary of the Sentinel-2 grid:
    $$\text{Col}_{\text{start}} = \lfloor \frac{X_{\text{ref\_origin}} - X_{\text{S2\_origin}}}{dx_{\text{S2}}} \rfloor, \quad \text{Row}_{\text{start}} = \lfloor \frac{Y_{\text{ref\_origin}} - Y_{\text{S2\_origin}}}{dy_{\text{S2}}} \rfloor$$
*   **Pixel Size Resolution Ratio**: The spatial resolution ratio must be exactly equal to the target upscale factor:
    $$\text{Resolution Ratio} = \frac{\text{Sentinel-2 Pixel Size}}{\text{Aligned Reference Pixel Size}} = \text{Upscale Factor}$$
    For $K=2$, the reference pixel resolution must be exactly **5.0m**. For $K=4$, the reference pixel resolution must be exactly **2.5m**.
*   **Geographic Intersection (Overlap)**: The spatial intersection bounds between Sentinel-2 and PlanetScope must be non-empty and fully cover the target AOI of Electronic City, Bengaluru.

---

## 4. Quality Control Criteria (PASS / WARN / FAIL)

### PASS (Dataset meets all criteria for immediate training)
*   **Readability**: Both GeoTIFF files open and read successfully using `rasterio`.
*   **CRS**: Both rasters are in `EPSG:32643` (or PlanetScope has been successfully reprojected to it).
*   **Overlap**: Both rasters have a non-empty spatial intersection bounds containing the Electronic City coordinates.
*   **Resolution Ratio**: Aligned reference pixel size is exactly $5.0\text{m} \times 5.0\text{m}$ (for upscale factor 2) or $2.5\text{m} \times 2.5\text{m}$ (for upscale factor 4).
*   **Band count**: PlanetScope file contains at least 4 bands (RGB + NIR).
*   **Temporal Gap**: The absolute temporal gap between Sentinel-2 and PlanetScope acquisition dates is **$\le 5$ days**.
*   **NoData Ratio**: The total fraction of NoData or zero-valued pixels in the overlapping scene area is **$< 1\%$**.
*   **Reflectance Range**: Pixel values in all bands lie strictly within the physical reflectance range of `[0.0, 1.0]` (or `[0, 10000]` for raw uint16).

### WARN (Dataset usable but generates training QC warnings)
*   **Temporal Gap**: Absolute temporal difference is **between 6 and 30 days**. Generates a warning because of seasonal vegetation growth or building shadow shifts.
*   **NoData Ratio**: NoData/zero pixel fraction is **between 1% and 10%**. Patches containing NoData values will be filtered out during slicing.
*   **Reflectance Clipping**: Minor out-of-bounds pixel values (e.g. slight negative reflectance due to atmospheric correction artifact or oversaturation values > 1.0) are present in less than 0.1% of pixels. These will be clipped to `[0.0, 1.0]`.

### FAIL (Dataset is rejected, training is aborted)
*   **File missing**: Either file cannot be located on the local disk.
*   **Raster unreadable**: Either file is corrupted or cannot be read by `rasterio`.
*   **No Overlap**: Geographic bounds do not intersect.
*   **Resolution mismatch**: Reference resolution is coarser than Sentinel-2 (e.g., ref resolution $\ge 10\text{m}$).
*   **Band mismatch**: PlanetScope file has $<4$ bands or is missing the NIR band.
*   **Temporal Gap**: Acquisition dates are separated by **$> 30$ days**. The image pair is rejected to prevent training on outdated land-use features.
*   **NoData Ratio**: Overlapping region contains **$> 10\%$ NoData/zero pixels**.
*   **Reflectance Corruption**: Reflectance ranges are wildly out of bounds (e.g. negative values or values exceeding 1.5 in more than 1% of the pixels, indicating wrong product type or scaling).
*   **Product Type Violation**: Reference file is a lossy RGB JPEG/PNG or processed visual basemap without NIR band.

---

## 5. Metadata Schema and Validation Matrix

Every input pair must be logged with the following metadata structure:

```json
{
    "validation_timestamp": "2026-08-31T12:10:00+05:30",
    "reference_file": "data/raw/planet/20260211_054815_64_254a_3m.tif",
    "sentinel_file": "data/raw/sentinel2/S2B_MSIL2A_20260211T050839_Red_10m.jp2",
    "checks": {
        "file_exists": "PASS",
        "raster_readability": "PASS",
        "crs_alignment": "PASS",
        "spatial_overlap": "PASS",
        "resolution_hierarchy": "PASS",
        "band_count_and_sequence": "PASS",
        "temporal_gap_days": 0,
        "nodata_fraction": 0.0,
        "reflectance_bounds": "PASS"
    },
    "status": "READY_FOR_REAL_TRAINING"
}
```
