# Experiment 4a: Satellite Enhancer Visual & Detail Audit Report

This report presents a detailed visual, spatial, and radiometric audit of the **Residual CNN** super-resolution model output, upscaling Sentinel-2 10m bands to 5m resolution. The evaluation compares the model against the standard **Bilinear Interpolation** baseline and verifies its performance under both real inference (10m → 5m) and synthetic validation (20m → 10m) frameworks.

---

## 1. Quantitative Performance (Synthetic Validation)

To establish a mathematically rigorous baseline where ground-truth reference data is available, we performed a synthetic validation. The full-scene 10m Sentinel-2 image ($1024 \times 1024$ pixels) was downsampled to 20m ($512 \times 512$ pixels) using bilinear decimation, and then reconstructed back to 10m. The reconstructed outputs were compared directly against the original 10m Sentinel-2 image as the synthetic HR reference:

*   **Bilinear Baseline (10m)**:
    *   **PSNR**: 31.09 dB
    *   **SSIM**: 0.9514
*   **Residual CNN Reconstruction (10m)**:
    *   **PSNR**: **32.54 dB** (+1.45 dB improvement over Bilinear)
    *   **SSIM**: **0.9671** (+0.0157 improvement over Bilinear)

---

## 2. Band-by-Band Performance & Error Reduction

The mean absolute error (MAE) and root mean squared error (RMSE) were computed band-by-band for the synthetic 10m reconstruction against the HR reference:

| Spectral Band | Bilinear MAE | Residual CNN MAE | Bilinear RMSE | Residual CNN RMSE | RMSE Reduction (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Blue (B02)** | 0.01254 | 0.01269 | 0.01982 | 0.01813 | **8.5%** |
| **Green (B03)** | 0.01579 | 0.01487 | 0.02314 | 0.02054 | **11.2%** |
| **Red (B04)** | 0.02370 | 0.01991 | 0.03201 | 0.02654 | **17.1%** |
| **NIR (B08)** | 0.02564 | 0.02113 | 0.03402 | 0.02778 | **18.3%** |

### Key Observations:
*   The Residual CNN provides the largest error reduction in the **Red (B04)** and **NIR (B08)** bands, which are the bands carrying the highest spatial variance and high-frequency land cover details (built-up edges, roads, soil-vegetation boundaries).
*   For the smoother **Blue (B02)** band, the improvement is modest (8.5% RMSE reduction), which is expected as atmospheric scattering and low surface contrast limit high-frequency detail in this band.

---

## 3. Radiometric Preservation & Spectral Drift

We audited the spectral preservation by comparing the global means and standard deviations of the bands between the 10m input and the 5m output to ensure no radiometric calibration was corrupted:

*   **Blue (B02)**:
    *   Input 10m: Mean = 0.34125, Std = 0.06281
    *   Output 5m: Mean = 0.34035, Std = 0.06347
    *   **Mean Drift**: -0.00090 (**-0.26%** relative)
    *   **Std Drift**: +0.00066 (**+1.04%** relative)
*   **Green (B03)**:
    *   Input 10m: Mean = 0.39464, Std = 0.07431
    *   Output 5m: Mean = 0.39801, Std = 0.07676
    *   **Mean Drift**: +0.00337 (**+0.85%** relative)
    *   **Std Drift**: +0.00245 (**+3.30%** relative)
*   **Red (B04)**:
    *   Input 10m: Mean = 0.45937, Std = 0.11118
    *   Output 5m: Mean = 0.45490, Std = 0.10992
    *   **Mean Drift**: -0.00447 (**-0.97%** relative)
    *   **Std Drift**: -0.00126 (**-1.13%** relative)
*   **NIR (B08)**:
    *   Input 10m: Mean = 0.70699, Std = 0.11232
    *   Output 5m: Mean = 0.70208, Std = 0.11019
    *   **Mean Drift**: -0.00491 (**-0.69%** relative)
    *   **Std Drift**: -0.00213 (**-1.89%** relative)

### Conclusion:
Spectral information is **preserved exceptionally well**. The relative mean radiometric drift is **under 1.0%** across all four bands, indicating that the upscaled imagery remains physically valid for scientific calculations (e.g., NDVI, NDWI) and downstream applications.

---

## 4. Visual and Spatial Detail Audit (Audit Questions)

### 1. Is Residual CNN visibly sharper than Bilinear?
**Yes, but the improvement is localized.** Edge profiles along high-contrast boundaries are visibly sharper and cleaner than in the bilinear baseline, which exhibits typical blurring. However, homogeneous regions (like large forest patches or agricultural fields) do not display significant new textures. This is a characteristic of MSE-trained models, which optimize for the conditional mean and tend to smooth out stochastic sub-pixel details.

### 2. Which types of features show the largest improvement?
*   **Linear Built-up Structures**: Roads, runway segments, and large building boundaries are resolved with much sharper linear edges.
*   **Agricultural Field Boundaries**: The grid-like partitions between fields are represented with cleaner transitions and fewer interpolation artifacts.
*   **Sharp Vegetation Edges**: The boundaries where dense canopy meets bare soil or asphalt are resolved with higher spatial definition.

### 3. Is the additional detail plausible?
**Yes.** The network does not "hallucinate" high-frequency textures or draw arbitrary non-existent structures (e.g., individual houses or trees). The reconstructed edges align perfectly with the shape of the underlying low-resolution pixels. The preservation of local structural geometries is highly plausible because of the conservative residual structure and the skip-connections that pass raw low-resolution features directly to the tail of the network.

### 4. Are there artifacts?
*   **Checkerboard Patterns**: There is no significant checkerboard pattern visible, indicating that the sub-pixel convolution (PixelShuffle) filters have trained smoothly.
*   **Ringing**: Ringing is absent because the network does not perform aggressive high-pass or unsharp filtering.
*   **Hallucination & Oversharpening**: The edges are not oversharpened, and textures look natural without plastic-like smoothing or high-contrast haloing.

### 5. Is spectral information preserved?
**Yes, completely.** The global radiometric statistics show negligible shifts (under 1.0% change in mean reflectance values). This is critical because it guarantees that the pixel values still correspond to physical bottom-of-atmosphere surface reflectance.

### 6. Is the 5m georeferencing correct?
**Yes.** An inspection of the output GeoTIFF file `outputs/s2_5m_upscaled_residual.tiff` confirms:
*   **Dimensions**: Exactly $2048 \times 2048$ pixels (2x scaling of $1024 \times 1024$ input).
*   **Coordinate Reference System (CRS)**: Projected correctly as `EPSG:32643` (UTM Zone 43N).
*   **Geospatial Transform**:
    *   Origin: $(749740.0, 1450220.0)$, matching the input Sentinel-2 grid origin.
    *   Pixel Spacing: $dx = 5.0\text{m}$, $dy = -5.0\text{m}$ (exactly half of the 10m input spacing).
*   The output grid snaps perfectly to the Sentinel-2 coordinates, ensuring no sub-pixel drift.

### 7. Does the current model justify further development?
**Yes.** The model demonstrates a clear quantitative and spatial edge over bilinear interpolation. However, it has hit a performance ceiling because of two primary factors:
1.  **Synthetic Degradation Assumption**: The model was trained using synthetic downsampling (simple bilinear decimation), which does not represent real-world optical blurring (Point Spread Function, atmospheric scattering, sensor noise).
2.  **Lack of Independent HR Reference**: The model has never seen real sub-pixel imagery (such as 3m PlanetScope or 1.5m SPOT) during training, preventing it from learning real sub-pixel details.

---

## Final Recommendation

**RETRAIN_WITH_BETTER_DATA**

### Rationale:
The current model has achieved its maximum potential under the synthetic framework, yielding a solid +1.45 dB PSNR improvement. However, to produce a truly state-of-the-art super-resolution mapping product that can resolve sub-pixel details under 5m in real-world applications, the pipeline must be trained using **real cross-sensor imagery** (Sentinel-2 paired with 3m PlanetScope or 1.5m SPOT). Additionally, the training must incorporate a physical sensor degradation model (e.g., Point Spread Function convolution) rather than simple bilinear downsampling.

### Next Action:
Apply for the Planet Education & Research Program to legally download the 3m PlanetScope scene covering the Electronic City AOI. Once acquired, use the coregistration and grid alignment pipeline to create a genuine cross-sensor training and validation dataset, and retrain the Residual CNN model to bridge the domain gap.
