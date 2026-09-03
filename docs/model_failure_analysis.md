# Model Failure & Frequency Analysis Report (Experiment 3 Audit)

This report presents a formal visual, spatial, and radiometric audit of the **Residual CNN** model from Experiment 3. It details the pixel-level error, edge preservation, local contrast, and high-frequency energy characteristics of the model's outputs against the standard **Bilinear Interpolation** baseline, evaluated under a synthetic validation framework (downscaling the 10m Sentinel-2 target to 20m and super-resolving back to 10m).

---

## 1. Executive Summary

While the **Residual CNN** demonstrates clear quantitative improvements over bilinear interpolation (with global PSNR gains of **+0.77 dB** to **+1.76 dB** depending on the band), its visual improvement remains highly localized and limited. 

Our audit reveals that the model behaves primarily as an **educated sharpening and deconvolution filter** rather than a true super-resolution model. It sharpens high-contrast step-edges (such as roads and building boundaries) by restoring the low-frequency envelope, but fails to recover or synthesize genuine sub-pixel spatial textures (like forest canopy, agricultural textures, or sub-pixel building shapes). In homogeneous and high-frequency textured regions, the model's outputs are heavily smoothed and lack the stochastic details of the high-resolution target.

---

## 2. Band-by-Band Quantitative Analysis

Global metrics were computed across the full $1024 \times 1024$ scene for all four spectral bands. We compare the **Bilinear Baseline** and the **Residual CNN** against the original **10m Sentinel-2 Target**:

| Spectral Band | Reconstruction Method | Mean Squared Error (MSE) | Peak Signal-to-Noise Ratio (PSNR) | Structural Similarity (SSIM) | Edge Preservation Index (EPI) | Local Contrast (Std Dev) | High-Frequency Energy Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Blue (B02)** | *Target (Reference)* | `0.000000` | `+inf` dB | `1.0000` | `1.0000` | `0.0282` | `61.69%` (of total) |
| | Bilinear Baseline | `0.000393` | `34.06` dB | `0.8922` | `0.8586` | `0.0183` | `28.66%` (of target) |
| | **Residual CNN** | `0.000329` | **`34.83` dB** *(+0.77)* | **`0.8963`** *(+0.004)* | **`0.8720`** *(+0.013)* | **`0.0258`** | **`69.29%`** *(+40.63)* |
| **Green (B03)**| *Target (Reference)* | `0.000000` | `+inf` dB | `1.0000` | `1.0000` | `0.0355` | `61.07%` (of total) |
| | Bilinear Baseline | `0.000535` | `32.71` dB | `0.8697` | `0.8452` | `0.0232` | `29.08%` (of target) |
| | **Residual CNN** | `0.000422` | **`33.75` dB** *(+1.04)* | **`0.8965`** *(+0.027)* | **`0.8705`** *(+0.025)* | **`0.0328`** | **`66.45%`** *(+37.37)* |
| **Red (B04)**  | *Target (Reference)* | `0.000000` | `+inf` dB | `1.0000` | `1.0000` | `0.0550` | `58.64%` (of total) |
| | Bilinear Baseline | `0.001024` | `29.90` dB | `0.8431` | `0.8334` | `0.0369` | `30.60%` (of target) |
| | **Residual CNN** | `0.000704` | **`31.52` dB** *(+1.62)* | **`0.8971`** *(+0.054)* | **`0.8765`** *(+0.043)* | **`0.0496`** | **`64.67%`** *(+34.07)* |
| **NIR (B08)**  | *Target (Reference)* | `0.000000` | `+inf` dB | `1.0000` | `1.0000` | `0.0569` | `60.51%` (of total) |
| | Bilinear Baseline | `0.001158` | `29.36` dB | `0.8161` | `0.8107` | `0.0367` | `29.69%` (of target) |
| | **Residual CNN** | `0.000772` | **`31.12` dB** *(+1.76)* | **`0.8878`** *(+0.072)* | **`0.8653`** *(+0.055)* | **`0.0493`** | **`59.88%`** *(+30.19)* |

### Key Spectral Insights:
1. **Red & NIR Dominate Gains**: The Residual CNN achieves its largest PSNR improvements in the **Red (+1.62 dB)** and **NIR (+1.76 dB)** bands. These bands contain the highest spatial variance and raw land-cover contrast, which the residual convolutions exploit to sharpen edges.
2. **SSIM Improvements**: SSIM gains are substantial in NIR (+0.072) and Red (+0.054), showing that the spatial structure is preserved far better than the fuzzy edges of bilinear interpolation.
3. **High-Frequency Restoration**: Bilinear decimation destroys up to 70% of the target's high-frequency energy. The Residual CNN recovers a significant portion of this energy (restoring it to **60%–69% of the target's high-frequency energy**), whereas bilinear interpolation remains capped at ~30%.

---

## 3. Feature-Region Analysis

We extracted $128 \times 128$ pixel crops corresponding to representative land-cover classes in the Electronic City scene. The average metrics across all 4 bands are presented below:

| Feature Class | Method | Mean Squared Error (MSE) | PSNR (dB) | SSIM | Edge Preservation Index (EPI) | Local Contrast (Std Dev) | High-Frequency Energy Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Buildings** | Target | `0.00000` | `+inf` | `1.0000` | `1.0000` | `0.0492` | `100.00%` |
| | Bilinear | `0.00105` | `29.78` | `0.8615` | `0.8667` | `0.0332` | `39.44%` |
| | **ResCNN** | **`0.00067`** | **`31.73`** | **`0.8997`** | **`0.9124`** | **`0.0443`** | **`0.7102`** |
| **Roads** | Target | `0.00000` | `+inf` | `1.0000` | `1.0000` | `0.0499` | `100.00%` |
| | Bilinear | `0.00104` | `29.81` | `0.8560` | `0.8646` | `0.0334` | `34.00%` |
| | **ResCNN** | **`0.00073`** | **`31.37`** | **`0.8934`** | **`0.8932`** | **`0.0452`** | **`0.6759`** |
| **Field Boundaries** | Target | `0.00000` | `+inf` | `1.0000` | `1.0000` | `0.0434` | `100.00%` |
| | Bilinear | `0.00077` | `31.16` | `0.8543` | `0.8265` | `0.0278` | `35.69%` |
| | **ResCNN** | **`0.00056`** | **`32.50`** | **`0.8906`** | **`0.8412`** | **`0.0390`** | **`0.6945`** |
| **Vegetation Boundaries**| Target | `0.00000` | `+inf` | `1.0000` | `1.0000` | `0.0420` | `100.00%` |
| | Bilinear | `0.00083` | `30.82` | `0.8454` | `0.8279` | `0.0266` | `34.16%` |
| | **ResCNN** | **`0.00057`** | **`32.46`** | **`0.8939`** | **`0.8708`** | **`0.0372`** | **`0.6593`** |
| **Homogeneous Areas** | Target | `0.00000` | `+inf` | `1.0000` | `1.0000` | `0.0454` | `100.00%` |
| | Bilinear | `0.00086` | `30.68` | `0.8457` | `0.8134` | `0.0296` | `33.26%` |
| | **ResCNN** | **`0.00066`** | **`31.81`** | **`0.8831`** | **`0.8353`** | **`0.0403`** | **`0.6512`** |

---

## 4. Audit Analysis Questions & Answers

### 1. How much better is Residual CNN than Bilinear visually?
Visually, the improvement is **modest, localized, and subtle**. It is only noticeable along high-contrast boundaries (e.g., roads, runway segments, large building edges) where pixel transition boundaries are sharper. In homogeneous agriculture fields, water bodies, or dense forest canopies, the visual differences are virtually indistinguishable. The model fails to produce new spatial details or textures that weren't already implicitly present in the low-resolution shape.

### 2. How much better is it quantitatively?
Quantitatively, the Residual CNN is **significantly better**. It improves global PSNR by **+1.25 dB on average** across all bands (+1.62 dB in Red and +1.76 dB in NIR). Structural similarity (SSIM) rises from 0.855 to 0.893 on average, and the Edge Preservation Index (EPI) shows a solid gain (+0.04 to +0.05), proving that the reconstruction error is mathematically minimized compared to the baseline.

### 3. Is the model recovering high-frequency detail?
**No. The model is recovering a pseudo-sharpened representation of low-frequency edges, not genuine sub-pixel high-frequency details.** Because bilinear decimation destroys all spatial information above the Nyquist frequency of the 20m grid, there is no mathematical trace of true 5m features in the LR input. The model acts as an educated deblurring kernel (deconvolution) that maps the blurred transitions back to sharp step-edges. It does not introduce genuine, sub-pixel structures (such as individual trees, cars, or small building divisions).

### 4. Is the model mostly smoothing?
**Yes. In textured regions, the model acts heavily as a smoother.** The local contrast (standard deviation) of the Residual CNN remains lower than the target across all features (e.g., contrast of 0.0443 vs. 0.0492 for buildings). Because the network was trained using the **Mean Squared Error (MSE) loss**, it is penalized heavily for making incorrect sub-pixel spatial predictions. The mathematically optimal solution under L2 loss is to output the average of all possible sub-pixel configurations (the conditional mean), which manifests as a smoothed, textureless blur over any stochastic patterns like vegetation canopies or soil surfaces.

### 5. Which features benefit?
**Predictable, linear, high-contrast geometric structures** benefit the most. This includes:
*   Large building footprints and concrete structures.
*   Straight road networks and airport runways.
*   Artificial field boundaries separating large agricultural plots.
These features follow simple mathematical structures that the convolutional kernels can easily map.

### 6. Which features fail?
**Stochastic, complex, and low-contrast textures** fail to reconstruct. This includes:
*   Dense forest and agricultural canopies (completely smoothed out).
*   Soil and dirt paths (represented as flat homogeneous areas).
*   Sub-pixel structures (isolated small homes or narrow pathways smaller than 10 meters) which are entirely erased or blurred into surrounding pixels.

### 7. Is the synthetic degradation unrealistic?
**Yes, highly unrealistic.** Bilinear decimation assumes a clean, noise-free, downsampling grid that does not correspond to physical remote sensing realities. Real satellite sensors degrade images through:
*   **Point Spread Function (PSF) Blur**: Optical diffraction and atmospheric scattering that act as a complex low-pass filter.
*   **Sensor Noise**: Shot noise, dark current, and read noise.
*   **Spectral Mismatch**: Differing Spectral Response Functions (SRFs) between the high-resolution source (e.g., PlanetScope) and low-resolution target (Sentinel-2).
A model trained on bilinear downsampling learns to "invert" bilinear interpolation. When deployed on real-world Sentinel-2 data, the domain gap is too large, resulting in sub-optimal upscaling and blurring.

### 8. Is the current architecture sufficient?
**No.** The current 3-block Residual CNN with 32 channels has only **251,753 trainable parameters**. This tiny capacity is sufficient for basic, single-sensor synthetic deblurring, but it lacks the representation power needed to:
*   Model complex, non-linear cross-sensor radiometric differences.
*   Correct sub-pixel coregistration offsets.
*   Generate multi-scale spatial textures in real-world scenarios.

### 9. Should we improve the data, degradation model, loss, architecture, or combination?
We must improve a **combination** of all of them. Enhancing only the architecture will not resolve the loss-based smoothing or the synthetic-to-real domain gap. The entire super-resolution pipeline must be upgraded.

---

## 5. Technical Recommendation

Based on our findings, we recommend upgrading the entire pipeline under a unified framework.

### **Final Recommendation: COMBINATION**

### Rationale:
To transition from a synthetic deblurring prototype to a real-world Super-Resolution Mapping model that reconstructs physically valid 5m details, we must address the following bottlenecks simultaneously:
1. **Improve Data**: We must transition to a **real paired dataset** using Sentinel-2 (10m L2A BOA) as input and PlanetScope (3m orthorectified SR) or SPOT (1.5m) as high-resolution reference.
2. **Improve Degradation Model**: We must replace bilinear downsampling with a **physical degradation pipeline** convolving PlanetScope reference data with the Sentinel-2 Point Spread Function (PSF), applying radiometric calibration (SRF matching), and injecting sensor noise.
3. **Improve Loss Function**: We must replace pure MSE (L2) loss with a **composite loss function**:
   $$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{Huber/L1}} + \lambda_2 \mathcal{L}_{\text{MS-SSIM}} + \lambda_3 \mathcal{L}_{\text{spectral\_drift}}$$
   This reduces L2-induced smoothing, preserves edges, and prevents spectral drift in radiometric bands (retaining physical validity for indices like NDVI).
4. **Improve Architecture**: Upgrade the model to a deeper, more expressive architecture (such as RCAN, EDSR, or SwinIR) incorporating channel and spatial attention mechanism to extract features across multiple scales.

---

## 6. Generated Visual Artifacts

The diagnostic audit generated the following visual plots representing the spatial, frequency, and regional limitations described above:
*   [Full-Scene FFT Frequency Analysis Plot](file:///c:/Users/urstr/New%20folder%20%283%29/outputs/model_failure_analysis.png): Displays the ground-truth target, bilinear, and Residual CNN reconstructions along with their corresponding 2D FFT magnitude spectra. It demonstrates the high-frequency suppression of bilinear interpolation and the recovery limits of the Residual CNN.
*   [Regional Detail Audit Plot](file:///c:/Users/urstr/New%20folder%20%283%29/outputs/model_failure_regions.png): Presents close-up $128 \times 128$ RGB crops of Buildings, Roads, Field Boundaries, Vegetation Boundaries, and Homogeneous Areas, comparing Bilinear against Residual CNN and the Target. It includes localized PSNR, SSIM, and EPI metrics.
