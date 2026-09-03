# Experiment 4b: Physical Degradation Implementation Report

This report documents the design, scientific formulation, and parameter requirements of the modular physical degradation pipeline implemented in [degradation.py](file:///c:/Users/urstr/New%20folder%20(3)/src/super_resolution/degradation.py).

---

## 1. Why Bilinear Degradation is Insufficient

Synthetic super-resolution models are typically trained by downsampling high-resolution (HR) target images using bilinear decimation. While mathematically simple, this approach is **scientifically invalid** and insufficient for real-world remote sensing for several reasons:
* **Aliasing and Nyquist Violations**: Bilinear interpolation is not an ideal low-pass filter. Simple decimation introduces aliasing artifacts by folding high-frequency information above the low-resolution Nyquist limit, which does not occur in physical optical systems.
* **No Optical Blur Representation**: Real satellite sensors have optical lenses, diffraction limits, and atmospheric scatter. This combined spatial blur, known as the Point Spread Function (PSF), is a complex low-pass filter that is not represented by bilinear interpolation.
* **No Sensor Noise**: Real Sentinel-2 imagery contains instrument noise (photon shot noise, dark current, quantization noise). A model trained on clean, downsampled images will overfit to noise-free inputs, leading to poor generalization and artifact generation when deployed on real noisy observations.
* **Spectral Discrepancy**: Different sensors (e.g., PlanetScope Dove vs. Sentinel-2 MSI) have different Spectral Response Functions (SRFs), meaning their bands cover slightly different wavelength ranges. Naive downsampling ignores this radiometric mismatch.

---

## 2. What Physical Degradation Represents

The physical degradation model simulates the path of light from the Earth's surface through the atmosphere, entering the optics of a high-resolution sensor, and projects what the equivalent low-resolution sensor (Sentinel-2) would observe under the same conditions. It is formulated as a sequence of modular physical steps:

```
High-resolution reference (PlanetScope 3m)
              ↓
    [1. Radiometric Normalization]  (Scale raw DN values to physical BOA reflectance)
              ↓
         [2. Spatial PSF]           (Convolve with band-specific Gaussian optical blur)
              ↓
  [3. Spectral Response Matching]   (Apply linear calibration to align band sensitivities)
              ↓
     [4. Sensor Noise Model]        (Inject additive instrument noise)
              ↓
   [5. Spatial Sampling/Decimation]  (Integrate pixels over low-res sensor IFOV)
              ↓
Low-resolution observation (Sentinel-2-like 10m equivalent)
```

---

## 3. Spatial Blurring: Point Spread Function (PSF)

The spatial blurring of the satellite optics and atmospheric scattering is modeled using a 2D circularly symmetric Gaussian Point Spread Function (PSF) kernel:

$$H(x, y) = \frac{1}{2\pi \sigma_c^2} \exp\left(-\frac{x^2 + y^2}{2\sigma_c^2}\right)$$

Where:
* $\sigma_c$ is the band-specific standard deviation of the blur kernel, specified in high-resolution pixel units.
* The standard deviation is related to the sensor's Full Width at Half Maximum (FWHM) by:

$$\text{FWHM} = 2\sqrt{2\ln 2} \cdot \sigma_c \approx 2.35482 \cdot \sigma_c$$

For Sentinel-2, the spatial FWHM is approximately 1.0 to 1.3 pixels on the 10m grid (representing 10m to 13m ground footprint), which must be scaled to the resolution of the high-resolution input grid during configuration.

---

## 4. Spectral Response Differences

Even though both PlanetScope and Sentinel-2 capture Blue, Green, Red, and Near-Infrared (NIR) bands, their Spectral Response Functions (SRFs) differ. This means they detect photons with different sensitivities across wavelengths. 

To prevent the super-resolution model from corrupting physical reflectance and downstream indices (like NDVI), we model this discrepancy using a linear band calibration regression:

$$I_{\text{matched}}[c] = \alpha_c \cdot I[c] + \beta_c$$

Where:
* $c$ represents the spectral band channel.
* $\alpha_c$ is the band-specific slope (gain).
* $\beta_c$ is the band-specific intercept (offset).

---

## 5. Radiometric Differences

High-resolution and low-resolution imagery may be delivered in different formats, bit depths, or scaling factors. For instance:
* Sentinel-2 L2A BOA reflectance values are stored as raw integers scaled by `10000.0` (Digital Numbers).
* PlanetScope Analytic SR imagery is also scaled by `10000.0` but may exhibit minor calibration offsets.

The **Radiometric Normalization** step divides input Digital Numbers by the configured `input_scale` to normalize reflectance to the physical range $[0.0, 1.0]$, and clips out-of-bounds artifacts (negative values or values $> 1.0$ due to atmospheric correction or saturation):

$$I_{\text{normalized}} = \text{clip}\left(\frac{I_{\text{raw}}}{\text{input\_scale}}, 0.0, 1.0\right)$$

---

## 6. Sensor Noise Model

Real observations contain instrument noise (e.g., thermal noise, shot noise). Adding noise prevents the neural network from overfitting to noise-free synthetic inputs and learning high-frequency interpolation artifacts. We model this as Additive White Gaussian Noise (AWGN):

$$I_{\text{noisy}}[c] = I_{\text{normalized}}[c] + \eta_c, \quad \eta_c \sim \mathcal{N}(0, \sigma_{\text{noise}, c}^2)$$

Where:
* $\sigma_{\text{noise}, c}$ is the standard deviation of noise in physical reflectance units for channel $c$.

---

## 7. Spatial Sampling / Decimation

The low-resolution sensor integrates incoming radiance over its Instantaneous Field of View (IFOV). We simulate this pixel area integration by using an area-weighted decimation mode (`area` interpolation) in PyTorch rather than standard bilinear or bicubic downsampling, which mimics the sensor's grid accumulation and minimizes artificial aliasing.

---

## 8. Parameters That Are Known (Scientifically Defensible)

The following parameters are determined by the mathematical design of the pipeline and target spatial grid properties:
* **Radiometric Normalization Inputs**: `input_scale = 10000.0` for Sentinel-2/PlanetScope surface reflectance, scaling integers to the physical range `clip_min = 0.0` and `clip_max = 1.0`.
* **Spatial Decimation Scale**: `upscale_factor = 2` (for 5m output) or `upscale_factor = 4` (for 2.5m output).
* **Decimation Mode**: `area` mode representing area-weighted IFOV pixel integration.
* **PSF Window**: `kernel_size = 15` (or any odd integer large enough to support the Gaussian kernel span without edge truncation).

---

## 9. Parameters Requiring Authoritative Sensor Documentation

To prevent parameter fabrication and ensure scientific validity, the following parameters are left unassigned (`null`) in the default config and **MUST** be obtained from authoritative documentation before real training:

| Config Parameter | Physical Meaning | Sourcing Requirement | Source Document / Method |
| :--- | :--- | :--- | :--- |
| `psf.band_sigmas` | Blur standard deviation (pixels) for [B, G, R, NIR] | **REQUIRED** | ESA Sentinel-2 MSI Technical Guide (FWHM specs) & PlanetScope Dove Optical Performance Specifications. |
| `spectral_matching.slopes` | Band calibration slope coefficients ($\alpha$) | **REQUIRED** | Computed from empirical linear regression over stable ground targets (e.g. runways, calm reservoirs) using coregistered cloud-free S2/Planet pairs. |
| `spectral_matching.intercepts` | Band calibration intercept coefficients ($\beta$) | **REQUIRED** | Computed from empirical linear regression over stable ground targets using coregistered S2/Planet pairs. |
| `sensor_noise.band_sigmas` | Noise standard deviation ($\sigma_{\text{noise}}$) | **REQUIRED** | Computed standard deviation of reflectance values inside uniform, homogeneous regions in the target Sentinel-2 scene. |

---

## 10. How the Pipeline Will Be Used in Training

Once real PlanetScope high-resolution reference data is available, the degradation pipeline will be integrated into the training flow as follows:

1. **Dataset Generation**: 
   A custom PyTorch Dataset class (e.g., `SatelliteSRDataset`) will load the high-resolution PlanetScope patches (`HR Target`).
2. **On-the-Fly Degradation**:
   During the forward pass of data loading, the `PhysicalDegradationPipeline` will process the `HR Target` to generate a realistic `LR Input` (Sentinel-2-like observation).
3. **Super-Resolution Training**:
   The model (e.g. `ResidualCNN`) takes the generated `LR Input`, performs spatial upscaling, and is trained to reconstruct the original high-resolution `HR Target` using L1/SSIM loss:
   
```
   [PlanetScope HR Target]
              │
              ├──────(Physical Degradation)──────> [Sentinel-2-like LR Input]
              │                                                │
              │                                        (Model Forward Pass)
              │                                                │
              │                                                ▼
              ▼                                        [Model HR Prediction]
              ├────────────────────────────────────────────────┤
              └──────────────(Compute L1 + SSIM Loss)──────────┘
```

This ensures the network learns to invert the physical degradation model, ensuring optimal generalization when applied directly to real Sentinel-2 observations during inference.
