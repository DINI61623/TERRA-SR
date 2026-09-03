# Experiment 4c: Authoritative Sensor Parameter Research Report (Revised)

This report presents a scientifically rigorous audit and classification of the sensor parameters used in our physical degradation pipeline. It documents their sources, conversion formulas, confidence levels, and appropriateness for training.

---

## 1. Parameter Classification System

To ensure transparency and prevent the promotion of nominal or scene-specific assumptions to "authoritative" specs, all parameters are classified into one of the following five tiers:

1.  **`SUPPORTED_DIRECTLY`**: Directly documented, invariant sensor specifications or format scales (e.g., product scale factors).
2.  **`DERIVED_FROM_SUPPORTED_VALUE`**: Derived mathematically from documented sensor parameters using verified physical equations (e.g., Gaussian sigma derived from MTF specs).
3.  **`NOMINAL_APPROXIMATION`**: Scientifically reasonable baseline approximations documented in literature, used when dynamic values are unavailable.
4.  **`SCENE_SPECIFIC`**: Parameters that vary by geographic pass, acquisition date, atmospheric state, or sensor calibration, and **must** be empirically estimated from the coregistered scene.
5.  **`UNKNOWN`**: Parameters with no scientific support or authoritative documentation.

---

## 2. Parameter Matrix

| Parameter Name | Spectral Band | Class | Nominal Value | Units | Source / Reference | Confidence | Safe for Training? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`radiometric_normalization.input_scale`** | All (B02-B08) | `SUPPORTED_DIRECTLY` | `10000.0` | Dimensionless scale | Copernicus S2 L2A & Planet Products specs | **High** | **Yes** |
| **`spatial_decimation.upscale_factor`** | All (B02-B08) | `SUPPORTED_DIRECTLY` | `2` (or `4`) | Dimensionless ratio | Target grid resolution transform design | **High** | **Yes** |
| **`psf.band_sigmas`** (S2 MSI Blur) | Blue (B02) | `DERIVED_FROM_SUPPORTED_VALUE` | `0.5225` (LR)<br>`1.045` (HR K=2) | Pixels | ESA SentiWiki & S2 MSI Cal/Val reports | **Medium** | **Yes** (With isotropic/isotropic-Gaussian approximations) |
| | Green (B03) | `DERIVED_FROM_SUPPORTED_VALUE` | `0.5115` (LR)<br>`1.023` (HR K=2) | Pixels | ESA SentiWiki & S2 MSI Cal/Val reports | **Medium** | **Yes** |
| | Red (B04) | `DERIVED_FROM_SUPPORTED_VALUE` | `0.5225` (LR)<br>`1.045` (HR K=2) | Pixels | ESA SentiWiki & S2 MSI Cal/Val reports | **Medium** | **Yes** |
| | NIR (B08) | `DERIVED_FROM_SUPPORTED_VALUE` | `0.5300` (LR)<br>`1.060` (HR K=2) | Pixels | ESA SentiWiki & S2 MSI Cal/Val reports | **Medium** | **Yes** |
| **`spectral_matching.slopes`** | All (B02-B08) | `SCENE_SPECIFIC` | `null` (requires regression) | Dimensionless gain | Roy et al. (2021); Planet Harmonization | **Low** | **No** (Must be empirically calculated per scene or disabled) |
| **`spectral_matching.intercepts`** | All (B02-B08) | `SCENE_SPECIFIC` | `null` (requires regression) | Reflectance | Roy et al. (2021); Planet Harmonization | **Low** | **No** |
| **`sensor_noise.band_sigmas`** | All (B02-B08) | `SCENE_SPECIFIC` | `null` (requires scene stats) | Reflectance | ESA S2 Performance specs ($NE\Delta R < 0.002$) | **Low** | **No** (Must be dynamically measured in uniform zones or disabled) |

---

## 3. Spatial Blurring (PSF) Analysis

### Distinguishing MTF, FWHM, and Gaussian Sigma
*   **Modulation Transfer Function (MTF)**: A frequency-domain representation of how the optical system modulates image contrast at different spatial frequencies. MTF at the Nyquist frequency ($f_N = 0.5$ cycles/pixel) describes the instrument's boundary limit sharpness.
*   **Full Width at Half Maximum (FWHM)**: A spatial-domain representation of the Point Spread Function (PSF) representing the width of the blur kernel at half its peak amplitude.
*   **Gaussian Sigma ($\sigma$)**: The standard deviation of the Gaussian spatial filter used to approximate the optical/atmospheric blur in the spatial domain.

### Mathematical Conversions
For a bi-dimensional Gaussian PSF, the spatial standard deviation $\sigma$ is converted from the FWHM using:

$$\sigma = \frac{\text{FWHM}}{2\sqrt{2\ln 2}} \approx \frac{\text{FWHM}}{2.35482}$$

Alternatively, the Gaussian standard deviation can be derived from the MTF at Nyquist frequency ($f_N = 0.5$) using:

$$\text{MTF}(f_N) = \exp\left(-2 \pi^2 \sigma^2 f_N^2\right) = \exp\left(-\frac{\pi^2 \sigma^2}{2}\right) \implies \sigma = \frac{\sqrt{-2 \ln(\text{MTF})}}{\pi}$$

Using ESA's officially monitored nominal MTF values at Nyquist frequency for Sentinel-2A MSI, we derive:
*   **Blue (B02)**: ALT MTF = 0.27, ACT MTF = 0.25. Averaging $\sigma$: $\sigma_{\text{LR}} \approx 0.5225$ pixels.
*   **Green (B03)**: ALT MTF = 0.28, ACT MTF = 0.27. Averaging $\sigma$: $\sigma_{\text{LR}} \approx 0.5115$ pixels.
*   **Red (B04)**: ALT MTF = 0.27, ACT MTF = 0.25. Averaging $\sigma$: $\sigma_{\text{LR}} \approx 0.5225$ pixels.
*   **NIR (B08)**: ALT MTF = 0.26, ACT MTF = 0.24. Averaging $\sigma$: $\sigma_{\text{LR}} \approx 0.5300$ pixels.

### Approximations Involved
1.  **Isotropic Gaussian PSF**: Real satellite PSFs are anisotropic (different ALT/ACT profiles). We average ALT and ACT values to create a symmetric, isotropic 2D Gaussian convolution kernel.
2.  **Instrument Averaging**: S2A and S2B sensors have minor manufacturing variances. We use nominal S2A MTF figures as a representative approximation for both units.

---

## 4. PlanetScope Spectral Response & Harmonization

### SuperDove Specifications vs. Ordered Product
*   **SuperDove Sensor (PSB.SD)**: Features 8 spectral bands. Its Spectral Response Functions (SRF) differ considerably from Dove Classic (4 bands) and Sentinel-2.
*   **Planet "Harmonize" Processing**: Planet provides an optional processing tool that applies a linear regression model to translate Planet surface reflectance products to Sentinel-2 MSI spectral equivalents.
*   **Limitations of Harmonization**: Equating harmonization with a perfect identity transform ($\alpha=1, \beta=0$) is an **invalid assumption**. Planet's harmonization regressions are calculated globally/regionally; residual band mismatches (spectral drift) of 2%–5% regularly remain, especially in the NIR band.
*   **Conclusion**: To ensure radiometric preservation, actual cross-sensor regression coefficients ($\alpha, \beta$) **must be dynamically estimated** using linear regressions over overlapping stable target pixels from the specific coregistered scene bounds.

---

## 5. Noise Modeling: Instrument specifications vs. Gaussian assumptions

*   **Instrument specs**:
    *   PlanetScope Dove Classic/SuperDove instruments specify a Signal-to-Noise Ratio (SNR) of $>100$ to $150$ under specific reference illuminations.
    *   Sentinel-2 MSI specifies a Noise Equivalent Delta Reflectance ($NE\Delta R < 0.002$).
*   **Gaussian Noise Assumption**:
    Adding additive white Gaussian noise (AWGN) to simulate sensor noise is a standard mathematical simplification. Converting a documented SNR (e.g., $150$) directly to a Gaussian standard deviation $\sigma_{\text{noise}} = 1/\text{SNR} = 0.0067$ assumes:
    1.  The noise is independent of the signal amplitude (which is false; photon shot noise follows Poisson distributions).
    2.  The noise is uniform across the entire dynamic range.
*   **Conclusion**:
    A fixed nominal noise value is an approximation. True scene-specific noise statistics must be derived by calculating the standard deviation of reflectance values within large, homogeneous land-cover zones (such as deep water bodies or concrete runways) in the target Sentinel-2 scene.
