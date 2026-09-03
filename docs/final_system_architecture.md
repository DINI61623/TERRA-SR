# Final System Architecture & Research Prototype Design Document

**Project**: Physically Informed Multispectral Super-Resolution Framework for Sentinel-2 Imagery  
**Document Identifier**: `docs/final_system_architecture.md`  
**Current Phase**: Research Prototype Architecture Design  
**Status**: `PROPOSED_ARCHITECTURE`  

---

## Executive Summary

The objective of this project is to develop a **physically informed multispectral super-resolution framework** that enhances Sentinel-2 Level-2A imagery from its native 10m spatial resolution toward **<4m spatial resolution (~3m ground sampling distance)** while strictly preserving spectral, spatial, and geographic consistency and providing explicit uncertainty quantification.

Previous iterations established baseline models under a synthetic decimation regime (Experiment 2: ESPCN, Experiment 3: Residual CNN). The comprehensive failure analysis of Experiment 3 (documented in `docs/model_failure_analysis.md`) proved that while shallow residual networks achieve mathematical PSNR/SSIM gains over bilinear interpolation (+0.77 dB to +1.76 dB), they behave primarily as localized edge-sharpening filters. Under L2/MSE loss and naive synthetic downsampling, shallow networks smooth out high-frequency stochastic textures, fail to recover genuine sub-pixel spatial patterns, and suffer from significant domain gap when applied to real optical data.

This document establishes the end-to-end architectural redesign for the research prototype, incorporating **Residual Channel Attention Networks (RCAN)**, the **Experiment 4 Physical Degradation Framework**, a **composite spectral-spatial loss function**, **uncertainty quantification**, and a **four-level scientific validation protocol**.

---

## 1. Problem Definition: Limitations of Sentinel-2 10m Imagery

Sentinel-2 MultiSpectral Instrument (MSI) provides invaluable global Earth observation data with a 5-day revisit cycle. However, its highest spatial resolution is capped at **10 meters per pixel** for the four visible and broad near-infrared bands (B02, B03, B04, B08), while Red Edge, SWIR, and narrow NIR bands are captured at 20m, and atmospheric bands at 60m.

```
+-----------------------------------------------------------------------------+
| Sentinel-2 10m Instantaneous Field of View (IFOV)                           |
| (10m x 10m Ground Footprint = 100 m^2 per pixel)                            |
|                                                                             |
|  +-----------------------------------------------------------------------+  |
|  |  Sub-pixel Realities (Averaged into 1 Single DN Value):                |  |
|  |   - Tree crown (3m) + Asphalt road edge (2m) + Soil patch (5m)        |  |
|  |   - Small building boundary (6m) + Lawn grass (4m)                    |  |
|  |   - Narrow canal / drainage ditch (2m) + Vegetation buffer (8m)        |  |
|  +-----------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------+
```

### Key Technical Limitations of 10m Resolution:
1. **The Mixed-Pixel Problem**: A single 10m pixel covers $100\text{ m}^2$ on the ground. When multiple distinct land covers coexist within this area (e.g., small buildings, narrow roads, localized crop patches, water boundaries), the sensor integrates incoming radiance into a single blended spectral signature, obscuring sub-pixel boundaries.
2. **Loss of High-Frequency Spatial Geometry**: Linear structures narrower than 10m (rural paths, agricultural field ditches, small streams, urban infrastructure) are blurred or completely lost in the background noise.
3. **Inadequacy for Precision Monitoring**:
   - *Precision Agriculture*: Cannot resolve individual tree canopies, row spacing, or intra-field localized stress in smallholder parcels.
   - *Urban Mapping*: Cannot separate dense individual building footprints or distinguish narrow roads from adjacent roofs.
   - *Disaster & Damage Assessment*: Cannot delineate localized flood boundaries along narrow drainage channels or structural damage to small infrastructure.
4. **Failure of Naive Interpolation**: Bilinear and bicubic upsampling merely smooth pixel grids without adding information. Synthetic super-resolution models trained with naive downsampling learn to invert mathematical interpolation filters rather than recovering optical and physical high-frequency features.

---

## 2. Input Specifications: Sentinel-2 Level-2A (L2A)

The primary input to the enhancement system is calibrated Bottom-Of-Atmosphere (BOA) Surface Reflectance from the Copernicus Sentinel-2 MSI sensor.

| Attribute | Specification |
| :--- | :--- |
| **Product Level** | Sentinel-2 Level-2A (L2A) Bottom-Of-Atmosphere (BOA) Surface Reflectance |
| **Spectral Bands** | **B02** (Blue, ~490 nm, 66 nm bandwidth)<br>**B03** (Green, ~560 nm, 36 nm bandwidth)<br>**B04** (Red, ~665 nm, 31 nm bandwidth)<br>**B08** (Broad NIR, ~842 nm, 115 nm bandwidth) |
| **Native Spatial Resolution** | **10.0 meters** Ground Sampling Distance (GSD) |
| **Coordinate Reference System** | Projected UTM (e.g., `EPSG:32643` - UTM Zone 43N for Electronic City, Bengaluru AOI) |
| **Radiometric Format & Range** | 16-bit unsigned integer Digital Numbers (DN), normalized to physical surface reflectance $[0.0, 1.0]$ via $I = \text{DN} / 10000.0$ |
| **Data Integrity Constraint** | Preserves all geospatial geotransforms, tie-points, nodata masks, and radiometric scaling |

---

## 3. Target Specifications: Enhanced Multispectral Product

The target output is an enhanced 4-band multispectral data product with spatial resolution matching high-resolution reference standards.

| Attribute | Target Specification |
| :--- | :--- |
| **Target Spatial Resolution** | **$\approx 3.0\text{ m}$ / $<4.0\text{ m}$** Ground Sampling Distance (nominal $3\times$ or $4\times$ spatial enhancement factor) |
| **Enhanced Channels** | 4 Spectral Bands corresponding to B02 (Blue), B03 (Green), B04 (Red), and B08 (NIR) |
| **Output Grid Alignment** | Snapped to integer sub-pixel grid offsets relative to the low-resolution Sentinel-2 bounding box |
| **Radiometric Consistency** | Reflectance values remain strictly bounded in $[0.0, 1.0]$, preserving physical multi-band ratios and indices (NDVI, NDWI) |
| **Associated Uncertainty** | Per-pixel, per-band confidence / variance estimation layer delivered alongside the enhanced reflectance cube |

---

## 4. Baselines (Fixed & Preserved)

To maintain scientific integrity and rigorous benchmark continuity, existing baseline models and artifacts are strictly preserved and remain unchanged.

```
                      +-----------------------------+
                      |     BASELINE REPOSITORY     |
                      |     (Frozen / Unmodified)   |
                      +--------------+--------------+
                                     |
         +---------------------------+---------------------------+
         |                                                       |
+--------v-------------------+                         +---------v-------------------+
| Baseline 1: Bilinear       |                         | Baseline 2: ESPCN           |
| Classical Interpolation    |                         | (Experiment 2)              |
| Fixed analytical baseline  |                         | Checkpoint:                 |
| No learned parameters      |                         | `models/espcn_srm_...pth`   |
+----------------------------+                         +-----------------------------+
                                     |
                                     |
                       +-------------v---------------+
                       | Baseline 3: Residual CNN    |
                       | (Experiment 3)              |
                       | Checkpoint:                 |
                       | `models/residual_srm_...pth`|
                       +-----------------------------+
```

1. **Bilinear Interpolation**:
   - Classical, non-learned interpolation baseline.
   - Continuous spatial interpolation without edge awareness or spectral cross-channel reasoning.
2. **ESPCN (Efficient Sub-Pixel Convolutional Network - Experiment 2)**:
   - Architecture: 3-layer convolutional feature extractor with Sub-Pixel PixelShuffle upscaling.
   - Status: Complete baseline, trained under synthetic $2\times$ decimation.
3. **Residual CNN (Experiment 3)**:
   - Architecture: Deep residual network with 3 Residual Blocks and residual skip connection (251,753 parameters).
   - Status: Complete baseline, trained under synthetic $2\times$ decimation (`models/residual_srm_experiment3.pth`).
   - Characteristics: Sharpens high-contrast linear edges (+1.25 dB avg PSNR gain over Bilinear), but acts as an educated deblurring filter and over-smooths stochastic textures under MSE loss.

> [!IMPORTANT]
> All baseline code, scripts (`train_srm.py`, `train_srm_synthetic.py`, `train_srm_residual_experiment3.py`), and weight checkpoints (`models/*.pth`) are preserved without modification.

---

## 5. Proposed Final Model: Residual Channel Attention Network (RCAN)

> [!NOTE]
> **Status**: `PROPOSED` (Architecture candidate designed; not implemented or trained yet).

For the final high-resolution prototype, we propose the **Residual Channel Attention Network (RCAN)** as the core super-resolution architecture.

```
                  +-------------------------------------------------------------+
                  |                 PROPOSED RCAN ARCHITECTURE                  |
                  +-------------------------------------------------------------+

Low-Res Input (4-Band S2, 10m) 
     │
     ├─────────────────────────────────────────┐ (Long Skip Connection)
     ▼                                         │
[ Shallow Feature Extraction (Conv 3x3) ]     │
     │                                         │
     ▼                                         │
[ Residual in Residual (RIR) Backbone ]        │
     │                                         │
     │  ┌─ Residual Group 1 (RG_1) ────────┐   │
     │  │   RCAB_1 -> RCAB_2 -> ... RCAB_N │   │
     │  │   + Short Skip Connection        │   │
     │  └──────────────────────────────────┘   │
     │                   │                     │
     │                  ...                    │
     │                   │                     │
     │  ┌─ Residual Group M (RG_M) ────────┐   │
     │  │   RCAB_1 -> ... -> RCAB_N        │   │
     │  │   + Short Skip Connection        │   │
     │  └──────────────────────────────────┘   │
     │                   │                     │
     │  [ RIR Conv 3x3 ] ◄─────────────────────┘
     │                   │
     ├───────────────────┘
     ▼
[ Sub-Pixel Upsampler (PixelShuffle x3 or x4) ]
     │
     ▼
[ Reconstruction Head (Conv 3x3) ]
     │
     ▼
Enhanced 4-Band High-Resolution Output (<4m)
```

### Why RCAN is the Primary Candidate Architecture:
1. **Deep Residual in Residual (RIR) Framework**:
   - Standard very deep CNNs suffer from vanishing gradients and performance saturation. RCAN organizes residual blocks into hierarchical **Residual Groups (RGs)** with **Short Skip Connections** within each group and a **Long Skip Connection** spanning the entire network.
   - Low-frequency information (base reflectance and coarse geometry) bypasses the deep network via the long skip connection, allowing the deep feature layers to focus purely on high-frequency residual structural information.
2. **Channel Attention (CA) Mechanism**:
   - In multispectral remote sensing, different spectral bands have distinct spatial variances, signal-to-noise ratios, and physical correlations (e.g., Red and NIR have strong anti-correlations over vegetation).
   - Standard convolutions treat all feature channels with equal weight. The **Channel Attention Block (RCAB)** explicitly models inter-channel dependencies by pooling global spatial context and adaptively re-scaling channel feature maps:
     $$s = \sigma\left(W_2 \cdot \text{ReLU}(W_1 \cdot \text{GAP}(f_c))\right)$$
     $$f_{\text{attended}} = s \odot f_c$$
   - This enables the model to dynamically prioritize the most informative spectral-spatial feature combinations.
3. **High Capacity for Multispectral Feature Representation**:
   - Unlike the 3-block Residual CNN (251k parameters), RCAN provides the representational capacity (e.g., 5–15 million parameters across 10 RGs and 20 RCABs per group) required to capture complex cross-sensor mapping without overfitting when guided by physical degradation constraints.

---

## 6. Training Data & Reference Framework

```
+------------------------------------------------------------------------------------+
| Real Paired Data Framework                                                         |
+------------------------------------------------------------------------------------+
|  Low-Resolution Source:       Sentinel-2 L2A (10m) - B02, B03, B04, B08            |
|  High-Resolution Reference:   PlanetScope Ortho Scene Surface Reflectance (~3.0m)  |
|  Target Scene Identifier:     20260211_054815_64_254a (Pass: 11 Feb 2026)          |
|  Target AOI:                  Electronic City, Bengaluru (EPSG:32643)              |
+------------------------------------------------------------------------------------+
```

### Rigorous Scientific Caveats on PlanetScope Reference Data:
1. **PlanetScope is NOT Ground Truth**:
   - PlanetScope imagery has its own optical Point Spread Function (PSF), atmospheric correction residuals, and sensor noise characteristics.
   - Coregistration errors of 1–2 meters (sub-pixel at S2 scale, multi-pixel at Planet scale) frequently occur between orbital passes.
2. **Spectral Response Discrepancies**:
   - PlanetScope Dove Classic / SuperDove instruments have different Spectral Response Functions (SRF) compared to Sentinel-2 MSI.
   - Even when Planet "Harmonize" processing is applied, residual spectral drift of 2%–5% persists, especially in the NIR band.
3. **Non-Trivial Cross-Sensor Gap**:
   - Directly minimizing pixel differences against PlanetScope without physical degradation modeling forces the network to learn sensor calibration artifacts rather than super-resolution.

---

## 7. Physical Degradation Framework (Experiment 4 Integration)

**Current Status**: `DEGRADATION_PARTIALLY_READY` (Implementation in `src/super_resolution/degradation.py` complete; awaiting real paired scene data for scene-specific regressions).

Rather than using naive bilinear or bicubic downsampling, the research prototype uses a physically informed forward degradation model simulating the optical path and sensor physics:

```
PlanetScope High-Resolution Reference (HR, ~3m)
               │
               ▼
   [ 1. Radiometric Normalization ]   (Scale DN by 10000.0 to physical BOA reflectance [0, 1])
               │
               ▼
        [ 2. Spatial PSF ]            (Band-specific 2D Gaussian optical blur based on MSI MTF)
               │
               ▼
 [ 3. Spectral Response Matching ]    (Linear cross-sensor gain/offset calibration)
               │
               ▼
     [ 4. Sensor Noise Model ]        (Additive noise simulating MSI NEΔR)
               │
               ▼
  [ 5. Spatial Sampling / Decimation ] (Area-weighted IFOV pixel integration)
               │
               ▼
Sentinel-2 Equivalent Observation (LR, 10m)
```

### Parameter Classification & Audit Summary (from Experiment 4C):

| Pipeline Stage | Parameter | Class | Value / Status | Confidence |
| :--- | :--- | :--- | :--- | :--- |
| **Radiometric Normalization** | `input_scale` | `SUPPORTED_DIRECTLY` | `10000.0` | **High** |
| **Spatial PSF** | Band Sigmas ($\sigma_c$) | `DERIVED_FROM_SUPPORTED_VALUE` | B02: 0.5225, B03: 0.5115,<br>B04: 0.5225, B08: 0.5300 px | **Medium** |
| **Spatial Decimation** | `upscale_factor`, `mode` | `SUPPORTED_DIRECTLY` | $K=3$ or $K=4$, mode=`area` | **High** |
| **Spectral Matching** | Slopes ($\alpha_c$), Intercepts ($\beta_c$) | `SCENE_SPECIFIC` | `null` (Awaiting empirical regression on Scene `20260211_054815_64_254a`) | **Low / Blocked** |
| **Sensor Noise** | Noise Sigmas ($\sigma_{\text{noise}, c}$) | `SCENE_SPECIFIC` | `null` (Awaiting dynamic homogeneous zone estimation) | **Low / Blocked** |

---

## 8. Proposed Composite Loss Function

> [!NOTE]
> **Status**: `PROPOSED` (Mathematical formulation designed; not implemented yet).

The failure analysis of Experiment 3 confirmed that Mean Squared Error (MSE / L2) loss forces the network to output the conditional mean of all possible sub-pixel configurations, resulting in severe over-smoothing of fine textures. 

To overcome this, we propose a multi-component composite loss function:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{rec}} \mathcal{L}_{\text{rec}} + \lambda_{\text{struct}} \mathcal{L}_{\text{struct}} + \lambda_{\text{spec}} \mathcal{L}_{\text{spec}} + \lambda_{\text{edge}} \mathcal{L}_{\text{edge}}$$

```
                               +---------------------------------------+
                               |        COMPOSITE LOSS FUNCTION        |
                               +-------------------+-------------------+
                                                   |
         +--------------------+--------------------+--------------------+--------------------+
         |                    |                                         |                    |
+--------v-----------+ +------v-------------+                  +--------v-----------+ +------v-----------+
| Reconstruction     | | Structural         |                  | Spectral           | | Spatial / Edge   |
| Loss (L1 / Huber)  | | Loss (MS-SSIM)     |                  | Loss (SAM + NDVI)  | | Gradient Loss    |
| Robust pixel-level | | Preserves contrast |                  | Preserves physical | | Sharpens parcel  |
| radiometric error  | | & multiscale edges |                  | spectral angles    | | & urban edges    |
+--------------------+ +--------------------+                  +--------------------+ +--------------------+
```

### Component Formulations:
1. **Reconstruction Loss ($\mathcal{L}_{\text{rec}}$)**:
   - Formulated as Charbonnier / Smooth L1 loss:
     $$\mathcal{L}_{\text{rec}} = \frac{1}{C \cdot H \cdot W} \sum_{c, x, y} \sqrt{(I_{\text{pred}}(x, y, c) - I_{\text{ref}}(x, y, c))^2 + \epsilon^2}$$
   - Penalizes absolute radiometric errors without overly penalizing high-frequency edge transitions or promoting blurry conditional means.
2. **Structural Loss ($\mathcal{L}_{\text{struct}}$)**:
   - Formulated using Multi-Scale Structural Similarity (MS-SSIM):
     $$\mathcal{L}_{\text{struct}} = 1 - \text{MS-SSIM}(I_{\text{pred}}, I_{\text{ref}})$$
   - Evaluates luminance, contrast, and structural correlation across multiple dyadic scales, preserving perceptual structures across both large features and fine boundaries.
3. **Spectral Consistency Loss ($\mathcal{L}_{\text{spec}}$)**:
   - Combines Spectral Angle Mapper (SAM) loss and index-consistency penalty:
     $$\mathcal{L}_{\text{SAM}} = \frac{1}{H \cdot W} \sum_{x, y} \arccos\left(\frac{I_{\text{pred}}(x,y) \cdot I_{\text{ref}}(x,y)}{\|I_{\text{pred}}(x,y)\|_2 \|I_{\text{ref}}(x,y)\|_2 + \epsilon}\right)$$
     $$\mathcal{L}_{\text{NDVI}} = \left\| \frac{I_{\text{pred}}^{\text{NIR}} - I_{\text{pred}}^{\text{Red}}}{I_{\text{pred}}^{\text{NIR}} + I_{\text{pred}}^{\text{Red}} + \epsilon} - \frac{I_{\text{ref}}^{\text{NIR}} - I_{\text{ref}}^{\text{Red}}}{I_{\text{ref}}^{\text{NIR}} + I_{\text{ref}}^{\text{Red}} + \epsilon} \right\|_1$$
   - Enforces that radiometric vectors maintain their true physical color and vegetation indices regardless of spatial sharpening.
4. **Spatial / Edge Gradient Loss ($\mathcal{L}_{\text{edge}}$)**:
   - Computes differences in spatial gradients using Sobel / Laplacian operators:
     $$\mathcal{L}_{\text{edge}} = \|\nabla_x I_{\text{pred}} - \nabla_x I_{\text{ref}}\|_1 + \|\nabla_y I_{\text{pred}} - \nabla_y I_{\text{ref}}\|_1$$
   - Prevents blurred transitions along parcel boundaries, roads, and building perimeters.

---

## 9. Uncertainty & Confidence Quantification

Because sub-pixel super-resolution is an ill-posed mathematical inverse problem, the model must not present inferred high-resolution details with unwarranted certainty. The final system is designed to provide an **explicit per-pixel uncertainty and confidence estimate**.

```
Input S2 (10m) ───► [ PI-RCAN Backbone ] ──┬──► Enhanced Reflectance Cube (4 Bands, ~3m)
                                           │
                                           └──► Uncertainty Map σ^2(x, y, c) & Confidence Mask
```

### Proposed Uncertainty Mechanisms:
1. **Heteroscedastic Aleatoric Uncertainty Head**:
   - The final layer branches into two heads: a **Mean Prediction Head** $\mu(x, y, c)$ (reflectance) and a **Variance Prediction Head** $\sigma^2(x, y, c)$ (observation uncertainty).
   - Trained via Gaussian Negative Log-Likelihood (NLL):
     $$\mathcal{L}_{\text{NLL}} = \frac{1}{2} \sum_{x, y, c} \left( \frac{(I_{\text{ref}}(x,y,c) - \mu(x,y,c))^2}{\sigma^2(x,y,c)} + \ln \sigma^2(x,y,c) \right)$$
   - Automatically assigns higher uncertainty to ambiguous sub-pixel boundaries, shadows, and stochastic textures.
2. **Epistemic Uncertainty via Monte Carlo Dropout / Ensemble**:
   - During inference, multiple stochastic forward passes with active dropout ($T=10$) quantify model epistemic uncertainty via the spatial variance of predictions across passes.
3. **Composite Confidence Index ($C_{\text{spatial}}$)**:
   - Delivers a normalized $[0.0, 1.0]$ confidence raster indicating where fine-scale detail is structurally well-supported by multi-band gradients vs. where it is inferred with lower certainty.

---

## 10. Four-Level Scientific Validation Framework

To ensure that performance is evaluated with complete scientific rigor, the prototype establishes a four-tier validation hierarchy:

```
+-----------------------------------------------------------------------------+
|                      FOUR-LEVEL VALIDATION FRAMEWORK                        |
+-----------------------------------------------------------------------------+
|  Level 1: Image Quality Metrics                                             |
|  - PSNR (Peak Signal-to-Noise Ratio)                                        |
|  - SSIM (Structural Similarity Index)                                       |
|  - MAE / RMSE (Radiometric Mean Absolute / Root Mean Square Error)          |
+-----------------------------------------------------------------------------+
|  Level 2: Spatial Fidelity Metrics                                          |
|  - Edge Preservation Index (EPI)                                            |
|  - High-Frequency Energy Ratio (Fourier / Wavelet PSD)                       |
|  - Local Contrast & Gradient Profile Preservation                           |
+-----------------------------------------------------------------------------+
|  Level 3: Spectral Fidelity Metrics                                         |
|  - Spectral Angle Mapper (SAM)                                               |
|  - ERGAS (Relative Dimensionless Global Error in Synthesis)                 |
|  - Universal Image Quality Index (UIQI / Q-Index)                           |
|  - Delta-NDVI and Delta-NDWI Cross-Band Consistency                         |
+-----------------------------------------------------------------------------+
|  Level 4: Downstream Analytical Utility Metrics                             |
|  - Land-Cover Classification Accuracy (Overall Accuracy, Kappa, F1-Score)   |
|  - Field Parcel Boundary Segmentation IoU                                   |
|  - Small Structure / Building Footprint Detection F1-Score                  |
+-----------------------------------------------------------------------------+
```

---

## 11. Downstream Applications: General-Purpose Earth Observation Layer

The super-resolution enhancer is designed as a **foundational, general-purpose Earth observation enhancement layer**. 

> [!NOTE]
> Specific downstream pipelines (such as dedicated oil-spill detection) are **NOT** implemented in this phase. The enhancer produces standard georeferenced multispectral GeoTIFFs that serve as high-resolution inputs for subsequent domain applications:

```
                               +-------------------------------------+
                               |   SATELLITE ENHANCEMENT ENGINE      |
                               |   (Sentinel-2 10m -> Enhanced <4m)  |
                               +------------------+------------------+
                                                  |
         +--------------------+-------------------+--------------------+--------------------+
         |                    |                   |                    |                    |
+--------v-----------+ +------v-----------+ +-----v------------+ +-----v------------+ +-----v------------+
| Precision          | | Land-Cover &     | | Disaster &       | | Urban &          | | Environmental &  |
| Agriculture        | | Parcel Mapping   | | Flood Monitoring | | Infrastructure   | | Marine Monitoring|
| - Intra-field crop | | - Cadastral      | | - Flood extent   | | - Informal road  | | - Water buffer   |
|   health & rows    |   boundaries       | - Landslide scars  |   mapping          |   zones            |
| - Canopy nitrogen  | - Smallholder plot | - Building damage  | - Building growth  | - Future coastal / |
|   assessment       |   classification   |   assessment       |   tracking         |   spill monitoring |
+--------------------+ +------------------+ +------------------+ +------------------+ +--------------------+
```

---

## 12. Experiment Roadmap: Current Status & Progression

| Experiment ID | Focus Area | Key Objective | Status | Blockers / Pre-conditions |
| :--- | :--- | :--- | :--- | :--- |
| **Experiment 2** | ESPCN Baseline | Fast sub-pixel convolution baseline on synthetic $2\times$ decimation | **COMPLETE** | None (Preserved) |
| **Experiment 3** | Residual CNN Baseline | Deep residual learning baseline on synthetic $2\times$ decimation | **COMPLETE** | None (Preserved) |
| **Experiment 4A** | Model Failure Analysis | In-depth spatial, frequency, and radiometric audit of Exp 3 | **COMPLETE** | None (Documented in `docs/model_failure_analysis.md`) |
| **Experiment 4B** | Physical Degradation | Implementation of modular physical PSF, noise, and decimation | **COMPLETE (Code)** /<br>`DEGRADATION_PARTIALLY_READY` | Awaiting real paired data for empirical regression |
| **Experiment 4C** | Sensor Parameter Audit | Rigorous parameter classification and MTF/PSF derivation | **COMPLETE** | Documented in `docs/experiment4_sensor_parameters.md` |
| **Experiment 5** | Real Data Paired Training + Proposed RCAN | End-to-end training of proposed RCAN with physical degradation and real PlanetScope reference | **PROPOSED / BLOCKED** | **Blocked on acquisition of authorized PlanetScope scene `20260211_054815_64_254a`** |

---

## 13. Important Scientific Limitation & Epistemic Boundaries

```
+------------------------------------------------------------------------------------+
|                         MANDATORY SCIENTIFIC STATEMENT                             |
|                                                                                    |
|  The <4m output produced by this framework is an INFERRED / SUPER-RESOLVED         |
|  representation, NOT a direct physical observation at <4m spatial resolution.      |
|                                                                                    |
|  The system does NOT, and CANNOT, claim to create physical information that        |
|  the Sentinel-2 MultiSpectral Instrument did not physically capture.               |
+------------------------------------------------------------------------------------+
```

### Epistemic Principles:
1. **Mathematical Reality of Inverse Problems**: Multiple distinct high-resolution surface configurations can map to the exact same 10m Sentinel-2 low-resolution observation after PSF convolution and pixel integration. The enhanced product represents the most statistically probable and physically consistent high-resolution estimate under the learned prior.
2. **Prohibition of Hallucination Claims**: Inferred details (such as apparent rooftop patterns or crop texture) are conditioned estimates, not verified ground truth.
3. **Obligation of Transparency**: All derivative maps, visual assets, and application pipelines utilizing this framework must explicitly label the data as super-resolved and display the accompanying uncertainty layer when making analytical decisions.

---

## 14. Recommended Final Prototype Architecture & Justification

### Formal Recommendation:
The recommended final prototype architecture is the **Physically Informed Residual Channel Attention Network (PI-RCAN)** trained under the **Experiment 4 Physical Degradation Model** with the **Composite Spectral-Spatial Loss Function** and **Dual-Head Uncertainty Estimation**.

```
+--------------------------------------------------------------------------------------------+
|                          RECOMMENDED RESEARCH PROTOTYPE STACK                              |
+--------------------------------------------------------------------------------------------+
|  Model Backbone:       Residual Channel Attention Network (RCAN)                           |
|  Degradation Model:    Modular Physical Degradation (MTF PSF + Decimation + Noise + SRF)   |
|  Training Objective:   Composite Loss (Charbonnier L1 + MS-SSIM + SAM/NDVI + Edge Grad)    |
|  Uncertainty Engine:   Heteroscedastic Aleatoric Variance Head + Epistemic MC Dropout      |
|  Reference Data:       Paired Sentinel-2 L2A & PlanetScope SR (Scene 20260211_054815_64)   |
|  Validation Hierarchy: Four-Level Framework (Image Quality, Spatial, Spectral, Downstream)  |
+--------------------------------------------------------------------------------------------+
```

### Why PI-RCAN is Preferable to Optimizing the Current Residual CNN:

| Dimension | Current Residual CNN (Exp 3) | Proposed PI-RCAN Prototype | Scientific Advantage |
| :--- | :--- | :--- | :--- |
| **Model Depth & Capacity** | 3 Shallow Residual Blocks (~251k parameters) | Deep Residual-in-Residual (10 RGs, ~5–12M parameters) | Able to learn complex hierarchical spatial-spectral features without gradient degradation. |
| **Spectral Channel Modeling** | Standard uniform convolutions across channels | Channel Attention Mechanism (RCAB) | Dynamically weights and exploits cross-band physical correlations (e.g., Red-NIR vegetation dynamics). |
| **Degradation Assumption** | Naive bilinear/bicubic synthetic decimation | Modular Physical Forward Model (PSF, IFOV, Noise) | Inverts the true physical sensor degradation, eliminating the synthetic-to-real domain gap. |
| **Loss Function** | Mean Squared Error (MSE / L2) | Composite Loss ($\text{L1} + \text{MS-SSIM} + \text{SAM} + \text{Edge}$) | Eliminates conditional-mean over-smoothing; enforces sharp edges while strictly preserving physical spectral angles. |
| **Uncertainty Quantification** | None (Deterministic point prediction) | Per-pixel Variance & Epistemic Uncertainty Maps | Empirically quantifies confidence, preventing misleading hallucination in downstream analytics. |

Continuing to optimize the 3-block Residual CNN under synthetic MSE conditions would yield diminishing returns, as a shallow network cannot overcome the structural limitations of uniform channel weighting, mathematical over-smoothing, and synthetic degradation mismatch. The proposed **PI-RCAN architecture** directly addresses each of these foundational bottlenecks.
