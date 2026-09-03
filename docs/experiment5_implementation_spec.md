# Experiment 5 Implementation Specification: Physically Informed RCAN (PI-RCAN) on Real Paired Reference Imagery

**Project**: Sentinel-2 Multispectral Super-Resolution Mapping (SRM) (SIH26142)  
**Document Identifier**: `docs/experiment5_implementation_spec.md`  
**Execution Stage**: Experiment 5 Implementation Specification (Design Only)  
**Status**: `SPECIFICATION_LOCKED` (Awaiting Real Data Delivery)  

---

## Executive Overview

This specification establishes the comprehensive technical, algorithmic, and procedural blueprint for **Experiment 5: Real-Data Training and Validation of the Physically Informed Residual Channel Attention Network (PI-RCAN)**. 

Experiment 5 transitions our super-resolution framework from synthetic decimation baselines (Experiment 2: ESPCN, Experiment 3: Residual CNN) to real-world optical reference pairing, integrating:
1. **Real Paired Satellite Data**: Coincident Sentinel-2 L2A (10m) and PlanetScope Ortho Scene Surface Reflectance (~3m).
2. **Physical Degradation Modeling**: Point Spread Function (PSF), sensor noise, spectral matching, and area-weighted decimation.
3. **Deep Residual Channel Attention (RCAN)**: Hierarchical Residual-in-Residual (RIR) backbone with Channel Attention Blocks (RCAB).
4. **Composite Spectral-Spatial Loss**: Charbonnier L1, MS-SSIM, Spectral Angle Mapper (SAM), NDVI consistency, and edge gradients.
5. **Dual-Head Uncertainty Estimation**: Heteroscedastic aleatoric variance prediction and epistemic confidence mapping.

> [!IMPORTANT]
> **STRICT CONSTRAINT ENFORCEMENT**:
> - No RCAN training or model code will be executed in this phase.
> - No synthetic or mock PlanetScope rasters will be generated.
> - No existing baseline scripts or checkpoints (`models/espcn_srm_synthetic.pth`, `models/residual_srm_experiment3.pth`) will be modified or overwritten.
> - All future Experiment 5 checkpoints must reside exclusively in `models/experiment4_real/`.

---

## 1. Input Data Specifications

The Experiment 5 pipeline is architected around the paired acquisition of cloud-free Sentinel-2 L2A observations and high-resolution PlanetScope reference imagery over the target Area of Interest (AOI).

```
+---------------------------------------------------------------------------------------------------+
|                                 EXPERIMENT 5 DATA SPECIFICATIONS                                  |
+------------------------------------+--------------------------------------------------------------+
| Parameter                          | Specification                                                |
+------------------------------------+--------------------------------------------------------------+
| Target AOI                         | Electronic City, Bengaluru, India                            |
| AOI Center Coordinate              | (12.85° N, 77.685° E)                                        |
| Coordinate Reference System (CRS)  | EPSG:32643 (WGS 84 / UTM Zone 43N)                          |
| Target Date                        | 11 February 2026 (Coincident Overpass)                       |
+------------------------------------+--------------------------------------------------------------+
| Low-Resolution (LR) Sensor         | Copernicus Sentinel-2 MSI (Level-2A BOA Surface Reflectance) |
| LR Spatial Resolution              | 10.0 meters Ground Sampling Distance (GSD)                  |
| LR Spectral Bands                  | B02 (Blue, 490nm), B03 (Green, 560nm),                       |
|                                    | B04 (Red, 665nm), B08 (Broad NIR, 842nm)                     |
| LR Radiometric Format              | 16-bit uint normalized to Float32 [0.0, 1.0] (DN / 10000.0)  |
+------------------------------------+--------------------------------------------------------------+
| High-Resolution (HR) Reference     | PlanetScope Ortho Scene (Surface Reflectance / SR Asset)     |
| PlanetScope Target Scene ID        | 20260211_054815_64_254a                                      |
| HR Spatial Resolution              | ~3.0 meters orthorectified GSD                               |
| HR Spectral Bands                  | 4-Band Multispectral (Blue, Green, Red, NIR)                 |
| HR Radiometric Format              | 16-bit uint normalized to Float32 [0.0, 1.0] (DN / 10000.0)  |
+------------------------------------+--------------------------------------------------------------+
```

### Spectral Band Correspondence Mapping:
The 4-channel tensor layout must strictly follow the physical wavelength order across both sensors:

| Channel Index | Spectral Band | Sentinel-2 L2A Band | PlanetScope 4-Band SR | Wavelength Interval |
| :---: | :---: | :---: | :---: | :---: |
| `Channel 0` | **Blue** | B02 (490 nm) | Band 1 (Classic/Dove-R) or Band 2 (SuperDove) | 450 – 515 nm |
| `Channel 1` | **Green** | B03 (560 nm) | Band 2 (Classic/Dove-R) or Band 3 (SuperDove) | 510 – 590 nm |
| `Channel 2` | **Red** | B04 (665 nm) | Band 3 (Classic/Dove-R) or Band 4 (SuperDove) | 600 – 690 nm |
| `Channel 3` | **NIR** | B08 (842 nm) | Band 4 (Classic/Dove-R) or Band 8 (SuperDove) | 760 – 890 nm |

---

## 2. Mandatory Data Quality Control (QC) Gates

Before any patch extraction or neural network forward pass, the input pair must pass a rigorous, automated **10-Gate Quality Control Audit**. 

```
                                  [ Raw S2 & PlanetScope Files ]
                                                │
                                                ▼
                                    +───────────────────────+
                                    |    DATA QC ENGINE     |
                                    +───────────┬───────────+
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
       [ ANY GATE FAILS ]                                            [ ALL GATES PASS ]
                 │                                                             │
                 ▼                                                             ▼
     ╔═══════════════════════╗                                     ╔═══════════════════════╗
     ║   TRAINING BLOCKED    ║                                     ║   PROCEED TO SPLIT    ║
     ║  (Raise QCException)  ║                                     ║      & EXTRACTION     ║
     ╚═══════════════════════╝                                     ╚═══════════════════════╝
```

### The 10 Mandatory Validation Gates:

| Gate # | QC Check | Validation Requirement | Rejection Threshold (FAIL) |
| :---: | :--- | :--- | :--- |
| **Gate 1** | **CRS Alignment** | Both rasters must resolve to `EPSG:32643`. Reprojection applied if reference is delivered in `EPSG:4326`. | Unresolvable CRS or datum mismatch. |
| **Gate 2** | **Geotransform Alignment** | HR pixel grid origin must snap exactly to discrete sub-pixel multiples of the S2 grid origin. | Origin drift $> 0.1\times$ HR pixel size. |
| **Gate 3** | **Spatial Overlap** | Geographic intersection of S2 and PlanetScope must completely enclose the Electronic City AOI. | Overlap area $< 95\%$ of target AOI. |
| **Gate 4** | **Resolution Hierarchy** | Resolution ratio $R = \text{GSD}_{\text{LR}} / \text{GSD}_{\text{HR}}$ must be an integer scaling ratio ($K=3$ or $K=4$). | Non-integer scaling ratio or HR coarser than LR. |
| **Gate 5** | **Band Correspondence** | File must contain 4 verified multispectral channels (RGB + NIR). Single-band pan or 3-band visual RGB rejected. | Missing NIR band or channel count $\ne 4$. |
| **Gate 6** | **NoData / Cloud Masking** | Total invalid/NoData pixel fraction across AOI must be $< 1.0\%$. | NoData fraction $> 10.0\%$ (or 1–10% without filtering). |
| **Gate 7** | **Reflectance Bounds** | Surface reflectance must reside in $[0.0, 1.0]$. Values $> 1.2$ or $< -0.05$ indicate corrupted scaling. | $> 0.1\%$ pixels outside valid physical bounds. |
| **Gate 8** | **Temporal Difference** | Acquisition time offset $|\Delta t| \le 5$ days. For Scene `20260211_054815_64_254a`, $\Delta t = 0$ days. | $|\Delta t| > 30$ days. |
| **Gate 9** | **Coregistration Audit** | Sub-pixel phase cross-correlation over high-contrast features (runways, road intersections) must show offset $\le 1.5$ px. | Sub-pixel shift $> 2.0$ HR pixels after alignment. |
| **Gate 10** | **Spatial Leakage Audit** | Geographic train/val/test split boundaries must maintain a minimum $200\text{ m}$ exclusion buffer. | Overlapping tiles between train and test splits. |

---

## 3. Paired Dataset Generation & Spatial Partitioning

To avoid spatial auto-correlation leakage—where random patch shuffling places adjacent pixels from the same building or field in both training and test sets—the scene is divided into **strictly isolated contiguous spatial blocks**.

```
+─────────────────────────────────────────────────────────────────────────────+
|               GEOGRAPHIC SCENE PARTITIONING (EPSG:32643)                    |
+─────────────────────────────────────────────────────────────────────────────+
|                                                                             |
|   +---------------------------------------------------------------------+   |
|   |                     TRAINING BLOCK (70% Area)                       |   |
|   |  - Sliced into 32x32 LR / 96x96 HR patches                          |   |
|   |  - Rotational and flip data augmentations                           |   |
|   +---------------------------------------------------------------------+   |
|                                                                             |
|   =================== SPATIAL EXCLUSION BUFFER (200m) ===================   |
|                                                                             |
|   +----------------------------------+  +-------------------------------+   |
|   |    VALIDATION BLOCK (15% Area)   |  |      TEST BLOCK (15% Area)    |   |
|   |    - Hyperparameter selection    |  |      - Final benchmark report |   |
|   |    - Early stopping trigger      |  |      - Multi-model evaluation |   |
|   +----------------------------------+  +-------------------------------+   |
|                                                                             |
+─────────────────────────────────────────────────────────────────────────────+
```

### Extraction Specifications:
1. **Patch Dimensions**:
   - **LR Input Patch**: $32 \times 32$ pixels at 10m ($320\text{ m} \times 320\text{ m}$ ground footprint).
   - **HR Target Patch ($K=3$)**: $96 \times 96$ pixels at ~3.33m ($320\text{ m} \times 320\text{ m}$ ground footprint).
   - **HR Target Patch ($K=4$)**: $128 \times 128$ pixels at 2.5m ($320\text{ m} \times 320\text{ m}$ ground footprint).
2. **Patch Stride & Overlap**:
   - Training Set: Stride of 16 LR pixels ($50\%$ spatial overlap for sample density).
   - Validation & Test Sets: Non-overlapping stride of 32 LR pixels to ensure independent statistical evaluation.
3. **NoData Filtering**:
   - Any extracted patch containing $> 1.0\%$ invalid, clipped, or shadowed pixels is discarded during extraction.

---

## 4. Physical Degradation Forward Pipeline

Experiment 5 utilizes the modular physical degradation engine implemented in `src/super_resolution/degradation.py`. During training, the high-resolution PlanetScope reference patches are physically degraded on-the-fly to generate synthetic Sentinel-2-like observations:

```
[ PlanetScope HR Reference (96x96, 4 Bands) ]
                     │
                     ▼
       ┌───────────────────────────┐
       │ Radiometric Normalization │  (Divide raw DN by 10000.0 -> [0.0, 1.0])
       └─────────────┬─────────────┘
                     │
                     ▼
       ┌───────────────────────────┐
       │    Spatial PSF Blurring   │  (Band-specific 2D Gaussian optical blur)
       └─────────────┬─────────────┘
                     │
                     ▼
       ┌───────────────────────────┐
       │ Spectral Response Match*  │  (Empirical linear regression gain/offset)
       └─────────────┬─────────────┘
                     │
                     ▼
       ┌───────────────────────────┐
       │     Sensor Noise Model*   │  (Additive Gaussian noise from uniform zones)
       └─────────────┬─────────────┘
                     │
                     ▼
       ┌───────────────────────────┐
       │ Spatial Decimation (Area) │  (Area-weighted IFOV pixel integration)
       └─────────────┬─────────────┘
                     │
                     ▼
[ Synthesized Sentinel-2 LR Observation (32x32, 4 Bands) ]
```

### Parameter Rigor & Scientific Defense:
- **PSF Gaussian Standard Deviations**:
  - Blue (B02): $\sigma_{\text{LR}} = 0.5225$ pixels ($\sigma_{\text{HR}} = 1.5675$ px for $K=3$).
  - Green (B03): $\sigma_{\text{LR}} = 0.5115$ pixels ($\sigma_{\text{HR}} = 1.5345$ px for $K=3$).
  - Red (B04): $\sigma_{\text{LR}} = 0.5225$ pixels ($\sigma_{\text{HR}} = 1.5675$ px for $K=3$).
  - NIR (B08): $\sigma_{\text{LR}} = 0.5300$ pixels ($\sigma_{\text{HR}} = 1.5900$ px for $K=3$).
- **Scene-Specific Spectral Matching & Noise Parameters**:
  - Slopes ($\alpha_c$) and intercepts ($\beta_c$) are **NOT hardcoded**. They are computed via linear regression over homogeneous ground targets across the coregistered scene pair.
  - Noise standard deviations ($\sigma_{\text{noise}, c}$) are estimated directly from uniform water bodies in the Sentinel-2 scene.

---

## 5. PI-RCAN Network Architecture Specifications

The **Physically Informed Residual Channel Attention Network (PI-RCAN)** provides deep representational capacity and explicit cross-spectral feature modeling.

```
                          +----------------------------------------------------+
                          |            PI-RCAN ARCHITECTURAL LAYOUT            |
                          +----------------------------------------------------+

Input Tensor: [Batch, 4, 32, 32] (LR S2 Bands: B02, B03, B04, B08)
     │
     ▼
[ Head: Conv 3x3 (4 -> 64 filters) ] ─────────────────────────┐ (Long Skip Connection)
     │                                                         │
     ▼                                                         │
[ Residual in Residual (RIR) Backbone ]                        │
     │                                                         │
     ├──► [ Residual Group 1 (RG_1) ]                          │
     │      ├─ RCAB_1 (Conv 3x3 -> ReLU -> Conv 3x3 -> CA)     │
     │      ├─ RCAB_2                                          │
     │      ├─ ...                                             │
     │      ├─ RCAB_10                                         │
     │      └─ Short Skip Connection (RG_1 Conv 3x3)           │
     │                                                         │
     ├──► [ Residual Group 2 (RG_2) ]                          │
     ├──► ...                                                  │
     ├──► [ Residual Group 8 (RG_8) ]                          │
     │                                                         │
     ▼                                                         │
[ RIR Tail: Conv 3x3 (64 -> 64 filters) ] ◄────────────────────┘
     │
     ├───────────────────────────────────┐
     ▼                                   ▼
[ Upsampler: PixelShuffle x3 or x4 ] [ Variance Branch: Conv + PixelShuffle ]
     │                                   │
     ▼                                   ▼
[ Reconstruction Head: Conv 3x3 ]   [ Aleatoric Variance Head: Conv 3x3 ]
     │                                   │
     ▼                                   ▼
Enhanced HR Reflectance Cube         Per-Pixel Uncertainty Map
[Batch, 4, 96, 96]                  [Batch, 4, 96, 96]
```

### Architectural Parameters:
- **Input Channels**: 4 (B02, B03, B04, B08).
- **Base Feature Channels ($C$)**: 64.
- **Residual Groups ($G$)**: 8 groups.
- **RCAB Blocks per Group ($B$)**: 8 blocks (Total: 64 Channel Attention Blocks).
- **Channel Attention Reduction Ratio ($r$)**: 16.
- **Upscaling Factor ($K$)**: 3 (or 4).
- **Output Channels**: 4 (Reconstructed Reflectance) + 4 (Aleatoric Variance).
- **Target Trainable Parameters**: $\approx 5.2 \text{ to } 7.8 \text{ Million}$ parameters.

### Memory & Hardware Optimization:
- **Precision**: Automatic Mixed Precision (`torch.cuda.amp.autocast`) with FP16/BF16.
- **Gradient Checkpointing**: Enabled across Residual Groups to reduce peak activation memory by $\approx 40\%$.
- **Inference Tiling**: Production tiled inferrer (`src/super_resolution/inference.py`) processes large $2048 \times 2048$ scenes in $128 \times 128$ overlapping tiles with cosine window blending to prevent seam artifacts while capping GPU VRAM usage at $< 4.0\text{ GB}$.

---

## 6. Composite Spectral-Spatial Loss Function

To eliminate the blurring and texture loss caused by L2/MSE loss, the model is trained with a multi-objective loss function configured via `configs/experiment5_pircan.yaml`:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{rec}} \mathcal{L}_{\text{rec}} + \lambda_{\text{struct}} \mathcal{L}_{\text{struct}} + \lambda_{\text{SAM}} \mathcal{L}_{\text{SAM}} + \lambda_{\text{NDVI}} \mathcal{L}_{\text{NDVI}} + \lambda_{\text{edge}} \mathcal{L}_{\text{edge}}$$

```
+----------------------------------------------------------------------------------------------------+
|                                    COMPOSITE LOSS CONFIGURATION                                    |
+-------------------+-----------------+---------------+----------------------------------------------+
| Loss Component    | Mathematical    | Default       | Scientific Justification                     |
|                   | Operator        | Weight        |                                              |
+-------------------+-----------------+---------------+----------------------------------------------+
| Reconstruction    | Charbonnier L1  | λ_rec = 1.0   | Robust pixel-level convergence; does not     |
|                   |                 |               | over-penalize large gradients like L2.       |
+-------------------+-----------------+---------------+----------------------------------------------+
| Structural        | Multi-Scale     | λ_struct = 0.2| Enforces structural luminance/contrast       |
|                   | SSIM            |               | correlations across multiple spatial scales. |
+-------------------+-----------------+---------------+----------------------------------------------+
| Spectral Angle    | Spectral Angle  | λ_SAM = 0.1   | Preserves physical spectral signature vector |
|                   | Mapper (SAM)    |               | across all 4 channels, preventing color skew.|
+-------------------+-----------------+---------------+----------------------------------------------+
| Biophysical Index | NDVI L1 Error   | λ_NDVI = 0.1  | Enforces precise non-linear Red-NIR band     |
|                   |                 |               | ratio preservation for vegetation analysis.  |
+-------------------+-----------------+---------------+----------------------------------------------+
| Spatial Edge      | Sobel Gradient  | λ_edge = 0.05 | Sharpens parcel boundaries, roads, and       |
|                   | L1 Difference   |               | building perimeters.                         |
+-------------------+-----------------+---------------+----------------------------------------------+
```

### Loss Formulas:
1. **Charbonnier Loss**:
   $$\mathcal{L}_{\text{rec}} = \frac{1}{C H W} \sum_{c, x, y} \sqrt{(I_{\text{pred}}(c, x, y) - I_{\text{ref}}(c, x, y))^2 + \epsilon^2}, \quad \epsilon = 10^{-3}$$
2. **Multi-Scale Structural Similarity (MS-SSIM)**:
   $$\mathcal{L}_{\text{struct}} = 1 - \prod_{m=1}^{M} \left[ l_M(x, y) \right]^{\alpha_M} \cdot \left[ c_m(x, y) \right]^{\beta_m} \cdot \left[ s_m(x, y) \right]^{\gamma_m}$$
3. **Spectral Angle Mapper (SAM)**:
   $$\mathcal{L}_{\text{SAM}} = \frac{1}{H W} \sum_{x, y} \arccos\left(\frac{\mathbf{v}_{\text{pred}}(x, y) \cdot \mathbf{v}_{\text{ref}}(x, y)}{\|\mathbf{v}_{\text{pred}}(x, y)\|_2 \|\mathbf{v}_{\text{ref}}(x, y)\|_2 + \epsilon}\right)$$
4. **NDVI Consistency**:
   $$\mathcal{L}_{\text{NDVI}} = \frac{1}{H W} \sum_{x, y} \left| \frac{I_{\text{pred}}^{\text{NIR}} - I_{\text{pred}}^{\text{Red}}}{I_{\text{pred}}^{\text{NIR}} + I_{\text{pred}}^{\text{Red}} + \epsilon} - \frac{I_{\text{ref}}^{\text{NIR}} - I_{\text{ref}}^{\text{Red}}}{I_{\text{ref}}^{\text{NIR}} + I_{\text{ref}}^{\text{Red}} + \epsilon} \right|$$
5. **Spatial Edge Gradient**:
   $$\mathcal{L}_{\text{edge}} = \frac{1}{C H W} \sum_{c} \left( \|\nabla_x I_{\text{pred}}[c] - \nabla_x I_{\text{ref}}[c]\|_1 + \|\nabla_y I_{\text{pred}}[c] - \nabla_y I_{\text{ref}}[c]\|_1 \right)$$

---

## 7. Uncertainty Quantification Architecture

To distinguish between confident super-resolved features and potential sub-pixel hallucinations, Experiment 5 implements explicit uncertainty quantification.

```
+─────────────────────────────────────────────────────────────────────────────+
|                        UNCERTAINTY ESTIMATION PHASING                       |
+─────────────────────────────────────────────────────────────────────────────+
|                                                                             |
|  [ PHASE 1: FIRST PROTOTYPE IMPLEMENTATION ]                                |
|  - Dual-Head Heteroscedastic Aleatoric Variance Head                        |
|  - Jointly optimized via Gaussian Negative Log-Likelihood (NLL)             |
|  - Generates 4-band variance map: σ^2(c, x, y)                              |
|  - Normalized spatial confidence raster: C(x, y) = exp(-mean(σ(c, x, y)))   |
|                                                                             |
|  [ PHASE 2: SUBSEQUENT RESEARCH ENHANCEMENTS ]                              |
|  - Monte Carlo (MC) Dropout Epistemic Uncertainty (T=10 forward passes)     |
|  - Deep Ensemble variance across diverse seed initializations               |
|  - Conformal prediction error bounds at 95% confidence intervals            |
|                                                                             |
+─────────────────────────────────────────────────────────────────────────────+
```

### Phase 1 Aleatoric Loss Formulation:
The network outputs both predicted reflectance $\hat{\mu}(c, x, y)$ and predicted log-variance $s(c, x, y) = \ln \hat{\sigma}^2(c, x, y)$:

$$\mathcal{L}_{\text{aleatoric}} = \frac{1}{2 C H W} \sum_{c, x, y} \left( \exp(-s(c, x, y)) \cdot (I_{\text{ref}}(c, x, y) - \hat{\mu}(c, x, y))^2 + s(c, x, y) \right)$$

This penalizes large errors while allowing the network to adaptively predict high variance in regions with intrinsic ambiguity (e.g., building shadow boundaries, high-frequency vegetation foliage).

---

## 8. Checkpoint Isolation & Directory Structure

To protect the integrity of completed experiments, all Experiment 5 outputs and models are strictly isolated.

```
models/
├── espcn_srm_synthetic.pth             <-- [READ-ONLY] Frozen Baseline (Exp 2)
├── residual_srm_experiment3.pth        <-- [READ-ONLY] Frozen Baseline (Exp 3)
└── experiment4_real/                   <-- [TARGET DIRECTORY FOR EXP 5]
    ├── pircan_best.pth                 <-- Best validation loss checkpoint
    ├── pircan_latest.pth               <-- Resume training checkpoint
    └── pircan_checkpoint_epoch_*.pth   <-- Periodic epoch checkpoints
```

> [!WARNING]
> Any execution script targeting `models/espcn_srm_synthetic.pth` or `models/residual_srm_experiment3.pth` will abort immediately with an overwrite violation error.

---

## 9. Output Artifacts & Deliverables

Upon completion of Experiment 5, the pipeline generates the following standardized deliverables in `outputs/experiment5/`:

1. **Enhanced 4-Band Multispectral GeoTIFF**:
   - Path: `outputs/experiment5/s2_pircan_3m_enhanced.tiff`
   - Format: 4-Band Float32 BOA Reflectance $[0.0, 1.0]$, georeferenced `EPSG:32643`, snaped pixel origin.
2. **Uncertainty & Confidence GeoTIFF**:
   - Path: `outputs/experiment5/s2_pircan_uncertainty.tiff`
   - Format: 5-Band Float32 (Bands 1–4: $\sigma_{\text{Blue}}, \sigma_{\text{Green}}, \sigma_{\text{Red}}, \sigma_{\text{NIR}}$; Band 5: Normalized Spatial Confidence Index $[0.0, 1.0]$).
3. **Quality Metrics JSON**:
   - Path: `outputs/experiment5/metrics_evaluation.json`
   - Content: Full numerical breakdown of PSNR, SSIM, SAM, ERGAS, EPI across all models on the held-out test split.
4. **Comprehensive Validation Report**:
   - Path: `docs/experiment5_validation_report.md`
   - Content: Multi-model comparative tables, frequency response audits, and scientific conclusions.
5. **Visual Comparison Assets**:
   - Path: `outputs/experiment5/visuals/`
   - Assets: Side-by-side true-color RGB crops, False-color NIR crops, NDVI difference maps, and uncertainty overlays.

---

## 10. Multi-Model Benchmark & Validation Protocol

The validation framework evaluates four candidate models against the real PlanetScope test split:

```
+─────────────────────────────────────────────────────────────────────────────+
|                        MULTI-MODEL BENCHMARK SUITE                          |
+─────────────────────────────────────────────────────────────────────────────+
|  1. Bilinear Baseline:       Standard non-learned spatial interpolation     |
|  2. ESPCN Baseline:          Lightweight sub-pixel CNN (Exp 2)              |
|  3. Residual CNN Baseline:   3-Block Residual Network (Exp 3)               |
|  4. PI-RCAN Prototype:       Physically Informed RCAN (Exp 5)               |
+─────────────────────────────────────────────────────────────────────────────+
```

### Complete Benchmark Evaluation Matrix:

| Category | Metric | Mathematical Definition / Objective | Target Direction |
| :--- | :--- | :--- | :---: |
| **Image Quality** | **PSNR (dB)** | Peak Signal-to-Noise Ratio over physical reflectance | Higher ($\ge +2.0\text{ dB}$ over Bilinear) |
| | **SSIM** | Structural Similarity Index (luminance, contrast, structure) | Higher ($\ge 0.90$) |
| | **MAE / RMSE** | Mean Absolute / Root Mean Square Reflectance Error | Lower |
| **Spatial Fidelity** | **EPI** | Edge Preservation Index along High-Pass Sobel gradients | Higher ($\ge 0.88$) |
| | **HF Energy Ratio** | Ratio of high-frequency Fourier energy relative to HR target | Higher ($\ge 75\%$) |
| | **Local Contrast** | Standard deviation ratio across $16 \times 16$ sliding windows | Closer to 1.0 |
| **Spectral Fidelity**| **SAM (deg)** | Spectral Angle Mapper across the 4-band spectral vectors | Lower ($< 2.5^\circ$) |
| | **ERGAS** | Relative Dimensionless Global Error in Synthesis | Lower ($< 3.0$) |
| | **UIQI (Q)** | Universal Image Quality Index across spectral bands | Higher ($\ge 0.92$) |
| | **$\Delta\text{NDVI}$** | Mean Absolute Error of computed NDVI vs. HR target NDVI | Lower ($< 0.02$) |
| **Downstream Utility**| **Parcel IoU** | Intersection-over-Union on segmented agricultural parcel bounds | Higher |
| | **Building F1** | F1-score of detected urban building boundaries | Higher |

---

## 11. Explicit Failure Conditions & Rejection Thresholds

The Experiment 5 execution and evaluation must be **immediately rejected and aborted** if any of the following failure conditions occur:

```
+─────────────────────────────────────────────────────────────────────────────+
|                         CRITICAL FAILURE CONDITIONS                         |
+─────────────────────────────────────────────────────────────────────────────+
|  TRAINING FAILURE CONDITIONS (Immediate Abort):                             |
|  - Loss Divergence: Total loss becomes NaN, Inf, or explodes > 10.0.        |
|  - Mode Collapse: Model outputs uniform mean reflectance across all patches.|
|  - Spectral Distortion: Mean SAM on validation split exceeds 5.0 degrees.   |
|  - Out-of-Bounds Generation: > 0.5% predicted pixels fall outside [0.0, 1.0]|
+─────────────────────────────────────────────────────────────────────────────+
|  EVALUATION REJECTION CONDITIONS (Benchmark Failure):                       |
|  - Zero PSNR Gain: PI-RCAN fails to exceed the Bilinear PSNR on test split. |
|  - Edge Blur: EPI of PI-RCAN is lower than the Residual CNN baseline.      |
|  - NDVI Corruption: Delta-NDVI MAE exceeds 0.05 (distorting vegetation).    |
|  - Coordinate Shift: Geotransform drift between input and output > 0.5 px.  |
|  - Uncertainty Anti-Correlation: Aleatoric variance negatively correlates   |
|    with ground-truth error residuals (r < 0.20).                            |
+─────────────────────────────────────────────────────────────────────────────+
```

---

## 12. Experiment 5 Acceptance Criteria (`REAL_DATA_VALIDATED`)

To officially certify Experiment 5 as **`REAL_DATA_VALIDATED`**, the pipeline must produce verifiable evidence satisfying **all seven acceptance criteria**:

```
+----------------------------------------------------------------------------------------------------+
|                                CRITERIA FOR REAL_DATA_VALIDATED                                    |
+----+-----------------------------+--------------------------------------------------+--------------+
| #  | Criterion                   | Required Threshold / Condition                   | Verification |
+----+-----------------------------+--------------------------------------------------+--------------+
| 1  | **Data QC Verification**    | 10/10 QC Gates pass on Scene 20260211_054815_64. | PASS         |
| 2  | **PSNR Superiority**        | PI-RCAN PSNR $\ge$ Bilinear PSNR + 2.0 dB on Test| Verified     |
| 3  | **Structural Fidelity**     | Test SSIM $\ge 0.90$ across all 4 spectral bands.| Verified     |
| 4  | **Spectral Preservation**   | Test SAM $< 2.5^\circ$ and $\Delta\text{NDVI} < 0.02$.       | Verified     |
| 5  | **Spatial Texture Recovery**| High-Frequency Energy Ratio $\ge 75\%$ of Target.| Verified     |
| 6  | **Uncertainty Calibration** | Variance positively correlates with error ($r > 0.65$). | Verified     |
| 7  | **Geospatial Integrity**    | Coordinate bounds & EPSG:32643 perfectly intact. | 100% Match   |
+----+-----------------------------+--------------------------------------------------+--------------+
```

---

## Summary of Execution Policy

When the authorized, authentic PlanetScope scene `20260211_054815_64_254a` is delivered to the repository:
1. Run automated Data QC (`src/super_resolution/quality_control.py`).
2. Extract geographically buffered train/val/test splits (`src/super_resolution/paired_dataset.py`).
3. Train PI-RCAN with physical degradation and composite loss (`models/experiment4_real/`).
4. Generate georeferenced enhanced TIFFs and uncertainty rasters (`outputs/experiment5/`).
5. Execute the multi-model benchmark and compile the validation report.
