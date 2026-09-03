# Satellite Super-Resolution Quality Improvement Plan: Architecture & Loss Redesign

**Project**: Sentinel-2 Multispectral Super-Resolution Mapping (SRM) (SIH26142)  
**Document Identifier**: `docs/sr_quality_improvement_plan.md`  
**Execution Phase**: Model & Loss Improvement Strategy (Pre-Implementation)  
**Status**: `PLAN_APPROVED_FOR_SPECIFICATION`  

---

## Executive Summary

A comprehensive visual and frequency audit of Experiment 3 (Residual CNN) revealed a critical bottleneck: while the network achieves mathematical gains over Bilinear interpolation (+1.25 dB PSNR, +0.05 SSIM), **the visual output is too close to Bilinear interpolation to be considered an effective high-resolution enhancement prototype**.

The shallow 3-block Residual CNN trained under Mean Squared Error (MSE / L2) acts primarily as a localized edge-sharpening/deblurring filter. It lacks the representational capacity and loss incentives required to recover high-frequency textures, resulting in over-smoothed natural surfaces and minimal visual gain in agricultural, forest, and textured urban regions.

This document establishes the **Super-Resolution Quality Improvement Plan**. It freezes downstream integration (such as oil-spill detection) to prioritize core super-resolution quality, evaluates five candidate architectures, specifies a composite spectral-spatial loss function, and defines a strict protocol to achieve **visibly distinct, scientifically defensible super-resolution**.

```
+─────────────────────────────────────────────────────────────────────────────+
|                          STRATEGIC PIVOT & FOCUS                            |
+─────────────────────────────────────────────────────────────────────────────+
|                                                                             |
|   [ PREVIOUS STATUS ]                                                       |
|   - 3-Block Residual CNN (251k params)                                      |
|   - MSE Loss (L2) -> Severe over-smoothing / conditional mean wash          |
|   - Visual outcome: Sharpens step edges, but looks ~85% like Bilinear       |
|                                                                             |
|   [ STRATEGIC PIVOT ]                                                       |
|   - PAUSE all downstream application pipelines (oil spill, AIS).            |
|   - Redesign model backbone & loss function to break performance ceiling.   |
|   - Objective: Achieve clearly noticeable, authentic spatial & spectral SR. |
|                                                                             |
+─────────────────────────────────────────────────────────────────────────────+
```

---

## 1. Analysis of Current Experiment 3 Results & Failure Modes

From the quantitative and visual audits documented in `docs/model_failure_analysis.md` and `docs/experiment4a_visual_audit.md`:

```
+----------------------------------------------------------------------------------------------------+
|                               EXPERIMENT 3 PERFORMANCE AUDIT SUMMARY                               |
+-------------------+-----------------+-----------------+--------------------------------------------+
| Metric / Feature  | Bilinear        | Residual CNN    | Visual / Physical Finding                  |
+-------------------+-----------------+-----------------+--------------------------------------------+
| **Global PSNR**   | 31.51 dB        | **32.80 dB**    | +1.29 dB gain (mostly in Red & NIR bands). |
| **Global SSIM**   | 0.8553          | **0.8944**      | +0.0391 gain in structural correlation.    |
| **High-Freq PSD** | 29.5% of target | **65.1%**       | Mid-frequencies restored; corners filtered.|
| **Local Contrast**| 0.0263 (Std)    | **0.0394**      | Still significantly below Target (0.0439). |
| **Buildings**     | 29.78 dB / 0.861| **31.73 / 0.900**| Sharper roof boundaries; interior blurred. |
| **Roads**         | 29.81 dB / 0.856| **31.37 / 0.893**| Clearer linear path; road width unchanged. |
| **Field Bounds**  | 31.16 dB / 0.854| **32.50 / 0.891**| Modest boundary tightening.                |
| **Forest Canopy** | 30.82 dB / 0.845| **32.46 / 0.894**| **Failure**: Airbrushed, smoothed wash.    |
| **Homogeneous**   | 30.68 dB / 0.846| **31.81 / 0.883**| Visually indistinguishable from Bilinear.  |
+-------------------+-----------------+-----------------+--------------------------------------------+
```

### Core Failure Mechanisms:
1. **L2 Conditional Mean Smoothing**: MSE penalizes any variance deviation from the ground truth quadratically. Faced with sub-pixel ambiguity, the network outputs the average of all possible pixel arrangements, wiping out fine natural textures (soil grain, canopy leaves).
2. **Receptive Field Limitation**: 3 convolutional layers provide a maximum effective receptive field of only ~15–20 pixels, preventing the network from utilizing broader contextual cues across the scene.
3. **Uniform Channel Treatment**: Standard convolutions treat visible RGB and NIR channels identically, ignoring the strong anti-correlations and physical spectral constraints between Red absorption and NIR reflectance.

---

## 2. Evaluation of Candidate Deep Architectures

We evaluate five state-of-the-art super-resolution architectures to identify the optimal primary model for our next development phase:

```
+----------------------------------------------------------------------------------------------------+
|                               CANDIDATE ARCHITECTURE EVALUATION                                    |
+----+-------------------+-----------------------+---------------------+-----------------------------+
| ID | Architecture      | Structural Mechanism  | Parameter / Compute | Suitability for Multispectral|
|    |                   |                       | Profile             | Remote Sensing              |
+----+-------------------+-----------------------+---------------------+-----------------------------+
| A  | **RCAN**          | Residual-in-Residual  | ~6.5M - 12.0M       | **EXCELLENT**               |
|    | (Residual Channel | (RIR) + Channel       | Moderate Compute    | Dynamically reweights cross-|
|    | Attention)        | Attention Blocks      | (Fast Convergence)  | band spectral features.     |
+----+-------------------+-----------------------+---------------------+-----------------------------+
| B  | **EDSR**          | Deep ResNet without   | ~10.0M - 43.0M      | **FAIR**                    |
|    | (Enhanced Deep    | Batch Normalization   | High Memory         | High capacity, but lacks    |
|    | Super-Resolution) | layers                | Footprint           | channel attention mechanism.|
+----+-------------------+-----------------------+---------------------+-----------------------------+
| C  | **SwinIR**        | Swin Transformer      | ~11.5M              | **GOOD (Data-Hungry)**      |
|    | (Image Restoration| Shifted Window Self-  | Very High Compute   | Superb spatial context, but |
|    | Transformer)      | Attention             | (Slow Training)     | overfits on small datasets. |
+----+-------------------+-----------------------+---------------------+-----------------------------+
| D  | **Multiscale      | Progressive Laplacian | ~8.0M               | **MODERATE**                |
|    | Residual Attn**   | feature pyramids      | Complex multi-branch| Unstable loss balancing for |
|    | (MS-ResAttn)      | + Spatial Attention   | routing             | multispectral bands.        |
+----+-------------------+-----------------------+---------------------+-----------------------------+
| E  | **Multispectral   | Dual Spectral-Spatial | ~5.5M - 8.5M        | **EXCELLENT**               |
|    | Channel-Attention | Attention (MS-RCAN)   | Balanced Memory     | Directly tailored to remote |
|    | SR (MS-RCAN)**    | with 4-Band Covariance| & Fast Inference    | sensing 4-band tensors.     |
+----+-------------------+-----------------------+---------------------+-----------------------------+
```

### Recommendation for Next Experiment:
**PRIMARY RECOMMENDED MODEL: Multispectral Residual Channel Attention Network (MS-RCAN / PI-RCAN)**.

#### Why MS-RCAN is the Decisive Winner:
1. **Residual-in-Residual (RIR) Depth**: Bypasses low-frequency radiometry via long and short skip connections, dedicating all parameter capacity (~6.5M parameters across 8 Residual Groups) to learning sub-pixel high-frequency residuals.
2. **Channel Attention (RCAB)**: Explicitly computes cross-band covariance weights, enabling the network to leverage the sharp spatial edges present in the NIR band (B08) to reconstruct details in the Red (B04) and Green (B03) bands.
3. **Training Stability & Data Efficiency**: Trains reliably on single-scene crops without the extreme data requirements or training instability of vision transformers (SwinIR).

---

## 3. Composite Spectral-Spatial Loss Function Design

To eliminate over-smoothing and enforce physical radiometric validity, we replace MSE with a four-component composite loss:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{rec}} \mathcal{L}_{\text{rec}} + \lambda_{\text{struct}} \mathcal{L}_{\text{struct}} + \lambda_{\text{spec}} \mathcal{L}_{\text{spec}} + \lambda_{\text{edge}} \mathcal{L}_{\text{edge}}$$

```
+─────────────────────────────────────────────────────────────────────────────+
|                         COMPOSITE LOSS ARCHITECTURE                         |
+─────────────────────────────────────────────────────────────────────────────+
|                                                                             |
|   1. Reconstruction Loss (Charbonnier / Smooth L1)     [Weight: λ_rec = 1.0] |
|      - Ensures accurate pixel-level reflectance without L2 over-smoothing.  |
|                                                                             |
|   2. Structural Loss (Multi-Scale SSIM)             [Weight: λ_struct = 0.2]|
|      - Preserves luminance, contrast, and multiscale spatial structure.      |
|                                                                             |
|   3. Spectral Consistency Loss (SAM + NDVI)            [Weight: λ_spec = 0.1]|
|      - Enforces cross-band angle alignment and biophysical index validity.  |
|                                                                             |
|   4. Spatial Edge / Gradient Loss (Sobel L1)           [Weight: λ_edge = 0.05]|
|      - Sharpens parcel boundaries, roads, and building perimeters.          |
|                                                                             |
+─────────────────────────────────────────────────────────────────────────────+
```

### Detailed Mathematical Formulations:

#### 1. Reconstruction Loss ($\mathcal{L}_{\text{rec}}$ - Charbonnier Loss):
$$\mathcal{L}_{\text{rec}} = \frac{1}{C \cdot H \cdot W} \sum_{c=1}^{C} \sum_{x, y} \sqrt{(I_{\text{pred}}(c, x, y) - I_{\text{target}}(c, x, y))^2 + \epsilon^2}, \quad \epsilon = 10^{-3}$$
*Role*: Replaces MSE. It behaves like L2 for tiny errors ($< \epsilon$) ensuring smooth convergence, but transitions to robust L1 for larger residuals, preventing the network from smoothing out sharp high-frequency transitions.

#### 2. Structural Loss ($\mathcal{L}_{\text{struct}}$ - Multi-Scale SSIM):
$$\mathcal{L}_{\text{struct}} = 1 - \text{MS-SSIM}(I_{\text{pred}}, I_{\text{target}})$$
*Role*: Evaluates structural correlation across multiple dyadic scales, forcing the model to reconstruct coherent spatial patterns rather than isolated bright pixels.

#### 3. Spectral Consistency Loss ($\mathcal{L}_{\text{spec}}$):
$$\mathcal{L}_{\text{spec}} = \mathcal{L}_{\text{SAM}} + \mathcal{L}_{\text{NDVI}}$$
$$\mathcal{L}_{\text{SAM}} = \frac{1}{H \cdot W} \sum_{x, y} \arccos\left(\frac{\mathbf{v}_{\text{pred}}(x, y) \cdot \mathbf{v}_{\text{target}}(x, y)}{\|\mathbf{v}_{\text{pred}}(x, y)\|_2 \|\mathbf{v}_{\text{target}}(x, y)\|_2 + \epsilon}\right)$$
$$\mathcal{L}_{\text{NDVI}} = \frac{1}{H \cdot W} \sum_{x, y} \left| \frac{I_{\text{pred}}^{\text{NIR}} - I_{\text{pred}}^{\text{Red}}}{I_{\text{pred}}^{\text{NIR}} + I_{\text{pred}}^{\text{Red}} + \epsilon} - \frac{I_{\text{target}}^{\text{NIR}} - I_{\text{target}}^{\text{Red}}}{I_{\text{target}}^{\text{NIR}} + I_{\text{target}}^{\text{Red}} + \epsilon} \right|$$
*Role*: Prevents spectral distortion and color bleaching. Enforces that super-resolved pixels preserve physical reflectance ratios and NDVI values for downstream agricultural and environmental analysis.

#### 4. Spatial Edge / Gradient Loss ($\mathcal{L}_{\text{edge}}$):
$$\mathcal{L}_{\text{edge}} = \frac{1}{C \cdot H \cdot W} \sum_{c=1}^{C} \left( \|\nabla_x I_{\text{pred}}[c] - \nabla_x I_{\text{target}}[c]\|_1 + \|\nabla_y I_{\text{pred}}[c] - \nabla_y I_{\text{target}}[c]\|_1 \right)$$
*Role*: Directly penalizes smoothed gradient transitions, tightening boundaries along roads, field edges, and buildings.

---

## 4. Strict Separation of Experimental Branches

To maintain scientific integrity, development is strictly bifurcated into two distinct phases:

```
+─────────────────────────────────────────────────────────────────────────────+
|                     EXPERIMENTAL BRANCH ISOLATION MATRIX                    |
+─────────────────────────────────────────────────────────────────────────────+
|                                                                             |
|  [ BRANCH 1: SYNTHETIC DEVELOPMENT EXPERIMENT ]                             |
|  - Scope: Architecture & Composite Loss Development Only.                   |
|  - Input / Target: 20m synthetic LR -> 10m Sentinel-2 target.               |
|  - Purpose: Prove that MS-RCAN + Composite Loss is VISIBLY superior to      |
|             the Experiment 3 Residual CNN under identical test conditions.  |
|  - Scientific Status: Methodological proof of concept.                      |
|    STRICT RULE: MUST NOT be claimed as proof of real <4m resolution!        |
|                                                                             |
|  [ BRANCH 2: REAL-DATA EXPERIMENT (EXPERIMENT 5) ]                          |
|  - Scope: Real-world operational super-resolution (<4m).                    |
|  - Input / Reference: Coincident Sentinel-2 L2A (10m) + PlanetScope (~3m).  |
|  - Degradation: Experiment 4 Modular Physical Degradation Framework.        |
|  - Scientific Status: Operational validation on real commercial reference.  |
|    BLOCKED: Awaiting delivery of authorized Scene 20260211_054815_64_254a.  |
|                                                                             |
+─────────────────────────────────────────────────────────────────────────────+
```

---

## 5. Synthetic Development Experiment Protocol (Experiment 4D)

The next immediate experimental step is **Experiment 4D: Synthetic Architecture & Loss Prototyping**.

```
+----------------------------------------------------------------------------------------------------+
|                             EXPERIMENT 4D SPECIFICATION MATRIX                                     |
+--------------------------+-------------------------------------------------------------------------+
| Parameter                | Specification                                                           |
+--------------------------+-------------------------------------------------------------------------+
| **Objective**            | Validate MS-RCAN + Composite Loss against Residual CNN & Bilinear       |
| **Input Data**           | Synthetic 20m LR Sentinel-2 (Downscaled from 10m ROI)                   |
| **Target Data**          | Original 10m Sentinel-2 ROI ($1024 \times 1024$ pixels, 4 Bands)        |
| **Scale Factor**         | $2\times$ ($20\text{m} \to 10\text{m}$) for synthetic validation       |
| **Patch Dimensions**     | LR: $32 \times 32$ pixels; HR: $64 \times 64$ pixels                    |
| **Model Architecture**   | MS-RCAN (8 Residual Groups, 8 RCAB blocks/group, 64 features, ~6.5M)    |
| **Loss Function**        | Composite ($\lambda_{\text{rec}}=1.0, \lambda_{\text{struct}}=0.2,     |
|                          | \lambda_{\text{SAM}}=0.1, \lambda_{\text{NDVI}}=0.1, \lambda_{\text{edge}}=0.05$) |
| **Optimizer**            | AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $= 10^{-4}$)          |
| **Learning Schedule**    | Cosine Annealing: $2 \times 10^{-4}$ decaying to $1 \times 10^{-6}$    |
| **Training Budget**      | 100 Epochs with Early Stopping (patience: 15 epochs)                    |
| **Spatial Partitioning** | 70% Train, 15% Val, 15% Test with 200m spatial buffer                  |
| **Target Checkpoint**    | `models/experiment4d_msrcan_synthetic.pth` (Strictly Isolated)          |
+--------------------------+-------------------------------------------------------------------------+
```

---

## 6. Rigorous Visual & Quantitative Evaluation Protocol

The next experiment will not rely solely on global PSNR/SSIM numbers. It enforces a **Three-Way Regional Evaluation Protocol** across five distinct land-cover classes:

```
+─────────────────────────────────────────────────────────────────────────────+
|                     THREE-WAY BENCHMARK EVALUATION MATRIX                   |
+─────────────────────────────────────────────────────────────────────────────+
|                                                                             |
|   Models Compared:                                                          |
|   1. Bilinear Baseline                                                      |
|   2. Residual CNN Baseline (Experiment 3)                                   |
|   3. NEW MS-RCAN Prototype (Experiment 4D)                                  |
|                                                                             |
|   Against: Original 10m Ground Truth Reference                              |
|                                                                             |
+─────────────────────────────────────────────────────────────────────────────+
```

### Land-Cover Test Regions ($128 \times 128$ Pixel Crops):
1. **Urban / Buildings**: Dense warehouse roofs, residential clusters, industrial boundaries.
2. **Roads & Transport**: Highway overpasses, straight asphalt roads, airport runways.
3. **Agricultural Field Boundaries**: Rectangular plot borders, irrigation ditches, crop rows.
4. **Forest & Vegetation Canopy**: Dense tree stands, parkland, vegetation-soil boundaries.
5. **Homogeneous Zones**: Deep water bodies, bare soil expanses.

### Comprehensive Metric Suite:
- **PSNR (dB) & SSIM**: Pixel fidelity and structural correlation.
- **Mean Absolute Error (MAE) & RMSE**: Radiometric reflectance error.
- **Edge Preservation Index (EPI)**: High-pass Sobel gradient sharpness.
- **High-Frequency Energy Ratio**: % of Target Fourier PSD energy recovered above Nyquist limit.
- **Local Contrast ($\sigma_{\text{local}}$)**: Standard deviation ratio across $16 \times 16$ sliding windows.
- **Spectral Angle Mapper (SAM)**: Degrees of spectral vector distortion ($< 2.0^\circ$ target).
- **NDVI Mean Absolute Error ($\Delta\text{NDVI}$)**: Biophysical index consistency ($< 0.015$ target).

### Visual & Artifact Audit Checklist:
- [ ] **Checkerboard Pattern Check**: Confirm sub-pixel convolution is artifact-free.
- [ ] **Ringing / Haloing Check**: Ensure sharp building boundaries have no dark/bright halos.
- [ ] **Plastic Smoothing Check**: Confirm forest canopy and soil retain natural grain.
- [ ] **Color Bleaching Check**: Verify RGB and False-Color NIR show no hue distortion.
- [ ] **Hallucination Check**: Confirm all reconstructed lines exist in the LR footprint.

---

## 7. Success Criteria for the Next Experiment

To certify that the new model has genuinely broken through the Experiment 3 ceiling:

1. **PSNR Improvement**: MS-RCAN must exceed Residual CNN by at least **$+0.75\text{ dB}$** (total $\ge +2.2\text{ dB}$ over Bilinear).
2. **High-Frequency Recovery**: High-Frequency Energy Ratio must reach **$\ge 75\%$** of target (up from 65% in Exp 3 and 29% in Bilinear).
3. **Local Contrast Restoration**: Local standard deviation in textured regions must reach **$\ge 0.042$** (target is 0.045; Exp 3 was 0.039; Bilinear was 0.026).
4. **Visual Disambiguation**: Clear visual separation between adjacent urban buildings and sharp, continuous tracing of narrow paths.
5. **Spectral Preservation**: $\text{SAM} < 2.0^\circ$ and $\Delta\text{NDVI} < 0.015$.

---

## Execution Constraints Summary

- **No Code Execution Yet**: This plan establishes the scientific architecture and evaluation contract.
- **No Overwriting**: Checkpoints `models/espcn_srm_synthetic.pth` and `models/residual_srm_experiment3.pth` remain untouched.
- **Synthetic Isolation**: Experiment 4D is labeled strictly as synthetic architectural prototyping.
- **Real Reference Integration**: Experiment 5 remains the designated real-data milestone once PlanetScope Scene `20260211_054815_64_254a` is ingested.
