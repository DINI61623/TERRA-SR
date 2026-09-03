# Downstream Oil-Spill Intelligence Module: Scientific Architecture & Prototype Design

**Project**: Satellite Multispectral Oil-Spill Intelligence System (SIH26142)  
**Document Identifier**: `docs/oil_spill_intelligence_design.md`  
**Execution Stage**: Downstream Intelligence Module Specification & Architecture Design  
**Status**: `DESIGN_COMPLETE` (Pre-Implementation Specification)  

---

## Executive Summary

This document specifies the scientific and technical architecture for the **Downstream Satellite Oil-Spill Intelligence Module**. The module operates downstream of the **Physically Informed Residual Channel Attention Network (PI-RCAN)**, utilizing enhanced ~3m multispectral imagery (B02, B03, B04, B08) to detect, segment, quantify, and map marine oil slicks.

The system is designed around physical optics, cross-sensor radiometric integrity, and strict anti-hallucination safeguards. Because super-resolved imagery contains inferred high-frequency spatial detail, the oil-spill module enforces **dual-scale co-verification** (cross-referencing detections against raw Sentinel-2 observations) and **uncertainty-gated segmentation** to prevent false positives from uncalibrated sub-pixel artifacts.

```
+─────────────────────────────────────────────────────────────────────────────────────────────+
|                                END-TO-END SYSTEM DATA FLOW                                  |
+─────────────────────────────────────────────────────────────────────────────────────────────+
|                                                                                             |
|   [ Sentinel-2 L2A (10m) ] ────► [ PI-RCAN Spatial Enhancer ] ──► [ Enhanced 4-Band (~3m) ]  |
|                                            │                                │               |
|                                            ▼                                │               |
|                              [ Aleatoric Uncertainty Map ]                  │               |
|                                            │                                │               |
|                                            └───────────────┬────────────────┘               |
|                                                            │                                |
|                                                            ▼                                |
|                                            [ Preprocessing & Water Gating ]                 |
|                                                            │                                |
|                                                            ▼                                |
|                                            [ Spectral Feature Extraction ]                  |
|                                                            │                                |
|                                                            ▼                                |
|                                            [ CNN Oil Slick Segmentation ]                   |
|                                                            │                                |
|                                                            ▼                                |
|                                            [ Anti-Hallucination Safeguards ]                |
|                                                            │                                |
|                                                            ▼                                |
|                                            [ GIS Intelligence & Reporting ]                 |
|                                                                                             |
+─────────────────────────────────────────────────────────────────────────────────────────────+
```

---

## 1. Problem Definition: Satellite Multispectral Oil-Spill Detection

Oil spills in marine environments represent critical ecological and economic hazards. Detecting and delineating oil slicks using optical satellite remote sensing relies on the physical interaction between solar irradiance, the sea surface, and petroleum hydrocarbon layers.

```
+─────────────────────────────────────────────────────────────────────────────────────────────+
|                                   OPTICAL DETECTION PHYSICS                                 |
+─────────────────────────────────────────────────────────────────────────────────────────────+
|                                                                                             |
|   Solar Irradiance (E_sun)                                                                  |
|             \                                                                               |
|              \          [ Thin Sheen (<0.1 mm) ]          [ Emulsified / Mousse (>1 mm) ]   |
|               \         - Damps capillary waves           - High volume scattering          |
|                \        - Lowers surface roughness        - Elevates Red/NIR reflectance    |
|                 \       - Alters specular sunglint        - Appears brighter than water     |
|                  ▼                 ▼                                   ▼                    |
|             ~~~~~~~~~~~~~~~[ Marine Surface ]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     |
|                    \                                                                        |
|                     ▼                                                                       |
|              [ Clean Seawater ]                                                             |
|              - Strong absorption in Red & NIR (Reflectance ~ 0.01 - 0.03)                   |
|              - Dominated by Blue/Green volume scattering                                    |
|                                                                                             |
+─────────────────────────────────────────────────────────────────────────────────────────────+
```

### Optical & Physical Characteristics of Oil Slicks:
1. **Thin Oil Sheens ($< 0.1\text{ mm}$)**:
   - Damps high-frequency capillary and gravity waves, creating a smooth surface.
   - Under non-sunglint geometry, sheens appear slightly darker or neutral relative to surrounding water due to suppressed surface scattering.
   - Under moderate sunglint, sheens act as specular mirrors, appearing significantly brighter than background water.
2. **Thick Emulsified Oil ("Mousse", $> 1.0\text{ mm}$)**:
   - Formed when seawater mixes into crude oil, increasing viscosity and optical scattering.
   - Generates elevated reflectance across Red (B04) and Near-Infrared (B08) wavelengths, creating a strong contrast against the near-zero NIR baseline of clean water.
3. **Core Challenges**:
   - *Spatial Resolution Limit*: Narrow oil filaments, sheen ribbons, and dispersant plumes are narrower than Sentinel-2's 10m pixels, causing severe mixed-pixel dilution.
   - *Look-Alike False Alarms*: Biogenic slicks (algal blooms, fish oils), suspended sediment plumes, cloud shadows, wind-sheltered zones, and ship wakes mimic oil optical signatures.

---

## 2. Input Specifications: Sentinel-2 Bands & Metadata

The intelligence module operates on the 4 primary multispectral bands enhanced by PI-RCAN, supplemented by native Sentinel-2 contextual auxiliary channels and observation geometry metadata.

| Input Element | Band / Metadata Tag | Wavelength / Type | Native GSD | Enhanced GSD | Role in Oil-Spill Pipeline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Blue Band** | **B02** | 490 nm (Vis Blue) | 10.0 m | **~3.0 m** | Water penetration, baseline oceanic background |
| **Green Band** | **B03** | 560 nm (Vis Green) | 10.0 m | **~3.0 m** | NDWI calculation, sediment / turbidity separation |
| **Red Band** | **B04** | 665 nm (Vis Red) | 10.0 m | **~3.0 m** | Emulsion scattering contrast, slick boundary tracing |
| **NIR Band** | **B08** | 842 nm (Broad NIR) | 10.0 m | **~3.0 m** | Primary oil contrast; clean water absorption baseline |
| **Red Edge 1** | **B05** | 705 nm (Context) | 20.0 m | N/A | Algal bloom / chlorophyll fluorescence discrimination |
| **SWIR 1** | **B11** | 1610 nm (Context) | 20.0 m | N/A | Cloud shadow & land/water masking verification |
| **Scene Class**| **SCL** | Classification raster | 20.0 m | N/A | Initial cloud, shadow, and water mask gating |
| **Geometry** | **SZA, SAA, VZA, VAA**| Solar/Sensor angles | Scene-level | N/A | Sunglint angle computation & illumination modeling |

---

## 3. Preprocessing Pipeline

Before segmentation, the scene undergoes a multi-stage physical preprocessing workflow:

```
[ Enhanced 4-Band Multispectral Cube (~3m) + Auxiliary S2 Metadata ]
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │ 1. Cloud & Cloud-Shadow Exclusion Gating     │
         │    - Exclude SCL classes 3, 8, 9, 10         │
         │    - Blue/SWIR thresholding + shadow projection
         └──────────────────────┬───────────────────────┘
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │ 2. Dynamic Marine Water-Body Masking         │
         │    - MNDWI = (Green - SWIR) / (Green + SWIR) │
         │    - Shoreline vector exclusion buffer (50m) │
         └──────────────────────┬───────────────────────┘
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │ 3. Radiometric & Sunglint Normalization      │
         │    - BOA Surface Reflectance in [0.0, 1.0]   │
         │    - Compute Sunglint Angle (θ_glint)        │
         └──────────────────────┬───────────────────────┘
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │ 4. Temporal Baseline Co-Registration         │
         │    - Retrieve pre-event baseline (T_-1)      │
         │    - Compute differential reflectance cube   │
         └──────────────────────┬───────────────────────┘
                                │
                                ▼
             [ Preprocessed Marine Region of Interest ]
```

### Preprocessing Stage Details:
1. **Cloud & Shadow Gating**:
   - Masks cloud pixels using SCL (classes 8, 9) and cirrus (class 10).
   - Projects cloud shadow polygons across the water surface using Solar Zenith ($SZA$) and Solar Azimuth ($SAA$) vectors, excluding ambiguous dark water zones.
2. **Dynamic Marine Water Masking**:
   - Computes Modified Normalized Difference Water Index ($\text{MNDWI}$):
     $$\text{MNDWI} = \frac{\text{B03} - \text{B11}}{\text{B03} + \text{B11} + \epsilon} > 0.0$$
   - Applies a 50m coastal shoreline buffer to exclude surf-zone breakers, intertidal mudflats, and coastal infrastructure.
3. **Sunglint Geometry Computation**:
   - Calculates the specular glint angle $\theta_{\text{glint}}$:
     $$\cos \theta_{\text{glint}} = \cos \theta_s \cos \theta_v - \sin \theta_s \sin \theta_v \cos(\phi_s - \phi_v)$$
   - When $\theta_{\text{glint}} < 20^\circ$, sunglint enhancement mode is activated; when $\theta_{\text{glint}} \ge 20^\circ$, standard absorption/emulsion contrast mode is used.
4. **Temporal Baseline Differencing**:
   - Compares the event scene ($T_{\text{event}}$) against a cloud-free historical baseline ($T_{\text{baseline}}$) acquired within 30 days prior:
     $$\Delta R(c, x, y) = R_{\text{event}}(c, x, y) - R_{\text{baseline}}(c, x, y)$$
   - Static features (submerged reefs, permanent sediment plumes) exhibit $\Delta R \approx 0$, isolating transient oil slicks.

---

## 4. Spectral & Spatial Feature Engineering

To discriminate between true petroleum oil and common marine look-alikes, the module computes a curated set of **physically justified spectral indices and spatial texture descriptors**.

```
+----------------------------------------------------------------------------------------------------+
|                                    SPECTRAL FEATURE MATRIX                                         |
+-------------------+-----------------+-----------------+--------------------------------------------+
| Target Class      | Blue / Green    | Red (B04)       | NIR (B08)       | Key Feature Indices      |
+-------------------+-----------------+-----------------+-----------------+--------------------------+
| **Clean Water**   | Low (0.03-0.08) | Very Low (<0.02)| Near Zero (<0.01)| NDWI > 0.3, NIR ≈ 0      |
| **Thin Sheen**    | Variable        | Low             | Slight Increase | Low contrast, FI > 0     |
| **Emulsified Oil**| Moderate        | High (0.08-0.18)| High (0.12-0.25)| SOSI > 0, NIR/Red > 1.2  |
| **Sediment Plume**| High (0.15-0.30)| High (0.15-0.25)| Moderate (0.05) | Red/NIR > 2.5, FAI ≈ 0   |
| **Algal Bloom**   | Moderate        | Low (Chlorophyll| Very High (FOLI)| FAI > 0.02, NDVI > 0.25  |
| **Cloud Shadow**  | Extremely Low   | Extremely Low   | Extremely Low   | Broad uniform drop       |
| **Vessel / Ship** | High (>0.40)    | High (>0.40)    | High (>0.40)    | Point anomaly, GLCM high |
+-------------------+-----------------+-----------------+-----------------+--------------------------+
```

### Scientifically Validated Indices:
1. **Normalized Difference Water Index (NDWI)**:
   $$\text{NDWI} = \frac{\text{B03} - \text{B08}}{\text{B03} + \text{B08} + \epsilon}$$
   - Separates open water ($\text{NDWI} > 0.0$) from land and dense vegetation.
2. **Surface Oil Spill Index (SOSI)**:
   $$\text{SOSI} = \frac{\text{B08} - \text{B04}}{\text{B08} + \text{B04} + \epsilon}$$
   - Identifies emulsified oil patches where NIR volume backscattering exceeds Red absorption.
3. **Floating Algae Index (FAI)**:
   $$\text{FAI} = \text{B08} - \left( \text{B04} + (\text{B11} - \text{B04}) \frac{\lambda_{\text{B08}} - \lambda_{\text{B04}}}{\lambda_{\text{B11}} - \lambda_{\text{B04}}} \right)$$
   - Distinguishes photosynthetic organisms (Sargassum, algae) from non-photosynthetic hydrocarbon slicks. True oil exhibits $\text{FAI} \approx 0$ or negative baseline deviation, whereas algae blooms exhibit strong positive $\text{FAI} > 0.02$.
4. **Fluorescence / Contrast Index (FI)**:
   $$\text{FI} = \frac{\text{B04} + \text{B03}}{\text{B02} + \epsilon}$$
   - Captures anomalous visible spectral tilt caused by surface hydrocarbons.
5. **Spatial Texture & Boundary Gradients**:
   - Gray-Level Co-occurrence Matrix (GLCM) Angular Second Moment (energy) and Contrast computed over $5 \times 5$ neighborhoods to capture slick ribbon boundaries.

---

## 5. Detection Strategy: Comparison & Recommended Architecture

We evaluate four candidate architectures for the oil-spill segmentation engine:

```
+----------------------------------------------------------------------------------------------------+
|                                    DETECTION STRATEGY COMPARISON                                   |
+-------------------+-------------------+-------------------+--------------------+-------------------+
| Criteria          | A. Rule-Based     | B. Classical ML   | C. CNN             | D. Transformer    |
|                   | Index Gating      | (Random Forest)   | (U-Net / ResNet)   | (SegFormer)       |
+-------------------+-------------------+-------------------+--------------------+-------------------+
| **Spatial Context | None (Pixel-wise) | Low (Handcrafted) | **High** (Learned  | **Very High**     |
| Modeling**        |                   |                   | multi-scale)       | (Self-attention)  |
| **Data Efficiency**| Needs 0 training | Moderate (~1000   | Moderate (~100-300 | Very Data-Hungry  |
|                   | samples           | pixels)           | scene patches)     | (>1000 scenes)    |
| **Filament Trace  | Poor (Noisy       | Fair (Fragmented  | **Excellent**      | Excellent         |
| Continuity**      | scatter)          | blobs)            | (Smooth ribbons)   | (Smooth ribbons)  |
| **Inference Speed**| Real-time (<10ms)| Fast (<50ms)      | Fast (~80ms/tile)  | Moderate (~250ms) |
| **Risk of         | Low               | Low               | Moderate           | High (Overfits    |
| Overfitting**     |                   |                   | (Manageable)       | on small data)    |
+-------------------+-------------------+-------------------+--------------------+-------------------+
```

### Recommendation for Research Prototype:
**Hybrid Two-Stage Architecture (Physics Gating + Lightweight Semantic CNN)**.

```
[ Preprocessed Enhanced Tensor: 8 Channels (4 Spectral + 4 Physical Indices) ]
                                │
                                ▼
       ┌─────────────────────────────────────────────────┐
       │ Stage 1: Physical Anomaly Gating & Pre-Filter   │
       │ - Exclude land, clouds, FAI algae, and sediment │
       │ - Yields Candidate Oil Mask (M_cand)            │
       └────────────────────────┬────────────────────────┘
                                │
                                ▼
       ┌─────────────────────────────────────────────────┐
       │ Stage 2: U-Net Segmentation with ResNet-34      │
       │ - Encoder-Decoder with Skip Connections         │
       │ - Predicts slick probability: P_oil(x, y) ∈ [0, 1]
       └────────────────────────┬────────────────────────┘
                                │
                                ▼
       ┌─────────────────────────────────────────────────┐
       │ Stage 3: Anti-Hallucination Verification Layer  │
       │ - Dual-scale check against native S2 10m bands  │
       │ - Uncertainty gating via PI-RCAN variance map   │
       └────────────────────────┬────────────────────────┘
                                │
                                ▼
                   [ Confirmed Oil-Spill Mask ]
```

### Why this approach is optimal:
- **Physical Safety**: Stage 1 guarantees that obvious physical look-alikes (algae, sediment) are rejected before the neural network evaluates spatial features.
- **Structural Integrity**: U-Net skip connections preserve fine sub-pixel filament geometry without requiring massive transformer training datasets.
- **Explainability**: Detections can be traced to both physical spectral indices and learned spatial morphology.

---

## 6. Temporal Change Detection & Baseline Subtraction

Optical characteristics of coastal water bodies vary with bathymetry, seasonal currents, and permanent runoff. Single-date imagery cannot always differentiate permanent dark water from fresh oil slicks.

```
+─────────────────────────────────────────────────────────────────────────────+
|                         TEMPORAL CO-VERIFICATION                            |
+─────────────────────────────────────────────────────────────────────────────+
|                                                                             |
|   Baseline Pass (T_-1: 1 Feb 2026)      Event Pass (T_0: 11 Feb 2026)       |
|   [ Clean Marine Baseline ]             [ Potential Slick Detected ]        |
|             \                                     /                         |
|              \                                   /                          |
|               ▼                                 ▼                           |
|        +───────────────────────────────────────────────+                    |
|        | Dynamic Temporal Difference Kernel:           |                    |
|        | ΔNIR(x, y) = NIR_event(x, y) - NIR_base(x, y) |                    |
|        | ΔNDWI(x, y) = NDWI_event(x, y) - NDWI_base(x) |                    |
|        +───────────────────────┬───────────────────────+                    |
|                                │                                            |
|                                ▼                                            |
|        [ Transient Hydrocarbon Signal Isolated (Static Noise Eliminated) ]   |
|                                                                             |
+─────────────────────────────────────────────────────────────────────────────+
```

1. **Temporal Anomaly Score ($S_{\text{temporal}}$)**:
   $$S_{\text{temporal}}(x, y) = \frac{|\Delta \text{B04}(x, y)| + |\Delta \text{B08}(x, y)|}{\sigma_{\text{baseline}}(x, y) + \epsilon}$$
2. **False Positive Suppression**:
   - If a dark anomaly was present in the historical pass ($T_{-1}$) with identical shape, it is flagged as **Bathymetric / Static Feature** and suppressed.
   - If the anomaly is transient ($S_{\text{temporal}} > 3.0$), it is confirmed as an active surface event.

---

## 7. Spatial Enhancement Integration & Anti-Hallucination Safeguards

```
+─────────────────────────────────────────────────────────────────────────────+
|                   ANTI-HALLUCINATION VERIFICATION PIPELINE                  |
+─────────────────────────────────────────────────────────────────────────────+
|                                                                             |
|  Enhanced Detection (~3m)              Raw S2 Observation (10m)             |
|  [ Candidate Slick Polygon ]           [ Native 10m Reflectance ]           |
|             │                                     │                         |
|             ├─────────────────┬───────────────────┤                         |
|             ▼                 ▼                   ▼                         |
|      [ Check 1: ]       [ Check 2: ]        [ Check 3: ]                    |
|      Dual-Scale         PI-RCAN             Spectral Angle                  |
|      Co-Verification    Uncertainty Gate    Consistency (SAM)               |
|      - Must show S2     - Drop pixels       - SAM(Enh, S2)                  |
|        10m anomaly        where σ^2 > 90%     must be < 3.0°                |
|             │                 │                   │                         |
|             └─────────────────┼───────────────────┘                         |
|                               ▼                                             |
|               [ Confirmed Real Slick Detection ]                            |
|                                                                             |
+─────────────────────────────────────────────────────────────────────────────+
```

> [!IMPORTANT]
> **MANDATORY ANTI-HALLUCINATION SAFEGUARDS**:
> 1. **Dual-Scale Co-Verification**:
>    Every connected component identified as an oil slick on the enhanced ~3m grid is down-projected to the original 10m Sentinel-2 grid. If the enclosed area exhibits zero detectable radiometric anomaly on native Sentinel-2 bands ($|\Delta R_{\text{S2}}| < 1.5 \times \text{NE}\Delta R$), the candidate polygon is rejected as a super-resolution artifact.
> 2. **Uncertainty-Gated Boundary Pruning**:
>    The PI-RCAN aleatoric uncertainty layer $\sigma^2(x, y, c)$ provides per-pixel variance. Pixels with uncertainty in the top 10th percentile ($\sigma^2 > \tau_{\text{unc}}$) are flagged as "Uncertain Sub-Pixel Detail" and excluded from confirmed area totals.
> 3. **Spectral Vector Invariance**:
>    $$\text{SAM}(\mathbf{v}_{\text{enhanced}}(x, y), \mathbf{v}_{\text{S2}}(x, y)) \le 3.0^\circ$$
>    If super-resolution altered the multi-band spectral ratio by $> 3^\circ$, the pixel is discarded.

---

## 8. Output Deliverables & GIS Intelligence Schema

The module outputs standardized geospatial intelligence products ready for operational maritime command systems.

```
+─────────────────────────────────────────────────────────────────────────────+
|                         FINAL DELIVERABLE PRODUCTS                          |
+─────────────────────────────────────────────────────────────────────────────+
|                                                                             |
|  1. Oil-Spill Classified Mask GeoTIFF (outputs/oil_spill/spill_mask.tiff)   |
|     - Pixel Values: 0=Clean Water, 1=Thin Sheen, 2=Thick Emulsion, 255=Mask |
|                                                                             |
|  2. Confidence & Uncertainty GeoTIFF (outputs/oil_spill/confidence.tiff)   |
|     - Band 1: Detection Probability [0.0, 1.0]                              |
|     - Band 2: PI-RCAN Epistemic/Aleatoric Uncertainty                       |
|                                                                             |
|  3. Vector Slick Polygons GeoJSON (outputs/oil_spill/spill_polygons.geojson)|
|     - Attributed vector boundaries in EPSG:32643                            |
|                                                                             |
|  4. Incident Intelligence Summary JSON (outputs/oil_spill/summary.json)     |
|                                                                             |
+─────────────────────────────────────────────────────────────────────────────+
```

### Geospatial Metadata & Attribute Schema (`summary.json`):
```json
{
  "incident_id": "OIL_SPILL_20260211_054815_EC01",
  "detection_timestamp": "2026-09-01T11:45:00+05:30",
  "satellite_overpass": {
    "sensor": "Sentinel-2B MSI (Enhanced via PI-RCAN)",
    "acquisition_time": "2026-02-11T05:08:39Z",
    "native_resolution_m": 10.0,
    "enhanced_resolution_m": 3.33,
    "crs": "EPSG:32643"
  },
  "slick_metrics": {
    "total_slick_area_km2": 4.82,
    "area_confidence_interval_95": [4.35, 5.29],
    "thick_emulsion_area_km2": 1.15,
    "thin_sheen_area_km2": 3.67,
    "centroid_coordinates": {
      "latitude": 12.8542,
      "longitude": 77.6891
    },
    "major_axis_length_km": 5.40,
    "dispersion_heading_deg": 142.5
  },
  "safeguard_audit": {
    "dual_scale_verification": "PASSED",
    "mean_uncertainty_score": 0.12,
    "hallucination_risk_flag": "LOW"
  }
}
```

---

## 9. Validation Protocol & Metrics

To establish true detection efficacy, model predictions are evaluated against certified ground-truth reference datasets (e.g. coincident SAR observations or aerial surveillance).

```
+─────────────────────────────────────────────────────────────────────────────+
|                         VALIDATION METRIC SUITE                             |
+─────────────────────────────────────────────────────────────────────────────+
|                                                                             |
|  1. Segmentation Overlap:     IoU = TP / (TP + FP + FN)                     |
|  2. Precision (Purity):       Precision = TP / (TP + FP)                    |
|  3. Recall (Completeness):    Recall = TP / (TP + FN)                       |
|  4. F1-Score:                 F1 = 2 * (Precision * Recall) / (P + R)       |
|  5. Boundary Accuracy:        Mean Hausdorff Distance (MHD in meters)       |
|  6. Area Quantification:      Area Error = |Area_pred - Area_gt| / Area_gt  |
|                                                                             |
+─────────────────────────────────────────────────────────────────────────────+
```

### Benchmark Acceptance Targets:
- **Intersection over Union (IoU)**: $\ge 0.75$ on thick emulsion; $\ge 0.65$ on total slick area.
- **Precision**: $\ge 0.85$ (strictly minimizing false alarms over clean water).
- **Recall**: $\ge 0.80$ (detecting at least 80% of confirmed slick area).
- **Boundary Precision**: Mean boundary error $< 1.5$ pixels ($< 5.0\text{ m}$).

---

## 10. False-Positive Handling & Look-Alike Mitigation Matrix

```
+----------------------------------------------------------------------------------------------------+
|                                    LOOK-ALIKE MITIGATION MATRIX                                    |
+-------------------+------------------------------+-------------------------------------------------+
| Look-Alike Entity | Optical / Spatial Signature  | Technical Mitigation Safeguard                  |
+-------------------+------------------------------+-------------------------------------------------+
| **Suspended       | High Red reflectance with    | **SWIR / Green Ratio**: Turbid water exhibits   |
| Sediment Plumes** | broad coastal river plume    | elevated B11/B03 ratio; FAI ≈ 0.                |
|                   | morphology.                  | Reject when Red > 0.15 and SOSI < 0.            |
+-------------------+------------------------------+-------------------------------------------------+
| **Intense         | Symmetrical specular flash   | **Glint Angle Model**: Compute θ_glint;         |
| Sunglint**        | across entire scene center.  | exclude or apply adaptive thresholding in       |
|                   |                              | high-glint zones (θ_glint < 15°).               |
+-------------------+------------------------------+-------------------------------------------------+
| **Cloud Shadows   | Sharp drop in reflectance    | **Geometric Cloud Projection**: Ray-trace cloud |
| on Water**        | matching cloud shapes.       | polygons along solar vector; mask shadow zone.  |
+-------------------+------------------------------+-------------------------------------------------+
| **Biogenic Algal  | High NIR reflection due to   | **FAI Gating**: Algae exhibits FAI > 0.02 and   |
| Blooms**          | chlorophyll fluorescence.    | NDVI > 0.25; true oil shows FAI ≤ 0.            |
+-------------------+------------------------------+-------------------------------------------------+
| **Dark Calm Water | Uniform low reflectance in   | **Temporal Baseline Differencing**: Anomaly must|
| (Wind Shadows)**  | lee of islands/coasts.       | show ΔR > 3σ over historical baseline.          |
+-------------------+------------------------------+-------------------------------------------------+
| **Ships & Large   | High-reflectance point       | **Morphological & AIS Filtering**: Exclude point|
| Vessels**         | targets with V-shaped wake.  | anomalies matching known vessel tracks.         |
+-------------------+------------------------------+-------------------------------------------------+
```

---

## 11. Interactive Demonstration UI Design

The prototype user interface is designed as a modern, dark-themed GIS dashboard providing synchronized multi-pane inspection:

```
+─────────────────────────────────────────────────────────────────────────────────────────────+
| SATELLITE OIL INTELLIGENCE DASHBOARD (SIH26142)                         [ 11-FEB-2026 05:08 ]|
+─────────────────────────────────────────────────────────────────────────────────────────────+
|                                                                                             |
|   [ VIEWPORT 1: SPLIT-SCREEN COMPARISON ]           [ VIEWPORT 2: GIS INTELLIGENCE HUD ]   |
|   +---------------------------------------+         +-------------------------------------+ |
|   |  Sentinel-2 10m  |  PI-RCAN 3m + Mask |         | INCIDENT METRICS:                   | |
|   |  (Original RGB)  |  (Enhanced Slick)  |         | - Spill Area: 4.82 km² (±0.4)       | |
|   |                  |                    |         | - Emulsion Core: 1.15 km²           | |
|   |       [=========||==============]       |         | - Centroid: 12.8542°N, 77.6891°E    | |
|   |                  |                    |         | - Confidence: 89.4% (HIGH)          | |
|   |                  |                    |         |                                     | |
|   |                  |                    |         | LAYER CONTROLS:                     | |
|   |                  |                    |         | [X] Oil Spill Binary Mask           | |
|   |                  |                    |         | [X] PI-RCAN Uncertainty Heatmap     | |
|   |                  |                    |         | [ ] FAI Algae Discrimination Layer  | |
|   +---------------------------------------+         +-------------------------------------+ |
|                                                                                             |
|   [ SPECTRAL PROFILE INSPECTOR ]                    [ EXPORT ACTIONS ]                      |
|   +───────────────────────────────────────+         +─────────────────────────────────────+ |
|   | Reflectance (%)                       |         | [ Download GeoTIFF Classification ] | |
|   | 20 |       * (Emulsion)               |         | [ Export GeoJSON Slick Boundary ]   | |
|   | 10 |   *-------* (Thin Sheen)         |         | [ Generate Incident PDF Report ]    | |
|   |  0 +---*-------*-------* (Clean Water)|         +─────────────────────────────────────+ |
|   |       B02     B04     B08             |                                                 |
|   +───────────────────────────────────────+                                                 |
+─────────────────────────────────────────────────────────────────────────────────────────────+
```

---

## 12. Legitimate Dataset Sourcing & Requirements

To train and validate the downstream oil-spill segmentation model in future phases, the following **authoritative, publicly available remote sensing datasets** are specified:

```
+----------------------------------------------------------------------------------------------------+
|                                  AUTHENTIC BENCHMARK DATASETS                                      |
+-------------------+--------------------+--------------------+--------------------------------------+
| Dataset Name      | Sensor Modality    | Spatial Resolution | Relevance / Application              |
+-------------------+--------------------+--------------------+--------------------------------------+
| **NOAA Deepwater  | Sentinel-2 / MODIS | 10m - 250m         | Ground-truth optical surface oil     |
| Horizon Archive** | / Landsat-7        |                    | signatures and emulsion calibration. |
+-------------------+--------------------+--------------------+--------------------------------------+
| **Mauritius MV    | Sentinel-2 L2A /   | 10m / 3m           | Coincident 10m Sentinel-2 and high-  |
| Wakashio (2020)** | PlanetScope        |                    | res optical oil spill in coral reef. |
+-------------------+--------------------+--------------------+--------------------------------------+
| **Peru Repsol     | Sentinel-2 L2A /   | 10m / 3m           | Coastal Pacific crude oil slick with |
| Spill (2022)**    | PlanetScope        |                    | heavy coastal sediment interaction.  |
+-------------------+--------------------+--------------------+--------------------------------------+
| **EMSA CleanSeaNet| Coincident S1/S2   | 10m - 20m          | European maritime certified spill    |
| Validation Data** | Sentinel pairs     |                    | detection polygon ground truth.      |
+-------------------+--------------------+--------------------+--------------------------------------+
```

> [!NOTE]
> No datasets will be downloaded or fabricated in this phase. Data ingestion will occur strictly during downstream training execution.

---

## 13. System Implementation Roadmap

```
+----------------------------------------------------------------------------------------------------+
|                                    PROJECT EXECUTION STATUS                                        |
+------------------------------------+-----------------------+---------------------------------------+
| Component                          | Status                | Notes / Blockers                      |
+------------------------------------+-----------------------+---------------------------------------+
| **Satellite Enhancer Architecture**| **COMPLETE**          | Fully specified in final_system_...   |
| **Experiment 5 Specification**     | **COMPLETE**          | Fully specified in exp5_spec.md       |
| **Physical Degradation Engine**    | **PARTIALLY READY**   | Implemented; needs scene parameters   |
| **Real PlanetScope Data Ingestion**| **BLOCKED**           | Awaiting authorized Scene 20260211... |
| **PI-RCAN Super-Resolution Model** | **PROPOSED**          | Architecture locked; training blocked |
| **Oil-Spill Intelligence Design**  | **DESIGN COMPLETE**   | Documented in this file               |
| **Oil-Spill Segmentation Training**| **PROPOSED**          | Blocked on real reference & enhancer  |
+------------------------------------+-----------------------+---------------------------------------+
```

### Next Implementation Steps:
1. **Acquisition & Ingestion**: Obtain authorized real reference PlanetScope scene `20260211_054815_64_254a`.
2. **Experiment 5 Execution**: Train and validate PI-RCAN with physical degradation and composite loss (`models/experiment4_real/`).
3. **Oil-Spill Preprocessing Implementation**: Implement water masking (`MNDWI`), cloud-shadow exclusion, and temporal baseline difference kernels in `src/oil_spill/`.
4. **Segmentation Engine & Safeguards**: Build the hybrid U-Net segmentation pipeline with dual-scale co-verification and uncertainty gating.
5. **Interactive GIS Dashboard**: Deploy the end-to-end multi-pane visualization UI.
