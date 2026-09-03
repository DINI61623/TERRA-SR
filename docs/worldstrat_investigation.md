# Technical Investigation Report: WorldStrat Dataset for Cross-Sensor SRM

**Project Code**: SIH26142  
**Subject**: Evaluation of the WorldStrat dataset for building a cross-sensor Super-Resolution Mapping (SRM) training pipeline (Sentinel-2 10m to Airbus SPOT 1.5m).

---

## 1. Dataset Structure & Modalities

The WorldStrat dataset is a curated benchmark pairing high-resolution (HR) satellite imagery with temporally matched low-resolution (LR) satellite time series.

*   **Low-Resolution (LR) Modality**: Sentinel-2 Level-2A (Bottom-of-Atmosphere reflectance) imagery.
*   **High-Resolution (HR) Modality**: Airbus SPOT 6 and SPOT 7 imagery.
*   **Temporal Matching**: For each high-resolution target image, the dataset provides **16 temporally matched Sentinel-2 revisits** (8 before and 8 after the acquisition date of the HR image) to support multi-frame and single-frame super-resolution mapping.

---

## 2. Spatial and Spectral Specifications

### Spatial Resolution
*   **LR (Sentinel-2)**: 10 m/pixel (Bands B02, B03, B04, B08).
*   **HR (Airbus SPOT 6/7)**: 1.5 m/pixel (pansharpened multi-spectral).

### Spectral Compatibility
SPOT 6/7 captures four spectral bands, which align perfectly with Sentinel-2’s 10m bands:
*   **Blue**: 450 – 520 nm (Sentinel-2 B02: 490 nm)
*   **Green**: 530 – 590 nm (Sentinel-2 B03: 560 nm)
*   **Red**: 625 – 695 nm (Sentinel-2 B04: 665 nm)
*   **Near-Infrared (NIR)**: 760 – 890 nm (Sentinel-2 B08: 842 nm)

### Image Dimensions & Bit Depth
*   **HR imagery**: Stored as 12-bit radiometry, typically packed in float32 reflectance ranges or normalized values.
*   **Spatial Dimensions**: The dataset is cropped into aligned patches (typically $128 \times 128$ pixels in LR corresponding to $853 \times 853$ pixels in HR).

---

## 3. Geographic Metadata & Splits

*   **Geographic Coverage**: Spans approximately 10,000 km² across **4,000 distinct Areas of Interest (AOIs)** globally.
*   **India Coverage**: India contains several stratified points of interest representing urban agricultural and forest mixtures (e.g. regions in northern India), but does **not** contain the Bengaluru/Electronic City AOI.
*   **Spatial Separation/Splits**: The dataset includes an official partition file (`stratified_train_val_test_split.csv`) defining train, validation, and test sets. These splits are separated geographically by distinct AOIs to ensure zero spatial leakage during model training.

---

## 4. Licensing & Use Considerations

The dataset components have split licensing:
*   **Sentinel-2 Imagery & Labels**: Licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)** (permitted for commercial/non-commercial use).
*   **High-Resolution Airbus SPOT Imagery**: Licensed under **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.
*   *Hackathon/SIH Compliance*: Permitted for academic, research, and non-commercial prototype presentations.

---

## 5. Download Constraints & Sample Access

### Zenodo Archives Size
*   `hr_dataset.tar.gz` (Airbus SPOT HR): **41.5 GB**
*   `lr_dataset_l2a.tar.gz` (Sentinel-2 L2A LR): **26.8 GB**
*   `metadata.csv` (Locations mapping): **14.6 MB**
*   `stratified_train_val_test_split.csv` (Official splits): **282 kB**

### Kaggle Archive Size
*   WorldStrat "Core" Dataset: **~53 GB** (zipped).

### Technical Download Constraint
> [!WARNING]
> **Zenodo and Kaggle distribute the dataset exclusively as massive monolithic archives (40GB+ and 53GB+).** 
> The repositories **do not** allow downloading individual patches, single scenes, or small regional subsets directly via standard HTTP requests. To get even a single paired LR-HR image patch from WorldStrat, we would have to download and decompress the entire **68.3 GB** Zenodo dataset.

---

## 6. Potential Alignment & Preprocessing Issues

1.  **Sensor Coregistration Error**: Although pre-aligned by the authors, cross-sensor data (SPOT vs. Sentinel-2) exhibits minor sub-pixel shifts due to differing orbital paths and viewing angles.
2.  **Spectral Radiance Differences**: SPOT 6/7 and Sentinel-2 sensors have differing Spectral Response Functions (SRF). A direct pixel-to-pixel training can cause the model to learn the radiometric transformation rather than the spatial upscaling.
3.  **Temporal Differences**: S2 images are captured at different times than the SPOT reference. Moving clouds, shadowing, and seasonal vegetation changes can introduce noise during training.

---

## 7. Recommended Preprocessing Pipeline

If using cross-sensor data, the pipeline must:
1.  **Radiometric Normalization**: Apply histogram matching or regression to normalize the radiometry of SPOT bands to Sentinel-2 spectral ranges.
2.  **Sub-pixel Registration**: Apply Phase Correlation or optical flow to align the SPOT HR image with the Sentinel-2 LR image to correct sub-pixel orthorectification offsets.
3.  **Spectral Calibration**: Convert both datasets to Surface Reflectance (L2A equivalent).

---

## 8. Final Recommendation

### **DO NOT PROCEED**

### Rationale:
1.  **Monolithic Storage Requirements**: Attempting to download the 68.3 GB monolithic tarballs in our current local workspace would exceed standard network/disk sandbox constraints, risking environment instability.
2.  **Bengaluru AOI Absence**: The WorldStrat dataset does not cover our project target area (Bengaluru/Electronic City).
3.  **More Efficient Alternative**: Since we need a cross-sensor high-resolution reference for validation in Bengaluru, the best alternative is to apply for the **Planet Education & Research Program** (free for university students and developers). This program allows us to download individual **3m PlanetScope scenes** of the exact Electronic City coordinates (size ~20-50MB per scene) directly using Planet's REST API. This is legal, geographically precise, and highly efficient.
