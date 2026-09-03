# High-Resolution Reference Imagery Options for Electronic City AOI (Experiment 4)

This document evaluates legitimate, legal, and scientifically valid options for acquiring a high-resolution reference dataset covering the Electronic City, Bengaluru Area of Interest (AOI) centered at **(12.85° N, 77.685° E)** for the target date of **11 February 2026** (or within a ±10-day temporal window). 

The goal of this evaluation is to find a dataset that meets the Experiment 4 requirements: **spatial resolution < 4m, multispectral (specifically containing Blue, Green, Red, and Near-Infrared bands mapped to Sentinel-2 B02, B03, B04, B08), genuine raw reflectance data (not processed basemaps), and legally compliant.**

---

## Evaluation of Alternatives

### 1. Planet Education & Research (E&R) Program
*   **Sensor**: PlanetScope (Dove Classic, Dove-R, or SuperDove)
*   **Spatial Resolution**: ~3.0 m (orthorectified, resampled from 3.7–4.1m nadir resolution)
*   **Spectral Bands**: 4-band multispectral (Blue, Green, Red, NIR) or 8-band multispectral (Coastal Blue, Blue, Green, Yellow, Red, Red Edge, NIR) on newer SuperDove assets. Mapped directly to S2 B02, B03, B04, and B08.
*   **Geographic Coverage**: Global coverage, fully covering the Electronic City, Bengaluru AOI.
*   **Acquisition Date Availability**: Daily revisit rate globally. The archive contains multiple cloud-free passes on or very close to **11 February 2026** (including the specific scene `20260211_054815_64_254a`).
*   **Download Availability**: Programmatic downloads are available via the Planet Orders API, Planet Python SDK, or Planet Explorer GUI. The individual Basic Plan grants a quota of **5,000 sq km per month** (or 3,000 sq km per month on some academic tiers), which easily covers our small local crop (~20–50 MB).
*   **Licensing/Academic-Use Restrictions**: Non-commercial academic research and educational use only. Eligible for university students, researchers, and faculty. Raw downloaded imagery cannot be shared publicly (only derivative outputs, maps, or statistics are permissible in publications).
*   **Experiment 4 Compatibility**: **Yes, perfectly.** It meets the spatial, spectral, temporal, and spatial georeferencing requirements.

---

### 2. Planet Insights Platform Free Trial / Sandbox Datasets
*   **Sensor**: PlanetScope (Dove / SuperDove)
*   **Spatial Resolution**: ~3.0 m
*   **Spectral Bands**: 4-band / 8-band multispectral (RGB + NIR)
*   **Geographic Coverage**: Restricted strictly to pre-defined global sandbox regions and demo sites curated by Planet. It does **not** cover the Electronic City, Bengaluru AOI.
*   **Acquisition Date Availability**: Predefined historical dates for the sandbox sites. Target date of **11 February 2026** is not available for custom regions.
*   **Download Availability**: Users get a 30-day trial with 30,000 processing units. However, custom AOI search and download are disabled for the sandbox dataset, meaning you cannot download arbitrary coordinates.
*   **Licensing/Academic-Use Restrictions**: Non-commercial evaluation license (Creative Commons Attribution-NonCommercial, CC BY-NC 4.0).
*   **Experiment 4 Compatibility**: **No.** It does not cover our geographic area of interest or target dates.

---

### 3. ISRO Bhoonidhi Portal / Bhuvan (Cartosat-2 Multispectral)
*   **Sensor**: High-Resolution Multispectral Radiometer (HRMX) onboard the Cartosat-2 series (e.g., Cartosat-2E)
*   **Spatial Resolution**: 4.0 m (Multispectral), 1.0 m (Panchromatic)
*   **Spectral Bands**: 4-band multispectral (Blue: 450-520 nm, Green: 520-590 nm, Red: 620-680 nm, NIR: 770-860 nm)
*   **Geographic Coverage**: National coverage over India, covering Bengaluru/Electronic City.
*   **Acquisition Date Availability**: Revisit rate of 4–5 days. Historical archives must be queried on Bhoonidhi to check for passes close to February 11, 2026.
*   **Download Availability**: High-resolution digital products (under 4m) are categorized as "Priced" on the Bhoonidhi portal. On Bhuvan, high-resolution data is hosted for **web-map visualization only** and cannot be downloaded as raw GeoTIFF files for free. Free downloads are restricted to lower-resolution sensors (e.g., LISS-III, AWiFS) or DEMs.
*   **Licensing/Academic-Use Restrictions**: Subject to the ISRO Remote Sensing Data Policy (RSDP). Academic/research use of high-resolution data requires institutional registration, project approval, and payment, or formal administrative clearance. Redistribution is prohibited.
*   **Experiment 4 Compatibility**: **No.** Raw multispectral digital data at <4m resolution is not downloadable for free.

---

### 4. ESA Third-Party Missions (TPM) - Airbus SPOT 6/7 Archive
*   **Sensor**: NAOMI (New Astrosat Optical Modular Instrument) onboard SPOT 6 and SPOT 7
*   **Spatial Resolution**: 1.5 m (Pansharpened multispectral or panchromatic), 6.0 m (Multispectral)
*   **Spectral Bands**: 4 multispectral bands (Blue: 450-520 nm, Green: 530-590 nm, Red: 625-695 nm, NIR: 760-890 nm) and 1 Panchromatic band.
*   **Geographic Coverage**: Global coverage, including Bengaluru.
*   **Acquisition Date Availability**: High historical archive coverage. Note that SPOT 7 ceased operations on March 17, 2023, so only SPOT 6 is active for the **11 February 2026** target date. Revisit availability depends on scheduling and tasking.
*   **Download Availability**: Requires submitting a formal scientific project proposal via the ESA Earthnet Online portal. Once approved, the data is delivered for free within allocated project quotas.
*   **Licensing/Academic-Use Restrictions**: Free for approved scientific research and application development. Strictly non-commercial.
*   **Experiment 4 Compatibility**: **Yes, but with caveats.** The administrative proposal review process takes several weeks/months, and SPOT 7's retirement reduces the probability of finding a tasked scene on the exact date.

---

### 5. ESA Third-Party Missions (TPM) - PlanetScope Archive
*   **Sensor**: PlanetScope (Dove Classic / SuperDove)
*   **Spatial Resolution**: ~3.0 m
*   **Spectral Bands**: 4-band / 8-band multispectral (RGB + NIR)
*   **Geographic Coverage**: Global coverage, including Bengaluru.
*   **Acquisition Date Availability**: Daily revisit rate; archived scenes exist for **11 February 2026**.
*   **Download Availability**: Requires submitting a formal project proposal to the ESA Earthnet Online portal.
*   **Licensing/Academic-Use Restrictions**: Non-commercial scientific research. Restricted to researchers affiliated with institutions in ESA Member States, EU Member States, or China (Dragon program).
*   **Experiment 4 Compatibility**: **Yes, but with eligibility and timeline constraints.** Only available to researchers within ESA/EU member states (or partners), and the project proposal review takes weeks, which is slower than applying directly to the Planet E&R Program.

---

### 6. NASA Commercial Smallsat Data Acquisition (CSDA) Program
*   **Sensor**: PlanetScope (Dove Classic / SuperDove) or Maxar (WorldView series)
*   **Spatial Resolution**: ~3.0 m (PlanetScope) or sub-meter (Maxar)
*   **Spectral Bands**: Multispectral (RGB + NIR)
*   **Geographic Coverage**: Global coverage.
*   **Acquisition Date Availability**: Daily / high frequency; covers February 11, 2026.
*   **Download Availability**: Programmatic download via NASA's SDX (Satellite Data Explorer) after authorization. There is a standard 30-day latency period for PlanetScope data.
*   **Licensing/Academic-Use Restrictions**: Strictly restricted to U.S. government civil servants, federal contractors, or researchers funded by U.S. federal agencies (such as NASA or NSF grants). Non-commercial use only.
*   **Experiment 4 Compatibility**: **No.** The research team is not affiliated with a U.S. federal agency or funded by a U.S. grant (NASA/NSF), making us ineligible.

---

### 7. WorldStrat Dataset
*   **Sensor**: Airbus SPOT 6 / SPOT 7 (paired with Sentinel-2)
*   **Spatial Resolution**: 1.5 m (Pansharpened multispectral)
*   **Spectral Bands**: 4-band multispectral (Blue, Green, Red, NIR)
*   **Geographic Coverage**: Spans 4,000 distinct AOIs globally (~10,000 km²). While it covers stratified sites in India, it does **not** cover the Electronic City, Bengaluru AOI.
*   **Acquisition Date Availability**: Multi-temporal pairs across historical dates, but does not contain data for our target date of **11 February 2026** in Bengaluru.
*   **Download Availability**: Openly available on Zenodo and Kaggle, but distributed **exclusively as massive monolithic archives** (41.5 GB HR dataset / 26.8 GB LR dataset). Individual tile/coordinate extraction is not supported.
*   **Licensing/Academic-Use Restrictions**: CC BY-NC 4.0 for Airbus SPOT imagery, CC BY 4.0 for Sentinel-2 data. Permitted for academic research and non-commercial prototyping.
*   **Experiment 4 Compatibility**: **No.** It does not cover our target AOI (Electronic City, Bengaluru) and cannot be downloaded selectively.

---

### 8. Maxar Open Data Program
*   **Sensor**: WorldView-1/2/3/4 or GeoEye-1
*   **Spatial Resolution**: Sub-meter (typically 0.3m – 0.5m)
*   **Spectral Bands**: Multispectral (RGB + NIR)
*   **Geographic Coverage**: Event-driven; restricted to active regions impacted by major natural disasters (e.g., floods, earthquakes, volcanic eruptions, humanitarian crises).
*   **Acquisition Date Availability**: Pre-event and post-event passes only. A scene for February 11, 2026, in Bengaluru is only available if a major disaster was declared there at that time.
*   **Download Availability**: Free download as GeoTIFF for active disaster response areas.
*   **Licensing/Academic-Use Restrictions**: Free for humanitarian, disaster response, and non-commercial research use.
*   **Experiment 4 Compatibility**: **No.** Electronic City, Bengaluru is not an active disaster zone in the Maxar Open Data Program catalog for February 11, 2026.

---

### 9. OpenEarthMap Dataset
*   **Sensor**: Various aerial and high-resolution satellite sensors
*   **Spatial Resolution**: 0.25m – 0.5m
*   **Spectral Bands**: **RGB only (3 bands)**. Completely lacks the Near-Infrared (NIR) band.
*   **Geographic Coverage**: 97 regions across 44 countries globally. Lacks specific coverage for the Electronic City, Bengaluru AOI.
*   **Acquisition Date Availability**: Pre-processed historical images, does not contain our target date of February 11, 2026.
*   **Download Availability**: Openly downloadable in patches.
*   **Licensing/Academic-Use Restrictions**: Creative Commons license for non-commercial research.
*   **Experiment 4 Compatibility**: **No.** It lacks the NIR band (which violates the Sentinel-2 B08 mapping contract) and does not cover the target AOI.

---

## Comparison Matrix

| Option | Sensor | Spatial Resolution | Bands | Covers Electronic City? | Target Date (11 Feb 2026)? | Download Availability (Free)? | Eligible? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Planet E&R** | PlanetScope | ~3.0m | RGB + NIR | **Yes** | **Yes** | **Yes** (5k sq km/mo) | **Yes** (University email) |
| **2. Planet Trial** | PlanetScope | ~3.0m | RGB + NIR | No | No | No (Sandbox only) | Yes |
| **3. ISRO Bhoonidhi** | Cartosat-2 | 4.0m | RGB + NIR | **Yes** | Subject to search | No (Priced) | No (Requires payment) |
| **4. ESA TPM SPOT** | SPOT 6/7 | 1.5m | RGB + NIR | **Yes** | Subject to search | **Yes** (Via Proposal) | **Yes** (Slow approval) |
| **5. ESA TPM Planet** | PlanetScope | ~3.0m | RGB + NIR | **Yes** | **Yes** | **Yes** (Via Proposal) | Restricted (ESA/EU) |
| **6. NASA CSDA** | PlanetScope | ~3.0m | RGB + NIR | **Yes** | **Yes** | **Yes** (Via SDX Portal) | No (US funding required) |
| **7. WorldStrat** | SPOT 6/7 | 1.5m | RGB + NIR | No | No | No (Monolithic 68GB) | Yes |
| **8. Maxar Open** | WorldView | <1.0m | RGB + NIR | No | No | No (Disaster only) | Yes |
| **9. OpenEarthMap** | Various | <0.5m | RGB only | No | No | Yes (Patches) | Yes |

---

RECOMMENDED_SOURCE: Planet Education & Research (E&R) Program
REASON: It is the only option that meets all scientific and operational requirements: provides 3m multispectral imagery (RGB + NIR) matching Sentinel-2 bands, covers the exact Electronic City coordinate bounds, contains the specific target scene from 11 February 2026, and provides free programmatic downloads for university-affiliated students and researchers under a legally compliant academic license.
NEXT_ACTION: Submit an application to the Planet Education & Research Program (https://www.planet.com/markets/education-and-research/) using a university email address. Once approved, use the Planet Orders API to query and download the 3m scene `20260211_054815_64_254a` covering the Electronic City AOI, placing it under `data/raw/planet/`.
