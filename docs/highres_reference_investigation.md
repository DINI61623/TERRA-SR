# Technical Investigation Report: High-Resolution Reference Imagery for Bengaluru AOI

**Project Code**: SIH26142  
**Subject**: Feasibility study on obtaining 1m–4m resolution imagery for the Bengaluru/Electronic City AOI (12.85° N, 77.685° E) for scientific validation of Sentinel-2 super-resolution.

---

## 1. Evaluation of Candidate Options

We investigated several possible sources of high-resolution imagery for our target Bengaluru coordinate bounds.

### Option 1: PlanetScope (Planet Education & Research Program)
*   **Spatial Resolution**: 3.0 m (orthorectified).
*   **Spectral Bands**: 8 bands available (Coastal Blue, Blue, Green, Yellow, Red, Red Edge, NIR). This corresponds perfectly to Sentinel-2 bands **B02, B03, B04, and B08**.
*   **Temporal Matching**: Planet operates a constellation of 200+ Dove satellites with **daily global revisit rates**. A cloud-free scene for Bengaluru on or very close to our target date (**11 February 2026**) is highly likely to exist in their archives.
*   **API & Download**: Access is provided via Planet Explorer (web UI) and Planet Python SDK. We can define our small bounding box and download only the target Electronic City tile.
*   **Download Size**: Small (approx. 20–50 MB for our local AOI crop).
*   **Licensing**: **Free academic license** (5,000 sq km/month quota) for university students and academic researchers. Non-commercial research publications are permitted.

### Option 2: ISRO Bhuvan Portal (Resourcesat / Cartosat)
*   **Spatial Resolution**: 
    *   Resourcesat-2 LISS-IV: 5.8 m multispectral.
    *   Cartosat-2 series: 1 m panchromatic, 4 m multispectral.
*   **Spectral Bands**: LISS-IV provides 3 bands (Green, Red, NIR). Cartosat-2 provides RGB + NIR.
*   **Temporal Matching**: Lower revisit rates. Finding an exact match for 11 February 2026 depends on archive passes.
*   **API & Download**: No official open REST API; files must be searched and downloaded manually via the Bhuvan NICES/Open Data portal.
*   **Licensing**: Free for Indian researchers and academic institutions upon registration.

### Option 3: ESA Third-Party Missions (Airbus SPOT 6/7 via Earthnet)
*   **Spatial Resolution**: SPOT 6/7 provides 1.5 m multispectral imagery.
*   **Spectral Bands**: RGB + NIR (matches S2 B02/B03/B04/B08).
*   **Temporal Matching**: High archive coverage.
*   **API & Download**: Requires submitting a formal scientific project proposal to the European Space Agency (ESA) Earthnet Online program. Approval takes several weeks/months.
*   **Licensing**: Free for approved scientific research, strictly non-commercial.

---

## 2. Legal & Scientific Audit of Basemap Imagery (Google/Bing/Esri)

A tempting shortcut is to scrape high-resolution tiles from Google Maps, Bing Maps, or Esri World Imagery basemaps. However, **this is scientifically invalid and legally prohibited**:

### Legal Restrictions
1.  **ToS Violations**: The Terms of Service of Google Maps (Section 3.2.3a) and Esri World Imagery explicitly prohibit downloading tiles, scraping data, creating offline databases, or using the imagery to **train machine learning/AI models**.
2.  **IP Infringement**: Scraped basemaps cannot be published in academic journals or presented at national hackathons (like SIH 2026) without exposing the project to intellectual property claims by commercial providers (Maxar/Airbus).

### Scientific Invalidation
1.  **Missing Spectral Bands**: Basemaps only contain 3-band RGB values. They **completely lack the Near-Infrared (NIR) band (B08)**, which is critical for agricultural monitoring, vegetation index calculation (NDVI), and multispectral super-resolution.
2.  **No Radiometric Meaning**: Basemaps undergo aggressive color-balancing, contrast stretching, and lossy JPEG compression (8-bit conversion) to look aesthetically pleasing to humans. The pixel values no longer represent physical surface reflectance (Digital Numbers have been corrupted), making them useless for physical remote sensing science.
3.  **Temporal Mismatch (Compositing)**: Basemaps are mosaic composites stitched together from multiple satellite passes over years to remove clouds. The exact date of any individual pixel is unknown, meaning they cannot be temporally matched to our February 11, 2026 Sentinel-2 scene.

---

## 3. Options Ranking

Based on spatial detail, spectral matching, local availability, temporal compatibility, and legal compliance, here is the ranking:

### **#1 Best Option: PlanetScope (Planet Education & Research Program)**
*   *Why*: Provides 3m multispectral imagery (RGB+NIR), has daily global passes (temporal match for 11 Feb 2026), permits local AOI downloads, and offers a legally clean academic license for university students.

### **#2 Backup Option: ISRO Bhuvan (Cartosat-2 Multispectral)**
*   *Why*: Officially open to Indian researchers and students, providing 4m multispectral imagery covering Bengaluru, though lacking programmatic download APIs.

### **#3 Backup Option: ESA Third-Party Missions (SPOT 6/7 Archive)**
*   *Why*: Excellent 1.5m spatial detail, but the formal proposal review process is too slow for rapid prototype deadlines.

---

## 4. Conclusion & Actionable Path

### **We DO have a practical path to genuine high-resolution validation.**

The path consists of:
1.  **Planet Application**: Apply immediately for the **Planet Education & Research Program** using university credentials.
2.  **Target Download**: Once approved, query the Planet API for the Electronic City AOI on **11 February 2026** (or closest cloud-free date) and download the 3m orthorectified GeoTIFF.
3.  **Validation**: Register this 3m scene with our 10m Sentinel-2 scene to validate the super-resolution output.
