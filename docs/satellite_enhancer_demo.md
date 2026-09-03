# Satellite Enhancer Demonstration Application Documentation

This document describes the structure, verification process, and user guide for the **Satellite Enhancer Demo Application** developed for project **SIH26142**.

The demo provides a zero-dependency, local web-based interactive interface that allows users to visual-audit and inspect the results of the **Residual CNN** model (upscaling 10m Sentinel-2 multi-spectral bands to 5m resolution).

---

## 1. Application Architecture and Codebase

The application is stored at:
*   **Demo Server Entrypoint**: [`app/satellite_enhancer.py`](file:///c:/Users/urstr/New%20folder%20(3)/app/satellite_enhancer.py)
*   **Static Assets & Dashboard Views**: stored at [`app/static/`](file:///c:/Users/urstr/New%20folder%20(3)/app/static/)

Reusing the production-ready tiled inference module (`src/super_resolution/inference.py`), the app:
1.  Warms up the model using the trained checkpoint weights: `models/residual_srm_experiment3.pth`.
2.  Runs tiled inference on the Sentinel-2 stacked GeoTIFF `data/processed/s2_10m_stacked_roi.tiff` (if the upscaled output doesn't already exist on disk).
3.  Saves the high-resolution output to `outputs/s2_5m_upscaled_residual.tiff`, preserving all 4 spectral bands and local georeferencing coordinate bounds.
4.  Launches a local, multi-threaded HTTP server on port **8080** and automatically opens the user's web browser to `http://localhost:8080/`.

---

## 2. Interactive Dashboard Features

The web interface is designed with a modern dark theme and glassmorphism styling to deliver a premium user experience.

### A. Interactive Split-Screen Image Slider
*   Users can interactively drag a center handle left and right to peel back the layers and directly compare the **Bilinear 5m baseline** and the **Residual CNN 5m output** in the same viewport.
*   The comparison is fully synced to prevent spatial coordinate mismatch during interaction.

### B. True-Color and False-Color (NIR) Visualization Toggles
*   **RGB (True Color)**: Combines Red (B04), Green (B03), and Blue (B02) bands with a 2%–98% contrast stretch.
*   **NIR (False Color)**: Combines NIR (B08), Red (B04), and Green (B03) bands to highlight vegetation structures, biomass, and chlorophyll density at 5m resolution.

### C. Urban sub-Pixel Zoom Comparison
*   Displays a side-by-side zoomed-in view of a highly textured urban area (containing roads, buildings, and clear asphalt-soil boundaries).
*   Allows the user to visually inspect the transition from blocky S2 10m pixels to smooth Bilinear 5m, and finally to the sharp, reconstructed linear details of the Residual CNN.

### D. Metadata and Grid Calibration Panel
*   Displays critical raster tags read directly from the GeoTIFFs:
    *   **Input Grid**: 10.0m resolution ($1024 \times 1024$ pixels)
    *   **Output Grid**: 5.0m resolution ($2048 \times 2048$ pixels)
    *   **Coordinate system**: `EPSG:32643` (UTM Zone 43N)
    *   **Model**: `ResidualCNN`
*   Includes a download action linking to `/api/download` that triggers the download of the georeferenced 5m GeoTIFF output file.

### E. Prominent Scientific Disclaimer
The dashboard includes the following banner:
> **SCIENTIFIC DISCLAIMER**: This 5m product is produced using a model trained on synthetic degradation. It has not yet been independently validated against real high-resolution reference imagery.

---

## 3. Launching the Application

Start the web server by running the entrypoint script in your terminal:
```bash
python app/satellite_enhancer.py
```

The script will automatically:
1.  Verify the integrity of input rasters and weights.
2.  Pre-generate visual JPEG/PNG assets inside `app/static/` from multi-spectral bands.
3.  Initialize the HTTP server.
4.  Open your default browser window pointing to `http://localhost:8080/`.

To stop the web server, press `Ctrl+C` in your terminal.

---

## 4. Verification and Security Log

### A. Non-Destructive Integrity Verification
*   The script reads weights from `models/residual_srm_experiment3.pth` in read-only mode (`model.eval()`).
*   No training routines are executed. No existing checkpoints (`models/espcn_srm_synthetic.pth` or `models/residual_srm_experiment3.pth`) are modified.

### B. Spectral Band Preservation
*   The output GeoTIFF `outputs/s2_5m_upscaled_residual.tiff` was verified using `rasterio` to contain exactly **4 spectral bands**. No channels were discarded or collapsed during upscaling.

### C. Georeferencing Preservation
*   The coordinate reference system of the output file matches `EPSG:32643`.
*   The geospatial transform scales from `dx = 10.0m`, `dy = -10.0m` to `dx = 5.0m`, `dy = -5.0m` with the exact top-left origin $(749740.0, 1450220.0)$ snapped to S2.

---

## 5. Verification Status

### **STATUS**: `DEMO_READY`

**Reason**:
The tiled production-ready inference pipeline successfully processes the Sentinel-2 stacked TIFF, outputs a georeferenced 4-band 5m TIFF, pre-generates the True-Color/NIR false-color assets, and launches the interactive dashboard server on port 8080. All components have been verified as correct, non-destructive, and structurally aligned with the Experiment 4 requirements.
