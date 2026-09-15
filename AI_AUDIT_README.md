# TERRA-SR — AI Implementation & Source-Code Audit Guide

## 1. Project Identification
- **Project Name:** TERRA-SR (AOI-Driven Satellite Earth Intelligence & Super-Resolution Platform)
- **Problem Statement:** SIH26142 — *Deep Learning Based Super Resolution Mapping (SRM) from Medium Resolution Satellite Imageries*
- **Competition:** Smart India Hackathon (SIH 2026)
- **Team Name:** VIBE-CODERS
- **Primary Domain:** Earth Observation (EO), Remote Sensing, Deep Learning, Geospatial Intelligence (GIS)

---

## 2. Auditor Directive
> **To the Receiving AI Auditor:**
>
> "This archive contains the complete TERRA-SR project for source-code audit and implementation verification."

Please execute a rigorous, systematic audit adhering strictly to the following 18 directives:

1. **Audit the entire codebase** across all architectural layers (`src/`, `app/`, `tests/`, `models/`, `configs/`, `scripts/`, `docs/`).
2. **Map every PPT/documentation claim to actual code** in the repository and verify functional backing.
3. **Identify implemented vs partially implemented vs missing features** with exact file citations and line ranges.
4. **Run the test suite** (`pytest` / `python -m unittest discover tests`) to confirm all 45 unit and integration tests execute successfully.
5. **Run the application** (`python app/satellite_enhancer.py`) and verify the UI dashboard, split slider, band switching, and GeoTIFF export on `http://localhost:8080/`.
6. **Verify the actual pipeline** from Sentinel-2 ingestion to SRM inference and downstream intelligence vectorization.
7. **Verify model implementations** (Bilinear, ESPCN, Residual CNN, MS-RCAN, HF-SRM, PI-RCAN Multi-Scale, U-Net ResNet-34).
8. **Verify preprocessing and tiling** (radiometric normalization $[0, 1]$, EPSG:4326/UTM CRS handling, $128\times 128$ sliding window patch extraction, and artifact-free mosaic stitching).
9. **Verify validation metrics** (PSNR, SSIM, Spectral Angle Mapper [SAM], ERGAS, Edge Preservation Index [EPI], NDVI/NDWI fidelity).
10. **Verify EO intelligence modules** across Water (NDWI shoreline), Agriculture (NDVI canopy & field boundaries), Urban (morphological building footprints & roads), Disaster (flood inundation & burn severity), and Oil Spill (SOSI/NDWI water-gated slick detection).
11. **Verify GIS outputs** ensuring valid GeoTIFF affine geotransforms and standard RFC 7946 GeoJSON feature collections.
12. **Verify APIs** including Copernicus Data Space Ecosystem (CDSE) OAuth2 token caching, STAC catalog search, and Process API integration.
13. **Verify CLI** (`cli.py`) supporting batch processing across models and intelligence domains.
14. **Verify Docker deployment** (`Dockerfile`, `DEPLOYMENT.md`, multi-stage container optimization, healthcheck endpoint `/health`).
15. **Identify bugs**, edge cases, potential memory leaks, or numerical instabilities.
16. **Implement missing features** or suggested refinements without breaking backwards compatibility.
17. **Re-run tests after fixes** to ensure zero regressions.
18. **Never fabricate results** — provide empirical metric computations, exact file paths, and verifiable facts.

---

## 3. Project Architecture & Code Navigation Matrix

### Core Application Entry Points
| Component | Entry File | Description / Command |
| :--- | :--- | :--- |
| **Web Dashboard** | `app/satellite_enhancer.py` | Full interactive platform (Port 8080): `python app/satellite_enhancer.py` |
| **Batch CLI** | `cli.py` | Headless execution CLI: `python cli.py --input outputs/s2_5m_upscaled_bilinear.tiff --model ResidualCNN --domain all` |
| **Test Suite** | `tests/` | Complete automated pytest suite (45 tests): `pytest` |
| **6-Way Benchmark** | `evaluate_6way_benchmark.py` | Rigorous evaluation harness: `python evaluate_6way_benchmark.py` |
| **Multi-Scale Benchmark** | `evaluate_multiscale_exp5.py` | <4m GSD verification: `python evaluate_multiscale_exp5.py` |
| **Controlled Study** | `run_controlled_sr_study.py` | Master 5-experiment controlled study: `python run_controlled_sr_study.py` |

### Key Modules & Directories
- `src/satellite/`: CDSE authentication (`copernicus_auth.py`), STAC catalog search (`catalog.py`), explainable scene ranker (`scene_ranker.py`), Process API ingestion (`process.py`), bbox validators (`validators.py`).
- `src/super_resolution/`: Model architectures (`pircan.py`, `hfsrm.py`, `msrcan.py`, `model.py`), degradation physics (`degradation.py`), 10-gate quality control (`quality_control.py`), sub-pixel coregistration (`coregistration.py`), production inference (`inference.py`).
- `src/intelligence/`: Base domain interface (`base.py`), Water (`water.py`), Agriculture (`agriculture.py`), Urban (`urban.py`), Disaster (`disaster.py`), Oil Spill Adapter (`oil_spill_adapter.py`), Module Registry (`__init__.py`).
- `src/oil_spill/`: Spectral slick indices (`indices.py`), U-Net architecture (`model.py`), water-gating (`water_gating.py`), anti-hallucination (`anti_hallucination.py`), end-to-end slick pipeline (`pipeline.py`).
- `src/core/`: Strict input/sensor validator (`input_validation.py`), georeferencing & CRS utilities (`georeference.py`), mosaic stitcher (`mosaic_stitcher.py`).
- `app/`: Interactive web server (`satellite_enhancer.py`) and static assets (`static/style.css`, pre-rendered overlays, UI icons).
- `models/`: PyTorch trained model checkpoints (`models/final_sr/best_model.pth`, `models/pircan_scale_x3.pth`, `models/residual_srm_experiment3.pth`, `models/msrcan_experiment4d.pth`, `models/hfsrm_experiment4e.pth`, `models/espcn_srm_synthetic.pth`).
- `configs/`: YAML definitions for physical degradation, sensor parameters, and training pipelines.
- `docs/`: Comprehensive technical architecture documentation, mathematical loss specifications, and visual audit reports.
- `tests/`: Unit and integration test suite covering input validation, satellite acquisition, intelligence modules, mosaic stitching, and CLI.

---

## 4. Security & Environment Configuration
- Real credentials and secrets have been intentionally excluded from this package (`src/.env`).
- A clean template is provided in `.env.example`:
  ```env
  CDSE_USERNAME=your_cdse_username@example.com
  CDSE_PASSWORD=your_cdse_password
  PLANET_API_KEY=your_planet_api_key_here
  ```
- If CDSE credentials are not set, the platform automatically runs in deterministic **Offline Demo Mode** using calibrated Sentinel-2 ROI data.
