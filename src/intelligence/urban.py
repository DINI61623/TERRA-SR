#!/usr/bin/env python3
"""
TERRA-SR Urban Intelligence Module
Extracts built-up extent, road/linear feature candidates, urban density heatmaps, and provides
Native (10m) vs. SR (<4m) feature interpretation metrics.
"""

import time
from typing import Dict, Any, Optional, Tuple
import numpy as np
import matplotlib as mpl

from src.intelligence.base import (
    BaseIntelligenceModule,
    apply_percentile_stretch,
    apply_percentile_stretch_uint8,
    apply_colormap_lut,
    compute_gradient_sharpness,
    extract_geojson_from_mask,
    tensor_morph_dilation,
    tensor_morph_erosion,
    tensor_morph_opening,
    tensor_morph_closing,
    tensor_white_tophat,
    tensor_gaussian_blur
)


class UrbanIntelligenceModule(BaseIntelligenceModule):
    def __init__(self):
        super().__init__(domain_name="Urban Intelligence", domain_key="urban")

    def compute_builtup_index(self, cube: np.ndarray) -> np.ndarray:
        """
        Computes Modified Built-Up Index using Red (B04), NIR (B08), and Blue (B02).
        Urban impervious surfaces exhibit high Red/Blue reflectance relative to NIR compared to vegetation.
        Formula: (Red - NIR) / (Red + NIR + 1e-7) + high-frequency spatial edge gradient.
        """
        blue, green, red, nir = cube[0], cube[1], cube[2], cube[3]
        ndvi = (nir - red) / (nir + red + 1e-7)
        ndwi = (green - nir) / (green + nir + 1e-7)
        
        # High-frequency structural texture
        gy, gx = np.gradient(red)
        edge_energy = np.sqrt(gx**2 + gy**2)
        del gy, gx
        edge_norm = edge_energy / (np.percentile(edge_energy[::4, ::4], 98) + 1e-7)
        del edge_energy
        
        raw_builtup = (red - nir) / (red + nir + 1e-7) + 0.5 * edge_norm
        del edge_norm
        # Suppress water and dense canopy
        raw_builtup[ndwi > 0.05] = -1.0
        raw_builtup[ndvi > 0.35] = -1.0
        del ndvi, ndwi
        return np.clip(raw_builtup, -1.0, 1.0)

    def extract_candidate_features(self, cube: np.ndarray, builtup_idx: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts candidate built-up clusters, road/linear corridor candidates, and density map.
        """
        red = cube[2]
        if builtup_idx is None:
            builtup_idx = self.compute_builtup_index(cube)
        
        # 1. Candidate Building / Roof Structures (Multi-scale morphological top-hat)
        top_hat = tensor_white_tophat(red, kernel_size=5)
        top_hat_norm = top_hat / (np.percentile(top_hat[::4, ::4], 99) + 1e-7)
        del top_hat
        building_candidates = (builtup_idx > 0.05) & (top_hat_norm > 0.25)
        del top_hat_norm
        building_candidates = tensor_morph_opening(building_candidates, kernel_size=3)
        
        # 2. Road / Linear Corridor Candidates (Ridge-like high-gradient elongation)
        gy, gx = np.gradient(red)
        grad_mag = np.sqrt(gx**2 + gy**2)
        del gy, gx
        grad_norm = grad_mag / (np.percentile(grad_mag[::4, ::4], 98) + 1e-7)
        del grad_mag
        linear_candidates = (grad_norm > 0.30) & (builtup_idx > -0.2) & (~building_candidates)
        del grad_norm
        linear_candidates = tensor_morph_closing(linear_candidates, kernel_size=3)
        
        # 3. Urban Density Heatmap (Continuous spatial Gaussian smoothing)
        density_raw = (builtup_idx > 0.0).astype(np.float32)
        density_heatmap = tensor_gaussian_blur(density_raw, kernel_size=15, sigma=4.0)
        del density_raw
        density_heatmap = density_heatmap / (np.max(density_heatmap) + 1e-7)
        
        return building_candidates, linear_candidates, density_heatmap

    def process(
        self,
        sr_cube: np.ndarray,
        lr_cube: Optional[np.ndarray] = None,
        gsd: float = 3.33,
        affine_transform: Any = None,
        crs: str = "EPSG:32643",
        bounds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        start_t = time.time()
        H, W = sr_cube.shape[1], sr_cube.shape[2]
        pixel_area_m2 = gsd * gsd
        pixel_area_ha = pixel_area_m2 / 10000.0
        
        # Ensure reflectance in [0, 1]
        sr_cube = np.clip(np.nan_to_num(sr_cube, nan=0.0), 0.0, 1.0)
        
        # True color RGB base
        sr_rgb = np.stack([sr_cube[2], sr_cube[1], sr_cube[0]], axis=-1)
        sr_rgb_stretched = apply_percentile_stretch_uint8(sr_rgb)
        del sr_rgb
        
        # 1. Compute Built-Up Index & Candidate Structures
        builtup_idx = self.compute_builtup_index(sr_cube)
        builtup_pixels = int(np.sum(builtup_idx > 0.0))
        building_mask, road_mask, density_map = self.extract_candidate_features(sr_cube, builtup_idx=builtup_idx)
        del builtup_idx
        
        # 2. Visual Overlays
        # A. Built-up Heatmap Overlay (Inferno Colormap via LUT)
        density_rgb = apply_colormap_lut(np.clip(density_map, 0.0, 1.0), "inferno")
        
        # B. Candidate Features Overlay (Cyan = Roads, Amber/Gold = Candidate Buildings)
        feature_overlay = sr_rgb_stretched.copy()
        feature_overlay[road_mask] = [0, 220, 255]       # Bright Cyan
        feature_overlay[building_mask] = [255, 170, 0]   # Amber / Orange
        
        # C. Built-up Boundary Demarcation
        builtup_dilated = tensor_morph_dilation(building_mask | road_mask, kernel_size=3)
        builtup_boundary = builtup_dilated ^ (building_mask | road_mask)
        boundary_overlay = sr_rgb_stretched.copy()
        boundary_overlay[builtup_boundary] = [255, 50, 50]  # Crimson outline
        
        # 3. Native vs. SR Impact Analysis
        impact_metrics = {}
        if lr_cube is not None:
            lr_cube = np.clip(np.nan_to_num(lr_cube, nan=0.0), 0.0, 1.0)
            lr_sharpness = compute_gradient_sharpness(lr_cube[2])
            sr_sharpness = compute_gradient_sharpness(sr_cube[2])
            
            sharpness_gain_pct = round(((sr_sharpness - lr_sharpness) / (lr_sharpness + 1e-7)) * 100.0, 1)
            
            impact_metrics = {
                "native_gsd": "10.0m",
                "sr_gsd": f"{gsd:.2f}m",
                "native_gradient_sharpness": round(lr_sharpness, 4),
                "sr_gradient_sharpness": round(sr_sharpness, 4),
                "edge_sharpness_gain_pct": f"+{sharpness_gain_pct}%",
                "interpretation_benefit": "Sharpened edge gradients allow distinction of sub-10m road corridors and compact structural boundaries that merge into homogeneous mixed pixels in native 10m Sentinel-2."
            }
        else:
            sr_sharpness = compute_gradient_sharpness(sr_cube[2])
            impact_metrics = {
                "sr_gsd": f"{gsd:.2f}m",
                "sr_gradient_sharpness": round(sr_sharpness, 4),
                "interpretation_benefit": "Sub-pixel reconstructed edges enhance linear infrastructure delineation."
            }
            
        # 4. Quantitative Statistics
        total_pixels = H * W
        candidate_building_pixels = int(np.sum(building_mask))
        candidate_road_pixels = int(np.sum(road_mask))
        
        builtup_area_ha = round(builtup_pixels * pixel_area_ha, 2)
        building_area_ha = round(candidate_building_pixels * pixel_area_ha, 2)
        road_area_ha = round(candidate_road_pixels * pixel_area_ha, 2)
        total_area_ha = round(total_pixels * pixel_area_ha, 2)
        
        builtup_density_pct = round((builtup_pixels / total_pixels) * 100.0, 1)
        
        # 5. GeoJSON Feature Extraction
        geojson_features = []
        if affine_transform is not None:
            geojson_features.extend(
                extract_geojson_from_mask(
                    building_mask,
                    affine_transform,
                    classification_name="Candidate Urban Building Structure",
                    class_id=1,
                    min_area_pixels=8
                )
            )
            geojson_features.extend(
                extract_geojson_from_mask(
                    road_mask,
                    affine_transform,
                    classification_name="Candidate Road Corridor / Linear Infrastructure",
                    class_id=2,
                    min_area_pixels=12
                )
            )
            
        elapsed = time.time() - start_t
        
        report = {
            "module": self.domain_name,
            "domain": self.domain_key,
            "status": "ANALYSIS_COMPLETE",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "processing_time_s": round(elapsed, 3),
            "spatial_resolution": f"{gsd:.2f}m GSD",
            "crs": crs,
            "scene_dimensions": f"{W} × {H} px",
            "summary": {
                "total_scene_area_ha": total_area_ha,
                "total_builtup_area_ha": builtup_area_ha,
                "builtup_density_percentage": f"{builtup_density_pct}%",
                "candidate_building_footprint_ha": building_area_ha,
                "candidate_road_infrastructure_ha": road_area_ha,
                "urban_density_classification": "High Density Urban" if builtup_density_pct > 40 else "Moderate Urban" if builtup_density_pct > 15 else "Low Density / Rural"
            },
            "sr_impact_analysis": impact_metrics,
            "scientific_safety_audit": {
                "product_type": "AI-reconstructed higher-resolution spatial product",
                "detection_classification": "Urban Feature Analysis / Candidate Map (Heuristic Edge-Morphological Extraction)",
                "validation_note": "Visual enhancement significantly increases linear and boundary contrast. Candidate extractions are heuristic spatial indicators and do not replace certified ground-truth building footprint vector databases without dedicated validated segmentation models."
            },
            "geojson_count": len(geojson_features),
            "geojson": {
                "type": "FeatureCollection",
                "features": geojson_features[:100]
            }
        }
        
        full_geojson = {
            "type": "FeatureCollection",
            "features": geojson_features
        }
        
        layers = {
            "builtup_density": density_rgb,
            "candidate_features": feature_overlay,
            "builtup_boundary": boundary_overlay,
            "rgb_base": sr_rgb_stretched
        }
        
        return {
            "report": report,
            "layers": layers,
            "geojson_full": full_geojson,
            "masks": {
                "building_candidates": building_mask,
                "road_candidates": road_mask,
                "density_map": density_map
            }
        }
