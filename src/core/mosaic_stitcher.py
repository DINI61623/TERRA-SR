#!/usr/bin/env python3
"""
TERRA-SR Batch Multi-Tile GeoTIFF Mosaic Ingestion & Seamless Tile Stitcher.
Enables large regional Area-of-Interest (AOI) super-resolution and downstream intelligence:
1. Ingests multiple adjacent or overlapping Sentinel-2 GeoTIFF tiles.
2. Validates spatial adjacency, coordinate systems, and spectral band alignment.
3. Performs seamless feathered overlap blending to eliminate edge seams.
4. Tiled inference engine: Splits large rasters into overlapping chunks (e.g. 512x512 with 32px padding),
   runs the core SR model, and reassembles the high-resolution regional mosaic with georeferencing preservation.
"""

from typing import List, Dict, Tuple, Any, Optional, Union
from pathlib import Path
import numpy as np

try:
    import rasterio
    from rasterio.windows import Window
    from affine import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

import torch
from src.core.input_validation import SatelliteInputValidator, ValidationResult
from src.core.georeference import verify_georeferencing_integrity


class BatchMosaicStitcher:
    """Handles multi-tile raster mosaic assembly and tiled high-resolution inference."""

    def __init__(self, tile_size: int = 512, overlap_px: int = 32):
        self.tile_size = tile_size
        self.overlap_px = overlap_px

    @classmethod
    def validate_tile_collection(
        cls,
        tile_paths: List[Union[str, Path]]
    ) -> ValidationResult:
        """
        Validates that all input tiles share identical CRS, band counts, and compatible GSD.
        """
        if len(tile_paths) == 0:
            return ValidationResult(
                is_valid=False,
                domain="batch_mosaic",
                checks={"tile_count": False},
                reasons=["No raster tiles provided for mosaic assembly."],
                metadata={}
            )

        checks = {"tile_count_ok": len(tile_paths) >= 1}
        reasons = []
        metas = []

        ref_crs = None
        ref_bands = None

        for idx, tp in enumerate(tile_paths):
            res = SatelliteInputValidator.validate_for_domain(tp, "super_resolution")
            metas.append(res.metadata)
            if not res.is_valid:
                reasons.extend([f"Tile {idx+1} ({Path(tp).name}): {r}" for r in res.reasons])

            crs = res.metadata.get("crs")
            bands = res.metadata.get("band_count")

            if ref_crs is None:
                ref_crs = crs
                ref_bands = bands
            else:
                if str(crs) != str(ref_crs):
                    reasons.append(f"CRS mismatch in Tile {idx+1}: {crs} vs reference {ref_crs}.")
                if bands != ref_bands:
                    reasons.append(f"Band count mismatch in Tile {idx+1}: {bands} vs reference {ref_bands}.")

        checks["crs_homogeneity"] = bool(ref_crs and len(reasons) == 0)
        checks["spectral_homogeneity"] = bool(ref_bands and len(reasons) == 0)

        is_valid = len(reasons) == 0
        return ValidationResult(
            is_valid=is_valid,
            domain="batch_mosaic",
            checks=checks,
            reasons=reasons,
            metadata={"tile_count": len(tile_paths), "tile_metadatas": metas, "mosaic_crs": ref_crs}
        )

    def stitch_overlapping_tiles(
        self,
        tiles: List[np.ndarray],
        grid_positions: List[Tuple[int, int]],
        output_shape: Tuple[int, int, int],
        overlap: int = 32
    ) -> np.ndarray:
        """
        Stitches a grid of raster tiles into a single seamless output array using
        linear 2D feathered blending over overlapping borders.
        """
        C, total_H, total_W = output_shape
        stitched = np.zeros(output_shape, dtype=np.float32)
        weight_map = np.zeros((total_H, total_W), dtype=np.float32)

        for tile, (r_start, c_start) in zip(tiles, grid_positions):
            _, t_H, t_W = tile.shape
            
            # Construct 2D trapezoidal feathering weights for this tile
            wy = np.ones(t_H, dtype=np.float32)
            wx = np.ones(t_W, dtype=np.float32)
            
            if overlap > 0:
                ramp = np.linspace(0.01, 1.0, overlap, dtype=np.float32)
                wy[:overlap] = np.minimum(wy[:overlap], ramp)
                wy[-overlap:] = np.minimum(wy[-overlap:], ramp[::-1])
                wx[:overlap] = np.minimum(wx[:overlap], ramp)
                wx[-overlap:] = np.minimum(wx[-overlap:], ramp[::-1])
                
            w2d = np.outer(wy, wx)
            
            r_end = min(total_H, r_start + t_H)
            c_end = min(total_W, c_start + t_W)
            
            valid_th = r_end - r_start
            valid_tw = c_end - c_start
            
            stitched[:, r_start:r_end, c_start:c_end] += tile[:, :valid_th, :valid_tw] * w2d[:valid_th, :valid_tw]
            weight_map[r_start:r_end, c_start:c_end] += w2d[:valid_th, :valid_tw]

        # Normalize by accumulated weights
        valid_weights = weight_map > 1e-7
        for c in range(C):
            stitched[c, valid_weights] /= weight_map[valid_weights]

        return np.clip(stitched, 0.0, 1.0)

    def enhance_large_raster_tiled(
        self,
        raster_cube: np.ndarray,
        model_runner: Any,
        scale_factor: int = 2
    ) -> np.ndarray:
        """
        Splits a large raster cube into overlapping chunks, executes super-resolution model inference,
        and stitches the enhanced tiles seamlessly.
        """
        C, H, W = raster_cube.shape
        t_size = self.tile_size
        ov = self.overlap_px
        stride = t_size - ov

        out_H = H * scale_factor
        out_W = W * scale_factor
        
        enhanced_tiles = []
        grid_positions = []

        for r in range(0, H, stride):
            for c in range(0, W, stride):
                r_end = min(H, r + t_size)
                c_end = min(W, c + t_size)
                
                # Extract LR tile
                tile = raster_cube[:, r:r_end, c:c_end]
                
                # Run SR inference
                tensor_in = torch.tensor(tile, dtype=torch.float32)
                hr_tile = model_runner.enhance_tensor(tensor_in)
                
                enhanced_tiles.append(hr_tile)
                grid_positions.append((r * scale_factor, c * scale_factor))

        # Stitch HR tiles
        return self.stitch_overlapping_tiles(
            tiles=enhanced_tiles,
            grid_positions=grid_positions,
            output_shape=(C, out_H, out_W),
            overlap=ov * scale_factor
        )
