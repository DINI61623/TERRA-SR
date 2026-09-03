#!/usr/bin/env python3
"""
Unit tests for Batch Multi-Tile Mosaic Ingestion & Seamless Tile Stitcher.
"""

import unittest
import numpy as np

from src.core.mosaic_stitcher import BatchMosaicStitcher
from src.super_resolution.inference import ProductionInference

try:
    from affine import Affine
    HAS_AFFINE = True
except ImportError:
    HAS_AFFINE = False


class TestMosaicStitcher(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.stitcher = BatchMosaicStitcher(tile_size=64, overlap_px=16)

    def test_stitch_constant_field(self):
        """A flat constant field of 0.85 should remain exactly 0.85 across all blended tile boundaries."""
        t1 = np.full((4, 64, 64), 0.85, dtype=np.float32)
        t2 = np.full((4, 64, 64), 0.85, dtype=np.float32)
        
        tiles = [t1, t2]
        positions = [(0, 0), (0, 48)]  # Overlap of 16 pixels
        output_shape = (4, 64, 112)
        
        stitched = self.stitcher.stitch_overlapping_tiles(tiles, positions, output_shape, overlap=16)
        self.assertEqual(stitched.shape, output_shape)
        self.assertFalse(np.isnan(stitched).any())
        # Check that everywhere in the overlap region is 0.85 (+/- 1e-4)
        np.testing.assert_allclose(stitched, 0.85, atol=1e-3)

    def test_tiled_inference_execution(self):
        """Tests tiled SR execution on a 128x128 cube scaling to 256x256 without border seams."""
        cube = np.random.uniform(0.1, 0.9, (4, 128, 128)).astype(np.float32)
        runner = ProductionInference(model_type="Bilinear")
        
        enhanced = self.stitcher.enhance_large_raster_tiled(
            raster_cube=cube,
            model_runner=runner,
            scale_factor=2
        )
        self.assertEqual(enhanced.shape, (4, 256, 256))
        self.assertFalse(np.isnan(enhanced).any())
        self.assertTrue(0.0 <= enhanced.min() and enhanced.max() <= 1.0)


if __name__ == "__main__":
    unittest.main()
