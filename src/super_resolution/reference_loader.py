#!/usr/bin/env python3
"""
Sentinel-2 Super-Resolution Mapping (SRM) - Reference Loader
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

Loads high-resolution multispectral reference GeoTIFF datasets (like PlanetScope)
and extracts geospatial metadata, CRS, affine transforms, and band details.
"""

import sys
from pathlib import Path

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


class ReferenceLoader:
    def __init__(self, file_path):
        """
        Args:
            file_path (str or Path): Path to the high-resolution reference GeoTIFF.
        """
        self.file_path = Path(file_path)

    def load_metadata(self):
        """
        Loads and validates metadata from the reference GeoTIFF.
        Returns a dictionary of raster properties.
        """
        if not HAS_RASTERIO:
            raise ImportError("The 'rasterio' library is required to load remote sensing imagery.")

        if not self.file_path.exists():
            raise FileNotFoundError(f"Reference imagery file not found at: {self.file_path}")

        try:
            with rasterio.open(self.file_path) as src:
                metadata = {
                    "filepath": str(self.file_path.resolve()),
                    "driver": src.driver,
                    "width": src.width,
                    "height": src.height,
                    "count": src.count,
                    "crs": src.crs.to_string() if src.crs else None,
                    "transform": list(src.transform) if src.transform else None,
                    "res": src.res,
                    "bounds": {
                        "left": src.bounds.left,
                        "bottom": src.bounds.bottom,
                        "right": src.bounds.right,
                        "top": src.bounds.top
                    },
                    "dtypes": [str(d) for d in src.dtypes],
                    "nodata": src.nodata
                }
                
            # Scientific validation: Check that we have at least 4 bands (RGB + NIR)
            if metadata["count"] < 4:
                raise ValueError(
                    f"Invalid band count: Reference image must contain at least 4 spectral bands "
                    f"(Blue, Green, Red, NIR). Found {metadata['count']} bands."
                )
                
            return metadata
            
        except Exception as e:
            raise RuntimeError(f"Failed to read metadata from {self.file_path.name}: {str(e)}")

    def read_bands_data(self, window=None):
        """
        Reads the spectral bands data.
        Returns a numpy array of shape (bands, height, width).
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Reference file not found: {self.file_path}")
            
        with rasterio.open(self.file_path) as src:
            # Read first 4 bands (RGB + NIR)
            data = src.read([1, 2, 3, 4], window=window)
            return data
