#!/usr/bin/env python3
"""
TERRA-SR Downstream Satellite Intelligence Framework - Base Application Interface
Standardized interface for all downstream satellite analytics consuming the single SR engine product.
Fully self-contained using PyTorch, NumPy, and Rasterio.
"""

import abc
import json
import time
import math
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn.functional as F

try:
    import rasterio
    from rasterio.features import shapes
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def apply_percentile_stretch(img: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    """Normalizes image array to [0.0, 1.0] using percentile clipping per channel."""
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    sample = img[::4, ::4] if img.shape[0] > 256 and img.shape[1] > 256 else img
    if img.ndim == 2:
        low = float(np.percentile(sample, low_pct))
        high = float(np.percentile(sample, high_pct))
        if high - low > 1e-6:
            return np.clip((img - low) / (high - low), 0.0, 1.0)
        return np.clip(img, 0.0, 1.0)
    
    stretched = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[-1]):
        low = float(np.percentile(sample[..., c], low_pct))
        high = float(np.percentile(sample[..., c], high_pct))
        if high - low > 1e-6:
            stretched[..., c] = np.clip((img[..., c] - low) / (high - low), 0.0, 1.0)
        else:
            stretched[..., c] = np.clip(img[..., c], 0.0, 1.0)
    return stretched


def tensor_morph_dilation(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Morphological dilation using PyTorch MaxPool2d."""
    t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    pad = kernel_size // 2
    dilated = F.max_pool2d(t, kernel_size=kernel_size, stride=1, padding=pad)
    return (dilated.squeeze().numpy() > 0.5)


def tensor_morph_erosion(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Morphological erosion using PyTorch inverted MaxPool2d."""
    t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    pad = kernel_size // 2
    eroded = -F.max_pool2d(-t, kernel_size=kernel_size, stride=1, padding=pad)
    return (eroded.squeeze().numpy() > 0.5)


def tensor_morph_opening(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Morphological opening (erosion followed by dilation)."""
    return tensor_morph_dilation(tensor_morph_erosion(mask, kernel_size), kernel_size)


def tensor_morph_closing(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Morphological closing (dilation followed by erosion)."""
    return tensor_morph_erosion(tensor_morph_dilation(mask, kernel_size), kernel_size)


def tensor_white_tophat(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """White Top-Hat transform (isolates elements brighter than surroundings)."""
    t = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    pad = kernel_size // 2
    # Opening on continuous grayscale: erosion then dilation
    eroded = -F.max_pool2d(-t, kernel_size=kernel_size, stride=1, padding=pad)
    opened = F.max_pool2d(eroded, kernel_size=kernel_size, stride=1, padding=pad)
    tophat = t - opened
    return np.clip(tophat.squeeze().numpy(), 0.0, None)


def tensor_gaussian_blur(img: np.ndarray, kernel_size: int = 7, sigma: float = 2.0) -> np.ndarray:
    """Applies Gaussian spatial smoothing using depthwise 2D convolution."""
    x = torch.arange(kernel_size) - (kernel_size - 1) / 2.0
    gauss_1d = torch.exp(-0.5 * (x / sigma)**2)
    gauss_2d = torch.outer(gauss_1d, gauss_1d)
    gauss_2d = (gauss_2d / gauss_2d.sum()).view(1, 1, kernel_size, kernel_size).float()
    
    t = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    pad = kernel_size // 2
    blurred = F.conv2d(t, gauss_2d, padding=pad)
    return blurred.squeeze().numpy()


def compute_gradient_sharpness(band: np.ndarray) -> float:
    """Computes mean gradient magnitude across a 2D band array."""
    gy, gx = np.gradient(band.astype(np.float32))
    mag = np.sqrt(gx**2 + gy**2)
    return float(np.mean(mag))


def calculate_polygon_area(coords: List[Tuple[float, float]]) -> float:
    """Calculates polygon area via Shoelace formula."""
    if len(coords) < 3:
        return 0.0
    n = len(coords)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += coords[i][0] * coords[j][1]
        area -= coords[j][0] * coords[i][1]
    return abs(area) / 2.0


def extract_geojson_from_mask(
    mask: np.ndarray,
    affine_transform,
    classification_name: str,
    class_id: int = 1,
    min_area_pixels: int = 4
) -> List[Dict[str, Any]]:
    """
    Extracts vector polygon features from a binary mask with affine transformation.
    """
    features = []
    if not HAS_RASTERIO or affine_transform is None or np.sum(mask) == 0:
        return features

    mask_uint8 = (mask > 0).astype(np.uint8)
    pixel_area = abs(affine_transform[0] * affine_transform[4]) if affine_transform else 25.0
    min_area_thresh = min_area_pixels * pixel_area

    for geom, val in shapes(mask_uint8, mask=(mask_uint8 > 0), transform=affine_transform):
        if geom.get("type") == "Polygon" and geom.get("coordinates"):
            exterior_ring = geom["coordinates"][0]
            poly_area = calculate_polygon_area(exterior_ring)
            if poly_area < min_area_thresh:
                continue
            features.append({
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "classification": classification_name,
                    "class_id": class_id,
                    "area_sq_m": round(float(poly_area), 2),
                    "area_ha": round(float(poly_area) / 10000.0, 4)
                }
            })
    return features


class BaseIntelligenceModule(abc.ABC):
    """
    Abstract Base Class for all downstream satellite intelligence modules.
    Standard input: 4-band Bottom-Of-Atmosphere reflectance cube (B02 Blue, B03 Green, B04 Red, B08 NIR)
    Standard output: Visual layers, quantitative metrics, GeoJSON features, and SR impact analysis.
    """
    
    def __init__(self, domain_name: str, domain_key: str):
        self.domain_name = domain_name
        self.domain_key = domain_key

    @abc.abstractmethod
    def process(
        self,
        sr_cube: np.ndarray,
        lr_cube: Optional[np.ndarray] = None,
        gsd: float = 3.33,
        affine_transform: Any = None,
        crs: str = "EPSG:32643",
        bounds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        pass
