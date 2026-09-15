#!/usr/bin/env python3
"""
TERRA-SR Satellite Data Acquisition Package
"""

from src.satellite.copernicus_auth import CopernicusAuthManager
from src.satellite.validators import validate_bbox, compute_bbox_area_km2, format_aoi_summary, AOIValidationError
from src.satellite.scene_ranker import SceneRanker
from src.satellite.catalog import search_copernicus_catalog
from src.satellite.process import retrieve_aoi_raster

__all__ = [
    "CopernicusAuthManager",
    "validate_bbox",
    "compute_bbox_area_km2",
    "format_aoi_summary",
    "AOIValidationError",
    "SceneRanker",
    "search_copernicus_catalog",
    "retrieve_aoi_raster"
]
