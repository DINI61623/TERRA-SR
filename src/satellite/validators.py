#!/usr/bin/env python3
"""
TERRA-SR Satellite AOI Coordinate & Geometry Validators
Handles:
1. Bounding box & polygon coordinate verification (WGS84 / EPSG:4326)
2. Geodesic area calculation (km² and hectares)
3. Coordinate ordering safety ([minLon, minLat, maxLon, maxLat])
"""

import math
from typing import List, Tuple, Dict, Any, Optional, Union


class AOIValidationError(ValueError):
    """Raised when an Area of Interest geometry or coordinate format is invalid."""
    pass


def validate_bbox(bbox: Union[List[float], Tuple[float, ...]]) -> List[float]:
    """
    Validates that a bounding box is in standard WGS84 format:
    [min_lon, min_lat, max_lon, max_lat]
    
    Checks:
    - Exactly 4 float numbers
    - Longitude in [-180.0, 180.0]
    - Latitude in [-90.0, 90.0]
    - min_lon < max_lon and min_lat < max_lat
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise AOIValidationError(
            f"Invalid bounding box format. Expected list of 4 floats [minLon, minLat, maxLon, maxLat], got: {bbox}"
        )

    min_lon, min_lat, max_lon, max_lat = [float(x) for x in bbox]

    # Coordinate ranges
    if not (-180.0 <= min_lon <= 180.0) or not (-180.0 <= max_lon <= 180.0):
        raise AOIValidationError(
            f"Longitude values must be between -180.0 and 180.0. Got min_lon={min_lon}, max_lon={max_lon}"
        )
    if not (-90.0 <= min_lat <= 90.0) or not (-90.0 <= max_lat <= 90.0):
        raise AOIValidationError(
            f"Latitude values must be between -90.0 and 90.0. Got min_lat={min_lat}, max_lat={max_lat}"
        )

    if min_lon >= max_lon:
        raise AOIValidationError(
            f"Invalid longitude order: min_lon ({min_lon}) must be strictly less than max_lon ({max_lon})"
        )
    if min_lat >= max_lat:
        raise AOIValidationError(
            f"Invalid latitude order: min_lat ({min_lat}) must be strictly less than max_lat ({max_lat})"
        )

    return [min_lon, min_lat, max_lon, max_lat]


def compute_bbox_area_km2(bbox: List[float]) -> float:
    """
    Calculates geographic surface area of a bounding box in square kilometers
    using ellipsoidal/spherical Earth approximation (WGS84 mean radius = 6371.0088 km).
    """
    min_lon, min_lat, max_lon, max_lat = validate_bbox(bbox)
    r_earth = 6371.0088  # Mean Earth radius in km

    lat1_rad = math.radians(min_lat)
    lat2_rad = math.radians(max_lat)
    lon_diff_rad = math.radians(max_lon - min_lon)

    # Spherical cap strip area: R^2 * (sin(lat2) - sin(lat1)) * dLon
    area_km2 = (r_earth ** 2) * abs(math.sin(lat2_rad) - math.sin(lat1_rad)) * lon_diff_rad
    return round(area_km2, 4)


def validate_geojson_polygon(geojson_geom: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates a GeoJSON Polygon or MultiPolygon geometry dictionary.
    Returns normalized GeoJSON geometry.
    """
    if not isinstance(geojson_geom, dict) or "type" not in geojson_geom:
        raise AOIValidationError("Invalid GeoJSON geometry: must be a dict with 'type' and 'coordinates'")

    geom_type = geojson_geom.get("type")
    coords = geojson_geom.get("coordinates")

    if geom_type not in ["Polygon", "MultiPolygon"]:
        raise AOIValidationError(f"Unsupported geometry type '{geom_type}'. Must be 'Polygon' or 'MultiPolygon'")

    if not coords or not isinstance(coords, list):
        raise AOIValidationError("Geometry 'coordinates' must be a non-empty list")

    return geojson_geom


def extract_bbox_from_geometry(geojson_geom: Dict[str, Any]) -> List[float]:
    """
    Extracts [min_lon, min_lat, max_lon, max_lat] from a GeoJSON geometry.
    """
    validate_geojson_polygon(geojson_geom)
    geom_type = geojson_geom["type"]
    coords = geojson_geom["coordinates"]

    all_points = []
    if geom_type == "Polygon":
        for ring in coords:
            for pt in ring:
                all_points.append(pt)
    elif geom_type == "MultiPolygon":
        for poly in coords:
            for ring in poly:
                for pt in ring:
                    all_points.append(pt)

    if not all_points:
        raise AOIValidationError("No coordinate vertices found in geometry")

    lons = [pt[0] for pt in all_points]
    lats = [pt[1] for pt in all_points]

    return validate_bbox([min(lons), min(lats), max(lons), max(lats)])


def format_aoi_summary(bbox: List[float], custom_polygon: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Produces a complete, structured metadata dictionary for the user-selected AOI.
    """
    valid_bbox = validate_bbox(bbox)
    min_lon, min_lat, max_lon, max_lat = valid_bbox
    center_lon = round((min_lon + max_lon) / 2.0, 6)
    center_lat = round((min_lat + max_lat) / 2.0, 6)
    area_km2 = compute_bbox_area_km2(valid_bbox)

    # Approximate width and height in km
    lat_mid_rad = math.radians(center_lat)
    km_per_deg_lat = 111.132
    km_per_deg_lon = 111.320 * math.cos(lat_mid_rad)

    width_km = round(abs(max_lon - min_lon) * km_per_deg_lon, 2)
    height_km = round(abs(max_lat - min_lat) * km_per_deg_lat, 2)

    return {
        "status": "VALID",
        "crs": "EPSG:4326 (WGS84)",
        "bbox": valid_bbox,
        "center": {
            "lat": center_lat,
            "lon": center_lon,
            "formatted": f"{center_lat:.4f}° N, {center_lon:.4f}° E" if center_lat >= 0 and center_lon >= 0 else f"{center_lat:.4f}°, {center_lon:.4f}°"
        },
        "dimensions_km": {
            "width_km": width_km,
            "height_km": height_km
        },
        "area_km2": area_km2,
        "area_ha": round(area_km2 * 100.0, 2),
        "geometry": custom_polygon if custom_polygon else {
            "type": "Polygon",
            "coordinates": [[
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat]
            ]]
        }
    }
