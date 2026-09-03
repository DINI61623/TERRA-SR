#!/usr/bin/env python3
"""
Copernicus Data Space Ecosystem (CDSE) STAC Search Utility
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM)
from Medium Resolution Satellite Imageries (SIH26142)
"""

import argparse
import sys
import requests
from datetime import datetime

# CDSE STAC Search Endpoint
STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac/search"

def main():
    # Setup CLI Argument Parser
    parser = argparse.ArgumentParser(
        description="Search Copernicus Data Space Ecosystem STAC catalog for satellite imagery."
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="sentinel-2-l2a",
        help="STAC collection name (default: sentinel-2-l2a)"
    )
    parser.add_argument(
        "--lat",
        type=float,
        required=True,
        help="Center latitude for spatial query (e.g. 12.85)"
    )
    parser.add_argument(
        "--lon",
        type=float,
        required=True,
        help="Center longitude for spatial query (e.g. 77.685)"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.03,
        help="Bounding box offset/radius in degrees (default: 0.03)"
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date in YYYY-MM-DD format (e.g. 2024-01-01)"
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date in YYYY-MM-DD format (e.g. 2026-08-30)"
    )
    parser.add_argument(
        "--cloud-cover",
        type=float,
        default=10.0,
        help="Maximum cloud cover percentage allowed (default: 10.0)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Query and display metadata without downloading anything"
    )

    args = parser.parse_args()

    # Convert coordinates and radius to an approximate bounding box [min_lon, min_lat, max_lon, max_lat]
    min_lon = args.lon - args.radius
    min_lat = args.lat - args.radius
    max_lon = args.lon + args.radius
    max_lat = args.lat + args.radius
    bbox = [min_lon, min_lat, max_lon, max_lat]

    # Build STAC Search Payload
    payload = {
        "collections": [args.collection],
        "bbox": bbox,
        "datetime": f"{args.start}T00:00:00Z/{args.end}T23:59:59Z",
        "query": {
            "eo:cloud_cover": {
                "lt": args.cloud_cover
            }
        },
        "limit": 100  # Pull up to 100 candidate scenes to allow sorting
    }

    print(f"Searching STAC catalog for collection: {args.collection}")
    print(f"Bounding Box: {bbox}")
    print(f"Date Range:   {args.start} to {args.end}")
    print(f"Max Clouds:   {args.cloud_cover}%")
    print("--------------------------------------------------")

    try:
        # Query CDSE STAC Search Endpoint
        response = requests.post(STAC_URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error querying Copernicus STAC API: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Server response: {response.text}")
        sys.exit(1)

    features = data.get("features", [])
    print(f"Products found: {len(features)}")

    # Sort results by lowest cloud cover first (Requirement 8)
    features.sort(key=lambda f: f.get("properties", {}).get("eo:cloud_cover", 100.0))

    # Display products
    for feature in features:
        properties = feature.get("properties", {})
        product_id = feature.get("id")
        acq_datetime = properties.get("datetime")
        cloud_val = properties.get("eo:cloud_cover")
        
        print(f"\nID: {product_id}")
        print(f"Date: {acq_datetime}")
        print(f"Cloud: {cloud_val:.1f}%" if cloud_val is not None else "Cloud: N/A")

if __name__ == "__main__":
    main()
