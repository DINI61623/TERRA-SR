#!/usr/bin/env python3
"""
PlanetScope Archive Search Script
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

Queries Planet's Data API to find cloud-free PlanetScope (PSScene) imagery
over our target Electronic City AOI (12.85, 77.685) in February 2026.
"""

import os
import sys
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ENV_PATH = Path(__file__).resolve().parent.parent / "src" / ".env"
load_dotenv(dotenv_path=ENV_PATH)

PLANET_API_KEY = os.getenv("PLANET_API_KEY")


# Target Search Parameters
AOI_LON = 77.685
AOI_LAT = 12.85
DATE_START = "2026-02-09T00:00:00Z"
DATE_END = "2026-02-16T23:59:59Z"
MAX_CLOUD_COVER = 0.1 # 10% max cloud cover


def build_search_query():
    """
    Constructs the Planet Data API search filter payload.
    """
    # Calculate bounding box coordinates with a 0.03 degree radius
    lon_min = AOI_LON - 0.03
    lon_max = AOI_LON + 0.03
    lat_min = AOI_LAT - 0.03
    lat_max = AOI_LAT + 0.03

    geometry_filter = {
        "type": "GeometryFilter",
        "field_name": "geometry",
        "config": {
            "type": "Polygon",
            "coordinates": [[
                [lon_min, lat_min],
                [lon_max, lat_min],
                [lon_max, lat_max],
                [lon_min, lat_max],
                [lon_min, lat_min]
            ]]
        }
    }
    
    date_filter = {
        "type": "DateRangeFilter",
        "field_name": "acquired",
        "config": {
            "gte": DATE_START,
            "lte": DATE_END
        }
    }
    
    cloud_filter = {
        "type": "RangeFilter",
        "field_name": "cloud_cover",
        "config": {
            "lte": MAX_CLOUD_COVER
        }
    }
    
    combined_filter = {
        "type": "AndFilter",
        "config": [geometry_filter, date_filter, cloud_filter]
    }
    
    query = {
        "item_types": ["PSScene"],
        "filter": combined_filter
    }
    return query


def main():
    print("==================================================")
    print("PlanetScope Scene Archive Search - February 2026")
    print("==================================================")
    print(f"Target AOI Center:  Lon {AOI_LON}, Lat {AOI_LAT}")
    print(f"Radius:             0.03 degrees (Bbox: [{AOI_LON-0.03:.3f}, {AOI_LAT-0.03:.3f}] to [{AOI_LON+0.03:.3f}, {AOI_LAT+0.03:.3f}])")
    print(f"Date Range:         {DATE_START} to {DATE_END}")
    print(f"Max Cloud Cover:    {MAX_CLOUD_COVER * 100}%")
    print("--------------------------------------------------")
    
    search_query = build_search_query()
    
    # Validation of API Key
    if not PLANET_API_KEY or PLANET_API_KEY.startswith("your_") or PLANET_API_KEY == "":
        print("\n[Status]: WAITING FOR PLANET API KEY")
        print("\nMissing or invalid PLANET_API_KEY environment variable.")
        print(f"Please add your Planet API Key to your env file at:\n  {ENV_PATH.resolve()}")
        print("\nExample line:")
        print("  PLANET_API_KEY=PL5a1bcde2fgh3ijklmn4opqrstuv5wxy")
        print("\nFor your reference, here is the exact JSON query body that will be sent:")
        print(json.dumps(search_query, indent=4))
        print("==================================================")
        sys.exit(0)
        
    print("\nSending query to Planet Quick-Search API...")
    
    session = requests.Session()
    session.auth = (PLANET_API_KEY, "")
    
    url = "https://api.planet.com/data/v1/quick-search"
    
    try:
        response = session.post(url, json=search_query)
        
        if response.status_code != 200:
            print(f"\nAPI Error (Status Code {response.status_code}):")
            print(response.text)
            sys.exit(1)
            
        data = response.json()
        features = data.get("features", [])
        
        # Sort features by cloud cover ascending (lowest cloud cover first)
        features.sort(key=lambda x: x.get("properties", {}).get("cloud_cover", 1.0))
        
        print(f"\nQuery completed. Found {len(features)} matching scenes:")
        print("--------------------------------------------------")
        
        if len(features) == 0:
            print("No cloud-free scenes found over Electronic City in the specified date range.")
            print("Try raising the MAX_CLOUD_COVER threshold or expanding the date window.")
            sys.exit(0)
            
        from datetime import datetime
        s2_date = datetime(2026, 2, 11)

        for i, feat in enumerate(features):
            props = feat.get("properties", {})
            feat_id = feat.get("id")
            acquired = props.get("acquired")
            cloud = props.get("cloud_cover", 0.0) * 100.0
            resolution = props.get("pixel_resolution", 3.0)
            bands = props.get("published_band_count", 4)
            instrument = props.get("instrument")
            
            # Calculate temporal gap to Sentinel-2 (Feb 11, 2026)
            try:
                acq_dt = datetime.strptime(acquired[:10], "%Y-%m-%d")
                gap = abs((acq_dt - s2_date).days)
                gap_str = f"{gap} days"
            except Exception:
                gap_str = "Unknown"
                
            permissions = feat.get("_permissions", [])
            downloadable = "Yes (orderable/downloadable via E&R license)" if ("assets:download" in permissions or len(permissions) > 0) else "Available to order on Planet Explorer"

            print(f"#{i+1} Scene ID: {feat_id}")
            print(f"   Acquired:   {acquired}")
            print(f"   Cloud Cover:{cloud:.2f}%")
            print(f"   Resolution: {resolution} m")
            print(f"   Bands:      {bands} bands (typically Coastal Blue, Blue, Green, Yellow, Red, Red Edge, NIR)")
            print(f"   AOI Intersect: Yes (centered at 12.85, 77.685 with 0.03 deg radius)")
            print(f"   Temporal Gap (S2 Feb 11): {gap_str}")
            print(f"   Download/Order Availability: {downloadable}")
            print(f"   Sensor:     {instrument}")
            print("--------------------------------------------------")

            
        print("\nRecommendation for Experiment 4:")
        print(f"Select the Scene closest to February 11, 2026, and download the 4-band analytic asset.")
        
    except Exception as e:
        print(f"\nConnection Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
