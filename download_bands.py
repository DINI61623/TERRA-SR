#!/usr/bin/env python3
"""
Sentinel-2 10m Bands Downloader
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

Downloads B02 (Blue), B03 (Green), B04 (Red), and B08 (NIR) directly from CDSE STAC assets.
"""

import os
import sys
import requests
from pathlib import Path

# Load credentials from .env (checking multiple standard paths)
try:
    from dotenv import load_dotenv
    # 1. Check current working directory
    load_dotenv()
    # 2. Check src/.env relative to script directory
    script_dir = Path(__file__).parent.resolve()
    load_dotenv(dotenv_path=script_dir / ".env")
    load_dotenv(dotenv_path=script_dir / "src" / ".env")
except ImportError:
    pass

# Try importing rasterio for image inspection fallback
try:
    import rasterio
except ImportError:
    rasterio = None


TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
STAC_SEARCH_URL = "https://catalogue.dataspace.copernicus.eu/stac/search"
OUTPUT_DIR = Path("data/raw/sentinel2")


def get_access_token(username, password):
    """
    Authenticate with CDSE to obtain OAuth2 token.
    """
    print("Authenticating with Copernicus CDSE...")
    data = {
        'client_id': 'cdse-public',
        'username': username,
        'password': password,
        'grant_type': 'password'
    }
    try:
        response = requests.post(TOKEN_URL, data=data, timeout=30)
        response.raise_for_status()
        return response.json()['access_token']
    except Exception as e:
        print(f"Authentication failed: {e}")
        print("Please check your CDSE_USERNAME and CDSE_PASSWORD in your .env file.")
        sys.exit(1)


def fetch_stac_item(product_id):
    """
    Query the STAC search API for the specific product ID.
    """
    print(f"Querying STAC API for product: {product_id}...")
    payload = {
        "collections": ["sentinel-2-l2a"],
        "ids": [product_id]
    }
    try:
        response = requests.post(STAC_SEARCH_URL, json=payload, timeout=30)
        response.raise_for_status()
        features = response.json().get("features", [])
        if not features:
            print(f"Product {product_id} not found in sentinel-2-l2a collection.")
            sys.exit(1)
        return features[0]
    except Exception as e:
        print(f"Error querying STAC API: {e}")
        sys.exit(1)


def download_file(url, dest_path, headers):
    """
    Download a file from an HTTPS URL with progress bar.
    """
    print(f"Downloading: {dest_path.name}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with requests.get(url, headers=headers, stream=True, timeout=60) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            chunk_size = 1024 * 1024  # 1MB chunks
            downloaded = 0
            
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            mb_downloaded = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            sys.stdout.write(
                                f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)"
                            )
                        else:
                            mb_downloaded = downloaded / (1024 * 1024)
                            sys.stdout.write(f"\r  Progress: {mb_downloaded:.1f} MB downloaded")
                        sys.stdout.flush()
            print("\n  Download complete!")
            return True
    except Exception as e:
        print(f"\n  Error downloading file: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def main():
    product_id = "S2B_MSIL2A_20260211T050839_N0512_R019_T43PGQ_20260211T085923"
    
    # Retrieve Credentials
    username = os.getenv("CDSE_USERNAME")
    password = os.getenv("CDSE_PASSWORD")
    
    if not username or not password:
        print("Error: CDSE credentials (CDSE_USERNAME and CDSE_PASSWORD) must be set in your environment or .env file.")
        print("Please set them up to proceed with downloading.")
        sys.exit(1)
        
    # Get OAuth token
    token = get_access_token(username, password)
    headers = {"Authorization": f"Bearer {token}"}
    
    # Get STAC Metadata
    item = fetch_stac_item(product_id)
    assets = item.get("assets", {})
    
    # Targets: 10m bands
    target_bands = {
        "B02_10m": "Blue",
        "B03_10m": "Green",
        "B04_10m": "Red",
        "B08_10m": "NIR"
    }
    
    downloaded_files = []
    
    print("\nStarting downloads for 10m bands...")
    for asset_key, band_name in target_bands.items():
        if asset_key not in assets:
            print(f"Warning: Band asset {asset_key} not found in metadata.")
            continue
            
        asset_info = assets[asset_key]
        
        # Extract HTTPS alternate href download URL
        download_url = asset_info.get("alternate", {}).get("https", {}).get("href")
        if not download_url:
            # Fallback to direct href if alternate HTTPS is not specified
            download_url = asset_info.get("href")
            
        if not download_url:
            print(f"Warning: No valid download URL found for {asset_key}.")
            continue
            
        # Define local filename based on path or name
        local_filename = f"{product_id}_{band_name}_10m.jp2"
        dest_path = OUTPUT_DIR / local_filename
        
        success = download_file(download_url, dest_path, headers)
        if success:
            downloaded_files.append((dest_path, band_name, asset_info))
            
    print("\n==================================================")
    print("Download Summary and Inspection Results:")
    print("==================================================")
    
    for path, band, metadata in downloaded_files:
        file_size_bytes = path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        # Read dimensions and CRS (Prefer direct file inspection using rasterio)
        dimensions = None
        crs = None
        
        if rasterio is not None:
            try:
                with rasterio.open(path) as src:
                    dimensions = f"{src.width} x {src.height}"
                    crs = src.crs.to_string() if src.crs else "None"
            except Exception as e:
                print(f"[Error inspecting file {path.name} with rasterio: {e}]")
                
        # Fallback to STAC metadata properties if rasterio is missing or failed
        if not dimensions:
            shape = metadata.get("proj:shape")
            if shape and len(shape) == 2:
                dimensions = f"{shape[1]} x {shape[0]} (from STAC metadata)"
            else:
                dimensions = "Unknown"
                
        if not crs:
            crs = metadata.get("proj:code", "Unknown (from STAC metadata)")
            
        print(f"\nBand Name:    {band} ({path.name})")
        print(f"File Size:    {file_size_mb:.2f} MB")
        print(f"Dimensions:   {dimensions}")
        print(f"CRS Code:     {crs}")
        print("-" * 50)


if __name__ == "__main__":
    main()
