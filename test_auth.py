#!/usr/bin/env python3
"""
Copernicus Data Space Ecosystem (CDSE) Authentication Tester
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

Tests connection to the CDSE authentication identity server using CDSE_USERNAME and CDSE_PASSWORD.
"""

import os
import sys
import requests

# Load dotenv to read .env file (checking multiple standard paths)
try:
    from dotenv import load_dotenv
    from pathlib import Path
    # 1. Check current working directory
    load_dotenv()
    # 2. Check src/.env relative to script directory
    script_dir = Path(__file__).parent.resolve()
    load_dotenv(dotenv_path=script_dir / ".env")
    load_dotenv(dotenv_path=script_dir / "src" / ".env")
except ImportError:
    pass

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"


def test_authentication():
    # Retrieve CDSE credentials
    username = os.getenv("CDSE_USERNAME")
    password = os.getenv("CDSE_PASSWORD")

    if not username or not password:
        print("Error: CDSE credentials not found.")
        print("Please verify that CDSE_USERNAME and CDSE_PASSWORD are set in your environment or .env file.")
        sys.exit(1)

    print("Checking CDSE authentication credentials...")
    
    # Official CDSE authentication flow
    data = {
        'client_id': 'cdse-public',
        'username': username,
        'password': password,
        'grant_type': 'password'
    }

    try:
        # Secure POST request to fetch token (timeout 15s)
        response = requests.post(TOKEN_URL, data=data, timeout=15)
        
        # Check HTTP status codes
        if response.status_code == 200:
            token_json = response.json()
            # Double check that we actually got an access token
            if 'access_token' in token_json:
                print("=========================================")
                print("CDSE Authentication Status: SUCCESS!")
                print("=========================================")
                print("Successfully obtained an access token securely.")
                print("(Note: Access token has been hidden for security)")
                return True
            else:
                print("Error: Server response did not contain 'access_token'.")
                return False
                
        elif response.status_code == 401:
            print("=========================================")
            print("CDSE Authentication Status: FAILED")
            print("=========================================")
            print("HTTP 401: Unauthorized. Invalid username or password.")
            return False
            
        else:
            print("=========================================")
            print("CDSE Authentication Status: FAILED")
            print("=========================================")
            print(f"Server returned unexpected status code: {response.status_code}")
            print(f"Details: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print("Error: Connection to CDSE identity server timed out. Please check your network connection.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Network error connecting to CDSE: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during authentication: {e}")
        return False


if __name__ == "__main__":
    success = test_authentication()
    sys.exit(0 if success else 1)
