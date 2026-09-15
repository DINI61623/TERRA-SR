#!/usr/bin/env python3
"""
TERRA-SR Copernicus Data Space Ecosystem (CDSE) Authentication Manager
Handles OAuth2 token acquisition, expiration management, and credential fallback.
"""

import os
import time
import requests
from typing import Optional, Dict, Any
from pathlib import Path

# Load environment variables if available
try:
    from dotenv import load_dotenv
    root_dir = Path(__file__).resolve().parent.parent.parent
    load_dotenv(dotenv_path=root_dir / ".env")
    load_dotenv(dotenv_path=root_dir / "src" / ".env")
except ImportError:
    pass

CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"


class CopernicusAuthManager:
    """
    Manages OAuth2 access tokens for Copernicus Data Space Ecosystem.
    Supports:
    1. Client Credentials (COPERNICUS_CLIENT_ID / COPERNICUS_CLIENT_SECRET)
    2. User Credentials (CDSE_USERNAME / CDSE_PASSWORD with client_id='cdse-public')
    """
    _cached_token: Optional[str] = None
    _token_expiry: float = 0.0
    _manual_connected: bool = False

    @classmethod
    def get_credentials(cls) -> Dict[str, Optional[str]]:
        """Extracts available credentials from environment variables."""
        return {
            "client_id": os.getenv("COPERNICUS_CLIENT_ID") or os.getenv("CDSE_CLIENT_ID"),
            "client_secret": os.getenv("COPERNICUS_CLIENT_SECRET") or os.getenv("CDSE_CLIENT_SECRET"),
            "username": os.getenv("CDSE_USERNAME") or os.getenv("COPERNICUS_USERNAME"),
            "password": os.getenv("CDSE_PASSWORD") or os.getenv("COPERNICUS_PASSWORD")
        }

    @classmethod
    def is_configured(cls) -> bool:
        """Returns True if minimum required credentials for CDSE exist in environment."""
        creds = cls.get_credentials()
        has_client = bool(creds["client_id"] and creds["client_secret"])
        has_user = bool(creds["username"] and creds["password"])
        return has_client or has_user

    @classmethod
    def get_access_token(cls, force_refresh: bool = False) -> Optional[str]:
        """
        Retrieves a valid OAuth2 bearer token. Caches token until 60s before expiry.
        Returns None if credentials are not configured or authentication fails.
        """
        now = time.time()
        if not force_refresh and cls._cached_token and now < cls._token_expiry:
            return cls._cached_token

        creds = cls.get_credentials()
        if not cls.is_configured():
            return None

        # 1. Attempt Client Credentials Grant
        if creds["client_id"] and creds["client_secret"]:
            data = {
                "grant_type": "client_credentials",
                "client_id": creds["client_id"],
                "client_secret": creds["client_secret"]
            }
        # 2. Attempt Password Grant
        elif creds["username"] and creds["password"]:
            data = {
                "grant_type": "password",
                "client_id": "cdse-public",
                "username": creds["username"],
                "password": creds["password"]
            }
        else:
            return None

        try:
            resp = requests.post(CDSE_TOKEN_URL, data=data, timeout=5)
            if resp.status_code == 200:
                body = resp.json()
                cls._cached_token = body.get("access_token")
                expires_in = body.get("expires_in", 300)
                cls._token_expiry = now + expires_in - 60.0  # 60s buffer
                cls._manual_connected = True
                return cls._cached_token
            else:
                print(f"[Copernicus Auth] Authentication failed ({resp.status_code}): {resp.text[:200]}")
                return None
        except Exception as e:
            print(f"[Copernicus Auth] Connection error: {e}")
            return None

    @classmethod
    def connect(cls, client_id: Optional[str] = None, client_secret: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes OAuth2 connection sequence using server-side environment variables or provided credentials.
        Never exposes secrets in response.
        """
        if client_id and client_secret:
            os.environ["COPERNICUS_CLIENT_ID"] = client_id
            os.environ["COPERNICUS_CLIENT_SECRET"] = client_secret
            cls._cached_token = None
            cls._token_expiry = 0.0

        if not cls.is_configured():
            return {
                "status": "UNCONFIGURED",
                "connected": False,
                "message": (
                    "Copernicus credentials not found in environment. "
                    "Please set COPERNICUS_CLIENT_ID & COPERNICUS_CLIENT_SECRET or run in Demo Mode."
                ),
                "access": "Offline / Demo Mission Available"
            }

        token = cls.get_access_token(force_refresh=True)
        if token:
            cls._manual_connected = True
            return {
                "status": "CONNECTED",
                "connected": True,
                "access": "Sentinel-2 L2A available",
                "auth_method": "OAuth2 Client Credentials" if os.getenv("COPERNICUS_CLIENT_ID") else "CDSE User Credentials",
                "expires_in_seconds": max(0, int(cls._token_expiry - time.time()))
            }
        else:
            return {
                "status": "ERROR",
                "connected": False,
                "message": "Copernicus authentication server rejected credentials or timed out.",
                "access": "Offline / Demo Mission Available"
            }

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        """
        Returns connection state without exposing secrets.
        """
        now = time.time()
        is_token_active = bool(cls._cached_token and now < cls._token_expiry)
        configured = cls.is_configured()
        
        return {
            "status": "CONNECTED" if (is_token_active or cls._manual_connected) else ("CONFIGURED" if configured else "DISCONNECTED"),
            "connected": bool(is_token_active or cls._manual_connected),
            "configured": configured,
            "token_valid": is_token_active,
            "access": "Sentinel-2 L2A available" if (is_token_active or cls._manual_connected) else "Offline / Demo Mode"
        }

    @classmethod
    def disconnect(cls) -> Dict[str, Any]:
        """
        Clears local cached session token.
        """
        cls._cached_token = None
        cls._token_expiry = 0.0
        cls._manual_connected = False
        return {
            "status": "DISCONNECTED",
            "connected": False,
            "message": "Copernicus session disconnected and token cache cleared."
        }

    @classmethod
    def get_auth_headers(cls) -> Dict[str, str]:
        """Returns HTTP headers dictionary with Authorization Bearer if available."""
        token = cls.get_access_token()
        if token:
            return {"Authorization": f"Bearer {token}"}
        return {}
