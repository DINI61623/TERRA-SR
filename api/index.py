import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import the canonical TERRA-SR request handler
from app.satellite_enhancer import UniversalRequestHandler

# Vercel Serverless Function entrypoint handler
# Vercel's Python builder recognizes BaseHTTPRequestHandler / SimpleHTTPRequestHandler subclasses
handler = UniversalRequestHandler

if __name__ == "__main__":
    from app.satellite_enhancer import run_server
    run_server()
