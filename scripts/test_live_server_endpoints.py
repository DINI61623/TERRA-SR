#!/usr/bin/env python3
"""
Test local server endpoints:
GET /
GET /health
POST /api/enhance
and measure peak RSS.
"""

import os
import sys
import json
import time
import urllib.request
import threading
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.profile_memory_stages import get_process_rss_mb
from app.satellite_enhancer import run_server, PORT

def test_endpoints():
    server_url = f"http://127.0.0.1:{PORT}"
    
    # Wait for server to start
    time.sleep(1.0)
    
    print(f"Testing server at {server_url}")
    
    # 1. GET /
    t0 = time.time()
    with urllib.request.urlopen(f"{server_url}/") as resp:
        code_root = resp.getcode()
        body_root = resp.read()
    print(f"GET / -> Status {code_root} (Length: {len(body_root)} bytes, Latency: {(time.time()-t0)*1000:.1f}ms)")
    assert code_root == 200, f"GET / failed: {code_root}"
    
    # 2. GET /health
    t0 = time.time()
    with urllib.request.urlopen(f"{server_url}/health") as resp:
        code_health = resp.getcode()
        body_health = resp.read().decode('utf-8')
    print(f"GET /health -> Status {code_health} (Body: {body_health.strip()}, Latency: {(time.time()-t0)*1000:.1f}ms)")
    assert code_health == 200, f"GET /health failed: {code_health}"
    
    # 3. POST /api/enhance
    mem_before = get_process_rss_mb()
    print(f"Memory before POST /api/enhance: {mem_before:.2f} MB")
    
    t0 = time.time()
    req = urllib.request.Request(
        f"{server_url}/api/enhance",
        data=json.dumps({"model": "ResidualCNN"}).encode('utf-8'),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        code_enhance = resp.getcode()
        body_enhance = json.loads(resp.read().decode('utf-8'))
        
    enhance_time = time.time() - t0
    mem_after = get_process_rss_mb()
    
    print(f"POST /api/enhance -> Status {code_enhance} (Latency: {enhance_time:.2f}s)")
    print(f"Memory after POST /api/enhance: {mem_after:.2f} MB (Peak Working Set Delta: +{mem_after - mem_before:.2f} MB)")
    print("Metrics returned:")
    print(json.dumps(body_enhance.get("metrics", {}), indent=2))
    assert code_enhance == 200, f"POST /api/enhance failed: {code_enhance}"
    assert body_enhance.get("status") == "SUCCESS", f"Enhance payload status: {body_enhance.get('status')}"
    
    print("\n--- ALL LOCAL LIVE SERVER ENDPOINT TESTS PASSED ---")

if __name__ == "__main__":
    from http.server import HTTPServer
    from app.satellite_enhancer import UniversalRequestHandler
    
    httpd = HTTPServer(('127.0.0.1', PORT), UniversalRequestHandler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    
    try:
        test_endpoints()
    finally:
        httpd.shutdown()
