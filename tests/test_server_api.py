import json
import threading
import time
import urllib.request
import urllib.parse
from pathlib import Path
import pytest
import http.server
import socketserver

from app.satellite_enhancer import UniversalRequestHandler, PORT

@pytest.fixture(scope="module")
def live_server():
    test_port = 8899
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    server = socketserver.ThreadingTCPServer(("127.0.0.1", test_port), UniversalRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.5)
    yield f"http://127.0.0.1:{test_port}"
    server.shutdown()
    server.server_close()


class TestServerEndpoints:
    def test_health_check(self, live_server):
        with urllib.request.urlopen(f"{live_server}/health") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode('utf-8'))
            assert data["status"] == "ok"
            assert data["service"] == "TERRA-SR"

    def test_copernicus_status_endpoint(self, live_server):
        with urllib.request.urlopen(f"{live_server}/api/copernicus/status") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode('utf-8'))
            assert "configured" in data
            assert "status" in data
            # Ensure client_secret is NEVER returned in response
            assert "client_secret" not in data
            assert "COPERNICUS_CLIENT_SECRET" not in data

    def test_copernicus_connect_and_disconnect(self, live_server):
        # Connect
        req = urllib.request.Request(
            f"{live_server}/api/copernicus/connect",
            data=json.dumps({"client_id": "test_id"}).encode('utf-8'),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode('utf-8'))
            assert "status" in data

        # Disconnect
        req_disc = urllib.request.Request(
            f"{live_server}/api/copernicus/disconnect",
            data=b"{}",
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req_disc) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode('utf-8'))
            assert data["status"] == "DISCONNECTED"

    def test_satellite_search(self, live_server):
        req = urllib.request.Request(
            f"{live_server}/api/satellite/search",
            data=json.dumps({
                "bbox": [77.65, 12.82, 77.72, 12.89],
                "startDate": "2026-01-01",
                "endDate": "2026-03-01",
                "maxCloud": 10.0,
                "minCoverage": 80.0
            }).encode('utf-8'),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode('utf-8'))
            assert data["status"] == "SUCCESS"
            assert "scenes" in data
            assert len(data["scenes"]) > 0
            assert "recommended_scene" in data

    def test_enhance_and_canonical_analysis_result(self, live_server):
        # Trigger Enhance
        req = urllib.request.Request(
            f"{live_server}/api/enhance",
            data=json.dumps({"model": "ResidualCNN"}).encode('utf-8'),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode('utf-8'))
            assert data["status"] == "SUCCESS"
            assert "metrics" in data
            assert data["metrics"]["psnr"] > 30.0

        # Fetch Canonical AnalysisResult
        with urllib.request.urlopen(f"{live_server}/api/analysis_result") as resp:
            assert resp.status == 200
            res_data = json.loads(resp.read().decode('utf-8'))
            assert "mission_id" in res_data
            assert "metrics" in res_data
            assert "difference" in res_data
            assert "intelligence" in res_data
            assert "evidence_level" in res_data
            assert res_data["evidence_level"] == "OPERATIONAL / NO HIGH-RESOLUTION REFERENCE"

    def test_deliverables_generation_endpoints(self, live_server):
        # 1. Report Generation
        req_rep = urllib.request.Request(f"{live_server}/api/generate_report", data=b"{}", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req_rep) as resp:
            assert resp.status == 200
            rep_data = json.loads(resp.read().decode('utf-8'))
            assert rep_data["status"] == "SUCCESS"
            assert rep_data["filename"] == "terra_sr_research_report.pdf"

        # 2. Narration Generation
        req_narr = urllib.request.Request(
            f"{live_server}/api/generate_narration",
            data=json.dumps({"mode": "Judge"}).encode('utf-8'),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req_narr) as resp:
            assert resp.status == 200
            narr_data = json.loads(resp.read().decode('utf-8'))
            assert narr_data["status"] == "SUCCESS"
            assert "script" in narr_data
            assert "TERRA-SR" in narr_data["script"]

        # 3. Package Export
        req_pkg = urllib.request.Request(f"{live_server}/api/export_package", data=b"{}", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req_pkg) as resp:
            assert resp.status == 200
            pkg_data = json.loads(resp.read().decode('utf-8'))
            assert pkg_data["status"] == "SUCCESS"
            assert pkg_data["filename"].endswith(".zip")

    def test_run_all_intelligence(self, live_server):
        req = urllib.request.Request(f"{live_server}/api/run_all_intelligence", data=b"{}", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode('utf-8'))
            assert data["status"] == "SUCCESS"
            assert "water" in data["reports"]
            assert "agriculture" in data["reports"]
            assert "urban" in data["reports"]
            assert "disaster" in data["reports"]
            assert "oil_spill" in data["reports"]

    def test_download_artifacts(self, live_server):
        # Test PDF download
        with urllib.request.urlopen(f"{live_server}/api/download_artifact?type=pdf") as resp:
            assert resp.status == 200
            assert resp.headers.get("Content-Type") == "application/pdf"
            content = resp.read()
            assert content.startswith(b"%PDF-")

        # Test Narration download
        with urllib.request.urlopen(f"{live_server}/api/download_artifact?type=narration") as resp:
            assert resp.status == 200
            assert "text/plain" in resp.headers.get("Content-Type")
            content = resp.read().decode('utf-8')
            assert len(content) > 50

    def test_ui_basemap_configuration_and_fallback(self, live_server):
        # 1. Verify GET / serves HTML with createBasemapTileLayer and OSM fallback URL
        with urllib.request.urlopen(f"{live_server}/") as resp:
            assert resp.status == 200
            html_text = resp.read().decode('utf-8')
            assert "createBasemapTileLayer" in html_text
            assert "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" in html_text
            assert "OpenStreetMap" in html_text
            # Verify raw template placeholder is replaced
            assert "__CARTO_API_KEY__" not in html_text

        # 2. Verify GET /api/satellite/config exposes carto_api_key_configured
        with urllib.request.urlopen(f"{live_server}/api/satellite/config") as resp:
            assert resp.status == 200
            config_data = json.loads(resp.read().decode('utf-8'))
            assert "carto_api_key_configured" in config_data
            assert isinstance(config_data["carto_api_key_configured"], bool)

