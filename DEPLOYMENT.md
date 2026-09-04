# TERRA-SR — Deployment & Operations Guide

## Overview
**TERRA-SR** is an AI Multispectral Super-Resolution & Geo-Accurate Satellite Intelligence Platform designed for high-throughput, offline-capable, and containerized deployment.

---

## 1. Prerequisites
- **Python**: Python 3.10+ (tested on Python 3.11 & Python 3.13)
- **Container Engine**: Docker 20.10+ / Podman (for containerized deployments)
- **Hardware Requirements**:
  - **CPU**: 2+ Cores (4+ Cores recommended for parallel downstream intelligence processing)
  - **RAM**: Minimum 4 GB (8 GB recommended for concurrent 2048×2048 tile inference)
  - **GPU**: Optional (NVIDIA CUDA automatically leveraged if available via PyTorch; CPU inference latency is ~850ms)

---

## 2. Docker Deployment

### A. Build Docker Image
```bash
docker build -t terra-sr:latest .
```

### B. Run Docker Container
```bash
docker run -d \
  --name terra-sr \
  -p 8080:8080 \
  --restart unless-stopped \
  terra-sr:latest
```

### C. Run with GPU Acceleration (NVIDIA Container Toolkit)
```bash
docker run -d \
  --name terra-sr-gpu \
  --gpus all \
  -p 8080:8080 \
  --restart unless-stopped \
  terra-sr:latest
```

---

## 3. Local / Bare-Metal Deployment

### A. Install Python Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### B. Start Application Server
```bash
python app/satellite_enhancer.py
```
The server will bind to `http://0.0.0.0:8080/`.

---

## 4. Configuration & Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PORT` | `8080` | HTTP port on which the daemon listens |
| `CDSE_USERNAME` | *(Optional)* | Copernicus Data Space Ecosystem API username |
| `CDSE_PASSWORD` | *(Optional)* | Copernicus Data Space Ecosystem API password |
| `PLANET_API_KEY` | *(Optional)* | PlanetScope API key for research validation scenes |

*(Note: Secrets are never baked into container images; pass environment variables via `-e` or `--env-file` if live satellite downloading is desired).*

---

## 5. Model Assets & Weights
The runtime requires the following model checkpoints in `models/`:
- `models/residual_srm_experiment3.pth` — Primary Validated Deep Residual CNN (5.0m GSD)
- `models/final_sr/best_model.pth` — Experimental PI-RCAN (3.33m reconstruction grid)
- `models/msrcan_experiment4d.pth` — MS-RCAN Channel Attention
- `models/hfsrm_experiment4e.pth` — High-Frequency Residual Attention

---

## 6. Health & Liveness Checks

### Health Endpoint
```http
GET /health
```
**Response**:
```json
{
  "status": "ok",
  "service": "TERRA-SR"
}
```

### Curl Verification
```bash
curl -f http://localhost:8080/health
```

---

## 7. Production Considerations & Reverse Proxy
For enterprise/production deployments, place an **Nginx** reverse proxy in front of the container:

```nginx
server {
    listen 80;
    server_name terra-sr.example.com;
    client_max_body_size 150M;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```
