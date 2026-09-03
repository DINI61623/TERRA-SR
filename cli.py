#!/usr/bin/env python3
"""
TERRA-SR Headless Batch Processing & Geospatial Intelligence CLI.
Usage:
    python cli.py --input <path_to_geotiff> --model <ResidualCNN|MSRCAN|HFSRM|PIRCAN|Bilinear> --domain <all|water|disaster|urban|agriculture|oil_spill> --output_dir <path>
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np

# Set stdout/stderr to UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add workspace root to Python path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

import torch
from src.core.input_validation import SatelliteInputValidator
from src.core.georeference import verify_georeferencing_integrity
from src.super_resolution.inference import ProductionInference, MODEL_REGISTRY
from src.intelligence import run_intelligence_pipeline, INTELLIGENCE_REGISTRY

try:
    import rasterio
    from affine import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def main():
    parser = argparse.ArgumentParser(
        description="TERRA-SR Headless Batch Processing & Geospatial Intelligence CLI"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="outputs/s2_5m_upscaled_bilinear.tiff",
        help="Path to input Sentinel-2 GeoTIFF raster"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="ResidualCNN",
        choices=["ResidualCNN", "MSRCAN", "HFSRM", "PIRCAN", "Bilinear"],
        help="Super-Resolution model architecture"
    )
    parser.add_argument(
        "--domain", "-d",
        type=str,
        default="all",
        choices=["all", "water", "disaster", "urban", "agriculture", "oil_spill"],
        help="Downstream satellite intelligence domain to execute"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="outputs/cli_exports",
        help="Directory to save generated GeoTIFF, GeoJSON, and report files"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-essential terminal outputs"
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print("=" * 65)
        print("TERRA-SR Geospatial Intelligence Batch CLI v3.0")
        print("=" * 65)
        print(f"* Input Scene:   {input_path.name}")
        print(f"* Model Engine:  {args.model}")
        print(f"* Target Domain: {args.domain.upper()}")
        print(f"* Output Dir:    {output_dir}")
        print("-" * 65)

    # 1. Strict Input Validation
    if not args.quiet:
        print("[1/4] Performing Strict Input Validation...", flush=True)
    val_res = SatelliteInputValidator.validate_for_domain(input_path, "super_resolution")
    if not val_res.is_valid:
        print(f"\n[FATAL ERROR] Input validation failed for {input_path.name}:", file=sys.stderr)
        for r in val_res.reasons:
            print(f"  [!] {r}", file=sys.stderr)
        sys.exit(1)
    
    if not args.quiet:
        print(f"      [OK] Status: {val_res.status} ({val_res.metadata.get('product_type')})")
        print(f"      [OK] CRS: {val_res.metadata.get('crs')} | GSD: {val_res.metadata.get('gsd'):.2f}m")

    # 2. Super-Resolution Inference
    if not args.quiet:
        print(f"\n[2/4] Executing AI Super-Resolution via {args.model}...", flush=True)
    start_sr = time.time()
    
    with rasterio.open(input_path) as src:
        lr_data = src.read([1, 2, 3, 4]).astype(np.float32)
        if src.dtypes[0] == 'uint16':
            lr_data /= 10000.0
        elif src.dtypes[0] == 'uint8':
            lr_data /= 255.0
        lr_data = np.clip(lr_data, 0.0, 1.0)
        src_transform = src.transform
        src_crs = src.crs.to_string() if src.crs else "EPSG:32643"

    runner = ProductionInference(model_type=args.model)
    tensor_in = torch.tensor(lr_data, dtype=torch.float32)
    hr_data = runner.enhance_tensor(tensor_in)
    
    scale_mult = hr_data.shape[1] // lr_data.shape[1]
    gsd_val = 10.0 / scale_mult
    hr_transform = src_transform * Affine.scale(1.0 / scale_mult) if src_transform else None
    elapsed_sr = time.time() - start_sr

    # Export Enhanced GeoTIFF
    enhanced_tiff_path = output_dir / f"enhanced_{args.model.lower()}_cube.tiff"
    if HAS_RASTERIO and hr_transform is not None:
        with rasterio.open(
            enhanced_tiff_path,
            "w",
            driver="GTiff",
            height=hr_data.shape[1],
            width=hr_data.shape[2],
            count=hr_data.shape[0],
            dtype="float32",
            crs=src_crs,
            transform=hr_transform
        ) as dst:
            for b_idx in range(hr_data.shape[0]):
                dst.write(hr_data[b_idx], b_idx + 1)

    if not args.quiet:
        print(f"      [OK] Enhanced shape: {hr_data.shape[1]} x {hr_data.shape[2]} px (<{gsd_val:.2f}m GSD)")
        print(f"      [OK] Inference latency: {elapsed_sr:.2f}s")
        print(f"      [OK] Exported GeoTIFF: {enhanced_tiff_path.name}")

    # 3. Downstream Intelligence Execution
    target_domains = list(INTELLIGENCE_REGISTRY.keys()) if args.domain == "all" else [args.domain]
    if not args.quiet:
        print(f"\n[3/4] Executing Downstream Intelligence for domains: {target_domains}...", flush=True)

    results_summary = {}
    for dom in target_domains:
        t_dom = time.time()
        res = run_intelligence_pipeline(
            domain=dom,
            sr_cube=hr_data,
            lr_cube=lr_data,
            gsd=gsd_val,
            affine_transform=hr_transform,
            crs=src_crs
        )
        e_dom = time.time() - t_dom
        
        # Save GeoJSON
        geojson_out = output_dir / f"{dom}_vectors.geojson"
        with open(geojson_out, "w") as f:
            json.dump(res.get("geojson_full", res["report"].get("geojson", {})), f, indent=2)
            
        # Save Report
        report_out = output_dir / f"{dom}_report.json"
        with open(report_out, "w") as f:
            json.dump(res["report"], f, indent=4)
            
        feat_cnt = res["report"].get("geojson_count", len(res.get("geojson_full", {}).get("features", [])))
        results_summary[dom] = {
            "status": res["report"].get("status"),
            "features_count": feat_cnt,
            "latency_s": round(e_dom, 3)
        }
        if not args.quiet:
            print(f"      [OK] [{dom.upper()}] Extracted {feat_cnt} features ({e_dom:.2f}s) -> {geojson_out.name}")

    # 4. Final Mission Portfolio
    summary_report = {
        "platform": "TERRA-SR Geospatial Intelligence Suite",
        "version": "3.0",
        "cli_execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_scene": str(input_path),
        "input_metadata": val_res.metadata,
        "super_resolution": {
            "model": args.model,
            "latency_s": round(elapsed_sr, 3),
            "output_gsd": f"{gsd_val:.2f}m",
            "output_dimensions": f"{hr_data.shape[2]} x {hr_data.shape[1]} px",
            "enhanced_geotiff": str(enhanced_tiff_path)
        },
        "intelligence_domains": results_summary
    }

    summary_file = output_dir / "mission_execution_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_report, f, indent=4)

    if not args.quiet:
        print("\n" + "=" * 65)
        print("EXECUTION COMPLETE [OK]")
        print(f"* Summary Report: {summary_file}")
        print(f"* Total Deliverables in: {output_dir}")
        print("=" * 65)

    return 0


if __name__ == "__main__":
    sys.exit(main())
