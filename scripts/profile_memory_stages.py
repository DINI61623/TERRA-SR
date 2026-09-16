#!/usr/bin/env python3
"""
TERRA-SR Memory Profiling Harness
Measures peak RSS and process memory for each stage of the pipeline.
"""

import os
import sys
import gc
import time
import tracemalloc
import ctypes
from ctypes import wintypes
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
torch.set_num_threads(1)
torch.set_grad_enabled(False)

from app.satellite_enhancer import generate_layer_assets, DEFAULT_INPUT_TIFF

class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ('cb', wintypes.DWORD),
        ('PageFaultCount', wintypes.DWORD),
        ('PeakWorkingSetSize', ctypes.c_size_t),
        ('WorkingSetSize', ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
        ('PagefileUsage', ctypes.c_size_t),
        ('PeakPagefileUsage', ctypes.c_size_t),
        ('PrivateUsage', ctypes.c_size_t),
    ]

def get_process_rss_mb():
    if sys.platform == "win32":
        try:
            counters = PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
            PROCESS_QUERY_INFORMATION = 0x0400
            PROCESS_VM_READ = 0x0010
            h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, os.getpid())
            get_mem_info = ctypes.windll.psapi.GetProcessMemoryInfo
            get_mem_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
            get_mem_info.restype = wintypes.BOOL
            res = get_mem_info(h, ctypes.byref(counters), counters.cb)
            ctypes.windll.kernel32.CloseHandle(h)
            if res:
                return counters.WorkingSetSize / (1024 * 1024)
        except Exception:
            pass
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0

def profile():
    gc.collect()
    start_rss = get_process_rss_mb()
    print(f"Startup RAM: {start_rss:.2f} MB")
    
    # 1. Input Loading
    import rasterio
    with rasterio.open(DEFAULT_INPUT_TIFF) as src:
        lr_data = src.read([1, 2, 3, 4]).astype(np.float32)
        if src.dtypes[0] == 'uint16':
            lr_data /= 10000.0
        elif src.dtypes[0] == 'uint8':
            lr_data /= 255.0
        lr_data = np.clip(lr_data, 0.0, 1.0)
        src_transform = src.transform
        src_crs = src.crs.to_string() if src.crs else "EPSG:32643"
    input_rss = get_process_rss_mb()
    print(f"Stage A (Input Loading): Current RSS = {input_rss:.2f} MB (Delta: +{input_rss - start_rss:.2f} MB)")
    
    # 2. Model Loading
    from src.super_resolution.inference import ProductionInference
    runner = ProductionInference(model_type="ResidualCNN")
    model_rss = get_process_rss_mb()
    print(f"Model RAM: {model_rss:.2f} MB")
    
    # 3. SR Inference
    tensor_in = torch.from_numpy(lr_data)
    hr_data = runner.enhance_tensor(tensor_in)
    del tensor_in
    sr_rss = get_process_rss_mb()
    print(f"SR peak RAM: {sr_rss:.2f} MB")
    
    # 4. Difference Analysis
    from src.intelligence.base import apply_colormap_lut, apply_percentile_stretch_uint8
    import torch.nn.functional as F
    scale_mult = hr_data.shape[1] // lr_data.shape[1]
    gsd_val = 10.0 / scale_mult
    from affine import Affine
    hr_transform = (src_transform @ Affine.scale(1.0 / scale_mult)) if src_transform else None
    
    lr_rgb = np.stack([lr_data[2], lr_data[1], lr_data[0]], axis=-1)
    hr_rgb = np.stack([hr_data[2], hr_data[1], hr_data[0]], axis=-1)
    lr_rgb_u8 = apply_percentile_stretch_uint8(lr_rgb)
    hr_rgb_u8 = apply_percentile_stretch_uint8(hr_rgb)
    del lr_rgb, hr_rgb
    
    with torch.inference_mode():
        t = torch.from_numpy(lr_rgb_u8).permute(2, 0, 1).unsqueeze(0).float()
        up = F.interpolate(t, size=(hr_rgb_u8.shape[0], hr_rgb_u8.shape[1]), mode='bilinear', align_corners=False)
        lr_upscaled_u8 = np.clip(up.squeeze(0).permute(1, 2, 0).numpy(), 0, 255).astype(np.uint8)
        del t, up
    diff_u8 = np.abs(hr_rgb_u8.astype(np.int16) - lr_upscaled_u8.astype(np.int16)).astype(np.uint8)
    del lr_upscaled_u8, lr_rgb_u8, hr_rgb_u8
    mean_abs_diff = float(np.mean(diff_u8)) / 255.0
    diff_mag = np.mean(diff_u8, axis=-1, dtype=np.float32) / 255.0
    del diff_u8
    diff_vis = apply_colormap_lut(np.clip(diff_mag * 3.5, 0.0, 1.0), "inferno")
    del diff_mag, diff_vis
    gc.collect()
    diff_rss = get_process_rss_mb()
    print(f"Difference peak RAM: {diff_rss:.2f} MB")
    
    # 5. Intelligence Stages
    from src.intelligence import run_intelligence_pipeline
    domains = ["water", "urban", "agriculture", "disaster", "oil_spill"]
    dom_rss = {}
    for dom in domains:
        gc.collect()
        rss_before = get_process_rss_mb()
        res = run_intelligence_pipeline(
            domain=dom,
            sr_cube=hr_data,
            lr_cube=lr_data,
            gsd=gsd_val,
            affine_transform=hr_transform,
            crs=src_crs
        )
        rss_after = get_process_rss_mb()
        dom_rss[dom] = rss_after
        del res
        gc.collect()
        
    print(f"Water peak RAM: {dom_rss['water']:.2f} MB")
    print(f"Urban peak RAM: {dom_rss['urban']:.2f} MB")
    print(f"Agriculture peak RAM: {dom_rss['agriculture']:.2f} MB")
    print(f"Disaster peak RAM: {dom_rss['disaster']:.2f} MB")
    print(f"Oil Spill peak RAM: {dom_rss['oil_spill']:.2f} MB")
    
    # 6. Full end-to-end generate_layer_assets
    print("\n--- Running Full End-to-End Pipeline ---")
    gc.collect()
    e2e_res = generate_layer_assets(DEFAULT_INPUT_TIFF, model_name="ResidualCNN")
    e2e_rss = get_process_rss_mb()
    print(f"Total /api/enhance peak RAM: {e2e_rss:.2f} MB")
    print("End-to-End Pipeline Completed Successfully!")

if __name__ == "__main__":
    profile()
