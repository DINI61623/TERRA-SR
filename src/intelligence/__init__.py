#!/usr/bin/env python3
"""
TERRA-SR Downstream Satellite Intelligence Suite
Package exposing standardized intelligence modules: Urban, Agriculture, Water, Oil Spill, and Disaster.
"""

from typing import Dict, Any, Optional
import numpy as np

from src.intelligence.base import BaseIntelligenceModule
from src.intelligence.urban import UrbanIntelligenceModule
from src.intelligence.agriculture import AgricultureIntelligenceModule
from src.intelligence.water import WaterIntelligenceModule
from src.intelligence.oil_spill_adapter import OilSpillIntelligenceModule
from src.intelligence.disaster import DisasterIntelligenceModule

INTELLIGENCE_REGISTRY = {
    "urban": UrbanIntelligenceModule,
    "agriculture": AgricultureIntelligenceModule,
    "water": WaterIntelligenceModule,
    "oil_spill": OilSpillIntelligenceModule,
    "disaster": DisasterIntelligenceModule
}

MODULE_INSTANCES = {}


def get_intelligence_module(domain: str) -> BaseIntelligenceModule:
    """Retrieves or instantiates the singleton module for a domain."""
    domain = domain.lower().replace("-", "_").replace(" ", "_")
    if domain not in INTELLIGENCE_REGISTRY:
        raise ValueError(f"Unknown intelligence domain: {domain}. Available: {list(INTELLIGENCE_REGISTRY.keys())}")
    
    if domain not in MODULE_INSTANCES:
        MODULE_INSTANCES[domain] = INTELLIGENCE_REGISTRY[domain]()
    return MODULE_INSTANCES[domain]


def run_intelligence_pipeline(
    domain: str,
    sr_cube: np.ndarray,
    lr_cube: Optional[np.ndarray] = None,
    gsd: float = 3.33,
    affine_transform: Any = None,
    crs: str = "EPSG:32643",
    bounds: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Unified entrypoint to run any downstream satellite intelligence module.
    """
    module = get_intelligence_module(domain)
    return module.process(
        sr_cube=sr_cube,
        lr_cube=lr_cube,
        gsd=gsd,
        affine_transform=affine_transform,
        crs=crs,
        bounds=bounds
    )
