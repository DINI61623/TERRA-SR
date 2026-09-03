#!/usr/bin/env python3
"""
Sentinel-2 Super-Resolution Mapping (SRM) - Physical Degradation Pipeline
For SIH 2026 - Deep Learning Based Super Resolution Mapping (SRM) (SIH26142)

Implements a modular, physically motivated degradation pipeline to simulate
low-resolution Sentinel-2-like observations from high-resolution reference imagery (e.g. PlanetScope).
Each stage of the pipeline can be independently enabled/disabled via configuration.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Union, Tuple


class PhysicalDegradationPipeline(nn.Module):
    """
    Modular pipeline that degrades a high-resolution reference image tensor (or numpy array)
    into a low-resolution Sentinel-2-like observation tensor.
    
    The pipeline consists of:
    1. Radiometric Normalization (scaling raw DNs to physical reflectance)
    2. Spatial PSF Blurring (band-specific Gaussian blur convolution)
    3. Spectral Response Matching (calibrating spectral bands via linear regression)
    4. Sensor Noise Model (injecting band-specific additive Gaussian instrument noise)
    5. Spatial Sampling / Decimation (downsampling convolved pixels to low-resolution grid)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the degradation pipeline and validates configuration parameters.
        
        Args:
            config (dict): Parsed configuration dictionary matching configs/experiment4_degradation.yaml structure.
        """
        super(PhysicalDegradationPipeline, self).__init__()
        self.config = config
        self._validate_config()

    def _validate_config(self):
        """
        Verifies that all required sensor-specific parameters are provided in the configuration.
        Raises ValueError with detailed documentation requirements if parameters are missing.
        """
        # Validate PSF
        psf_cfg = self.config.get("psf", {})
        if psf_cfg.get("enabled", False):
            sigmas = psf_cfg.get("band_sigmas")
            if sigmas is None:
                raise ValueError(
                    "[Degradation Pipeline Blocked] Missing required parameter 'psf.band_sigmas'.\n"
                    "Requirement Source: Authoritative ESA Sentinel-2 Point Spread Function (PSF) specs.\n"
                    "Description: Must provide a list of 4 floats representing Gaussian blur sigma (in HR pixels) "
                    "for [Blue, Green, Red, NIR] bands to model optical and atmospheric light scattering."
                )
            if not isinstance(sigmas, (list, tuple)) or len(sigmas) != 4:
                raise ValueError("'psf.band_sigmas' must be a list/tuple of 4 floats.")

        # Validate Spectral Matching
        spec_cfg = self.config.get("spectral_matching", {})
        if spec_cfg.get("enabled", False):
            slopes = spec_cfg.get("slopes")
            intercepts = spec_cfg.get("intercepts")
            if slopes is None or intercepts is None:
                raise ValueError(
                    "[Degradation Pipeline Blocked] Missing required parameters 'spectral_matching.slopes' or 'spectral_matching.intercepts'.\n"
                    "Requirement Source: Empirical linear regression over overlapping cloud-free targets.\n"
                    "Description: Must provide lists of 4 floats representing slopes (alpha) and intercepts (beta) "
                    "for [Blue, Green, Red, NIR] bands to adjust PlanetScope sensor sensitivities to Sentinel-2 equivalents."
                )
            if not isinstance(slopes, (list, tuple)) or len(slopes) != 4:
                raise ValueError("'spectral_matching.slopes' must be a list/tuple of 4 floats.")
            if not isinstance(intercepts, (list, tuple)) or len(intercepts) != 4:
                raise ValueError("'spectral_matching.intercepts' must be a list/tuple of 4 floats.")

        # Validate Sensor Noise
        noise_cfg = self.config.get("sensor_noise", {})
        if noise_cfg.get("enabled", False):
            sigmas = noise_cfg.get("band_sigmas")
            if sigmas is None:
                raise ValueError(
                    "[Degradation Pipeline Blocked] Missing required parameter 'sensor_noise.band_sigmas'.\n"
                    "Requirement Source: Noise estimations from homogeneous targets in target Sentinel-2 scenes.\n"
                    "Description: Must provide a list of 4 floats representing additive white Gaussian noise standard deviation "
                    "(in reflectance units) for [Blue, Green, Red, NIR] bands to model sensor instrumentation noise."
                )
            if not isinstance(sigmas, (list, tuple)) or len(sigmas) != 4:
                raise ValueError("'sensor_noise.band_sigmas' must be a list/tuple of 4 floats.")

        # Validate Spatial Decimation
        dec_cfg = self.config.get("spatial_decimation", {})
        if dec_cfg.get("enabled", False):
            factor = dec_cfg.get("upscale_factor")
            mode = dec_cfg.get("interpolation_mode")
            if factor is None:
                raise ValueError("Missing 'spatial_decimation.upscale_factor'. Must specify integer (e.g. 2 or 4).")
            if mode not in ["area", "bilinear", "bicubic", "nearest"]:
                raise ValueError(
                    f"Unsupported interpolation mode '{mode}'. Must be one of: ['area', 'bilinear', 'bicubic', 'nearest']."
                )

    def _gaussian_kernel_2d(self, kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
        """
        Dynamically constructs a 2D circularly symmetric Gaussian kernel in PyTorch.
        """
        if sigma <= 0.0:
            # Dirac delta equivalent if sigma is zero or negative
            kernel = torch.zeros((kernel_size, kernel_size), dtype=torch.float32, device=device)
            kernel[kernel_size // 2, kernel_size // 2] = 1.0
            return kernel

        grid = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32, device=device)
        y, x = torch.meshgrid(grid, grid, indexing='ij')
        kernel = torch.exp(-(x**2 + y**2) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel

    def apply_radiometric_normalization(self, x: torch.Tensor) -> torch.Tensor:
        """
        Divides by the input scale (e.g., to convert raw DNs to reflectance) and clips output range.
        """
        rad_cfg = self.config.get("radiometric_normalization", {})
        if not rad_cfg.get("enabled", False):
            return x

        input_scale = rad_cfg.get("input_scale", 1.0)
        clip_min = rad_cfg.get("clip_min", 0.0)
        clip_max = rad_cfg.get("clip_max", 1.0)

        # Scale raw input values
        if input_scale != 1.0:
            x = x / input_scale

        # Clip values to valid reflectance bounds
        x = torch.clamp(x, min=clip_min, max=clip_max)
        return x

    def apply_psf_blur(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies band-specific 2D Gaussian blur convolution using depthwise 2D convolutions.
        """
        psf_cfg = self.config.get("psf", {})
        if not psf_cfg.get("enabled", False):
            return x

        kernel_size = psf_cfg.get("kernel_size", 15)
        band_sigmas = psf_cfg.get("band_sigmas")
        channels = x.shape[1]

        # Construct depthwise separable convolution weights
        kernels = []
        for c in range(channels):
            sigma = band_sigmas[c]
            kernels.append(self._gaussian_kernel_2d(kernel_size, sigma, x.device))
        
        weights = torch.stack(kernels).unsqueeze(1) # shape (C, 1, K, K)
        padding = kernel_size // 2

        # Convolve each channel independently
        return nn.functional.conv2d(x, weights, groups=channels, padding=padding)

    def apply_spectral_matching(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies linear calibration: matched = slope * x + intercept for each channel.
        """
        spec_cfg = self.config.get("spectral_matching", {})
        if not spec_cfg.get("enabled", False):
            return x

        slopes = torch.tensor(spec_cfg.get("slopes"), dtype=torch.float32, device=x.device)
        intercepts = torch.tensor(spec_cfg.get("intercepts"), dtype=torch.float32, device=x.device)

        # Apply calibration channel-wise: shape (B, C, H, W)
        # Reshape coefficient tensors for broadcasting: (1, C, 1, 1)
        slopes = slopes.view(1, -1, 1, 1)
        intercepts = intercepts.view(1, -1, 1, 1)

        return x * slopes + intercepts

    def apply_sensor_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds independent zero-mean additive white Gaussian noise (AWGN) to each band.
        """
        noise_cfg = self.config.get("sensor_noise", {})
        if not noise_cfg.get("enabled", False):
            return x

        band_sigmas = noise_cfg.get("band_sigmas")
        noise = torch.zeros_like(x)
        
        for c in range(x.shape[1]):
            sigma = band_sigmas[c]
            if sigma > 0.0:
                noise[:, c] = torch.randn_like(x[:, c]) * sigma

        return x + noise

    def apply_decimation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Downsamples high-resolution feature maps to simulate sensor decimation.
        """
        dec_cfg = self.config.get("spatial_decimation", {})
        if not dec_cfg.get("enabled", False):
            return x

        upscale_factor = dec_cfg.get("upscale_factor", 2)
        interpolation_mode = dec_cfg.get("interpolation_mode", "area")
        
        downscale_ratio = 1.0 / upscale_factor

        if interpolation_mode in ["bilinear", "bicubic"]:
            return nn.functional.interpolate(
                x, scale_factor=downscale_ratio, mode=interpolation_mode, align_corners=False
            )
        else:
            return nn.functional.interpolate(
                x, scale_factor=downscale_ratio, mode=interpolation_mode
            )

    def forward(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Executes the physical degradation pipeline forward pass.
        
        Args:
            x (Tensor or ndarray): Input high-resolution reference image.
                                  Can be shape (C, H, W) or (B, C, H, W).
                                  Values can be raw integers (e.g. 0-10000) or normalized float.
                                  
        Returns:
            Tensor or ndarray: Degraded low-resolution observation matching shape structure of input.
        """
        is_numpy = isinstance(x, np.ndarray)
        original_shape = x.shape

        # Convert input to PyTorch tensor with shape (B, C, H, W)
        if is_numpy:
            x_tensor = torch.from_numpy(x).float()
        else:
            x_tensor = x.clone().float()

        if len(x_tensor.shape) == 3:
            # (C, H, W) -> (1, C, H, W)
            x_tensor = x_tensor.unsqueeze(0)
        elif len(x_tensor.shape) != 4:
            raise ValueError("Input tensor must have shape (C, H, W) or (B, C, H, W).")

        # Execute modular degradation steps in order
        out = self.apply_radiometric_normalization(x_tensor)
        out = self.apply_psf_blur(out)
        out = self.apply_spectral_matching(out)
        out = self.apply_sensor_noise(out)
        out = self.apply_decimation(out)

        # Convert back to original structure
        if len(original_shape) == 3:
            # (1, C, H, W) -> (C, H, W)
            out = out.squeeze(0)

        if is_numpy:
            return out.cpu().numpy()
        return out
