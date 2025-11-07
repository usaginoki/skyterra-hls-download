"""
SkyTerra HLS Downloader

A collection of Python functions for searching and downloading satellite imagery
from NASA's Harmonized Landsat Sentinel-2 (HLS) dataset using Google Earth Engine.
"""

from .hls_downloader import (
    find_single_image,
    find_multiple_images,
    batch_download,
    download_images_end_to_end,
)

from .image_utils import (
    merge_temporal_images,
    crop_image,
    extract_rgb_images,
)

__all__ = [
    # Download functions
    'find_single_image',
    'find_multiple_images',
    'batch_download',
    'download_images_end_to_end',
    # Utility functions
    'merge_temporal_images',
    'crop_image',
    'extract_rgb_images',
]

__version__ = '0.1.0'
