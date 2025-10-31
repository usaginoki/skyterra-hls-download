"""
Image processing utilities for HLS satellite imagery.

This module provides functions for temporal merging, spatial cropping, and RGB extraction of GeoTIFF images.
"""

import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import rowcol
import numpy as np
from typing import List, Tuple, Union, Optional
import os
from datetime import datetime
from PIL import Image


def merge_temporal_images(
    image_paths: List[str],
    output_path: Optional[str] = None
) -> Union[str, np.ndarray]:
    """
    Merge multiple temporal images into a single multi-band GeoTIFF.

    Images are stacked chronologically, preserving band order within each temporal slice.
    For example, if each image has 6 bands (B2, B3, B4, B8A, B11, B12), the output will have:
    - Bands 1-6: First image (t0_B2, t0_B3, t0_B4, t0_B8A, t0_B11, t0_B12)
    - Bands 7-12: Second image (t1_B2, t1_B3, ...)
    - And so on...

    Parameters:
    -----------
    image_paths : List[str]
        List of GeoTIFF file paths in chronological order [t0, t1, t2, ...]
    output_path : Optional[str]
        Path to save the merged GeoTIFF. If None, returns array instead.

    Returns:
    --------
    Union[str, np.ndarray]
        If output_path provided: returns the output file path
        If output_path is None: returns the merged numpy array

    Raises:
    -------
    ValueError
        If image_paths is empty or images have mismatched dimensions/CRS
    FileNotFoundError
        If any input file does not exist

    Example:
    --------
    >>> # Merge 3 temporal images
    >>> merge_temporal_images(
    ...     ['image_t0.tif', 'image_t1.tif', 'image_t2.tif'],
    ...     'merged_temporal.tif'
    ... )
    'merged_temporal.tif'
    """
    if not image_paths:
        raise ValueError("image_paths cannot be empty")

    # Verify all files exist
    for path in image_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")

    # Read first image to get reference metadata
    with rasterio.open(image_paths[0]) as src0:
        ref_meta = src0.meta.copy()
        ref_width = src0.width
        ref_height = src0.height
        ref_crs = src0.crs
        ref_transform = src0.transform
        ref_bounds = src0.bounds

        # Get band names/descriptions if available
        ref_band_descriptions = src0.descriptions

    # Validate all images have same dimensions and CRS
    total_bands = 0
    band_descriptions = []
    acquisition_dates = []

    for idx, image_path in enumerate(image_paths):
        with rasterio.open(image_path) as src:
            if src.width != ref_width or src.height != ref_height:
                raise ValueError(
                    f"Image {idx} has mismatched dimensions: "
                    f"expected {ref_width}x{ref_height}, got {src.width}x{src.height}"
                )

            if src.crs != ref_crs:
                raise ValueError(
                    f"Image {idx} has mismatched CRS: "
                    f"expected {ref_crs}, got {src.crs}"
                )

            # Count bands
            num_bands = src.count
            total_bands += num_bands

            # Create band descriptions with temporal labels
            for band_idx in range(1, num_bands + 1):
                band_desc = src.descriptions[band_idx - 1] if src.descriptions[band_idx - 1] else f"Band{band_idx}"
                band_descriptions.append(f"t{idx}_{band_desc}")

            # Try to extract acquisition date from filename or metadata
            # Format: imageid_YYYY-MM-DD.tif
            filename = os.path.basename(image_path)
            date_str = None
            if '_' in filename:
                parts = filename.split('_')
                for part in parts:
                    # Look for YYYY-MM-DD pattern
                    if len(part) >= 10 and part[:10].count('-') == 2:
                        date_str = part[:10]
                        break

            acquisition_dates.append(date_str if date_str else f"t{idx}")

    # Read all images and stack bands
    all_bands = []

    print(f"Merging {len(image_paths)} temporal images with {total_bands} total bands...")

    for idx, image_path in enumerate(image_paths):
        with rasterio.open(image_path) as src:
            print(f"  Reading {os.path.basename(image_path)} ({src.count} bands)...")
            for band_idx in range(1, src.count + 1):
                band_data = src.read(band_idx)
                all_bands.append(band_data)

    # Stack all bands
    merged_array = np.stack(all_bands, axis=0)

    print(f"  Total bands in merged image: {merged_array.shape[0]}")

    if output_path is None:
        # Return array only
        return merged_array

    # Update metadata for merged image
    ref_meta.update({
        'count': total_bands,
        'dtype': merged_array.dtype
    })

    # Write merged GeoTIFF
    with rasterio.open(output_path, 'w', **ref_meta) as dst:
        for band_idx, band_data in enumerate(all_bands, 1):
            dst.write(band_data, band_idx)
            # Set band description
            dst.set_band_description(band_idx, band_descriptions[band_idx - 1])

        # Add acquisition dates to metadata
        dst.update_tags(
            temporal_images=len(image_paths),
            acquisition_dates=','.join(acquisition_dates),
            description=f"Multi-temporal merge of {len(image_paths)} images"
        )

    print(f"  ✓ Saved merged image to {output_path}")
    return output_path


def crop_image(
    input_path: str,
    coordinates: Union[List[float], Tuple],
    output_path: Optional[str] = None,
    coordinate_type: str = 'bbox'
) -> Union[str, Tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS]]:
    """
    Crop a region from a GeoTIFF image.

    Works with both single-temporal and multi-temporal (merged) images,
    preserving all bands in the cropped output.

    Parameters:
    -----------
    input_path : str
        Path to input GeoTIFF file
    coordinates : Union[List[float], Tuple]
        Cropping coordinates:
        - If coordinate_type='bbox': [min_lon, min_lat, max_lon, max_lat]
        - If coordinate_type='point_offset': ((lon, lat), x_distance_m, y_distance_m)
    output_path : Optional[str]
        Path to save cropped GeoTIFF. If None, returns array/metadata instead.
    coordinate_type : str
        Either 'bbox' for bounding box or 'point_offset' for point with distances

    Returns:
    --------
    Union[str, Tuple]
        If output_path provided: returns the output file path
        If output_path is None: returns (cropped_array, transform, crs)

    Raises:
    -------
    ValueError
        If coordinates are invalid or coordinate_type is not recognized
    FileNotFoundError
        If input file does not exist

    Examples:
    ---------
    >>> # Crop using bounding box
    >>> crop_image(
    ...     'image.tif',
    ...     [71.37, 51.12, 71.52, 51.21],
    ...     'cropped.tif',
    ...     coordinate_type='bbox'
    ... )
    'cropped.tif'

    >>> # Crop using point and offsets (5km in each direction)
    >>> crop_image(
    ...     'image.tif',
    ...     ((71.4491, 51.1694), 5000, 5000),
    ...     'cropped.tif',
    ...     coordinate_type='point_offset'
    ... )
    'cropped.tif'

    >>> # Return array without saving
    >>> arr, transform, crs = crop_image(
    ...     'image.tif',
    ...     [71.37, 51.12, 71.52, 51.21],
    ...     coordinate_type='bbox'
    ... )
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Parse coordinates based on type
    if coordinate_type == 'bbox':
        if len(coordinates) != 4:
            raise ValueError("Bounding box coordinates must be [min_lon, min_lat, max_lon, max_lat]")
        min_lon, min_lat, max_lon, max_lat = coordinates
        bounds = (min_lon, min_lat, max_lon, max_lat)

    elif coordinate_type == 'point_offset':
        if len(coordinates) != 3:
            raise ValueError("Point offset coordinates must be ((lon, lat), x_distance, y_distance)")

        point, x_distance, y_distance = coordinates
        lon, lat = point

        # Convert distances from meters to degrees (approximate)
        # At the equator: 1 degree ≈ 111,320 meters
        # This is an approximation; for more accuracy, use pyproj
        meters_per_degree = 111320

        lon_offset = x_distance / (meters_per_degree * np.cos(np.radians(lat)))
        lat_offset = y_distance / meters_per_degree

        min_lon = lon - lon_offset
        max_lon = lon + lon_offset
        min_lat = lat - lat_offset
        max_lat = lat + lat_offset

        bounds = (min_lon, min_lat, max_lon, max_lat)

    else:
        raise ValueError(f"Unknown coordinate_type: {coordinate_type}. Use 'bbox' or 'point_offset'")

    # Open image and crop
    with rasterio.open(input_path) as src:
        # Calculate window from bounds
        window = from_bounds(*bounds, transform=src.transform)

        # Round window to integer pixel coordinates
        window = window.round_offsets().round_lengths()

        # Read cropped data for all bands
        cropped_data = src.read(window=window)

        # Calculate transform for cropped area
        cropped_transform = src.window_transform(window)

        # Get metadata
        cropped_meta = src.meta.copy()
        cropped_meta.update({
            'height': window.height,
            'width': window.width,
            'transform': cropped_transform
        })

        # Get band descriptions
        band_descriptions = src.descriptions

    print(f"Cropped from {src.width}x{src.height} to {window.width}x{window.height} pixels")
    print(f"Bounds: {bounds}")

    if output_path is None:
        # Return array and metadata
        return cropped_data, cropped_transform, src.crs

    # Save cropped image
    with rasterio.open(output_path, 'w', **cropped_meta) as dst:
        dst.write(cropped_data)

        # Preserve band descriptions
        if band_descriptions:
            for idx, desc in enumerate(band_descriptions, 1):
                if desc:
                    dst.set_band_description(idx, desc)

        # Add cropping metadata
        dst.update_tags(
            cropped_from=input_path,
            crop_bounds=f"{bounds}",
            crop_type=coordinate_type
        )

    print(f"  ✓ Saved cropped image to {output_path}")
    return output_path
