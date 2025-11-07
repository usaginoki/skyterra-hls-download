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


def extract_rgb_images(
    input_path: str,
    output_directory: str,
    scale_factor: float = 3.0,
    bands_per_temporal: int = 6
) -> List[str]:
    """
    Extract RGB images (B4-Red, B3-Green, B2-Blue) for each temporal step.

    Analyzes the input image to determine the number of temporal steps based on
    the total band count, then extracts and saves RGB composites for each time period.

    Parameters:
    -----------
    input_path : str
        Path to input multi-band GeoTIFF
    output_directory : str
        Directory to save RGB images
    scale_factor : float
        Multiplier for brightness adjustment (default: 3.0)
    bands_per_temporal : int
        Number of bands per temporal step (default: 6 for HLS standard bands)

    Returns:
    --------
    List[str] : List of paths to created RGB images

    Raises:
    -------
    ValueError
        If band count is not divisible by bands_per_temporal
    FileNotFoundError
        If input file does not exist

    Example:
    --------
    >>> # Extract RGB from merged temporal image with 18 bands
    >>> rgb_files = extract_rgb_images(
    ...     'merged_temporal.tif',
    ...     './rgb_output'
    ... )
    >>> # Creates: t0_rgb.png, t1_rgb.png, t2_rgb.png

    Notes:
    ------
    - B2 (Blue), B3 (Green), B4 (Red) are expected at positions 0, 1, 2
      within each temporal step
    - Output images are scaled and contrast-stretched for visualization
    - Single temporal images (6 bands) will create one RGB output
    - Multi-temporal images create separate RGB for each time period
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Create output directory
    os.makedirs(output_directory, exist_ok=True)

    print(f"Extracting RGB images from: {input_path}")
    print(f"Output directory: {output_directory}")
    print("-" * 70)

    # Open the image and analyze bands
    with rasterio.open(input_path) as src:
        total_bands = src.count
        width = src.width
        height = src.height

        # Calculate number of temporal steps
        if total_bands % bands_per_temporal != 0:
            raise ValueError(
                f"Total bands ({total_bands}) is not divisible by "
                f"bands_per_temporal ({bands_per_temporal}). "
                f"Cannot determine number of temporal steps."
            )

        num_temporal = total_bands // bands_per_temporal

        print(f"Image analysis:")
        print(f"  Total bands: {total_bands}")
        print(f"  Bands per temporal step: {bands_per_temporal}")
        print(f"  Number of temporal steps: {num_temporal}")
        print(f"  Dimensions: {width} × {height} pixels")
        print()

        rgb_files = []

        # Extract RGB for each temporal step
        for temporal_idx in range(num_temporal):
            # Calculate band indices for this temporal step
            # Band layout: t0_B2, t0_B3, t0_B4, t0_B8A, t0_B11, t0_B12, t1_B2, ...
            base_idx = temporal_idx * bands_per_temporal

            # HLS bands: B2 (Blue), B3 (Green), B4 (Red)
            # Rasterio uses 1-based indexing
            blue_idx = base_idx + 1   # B2
            green_idx = base_idx + 2  # B3
            red_idx = base_idx + 3    # B4

            print(f"Processing temporal step {temporal_idx} (t{temporal_idx})...")
            print(f"  Reading bands: R=B4 (band {red_idx}), G=B3 (band {green_idx}), B=B2 (band {blue_idx})")

            # Read the RGB bands
            red = src.read(red_idx).astype(np.float32)
            green = src.read(green_idx).astype(np.float32)
            blue = src.read(blue_idx).astype(np.float32)

            # Get band descriptions for metadata
            red_desc = src.descriptions[red_idx - 1] if src.descriptions[red_idx - 1] else f"Band {red_idx}"
            green_desc = src.descriptions[green_idx - 1] if src.descriptions[green_idx - 1] else f"Band {green_idx}"
            blue_desc = src.descriptions[blue_idx - 1] if src.descriptions[blue_idx - 1] else f"Band {blue_idx}"

            # Apply scaling and contrast stretching
            def scale_band(band, scale=scale_factor):
                """Scale and clip band values to 0-255 range"""
                # Apply scaling factor
                scaled = band * scale

                # Clip to valid range
                scaled = np.clip(scaled, 0, 1)

                # Convert to 8-bit
                return (scaled * 255).astype(np.uint8)

            red_scaled = scale_band(red)
            green_scaled = scale_band(green)
            blue_scaled = scale_band(blue)

            print(f"  Value ranges (after scaling):")
            print(f"    Red:   min={red_scaled.min()}, max={red_scaled.max()}, mean={red_scaled.mean():.1f}")
            print(f"    Green: min={green_scaled.min()}, max={green_scaled.max()}, mean={green_scaled.mean():.1f}")
            print(f"    Blue:  min={blue_scaled.min()}, max={blue_scaled.max()}, mean={blue_scaled.mean():.1f}")

            # Stack into RGB array
            rgb_array = np.stack([red_scaled, green_scaled, blue_scaled], axis=-1)

            # Create PIL Image
            rgb_image = Image.fromarray(rgb_array, mode='RGB')

            # Generate output filename
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            if num_temporal > 1:
                output_filename = f"t{temporal_idx}_rgb.png"
            else:
                output_filename = f"{base_name}_rgb.png"

            output_path = os.path.join(output_directory, output_filename)

            # Save the image
            rgb_image.save(output_path, 'PNG')
            rgb_files.append(output_path)

            print(f"  ✓ Saved: {output_path}")
            print(f"    Bands: {red_desc} (R), {green_desc} (G), {blue_desc} (B)")
            print()

    print("=" * 70)
    print(f"RGB extraction complete! Created {len(rgb_files)} image(s)")
    print("=" * 70)

    return rgb_files
