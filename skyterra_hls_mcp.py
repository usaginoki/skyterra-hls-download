#!/usr/bin/env python3
"""
MCP Server for SkyTerra HLS (Harmonized Landsat Sentinel-2) Image Downloader.

This server provides tools to search, download, and process NASA HLS satellite imagery
using Google Earth Engine. It enables agents to find cloud-free imagery, download temporal
sequences, merge multi-temporal data, crop regions of interest, and extract RGB visualizations.
"""

import json
import os
from typing import List, Optional, Tuple
from enum import Enum
from datetime import datetime

from pydantic import BaseModel, Field, field_validator, ConfigDict
from fastmcp import FastMCP

# Import the HLS downloader library
from src import hls_downloader, image_utils

# Initialize the MCP server
mcp = FastMCP("skyterra_hls_mcp")

# Constants
DEFAULT_BANDS = ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12']
DEFAULT_IMAGE_COLLECTION = "NASA/HLS/HLSS30/v002"
CHARACTER_LIMIT = 25000  # Maximum response size in characters


# ============================================================================
# ENUMS
# ============================================================================

class CoordinateType(str, Enum):
    """Coordinate input type for spatial operations."""
    POINT = "point"
    POLYGON = "polygon"
    BBOX = "bbox"
    POINT_OFFSET = "point_offset"


class ResponseFormat(str, Enum):
    """Output format for tool responses."""
    JSON = "json"
    MARKDOWN = "markdown"


# ============================================================================
# PYDANTIC MODELS FOR INPUT VALIDATION
# ============================================================================

class HLSFindImagesInput(BaseModel):
    """Input model for finding HLS images."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    start_date: str = Field(
        ...,
        description="Start date in YYYY-MM-DD format (e.g., '2024-04-01', '2025-06-15')",
        pattern=r'^\d{4}-\d{2}-\d{2}$'
    )
    end_date: str = Field(
        ...,
        description="End date in YYYY-MM-DD format (e.g., '2024-09-30', '2025-12-31')",
        pattern=r'^\d{4}-\d{2}-\d{2}$'
    )
    longitude: float = Field(
        ...,
        description="Longitude of center point in decimal degrees (e.g., 70.4991, -109.53)",
        ge=-180.0,
        le=180.0
    )
    latitude: float = Field(
        ...,
        description="Latitude of center point in decimal degrees (e.g., 51.1694, 29.19)",
        ge=-90.0,
        le=90.0
    )
    number_of_images: int = Field(
        default=3,
        description="Number of images to find, evenly spaced across date range (e.g., 1, 3, 5)",
        ge=1,
        le=10
    )
    cloud_coverage: float = Field(
        default=10.0,
        description="Maximum cloud coverage percentage allowed (e.g., 10.0 for 10%, 20.0 for 20%)",
        ge=0.0,
        le=100.0
    )
    buffer_distance: int = Field(
        default=10000,
        description="Buffer distance around point in meters for area of interest (e.g., 5000 for 5km, 10000 for 10km)",
        ge=100,
        le=100000
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )

    @field_validator('end_date')
    @classmethod
    def validate_date_range(cls, v: str, info) -> str:
        """Ensure end_date is after start_date."""
        if 'start_date' in info.data:
            start = datetime.strptime(info.data['start_date'], '%Y-%m-%d')
            end = datetime.strptime(v, '%Y-%m-%d')
            if end <= start:
                raise ValueError("end_date must be after start_date")
        return v


class HLSDownloadImagesInput(BaseModel):
    """Input model for downloading HLS images."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    start_date: str = Field(
        ...,
        description="Start date in YYYY-MM-DD format (e.g., '2024-04-01', '2025-06-15')",
        pattern=r'^\d{4}-\d{2}-\d{2}$'
    )
    end_date: str = Field(
        ...,
        description="End date in YYYY-MM-DD format (e.g., '2024-09-30', '2025-12-31')",
        pattern=r'^\d{4}-\d{2}-\d{2}$'
    )
    longitude: float = Field(
        ...,
        description="Longitude of center point in decimal degrees (e.g., 70.4991, -109.53)",
        ge=-180.0,
        le=180.0
    )
    latitude: float = Field(
        ...,
        description="Latitude of center point in decimal degrees (e.g., 51.1694, 29.19)",
        ge=-90.0,
        le=90.0
    )
    output_directory: str = Field(
        ...,
        description="Directory path to save downloaded images (e.g., './downloads', '/tmp/hls_data')",
        min_length=1
    )
    number_of_images: int = Field(
        default=3,
        description="Number of images to download (e.g., 1, 3, 5)",
        ge=1,
        le=10
    )
    cloud_coverage: float = Field(
        default=10.0,
        description="Maximum cloud coverage percentage (e.g., 10.0, 20.0)",
        ge=0.0,
        le=100.0
    )
    buffer_distance: int = Field(
        default=10000,
        description="Buffer distance around point in meters (e.g., 5000, 10000)",
        ge=100,
        le=100000
    )
    scale: int = Field(
        default=30,
        description="Resolution in meters per pixel (e.g., 10, 30, 60)",
        ge=10,
        le=100
    )


class HLSMergeTemporalInput(BaseModel):
    """Input model for merging temporal images."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    image_paths: List[str] = Field(
        ...,
        description="List of GeoTIFF file paths to merge in chronological order (e.g., ['img1.tif', 'img2.tif', 'img3.tif'])",
        min_items=2,
        max_items=10
    )
    output_path: str = Field(
        ...,
        description="Path to save merged multi-temporal GeoTIFF (e.g., './merged/temporal_stack.tif')",
        min_length=1
    )

    @field_validator('image_paths')
    @classmethod
    def validate_paths_exist(cls, v: List[str]) -> List[str]:
        """Ensure all input files exist."""
        for path in v:
            if not os.path.exists(path):
                raise ValueError(f"Image file not found: {path}")
        return v


class HLSCropImageInput(BaseModel):
    """Input model for cropping images."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    input_path: str = Field(
        ...,
        description="Path to input GeoTIFF file to crop (e.g., './merged/temporal_stack.tif')",
        min_length=1
    )
    output_path: str = Field(
        ...,
        description="Path to save cropped GeoTIFF (e.g., './cropped/region.tif')",
        min_length=1
    )
    coordinate_type: CoordinateType = Field(
        ...,
        description="Type of coordinates: 'bbox' for bounding box [min_lon, min_lat, max_lon, max_lat], "
                    "'point_offset' for center point with distances"
    )
    # For bbox mode
    min_longitude: Optional[float] = Field(
        default=None,
        description="Minimum longitude for bbox mode (e.g., 70.37)",
        ge=-180.0,
        le=180.0
    )
    min_latitude: Optional[float] = Field(
        default=None,
        description="Minimum latitude for bbox mode (e.g., 51.12)",
        ge=-90.0,
        le=90.0
    )
    max_longitude: Optional[float] = Field(
        default=None,
        description="Maximum longitude for bbox mode (e.g., 70.52)",
        ge=-180.0,
        le=180.0
    )
    max_latitude: Optional[float] = Field(
        default=None,
        description="Maximum latitude for bbox mode (e.g., 51.21)",
        ge=-90.0,
        le=90.0
    )
    # For point_offset mode
    center_longitude: Optional[float] = Field(
        default=None,
        description="Center longitude for point_offset mode (e.g., 70.4491)",
        ge=-180.0,
        le=180.0
    )
    center_latitude: Optional[float] = Field(
        default=None,
        description="Center latitude for point_offset mode (e.g., 51.1694)",
        ge=-90.0,
        le=90.0
    )
    x_distance_m: Optional[int] = Field(
        default=None,
        description="Distance in meters from center point in X direction (e.g., 1000 for 1km, 5000 for 5km)",
        ge=100,
        le=100000
    )
    y_distance_m: Optional[int] = Field(
        default=None,
        description="Distance in meters from center point in Y direction (e.g., 1000 for 1km, 5000 for 5km)",
        ge=100,
        le=100000
    )

    @field_validator('input_path')
    @classmethod
    def validate_input_exists(cls, v: str) -> str:
        """Ensure input file exists."""
        if not os.path.exists(v):
            raise ValueError(f"Input file not found: {v}")
        return v


class HLSExtractRGBInput(BaseModel):
    """Input model for extracting RGB images."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    input_path: str = Field(
        ...,
        description="Path to input multi-band GeoTIFF (e.g., './merged/temporal_stack.tif')",
        min_length=1
    )
    output_directory: str = Field(
        ...,
        description="Directory to save RGB PNG images (e.g., './rgb')",
        min_length=1
    )
    scale_factor: float = Field(
        default=3.0,
        description="Brightness multiplier for visualization (e.g., 2.0, 3.0, 4.0)",
        ge=1.0,
        le=10.0
    )
    bands_per_temporal: int = Field(
        default=6,
        description="Number of bands per temporal step (default 6 for HLS standard bands)",
        ge=3,
        le=12
    )

    @field_validator('input_path')
    @classmethod
    def validate_input_exists(cls, v: str) -> str:
        """Ensure input file exists."""
        if not os.path.exists(v):
            raise ValueError(f"Input file not found: {v}")
        return v


class HLSFullPipelineInput(BaseModel):
    """Input model for full HLS processing pipeline."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    start_date: str = Field(
        ...,
        description="Start date in YYYY-MM-DD format (e.g., '2024-04-01', '2025-06-15')",
        pattern=r'^\d{4}-\d{2}-\d{2}$'
    )
    end_date: str = Field(
        ...,
        description="End date in YYYY-MM-DD format (e.g., '2024-09-30', '2025-12-31')",
        pattern=r'^\d{4}-\d{2}-\d{2}$'
    )
    center_longitude: float = Field(
        ...,
        description="Longitude of center point in decimal degrees (e.g., 70.4991, -109.53)",
        ge=-180.0,
        le=180.0
    )
    center_latitude: float = Field(
        ...,
        description="Latitude of center point in decimal degrees (e.g., 51.1694, 29.19)",
        ge=-90.0,
        le=90.0
    )
    area_size_m: int = Field(
        ...,
        description="Size of area to crop in meters (e.g., 1000 for 1km x 1km, 5000 for 5km x 5km)",
        ge=100,
        le=100000
    )
    output_directory: str = Field(
        ...,
        description="Base output directory for all results (e.g., './hls_output')",
        min_length=1
    )
    number_of_images: int = Field(
        default=3,
        description="Number of temporal images to download (default: 3)",
        ge=1,
        le=10
    )
    cloud_coverage: float = Field(
        default=10.0,
        description="Maximum cloud coverage percentage (default: 10.0)",
        ge=0.0,
        le=100.0
    )
    scale: int = Field(
        default=30,
        description="Resolution in meters per pixel (default: 30)",
        ge=10,
        le=100
    )
    rgb_scale_factor: float = Field(
        default=3.0,
        description="Brightness multiplier for RGB visualization (default: 3.0)",
        ge=1.0,
        le=10.0
    )

    @field_validator('end_date')
    @classmethod
    def validate_date_range(cls, v: str, info) -> str:
        """Ensure end_date is after start_date."""
        if 'start_date' in info.data:
            start = datetime.strptime(info.data['start_date'], '%Y-%m-%d')
            end = datetime.strptime(v, '%Y-%m-%d')
            if end <= start:
                raise ValueError("end_date must be after start_date")
        return v


# ============================================================================
# SHARED UTILITY FUNCTIONS
# ============================================================================

def _handle_error(e: Exception) -> str:
    """
    Consistent error formatting across all tools.

    Returns clear, actionable error messages for common failure scenarios.
    """
    error_msg = str(e)

    # Common error patterns and helpful responses
    if "authenticate" in error_msg.lower() or "credentials" in error_msg.lower():
        return (
            f"Error: Earth Engine authentication required. "
            f"Please run 'earthengine authenticate' in your terminal first. "
            f"Details: {error_msg}"
        )
    elif "not found" in error_msg.lower():
        return f"Error: File or resource not found. Please check the path is correct. Details: {error_msg}"
    elif "permission" in error_msg.lower() or "access" in error_msg.lower():
        return f"Error: Permission denied. Check file permissions and directory access. Details: {error_msg}"
    elif "mismatched" in error_msg.lower():
        return f"Error: Image dimensions or CRS mismatch. Ensure all images are compatible. Details: {error_msg}"
    elif "no image" in error_msg.lower() or "could not find" in error_msg.lower():
        return (
            f"Error: No images found matching criteria. Try: "
            f"(1) Expanding the date range, "
            f"(2) Increasing cloud_coverage threshold, "
            f"(3) Increasing buffer_distance. "
            f"Details: {error_msg}"
        )
    else:
        return f"Error: {error_msg}"


def _create_directory_structure(base_dir: str) -> dict:
    """
    Create organized directory structure for HLS outputs.

    Returns dictionary with paths to subdirectories.
    """
    dirs = {
        'base': base_dir,
        'downloads': os.path.join(base_dir, 'downloads'),
        'merged': os.path.join(base_dir, 'merged'),
        'cropped': os.path.join(base_dir, 'cropped'),
        'rgb': os.path.join(base_dir, 'rgb')
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def _format_image_metadata_markdown(metadata: dict, idx: int) -> str:
    """Format image metadata as markdown."""
    return f"""
## Image {idx}
- **Date**: {metadata['date']}
- **Cloud Coverage**: {metadata['cloud_coverage']:.1f}%
- **Image ID**: {metadata['image_id']}
"""


def _format_file_info(filepath: str) -> str:
    """Format file information as markdown."""
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        return f"- `{filepath}` ({size_mb:.2f} MB)"
    return f"- `{filepath}` (file not found)"


# ============================================================================
# MCP TOOL IMPLEMENTATIONS
# ============================================================================

@mcp.tool(
    name="hls_find_images",
    annotations={
        "title": "Find HLS Satellite Images",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def hls_find_images(params: HLSFindImagesInput) -> str:
    """
    Search for HLS satellite images matching specified criteria.

    This tool searches NASA's Harmonized Landsat Sentinel-2 (HLS) collection in Google Earth Engine
    to find cloud-free images for a specified location and date range. It does NOT download images,
    only searches and returns metadata about available images.

    Args:
        params (HLSFindImagesInput): Validated input parameters containing:
            - start_date (str): Start date in YYYY-MM-DD format
            - end_date (str): End date in YYYY-MM-DD format
            - longitude (float): Center point longitude in decimal degrees
            - latitude (float): Center point latitude in decimal degrees
            - number_of_images (int): Number of images to find (default: 3)
            - cloud_coverage (float): Maximum cloud coverage % (default: 10.0)
            - buffer_distance (int): Buffer around point in meters (default: 10000)
            - response_format (str): 'markdown' or 'json' (default: 'markdown')

    Returns:
        str: Formatted response containing found images with metadata

        Success response (markdown):
        # Found Images
        Found X images matching criteria

        ## Image 1
        - Date: YYYY-MM-DD
        - Cloud Coverage: X.X%
        - Image ID: <id>

        Success response (json):
        {
            "found_count": int,
            "requested_count": int,
            "images": [
                {
                    "date": "YYYY-MM-DD",
                    "cloud_coverage": float,
                    "image_id": str
                }
            ]
        }

        Error response:
        "Error: <error message>"

    Examples:
        - Use when: "Find 3 cloud-free images near coordinates (70.5, 51.2) from April to September 2024"
        - Use when: "Search for satellite images with less than 5% cloud cover"
        - Don't use when: You need to actually download the images (use hls_download_images instead)
        - Don't use when: You already have image files and need to process them

    Error Handling:
        - Returns "Error: Earth Engine authentication required" if EE not authenticated
        - Returns "Error: No images found matching criteria" if no suitable images exist
        - Suggests adjustments: expand date range, increase cloud threshold, increase buffer
    """
    try:
        # Find images using the HLS downloader library
        image_objects = hls_downloader.find_multiple_images(
            date_range=(params.start_date, params.end_date),
            coordinates=(params.longitude, params.latitude),
            image_collection=DEFAULT_IMAGE_COLLECTION,
            bands=DEFAULT_BANDS,
            cloud_coverage=params.cloud_coverage,
            number_of_images=params.number_of_images,
            buffer_distance=params.buffer_distance
        )

        if not image_objects:
            return (
                f"No images found matching criteria. Try: "
                f"(1) Expanding date range, "
                f"(2) Increasing cloud_coverage to {min(params.cloud_coverage + 10, 30)}%, "
                f"(3) Increasing buffer_distance to {params.buffer_distance * 2}m"
            )

        # Format response
        if params.response_format == ResponseFormat.MARKDOWN:
            lines = [
                f"# Found HLS Images",
                f"",
                f"Found **{len(image_objects)}** out of **{params.number_of_images}** requested images",
                f"",
                f"**Search Criteria:**",
                f"- Date Range: {params.start_date} to {params.end_date}",
                f"- Location: ({params.latitude:.4f}, {params.longitude:.4f})",
                f"- Max Cloud Coverage: {params.cloud_coverage}%",
                f"- Buffer Distance: {params.buffer_distance}m",
                f""
            ]

            for idx, img_obj in enumerate(image_objects, 1):
                lines.append(_format_image_metadata_markdown(img_obj['metadata'], idx))

            return "\n".join(lines)

        else:  # JSON format
            result = {
                "found_count": len(image_objects),
                "requested_count": params.number_of_images,
                "search_criteria": {
                    "date_range": [params.start_date, params.end_date],
                    "location": [params.latitude, params.longitude],
                    "cloud_coverage": params.cloud_coverage,
                    "buffer_distance": params.buffer_distance
                },
                "images": [
                    {
                        "date": img_obj['metadata']['date'],
                        "cloud_coverage": img_obj['metadata']['cloud_coverage'],
                        "image_id": img_obj['metadata']['image_id']
                    }
                    for img_obj in image_objects
                ]
            }
            return json.dumps(result, indent=2)

    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="hls_download_images",
    annotations={
        "title": "Download HLS Satellite Images",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def hls_download_images(params: HLSDownloadImagesInput) -> str:
    """
    Download HLS satellite images from Google Earth Engine.

    This tool searches for and downloads HLS images matching the specified criteria.
    Images are saved as multi-band GeoTIFF files in the specified output directory.
    Each image contains 6 bands: B2, B3, B4, B8A, B11, B12 (Blue, Green, Red, NIR, SWIR1, SWIR2).

    Args:
        params (HLSDownloadImagesInput): Validated input parameters containing:
            - start_date (str): Start date in YYYY-MM-DD format
            - end_date (str): End date in YYYY-MM-DD format
            - longitude (float): Center point longitude
            - latitude (float): Center point latitude
            - output_directory (str): Directory to save images
            - number_of_images (int): Number of images (default: 3)
            - cloud_coverage (float): Max cloud % (default: 10.0)
            - buffer_distance (int): Buffer in meters (default: 10000)
            - scale (int): Resolution in meters/pixel (default: 30)

    Returns:
        str: JSON-formatted response containing download results

        Success response:
        {
            "status": "success",
            "downloaded_count": int,
            "requested_count": int,
            "output_directory": str,
            "files": [
                {
                    "path": str,
                    "date": str,
                    "cloud_coverage": float,
                    "image_id": str,
                    "size_mb": float
                }
            ]
        }

        Error response:
        "Error: <error message>"

    Examples:
        - Use when: "Download 3 satellite images from April to September near (70.5, 51.2)"
        - Use when: "Get HLS imagery with less than 10% cloud cover for my study area"
        - Don't use when: You only want to search without downloading (use hls_find_images)
        - Don't use when: You need a complete workflow with merge/crop (use hls_full_pipeline)

    Error Handling:
        - Returns error if Earth Engine not authenticated
        - Returns error if output directory cannot be created
        - Includes partial results if some downloads fail
        - Suggests increasing cloud threshold or date range if no images found
    """
    try:
        # Download images using the HLS downloader library
        downloaded_files = hls_downloader.download_images_end_to_end(
            date_range=(params.start_date, params.end_date),
            coordinates=(params.longitude, params.latitude),
            output_directory=params.output_directory,
            image_collection=DEFAULT_IMAGE_COLLECTION,
            bands=DEFAULT_BANDS,
            cloud_coverage=params.cloud_coverage,
            number_of_images=params.number_of_images,
            buffer_distance=params.buffer_distance,
            scale=params.scale
        )

        if not downloaded_files:
            return json.dumps({
                "status": "error",
                "message": "No images downloaded. Try expanding date range or increasing cloud coverage threshold."
            }, indent=2)

        # Build response with file info
        files_info = []
        for filepath in downloaded_files:
            size_mb = os.path.getsize(filepath) / (1024 * 1024) if os.path.exists(filepath) else 0
            # Extract metadata from filename: imageid_YYYY-MM-DD.tif
            filename = os.path.basename(filepath)
            parts = filename.replace('.tif', '').split('_')
            date_str = parts[-1] if len(parts) > 1 else "unknown"
            image_id = '_'.join(parts[:-1]) if len(parts) > 1 else filename

            files_info.append({
                "path": filepath,
                "date": date_str,
                "image_id": image_id,
                "size_mb": round(size_mb, 2)
            })

        result = {
            "status": "success",
            "downloaded_count": len(downloaded_files),
            "requested_count": params.number_of_images,
            "output_directory": params.output_directory,
            "files": files_info
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="hls_merge_temporal",
    annotations={
        "title": "Merge Temporal HLS Images",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def hls_merge_temporal(params: HLSMergeTemporalInput) -> str:
    """
    Merge multiple temporal HLS images into a single multi-band GeoTIFF.

    This tool stacks multiple temporal images chronologically, creating a multi-band file
    where bands are organized as: t0_B2, t0_B3, ..., t0_B12, t1_B2, t1_B3, ..., t1_B12, etc.
    Each temporal step preserves all 6 HLS bands in spectral order.

    Args:
        params (HLSMergeTemporalInput): Validated input parameters containing:
            - image_paths (List[str]): List of GeoTIFF paths in chronological order
            - output_path (str): Path to save merged GeoTIFF

    Returns:
        str: JSON-formatted response with merge results

        Success response:
        {
            "status": "success",
            "output_path": str,
            "temporal_count": int,
            "total_bands": int,
            "input_files": [str],
            "size_mb": float
        }

        Error response:
        "Error: <error message>"

    Examples:
        - Use when: "Merge 3 downloaded images into a temporal stack"
        - Use when: "Combine multiple dates into one file for time series analysis"
        - Don't use when: You only have one image (no merging needed)
        - Don't use when: Images have different dimensions or CRS (will fail validation)

    Error Handling:
        - Returns error if input files don't exist
        - Returns error if images have mismatched dimensions or CRS
        - Validates all inputs before processing
        - Provides clear message about dimension/CRS requirements
    """
    try:
        # Merge images using the image_utils library
        output_path = image_utils.merge_temporal_images(
            image_paths=params.image_paths,
            output_path=params.output_path
        )

        # Get output file info
        size_mb = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0

        # Count bands using rasterio
        import rasterio
        with rasterio.open(output_path) as src:
            total_bands = src.count

        result = {
            "status": "success",
            "output_path": output_path,
            "temporal_count": len(params.image_paths),
            "total_bands": total_bands,
            "input_files": params.image_paths,
            "size_mb": round(size_mb, 2)
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="hls_crop_image",
    annotations={
        "title": "Crop HLS Image to Region",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def hls_crop_image(params: HLSCropImageInput) -> str:
    """
    Crop a spatial region from an HLS image.

    This tool extracts a rectangular region from a GeoTIFF image, preserving all bands.
    Works with both single-temporal and multi-temporal (merged) images. Supports two
    coordinate modes: bounding box (exact coordinates) or point with offsets (distances).

    Args:
        params (HLSCropImageInput): Validated input parameters containing:
            - input_path (str): Path to input GeoTIFF
            - output_path (str): Path to save cropped GeoTIFF
            - coordinate_type (str): 'bbox' or 'point_offset'

            For bbox mode:
            - min_longitude, min_latitude, max_longitude, max_latitude

            For point_offset mode:
            - center_longitude, center_latitude, x_distance_m, y_distance_m

    Returns:
        str: JSON-formatted response with crop results

        Success response:
        {
            "status": "success",
            "output_path": str,
            "input_dimensions": [width, height],
            "output_dimensions": [width, height],
            "bands": int,
            "bounds": [min_lon, min_lat, max_lon, max_lat],
            "size_mb": float
        }

        Error response:
        "Error: <error message>"

    Examples:
        - Use when: "Crop a 2km x 2km area around coordinates (70.5, 51.2)"
        - Use when: "Extract region bounded by (70.3, 51.1) to (70.7, 51.3)"
        - Don't use when: Input file doesn't exist (will fail validation)
        - Don't use when: You need to download images first (use hls_download_images)

    Error Handling:
        - Returns error if input file not found
        - Returns error if coordinates are outside image bounds
        - Validates coordinate parameters based on coordinate_type
        - Provides guidance on coordinate format requirements
    """
    try:
        # Parse coordinates based on type
        if params.coordinate_type == CoordinateType.BBOX:
            if None in [params.min_longitude, params.min_latitude, params.max_longitude, params.max_latitude]:
                return "Error: For bbox mode, must provide min_longitude, min_latitude, max_longitude, max_latitude"

            coordinates = [params.min_longitude, params.min_latitude, params.max_longitude, params.max_latitude]
            coord_type = 'bbox'

        elif params.coordinate_type == CoordinateType.POINT_OFFSET:
            if None in [params.center_longitude, params.center_latitude, params.x_distance_m, params.y_distance_m]:
                return "Error: For point_offset mode, must provide center_longitude, center_latitude, x_distance_m, y_distance_m"

            coordinates = ((params.center_longitude, params.center_latitude), params.x_distance_m, params.y_distance_m)
            coord_type = 'point_offset'

        else:
            return f"Error: Unknown coordinate_type: {params.coordinate_type}"

        # Get input dimensions
        import rasterio
        with rasterio.open(params.input_path) as src:
            input_width = src.width
            input_height = src.height
            input_bands = src.count

        # Crop image using image_utils library
        output_path = image_utils.crop_image(
            input_path=params.input_path,
            coordinates=coordinates,
            output_path=params.output_path,
            coordinate_type=coord_type
        )

        # Get output info
        size_mb = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0

        with rasterio.open(output_path) as src:
            output_width = src.width
            output_height = src.height
            output_bands = src.count
            bounds = src.bounds

        result = {
            "status": "success",
            "output_path": output_path,
            "input_dimensions": [input_width, input_height],
            "output_dimensions": [output_width, output_height],
            "bands": output_bands,
            "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
            "size_mb": round(size_mb, 2)
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="hls_extract_rgb",
    annotations={
        "title": "Extract RGB Visualizations",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def hls_extract_rgb(params: HLSExtractRGBInput) -> str:
    """
    Extract RGB images (natural color) from multi-temporal HLS data.

    This tool creates RGB PNG visualizations from HLS GeoTIFF files by extracting
    B4 (Red), B3 (Green), B2 (Blue) bands for each temporal step. Output images
    are scaled and contrast-enhanced for visual interpretation.

    Args:
        params (HLSExtractRGBInput): Validated input parameters containing:
            - input_path (str): Path to multi-band GeoTIFF
            - output_directory (str): Directory to save RGB PNGs
            - scale_factor (float): Brightness multiplier (default: 3.0)
            - bands_per_temporal (int): Bands per time step (default: 6)

    Returns:
        str: JSON-formatted response with extraction results

        Success response:
        {
            "status": "success",
            "output_directory": str,
            "temporal_count": int,
            "rgb_files": [
                {
                    "path": str,
                    "temporal_index": int,
                    "size_kb": float
                }
            ]
        }

        Error response:
        "Error: <error message>"

    Examples:
        - Use when: "Create RGB visualizations from merged temporal stack"
        - Use when: "Generate natural color images for each date"
        - Don't use when: You need false color composites (NIR-Red-Green)
        - Don't use when: Input doesn't have standard HLS band structure

    Error Handling:
        - Returns error if input file not found
        - Returns error if band count not divisible by bands_per_temporal
        - Validates band structure before processing
        - Provides guidance on expected band layout
    """
    try:
        # Extract RGB images using image_utils library
        rgb_files = image_utils.extract_rgb_images(
            input_path=params.input_path,
            output_directory=params.output_directory,
            scale_factor=params.scale_factor,
            bands_per_temporal=params.bands_per_temporal
        )

        # Build response with file info
        rgb_info = []
        for idx, filepath in enumerate(rgb_files):
            size_kb = os.path.getsize(filepath) / 1024 if os.path.exists(filepath) else 0
            rgb_info.append({
                "path": filepath,
                "temporal_index": idx,
                "size_kb": round(size_kb, 2)
            })

        result = {
            "status": "success",
            "output_directory": params.output_directory,
            "temporal_count": len(rgb_files),
            "rgb_files": rgb_info
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="hls_full_pipeline",
    annotations={
        "title": "Full HLS Processing Pipeline",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def hls_full_pipeline(params: HLSFullPipelineInput) -> str:
    """
    Execute complete HLS processing workflow: find, download, merge, crop, and extract RGB.

    This is the primary workflow tool that orchestrates the entire HLS processing pipeline:
    1. Search for images matching criteria (find N images in date range)
    2. Download images from Earth Engine
    3. Merge temporal images into multi-band stack
    4. Crop to specified area size around center point
    5. Extract RGB visualizations for each temporal step

    All outputs are organized in a neat directory structure:
    - downloads/: Raw downloaded images
    - merged/: Merged temporal stack
    - cropped/: Cropped region of interest
    - rgb/: RGB PNG visualizations

    Args:
        params (HLSFullPipelineInput): Validated input parameters containing:
            - start_date (str): Start date YYYY-MM-DD
            - end_date (str): End date YYYY-MM-DD
            - center_longitude (float): Center point longitude
            - center_latitude (float): Center point latitude
            - area_size_m (int): Size of crop area in meters (e.g., 2000 = 2km x 2km)
            - output_directory (str): Base output directory
            - number_of_images (int): Number of images (default: 3)
            - cloud_coverage (float): Max cloud % (default: 10.0)
            - scale (int): Resolution m/pixel (default: 30)
            - rgb_scale_factor (float): RGB brightness (default: 3.0)

    Returns:
        str: JSON-formatted response with complete pipeline results

        Success response:
        {
            "status": "success",
            "pipeline_steps": {
                "1_download": {
                    "count": int,
                    "directory": str,
                    "files": [str]
                },
                "2_merge": {
                    "output_path": str,
                    "total_bands": int,
                    "size_mb": float
                },
                "3_crop": {
                    "output_path": str,
                    "dimensions": [width, height],
                    "size_mb": float
                },
                "4_rgb_extract": {
                    "count": int,
                    "directory": str,
                    "files": [str]
                }
            },
            "summary": {
                "temporal_images": int,
                "final_cropped_image": str,
                "rgb_visualizations": int,
                "total_processing_time": str
            }
        }

        Error response:
        "Error: <error message>"

    Examples:
        - Use when: "Download 3 cloud-free images from April-September 2024 near (70.5, 51.2), "
                    "merge them, and crop to 2km x 2km area"
        - Use when: "Get me a temporal stack of HLS imagery for my study area with RGB visualizations"
        - Use when: "Complete workflow: find images → download → merge → crop → visualize"
        - Don't use when: You only need part of the workflow (use specific tools instead)
        - Don't use when: You already have downloaded images (use merge/crop tools directly)

    Error Handling:
        - Returns detailed error if any pipeline step fails
        - Includes information about which step failed
        - Provides partial results if early steps succeeded
        - Suggests remediation for common issues (auth, no images, etc.)
    """
    import time
    start_time = time.time()

    try:
        # Create organized directory structure
        dirs = _create_directory_structure(params.output_directory)

        pipeline_results = {}

        # STEP 1: Download images
        print(f"\n{'='*70}")
        print(f"STEP 1: Downloading {params.number_of_images} images...")
        print(f"{'='*70}")

        downloaded_files = hls_downloader.download_images_end_to_end(
            date_range=(params.start_date, params.end_date),
            coordinates=(params.center_longitude, params.center_latitude),
            output_directory=dirs['downloads'],
            image_collection=DEFAULT_IMAGE_COLLECTION,
            bands=DEFAULT_BANDS,
            cloud_coverage=params.cloud_coverage,
            number_of_images=params.number_of_images,
            buffer_distance=params.area_size_m * 2,  # Download larger area than crop
            scale=params.scale
        )

        if not downloaded_files:
            return json.dumps({
                "status": "error",
                "step": "download",
                "message": "No images found or downloaded. Try expanding date range or increasing cloud coverage threshold."
            }, indent=2)

        pipeline_results['1_download'] = {
            "count": len(downloaded_files),
            "directory": dirs['downloads'],
            "files": downloaded_files
        }

        # STEP 2: Merge temporal images
        print(f"\n{'='*70}")
        print(f"STEP 2: Merging {len(downloaded_files)} temporal images...")
        print(f"{'='*70}")

        merged_filename = f"merged_temporal_{params.start_date}_{params.end_date}.tif"
        merged_path = os.path.join(dirs['merged'], merged_filename)

        merged_output = image_utils.merge_temporal_images(
            image_paths=downloaded_files,
            output_path=merged_path
        )

        import rasterio
        with rasterio.open(merged_path) as src:
            merged_bands = src.count

        merged_size_mb = os.path.getsize(merged_path) / (1024 * 1024)

        pipeline_results['2_merge'] = {
            "output_path": merged_path,
            "total_bands": merged_bands,
            "size_mb": round(merged_size_mb, 2)
        }

        # STEP 3: Crop to desired area
        print(f"\n{'='*70}")
        print(f"STEP 3: Cropping to {params.area_size_m}m x {params.area_size_m}m area...")
        print(f"{'='*70}")

        cropped_filename = f"cropped_{params.area_size_m}m_{params.start_date}_{params.end_date}.tif"
        cropped_path = os.path.join(dirs['cropped'], cropped_filename)

        # Use point_offset mode for cropping
        coordinates = (
            (params.center_longitude, params.center_latitude),
            params.area_size_m,
            params.area_size_m
        )

        cropped_output = image_utils.crop_image(
            input_path=merged_path,
            coordinates=coordinates,
            output_path=cropped_path,
            coordinate_type='point_offset'
        )

        with rasterio.open(cropped_path) as src:
            cropped_width = src.width
            cropped_height = src.height
            cropped_bands = src.count

        cropped_size_mb = os.path.getsize(cropped_path) / (1024 * 1024)

        pipeline_results['3_crop'] = {
            "output_path": cropped_path,
            "dimensions": [cropped_width, cropped_height],
            "bands": cropped_bands,
            "size_mb": round(cropped_size_mb, 2)
        }

        # STEP 4: Extract RGB visualizations
        print(f"\n{'='*70}")
        print(f"STEP 4: Extracting RGB visualizations...")
        print(f"{'='*70}")

        rgb_files = image_utils.extract_rgb_images(
            input_path=cropped_path,
            output_directory=dirs['rgb'],
            scale_factor=params.rgb_scale_factor,
            bands_per_temporal=6
        )

        pipeline_results['4_rgb_extract'] = {
            "count": len(rgb_files),
            "directory": dirs['rgb'],
            "files": rgb_files
        }

        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time

        # Build final response
        result = {
            "status": "success",
            "pipeline_steps": pipeline_results,
            "summary": {
                "temporal_images": len(downloaded_files),
                "final_merged_image": merged_path,
                "final_cropped_image": cropped_path,
                "rgb_visualizations": len(rgb_files),
                "total_processing_time": f"{processing_time:.1f} seconds"
            },
            "directory_structure": {
                "base": params.output_directory,
                "downloads": dirs['downloads'],
                "merged": dirs['merged'],
                "cropped": dirs['cropped'],
                "rgb": dirs['rgb']
            }
        }

        print(f"\n{'='*70}")
        print(f"PIPELINE COMPLETE!")
        print(f"{'='*70}")
        print(f"Final cropped image: {cropped_path}")
        print(f"RGB visualizations: {dirs['rgb']}")
        print(f"Total time: {processing_time:.1f}s")
        print(f"{'='*70}\n")

        return json.dumps(result, indent=2)

    except Exception as e:
        return _handle_error(e)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    mcp.run()
