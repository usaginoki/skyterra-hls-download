# SkyTerra HLS MCP Server

Model Context Protocol (MCP) server for downloading and processing NASA HLS (Harmonized Landsat Sentinel-2) satellite imagery using Google Earth Engine.

## Overview

This MCP server enables AI agents to search, download, and process HLS satellite imagery through a set of composable tools. It provides both granular control over individual processing steps and a complete pipeline for end-to-end workflows.

## Features

- **Search HLS Images**: Find cloud-free satellite images for any location and date range
- **Download Images**: Retrieve multi-band GeoTIFF files from Earth Engine
- **Merge Temporal Data**: Stack multiple dates into multi-temporal datasets
- **Crop Regions**: Extract spatial regions of interest
- **Extract RGB**: Generate natural color visualizations
- **Full Pipeline**: Complete workflow from search to final outputs

## Installation

### Prerequisites

- Python 3.11 or higher
- Google Earth Engine account ([sign up here](https://earthengine.google.com/))
- Authenticated Earth Engine API (`earthengine authenticate`)

### Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Authenticate Earth Engine

```bash
earthengine authenticate
```

## Running the Server

### Stdio Transport (for Claude Desktop, etc.)

```bash
uv run python skyterra_hls_mcp.py
```

### Add to Claude Desktop

Add this configuration to your Claude Desktop config:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**Linux**: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "skyterra-hls": {
      "command": "uv",
      "args": [
        "run",
        "python",
        "/absolute/path/to/skyterra-hls-download/skyterra_hls_mcp.py"
      ]
    }
  }
}
```

## Available Tools

### 1. `hls_find_images`

Search for HLS images matching specified criteria. Returns metadata without downloading.

**Parameters:**
- `start_date`: Start date (YYYY-MM-DD)
- `end_date`: End date (YYYY-MM-DD)
- `longitude`: Center point longitude (-180 to 180)
- `latitude`: Center point latitude (-90 to 90)
- `number_of_images`: Number of images to find (1-10, default: 3)
- `cloud_coverage`: Maximum cloud coverage % (0-100, default: 10.0)
- `buffer_distance`: Buffer around point in meters (100-100000, default: 10000)
- `response_format`: 'markdown' or 'json' (default: 'markdown')

**Example:**
```
Find 3 HLS images near coordinates (70.4991, 51.1694) from April to September 2024
with less than 10% cloud coverage
```

### 2. `hls_download_images`

Download HLS images from Google Earth Engine.

**Parameters:**
- `start_date`: Start date (YYYY-MM-DD)
- `end_date`: End date (YYYY-MM-DD)
- `longitude`: Center point longitude
- `latitude`: Center point latitude
- `output_directory`: Directory to save images
- `number_of_images`: Number of images (1-10, default: 3)
- `cloud_coverage`: Maximum cloud % (0-100, default: 10.0)
- `buffer_distance`: Buffer in meters (100-100000, default: 10000)
- `scale`: Resolution in meters/pixel (10-100, default: 30)

**Example:**
```
Download 3 HLS images near (70.5, 51.2) from June to August 2024
and save to ./downloads
```

### 3. `hls_merge_temporal`

Merge multiple temporal images into a single multi-band GeoTIFF stack.

**Parameters:**
- `image_paths`: List of GeoTIFF paths in chronological order (2-10 files)
- `output_path`: Path to save merged file

**Example:**
```
Merge these 3 images into a temporal stack:
['./downloads/img1.tif', './downloads/img2.tif', './downloads/img3.tif']
and save to ./merged/temporal_stack.tif
```

### 4. `hls_crop_image`

Crop a spatial region from an image.

**Parameters:**
- `input_path`: Path to input GeoTIFF
- `output_path`: Path to save cropped file
- `coordinate_type`: 'bbox' or 'point_offset'

**For bbox mode:**
- `min_longitude`, `min_latitude`, `max_longitude`, `max_latitude`

**For point_offset mode:**
- `center_longitude`, `center_latitude`
- `x_distance_m`: Distance in X direction (meters)
- `y_distance_m`: Distance in Y direction (meters)

**Example:**
```
Crop a 2km x 2km area around coordinates (70.4991, 51.1694)
from ./merged/temporal_stack.tif
```

### 5. `hls_extract_rgb`

Extract RGB (natural color) visualizations from multi-temporal data.

**Parameters:**
- `input_path`: Path to multi-band GeoTIFF
- `output_directory`: Directory to save RGB PNGs
- `scale_factor`: Brightness multiplier (1.0-10.0, default: 3.0)
- `bands_per_temporal`: Bands per time step (3-12, default: 6)

**Example:**
```
Extract RGB images from ./cropped/region.tif with brightness factor 3.0
and save to ./rgb
```

### 6. `hls_full_pipeline` ⭐

**Complete workflow: find → download → merge → crop → extract RGB**

This is the primary tool that executes the entire HLS processing pipeline in one step.

**Parameters:**
- `start_date`: Start date (YYYY-MM-DD)
- `end_date`: End date (YYYY-MM-DD)
- `center_longitude`: Center point longitude
- `center_latitude`: Center point latitude
- `area_size_m`: Size of crop area in meters (e.g., 2000 = 2km x 2km)
- `output_directory`: Base output directory
- `number_of_images`: Number of images (1-10, default: 3)
- `cloud_coverage`: Maximum cloud % (0-100, default: 10.0)
- `scale`: Resolution m/pixel (10-100, default: 30)
- `rgb_scale_factor`: RGB brightness (1.0-10.0, default: 3.0)

**Output Structure:**
```
output_directory/
├── downloads/       # Raw downloaded images
├── merged/          # Merged temporal stack
├── cropped/         # Cropped region of interest
└── rgb/             # RGB PNG visualizations
```

**Example:**
```
Execute full HLS pipeline:
- Find 3 cloud-free images from April to September 2024
- Location: (70.4991, 51.1694) near Karaganda, Kazakhstan
- Crop to 2km x 2km area around center point
- Save all outputs to ./hls_output
- Use 30m resolution and brightness factor 3.0 for RGB
```

## Usage Examples

### Example 1: Quick Search

```
Use the hls_find_images tool to find 5 images near coordinates
(70.4991, 51.1694) from May to September 2025 with less than 15% cloud coverage
```

### Example 2: Download and Process Separately

```
1. Download 3 images near (-109.53, 29.19) from April 2024 to ./data/downloads
2. Merge them into ./data/merged/stack.tif
3. Crop to 5km x 5km around the center point to ./data/cropped/region.tif
4. Extract RGB visualizations to ./data/rgb
```

### Example 3: Complete Pipeline (Recommended)

```
Run the full HLS pipeline:
- Dates: June 1, 2024 to September 30, 2024
- Location: (70.4991, 51.1694)
- Area: 3000m x 3000m (3km x 3km)
- Number of images: 3
- Cloud coverage: max 10%
- Output directory: ./my_study_area
```

## Output Files

### Downloaded Images
- **Format**: Multi-band GeoTIFF (.tif)
- **Bands**: 6 bands per image (B2, B3, B4, B8A, B11, B12)
- **Naming**: `{image_id}_{YYYY-MM-DD}.tif`
- **Size**: ~3-5 MB per image (30m resolution, 10km x 10km area)

### Merged Temporal Stack
- **Format**: Multi-band GeoTIFF
- **Bands**: N × 6 bands (N = number of temporal images)
- **Band Labels**: t0_B2, t0_B3, ..., t1_B2, t1_B3, ..., tN_B12
- **Size**: ~9-15 MB for 3 temporal images

### Cropped Region
- **Format**: Multi-band GeoTIFF
- **Bands**: Same as input (preserves all bands)
- **Size**: Depends on crop area (smaller than merged)

### RGB Visualizations
- **Format**: PNG images
- **Files**: One per temporal step (t0_rgb.png, t1_rgb.png, ...)
- **Bands**: 3 (Red, Green, Blue)
- **Size**: ~20-50 KB per PNG

## HLS Band Information

The Harmonized Landsat Sentinel-2 (HLS) dataset provides 6 standard optical/SWIR bands:

| Band | Name | Wavelength | Description |
|------|------|------------|-------------|
| B2 | Blue | 0.48 µm | Blue visible light |
| B3 | Green | 0.56 µm | Green visible light |
| B4 | Red | 0.65 µm | Red visible light |
| B8A | NIR | 0.86 µm | Near-infrared |
| B11 | SWIR1 | 1.6 µm | Short-wave infrared 1 |
| B12 | SWIR2 | 2.2 µm | Short-wave infrared 2 |

**RGB Composite:** B4-B3-B2 (Red-Green-Blue) = Natural color
**False Color:** B8A-B4-B3 (NIR-Red-Green) = Vegetation analysis

## Troubleshooting

### Earth Engine Authentication Error

```
Error: Earth Engine authentication required
```

**Solution:**
```bash
earthengine authenticate
```

Follow the prompts to authenticate with your Google account.

### No Images Found

```
Error: No images found matching criteria
```

**Solutions:**
1. Expand the date range (try a wider time window)
2. Increase `cloud_coverage` threshold (try 20-30%)
3. Increase `buffer_distance` (try 20000m instead of 10000m)
4. Check if coordinates are over land (ocean areas may have no coverage)

### Download Size Limit

Earth Engine has a 50MB download limit per request.

**Solutions:**
1. Reduce `buffer_distance` (smaller area = smaller files)
2. Increase `scale` (lower resolution = smaller files, e.g., 60m instead of 30m)
3. Use the crop tool after downloading to extract smaller regions

### Mismatched Dimensions Error

```
Error: Image X has mismatched dimensions
```

**Cause:** Trying to merge images with different sizes or CRS.

**Solution:** Ensure all images are from the same location and have the same spatial extent.

## Best Practices

### 1. Use Full Pipeline for New Workflows
Start with `hls_full_pipeline` for most use cases. It handles the complete workflow efficiently.

### 2. Optimize Download Area
- Download a larger area than you need initially (use higher `buffer_distance`)
- Crop to exact region of interest after downloading
- This avoids re-downloading if you need to adjust the crop area

### 3. Cloud Coverage Threshold
- Start with 10% for optimal quality
- Increase to 20-30% if no images are found
- Very low thresholds (< 5%) may find no images in some regions/seasons

### 4. Temporal Coverage
- For seasonal analysis: Use 6-month date ranges (e.g., April-September for growing season)
- For change detection: Request 3-5 images evenly spaced across the range
- For time series: Use shorter date ranges with more frequent images

### 5. Resolution Trade-offs
- 30m (default): Good balance of detail and file size
- 10m: Maximum detail, but larger files (use for small areas)
- 60m: Faster downloads, smaller files (use for regional analysis)

## Error Codes and Messages

All tools return structured error messages with actionable guidance:

- **Authentication errors**: Include instructions to run `earthengine authenticate`
- **No images found**: Suggest adjustments to search parameters
- **File not found**: Verify paths and file existence
- **Dimension mismatch**: Explain compatibility requirements
- **Permission denied**: Check directory write permissions

## Limitations

1. **Download Size**: Earth Engine limits downloads to 50MB per request
2. **Processing Time**: Large areas or high-resolution downloads may take several minutes
3. **Cloud Coverage**: Filter only, not a mask (cloudy pixels remain in imagery)
4. **Coordinate Precision**: Point-offset mode uses approximate degree conversion
5. **Date Availability**: HLS data available from 2013-present (Landsat 8 + Sentinel-2)

## Technical Details

### MCP Server Information
- **Server Name**: `skyterra_hls_mcp`
- **Protocol**: MCP (Model Context Protocol)
- **Transport**: Stdio (standard input/output)
- **Python Version**: 3.11+
- **Framework**: FastMCP (mcp>=1.3.0)

### Dependencies
- `earthengine-api`: Google Earth Engine API
- `rasterio`: GeoTIFF reading/writing
- `pillow`: PNG image creation
- `pydantic`: Input validation
- `mcp`: MCP server framework
- `requests`: HTTP downloads

### Data Source
- **Collection**: NASA/HLS/HLSS30/v002
- **Platform**: Google Earth Engine
- **Coverage**: Global
- **Temporal Range**: 2013-present
- **Update Frequency**: Near real-time (2-5 day latency)

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html
```

### Project Structure

```
skyterra-hls-download/
├── skyterra_hls_mcp.py         # MCP server implementation
├── src/
│   ├── __init__.py
│   ├── hls_downloader.py       # Core download functions
│   └── image_utils.py          # Image processing utilities
├── tests/
│   ├── test_hls_downloader.py
│   └── test_image_utils.py
├── pyproject.toml              # Project dependencies
├── README.md                   # Library documentation
└── MCP_README.md               # This file (MCP server docs)
```

## Support and Resources

- **Earth Engine Documentation**: https://developers.google.com/earth-engine
- **HLS Project**: https://hls.gsfc.nasa.gov/
- **MCP Documentation**: https://modelcontextprotocol.io/
- **Issue Reporting**: See repository README for contact information

## License

See project LICENSE file for details.

## Acknowledgments

- NASA HLS Project - Harmonized Landsat Sentinel-2 dataset
- Google Earth Engine - Cloud processing platform
- USGS - Landsat data
- ESA - Sentinel-2 data
- Model Context Protocol - AI integration framework
