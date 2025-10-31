# SkyTerra HLS Downloader

A Python library for searching, downloading, and processing NASA's Harmonized Landsat Sentinel-2 (HLS) satellite imagery using Google Earth Engine.

## Features

### Download & Search
- **Find Single Images** - Search for the least cloudy image matching your criteria
- **Find Multiple Images** - Get temporally-spaced images across a date range
- **Batch Download** - Download multiple images with automatic multi-band merging
- **Flexible Coordinates** - Support for point coordinates, polygons, or Earth Engine geometries
- **Cloud Filtering** - Configurable cloud coverage thresholds
- **Smart Search** - Automatic window expansion to find evenly-spaced images

### Processing & Analysis
- **Temporal Merging** - Stack multiple temporal images into a single multi-band GeoTIFF
- **Spatial Cropping** - Extract regions using bounding boxes or point+distance
- **Band Ordering** - Automatic sorting of bands to standard order (B2, B3, B4, B8A, B11, B12)
- **Smart Labeling** - Temporal band labels (t0_B2, t1_B2, etc.) with metadata preservation

### Quality & Testing
- **Well Tested** - 41 passing tests with comprehensive coverage
- **Production Ready** - Tested with real HLS data from multiple locations

## Installation

### Prerequisites

1. Python 3.11 or higher
2. Google Earth Engine account ([sign up here](https://earthengine.google.com/signup/))
3. Earth Engine authentication

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd skyterra-hls-download

# Install dependencies using uv
uv sync

# Authenticate with Earth Engine (one-time setup)
earthengine authenticate
```

## Quick Start

### Example 1: Download a Single Image

```python
from src.hls_downloader import download_images_end_to_end

# Download a single image
files = download_images_end_to_end(
    date_range=('2024-09-25', '2024-10-05'),
    coordinates=(71.4491, 51.1694),  # (lon, lat) for Astana, Kazakhstan
    output_directory='./downloads',
    number_of_images=1,
    cloud_coverage=30,
    buffer_distance=5000  # 5km radius around point
)

print(f"Downloaded: {files[0]}")
# Result: 6-band GeoTIFF with bands B2, B3, B4, B8A, B11, B12
```

### Example 2: Download Multiple Temporal Images

```python
from src.hls_downloader import download_images_end_to_end

# Download 3 evenly-spaced images over a month
images = download_images_end_to_end(
    date_range=('2024-04-01', '2024-04-30'),
    coordinates=(71.4491, 51.1694),
    output_directory='./downloads',
    number_of_images=3,
    cloud_coverage=30
)

print(f"Downloaded {len(images)} images:")
for img in images:
    print(f"  - {img}")
```

### Example 3: Merge Temporal Images

```python
from src.image_utils import merge_temporal_images

# Merge 3 temporal images into one multi-band file
# Each image has 6 bands, result will have 18 bands (6 × 3)
merged_file = merge_temporal_images(
    image_paths=[
        './downloads/image_2024-04-05.tif',
        './downloads/image_2024-04-15.tif',
        './downloads/image_2024-04-25.tif'
    ],
    output_path='./merged_temporal.tif'
)

# Bands are labeled: t0_B2, t0_B3, ..., t1_B2, t1_B3, ..., t2_B2, t2_B3, ...
print(f"Merged temporal stack: {merged_file}")
```

### Example 4: Crop to Region of Interest

```python
from src.image_utils import crop_image

# Crop using bounding box
cropped = crop_image(
    input_path='./merged_temporal.tif',
    coordinates=[71.37, 51.12, 71.52, 51.21],  # [min_lon, min_lat, max_lon, max_lat]
    output_path='./cropped.tif',
    coordinate_type='bbox'
)

# Or crop using point and distance
cropped = crop_image(
    input_path='./merged_temporal.tif',
    coordinates=((71.4491, 51.1694), 5000, 5000),  # ((lon, lat), x_distance_m, y_distance_m)
    output_path='./cropped_point.tif',
    coordinate_type='point_offset'
)
```

### Example 5: Complete Multi-Temporal Workflow

```python
from src.hls_downloader import download_images_end_to_end
from src.image_utils import merge_temporal_images, crop_image

# Step 1: Download 3 temporal images
print("Downloading temporal images...")
images = download_images_end_to_end(
    date_range=('2024-04-01', '2024-04-30'),
    coordinates=(71.4491, 51.1694),
    output_directory='./downloads',
    number_of_images=3,
    cloud_coverage=30
)

# Step 2: Merge into single temporal stack (18 bands)
print("Merging temporal images...")
merged = merge_temporal_images(
    image_paths=images,
    output_path='./merged_stack.tif'
)

# Step 3: Crop to region of interest
print("Cropping to ROI...")
cropped = crop_image(
    input_path=merged,
    coordinates=[71.40, 51.14, 71.50, 51.20],
    output_path='./final_roi.tif',
    coordinate_type='bbox'
)

print(f"Final output: {cropped}")
print("Ready for analysis!")
```

## API Reference

### Download Functions

#### `find_single_image()`

Find a single image with the least cloud coverage matching the criteria.

```python
from src.hls_downloader import find_single_image

result = find_single_image(
    date_range=('2024-04-25', '2024-04-26'),
    coordinates=(71.4491, 51.1694),
    image_collection="NASA/HLS/HLSS30/v002",  # optional
    bands=['B2', 'B3', 'B4', 'B8A', 'B11', 'B12'],  # optional
    cloud_coverage=10.0,  # optional
    buffer_distance=10000  # optional, meters
)
```

**Parameters:**
- `date_range` *(required)*: (start_date, end_date) in 'YYYY-MM-DD' format
- `coordinates` *(required)*: Point (lon, lat), polygon coordinates, or ee.Geometry
- `image_collection`: Earth Engine collection ID (default: "NASA/HLS/HLSS30/v002")
- `bands`: Band names (default: ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12'])
- `cloud_coverage`: Maximum cloud coverage % (default: 10.0)
- `buffer_distance`: Buffer in meters for point coordinates (default: 10000)

**Returns:** Dict with 'image' (ee.Image) and 'metadata' (Dict), or None if no image found

#### `find_multiple_images()`

Find multiple images evenly spaced across the date range.

```python
from src.hls_downloader import find_multiple_images

results = find_multiple_images(
    date_range=('2024-04-01', '2024-04-30'),
    coordinates=(71.4491, 51.1694),
    number_of_images=3,  # optional
    max_expansion_days=30  # optional
    # ... same parameters as find_single_image()
)
```

**Parameters:** Same as `find_single_image()` plus:
- `number_of_images`: Number of images to find (default: 3)
- `max_expansion_days`: Maximum days to expand search window (default: 30)

**Returns:** List of dicts with 'image' and 'metadata'

#### `batch_download()`

Download multiple images to a directory.

```python
from src.hls_downloader import batch_download

downloaded = batch_download(
    image_objects=results,  # from find_*_images()
    output_directory='./downloads',
    scale=30,  # optional
    crs='EPSG:4326'  # optional
)
```

**Parameters:**
- `image_objects`: List of image objects from find functions
- `output_directory`: Directory to save images
- `scale`: Resolution in meters (default: 30)
- `crs`: Coordinate reference system (default: 'EPSG:4326')

**Returns:** List of successfully downloaded file paths

**Note:** Automatically merges multi-band images from Earth Engine's separate band files.

#### `download_images_end_to_end()`

Complete workflow: find and download images in one function.

```python
from src.hls_downloader import download_images_end_to_end

files = download_images_end_to_end(
    date_range=('2024-04-01', '2024-04-30'),
    coordinates=(71.4491, 51.1694),
    output_directory='./downloads',
    number_of_images=3,
    # ... combines all parameters from above functions
)
```

**Returns:** List of downloaded file paths

### Processing Functions

#### `merge_temporal_images()`

Merge multiple temporal images into a single multi-band GeoTIFF with chronological band stacking.

```python
from src.image_utils import merge_temporal_images

# Save to file
merged_path = merge_temporal_images(
    image_paths=['img1.tif', 'img2.tif', 'img3.tif'],
    output_path='merged.tif'
)

# Return array without saving
merged_array = merge_temporal_images(
    image_paths=['img1.tif', 'img2.tif', 'img3.tif'],
    output_path=None
)
```

**Parameters:**
- `image_paths`: List of GeoTIFF paths in chronological order [t0, t1, t2, ...]
- `output_path`: Path to save merged GeoTIFF, or None to return array

**Returns:**
- If output_path provided: str (file path)
- If output_path is None: np.ndarray (merged array)

**Band Organization:**
- Bands from each image are stacked sequentially
- Band names include temporal prefix: `t0_B2, t0_B3, ..., t1_B2, t1_B3, ...`
- Metadata includes acquisition dates and temporal count

**Example:** 3 images with 6 bands each → 18-band output

**Validation:**
- All images must have matching dimensions and CRS
- Raises `ValueError` if mismatched

#### `crop_image()`

Crop a rectangular region from a GeoTIFF image, preserving all bands.

```python
from src.image_utils import crop_image

# Bounding box crop
cropped_path = crop_image(
    input_path='input.tif',
    coordinates=[71.37, 51.12, 71.52, 51.21],
    output_path='cropped.tif',
    coordinate_type='bbox'
)

# Point + distance crop
cropped_path = crop_image(
    input_path='input.tif',
    coordinates=((71.45, 51.17), 5000, 5000),
    output_path='cropped.tif',
    coordinate_type='point_offset'
)

# Return array without saving
array, transform, crs = crop_image(
    input_path='input.tif',
    coordinates=[71.37, 51.12, 71.52, 51.21],
    output_path=None,
    coordinate_type='bbox'
)
```

**Parameters:**
- `input_path`: Path to input GeoTIFF
- `coordinates`: Cropping coordinates (see coordinate types below)
- `output_path`: Path to save cropped image, or None to return array
- `coordinate_type`: Either 'bbox' or 'point_offset'

**Coordinate Types:**
- `'bbox'`: `[min_lon, min_lat, max_lon, max_lat]` - Direct bounding box
- `'point_offset'`: `((lon, lat), x_distance_m, y_distance_m)` - Point with meter offsets

**Returns:**
- If output_path provided: str (file path)
- If output_path is None: Tuple[np.ndarray, Affine, CRS]

**Features:**
- Works with both single-temporal and multi-temporal images
- Preserves all bands and band descriptions
- Maintains spatial reference and metadata

## Band Information

### Default Bands

The library uses 6 standard HLS spectral bands by default:

| Band | Name | Wavelength | Description |
|------|------|------------|-------------|
| B2 | Blue | 0.459–0.525 μm | Visible blue light |
| B3 | Green | 0.525–0.600 μm | Visible green light |
| B4 | Red | 0.630–0.690 μm | Visible red light |
| B8A | NIR Narrow | 0.846–0.885 μm | Near-infrared |
| B11 | SWIR 1 | 1.560–1.660 μm | Short-wave infrared 1 |
| B12 | SWIR 2 | 2.100–2.300 μm | Short-wave infrared 2 |

### Band Ordering

Downloaded images have bands in this standard spectral order:
1. Band 1: B2 (Blue)
2. Band 2: B3 (Green)
3. Band 3: B4 (Red)
4. Band 4: B8A (NIR)
5. Band 5: B11 (SWIR 1)
6. Band 6: B12 (SWIR 2)

This ordering is automatically enforced during download, regardless of how Earth Engine returns the data.

### Temporal Band Labels

When using `merge_temporal_images()`, bands are labeled with temporal prefixes:

```
# 3 images merged:
Band 1:  t0_B2   (Time 0, Blue)
Band 2:  t0_B3   (Time 0, Green)
Band 3:  t0_B4   (Time 0, Red)
Band 4:  t0_B8A  (Time 0, NIR)
Band 5:  t0_B11  (Time 0, SWIR 1)
Band 6:  t0_B12  (Time 0, SWIR 2)
Band 7:  t1_B2   (Time 1, Blue)
Band 8:  t1_B3   (Time 1, Green)
...
Band 18: t2_B12  (Time 2, SWIR 2)
```

## Coordinate Formats

### Point Coordinates

Simple lon, lat tuple:
```python
coordinates = (71.4491, 51.1694)  # Astana, Kazakhstan
```

Automatically buffered to create a circular region (default 10km radius).

### Polygon Coordinates

List of [lon, lat] pairs:
```python
coordinates = [
    [71.37, 51.12],
    [71.52, 51.12],
    [71.52, 51.21],
    [71.37, 51.21],
    [71.37, 51.12]  # Close the polygon
]
```

### Earth Engine Geometry

Pass ee.Geometry objects directly:
```python
coordinates = ee.Geometry.Rectangle([71.37, 51.12, 71.52, 51.21])
```

## Important Notes

### Earth Engine Download Limits

Earth Engine has a 50MB limit for direct downloads. If you encounter size errors:

1. **Reduce buffer distance** - Use smaller areas (e.g., 5km instead of 15km)
2. **Increase scale** - Use coarser resolution (e.g., 60m instead of 30m)
3. **Select fewer bands** - Download only necessary bands
4. **Crop after download** - Download larger area, then crop to ROI

### Cloud Coverage

- HLS images use the `CLOUD_COVERAGE` metadata field
- Values range from 0-100 (percentage)
- Lower thresholds (5-10%) may limit available images
- Higher thresholds (20-30%) provide more options
- The algorithm selects the least cloudy image within criteria

### Temporal Spacing

The `find_multiple_images()` function:
1. Divides date range into equal intervals
2. Searches for images near each target date
3. Gradually expands search window if needed
4. Avoids duplicate images
5. May expand beyond date range if `max_expansion_days` allows

## Testing

### Run Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_image_utils.py -v
```

### Test Results

- **41 tests passing**
- **1 skipped** (requires EE credentials)
- **Coverage:** Comprehensive coverage of all functions

### Test Categories

- **Download Functions** (21 tests)
  - Coordinate parsing
  - Metadata extraction
  - Single/multiple image search
  - Batch download
  - End-to-end workflows

- **Processing Functions** (20 tests)
  - Band sorting
  - Temporal merging (2-5 images)
  - Spatial cropping (bbox and point+offset)
  - Error handling
  - Integration workflows

## Project Structure

```
skyterra-hls-download/
├── src/
│   ├── __init__.py              # Package exports
│   ├── hls_downloader.py        # Download functions
│   └── image_utils.py           # Processing utilities
├── tests/
│   ├── __init__.py
│   ├── test_hls_downloader.py   # Download tests
│   └── test_image_utils.py      # Processing tests
├── pyproject.toml               # Project configuration
├── uv.lock                      # Dependency lock file
└── README.md                    # This file
```

## Dependencies

### Core Dependencies
- `earthengine-api>=1.6.14` - Google Earth Engine Python API
- `requests>=2.31.0` - HTTP library for downloads
- `rasterio>=1.4.3` - GeoTIFF I/O and processing
- `numpy` - Array operations
- `seaborn>=0.13.2` - Statistical visualization

### Development Dependencies
- `pytest>=8.4.2` - Testing framework
- `pytest-cov>=7.0.0` - Coverage reporting

### Installation
```bash
# Install core dependencies
uv sync

# Install with dev dependencies
uv add --dev pytest pytest-cov
```

## Configuration

Update the Earth Engine project ID in `src/hls_downloader.py`:

```python
ee.Initialize(project='your-project-id')
```

Or set it as an environment variable before running.

## Real-World Example

Successfully tested with actual HLS data from Astana, Kazakhstan:

```python
# Download image from ~1 month ago
files = download_images_end_to_end(
    date_range=('2024-09-25', '2024-10-05'),
    coordinates=(71.4491, 51.1694),  # Astana center
    output_directory='./downloads',
    number_of_images=1,
    cloud_coverage=30,
    buffer_distance=5000
)
```

**Result:**
- Found image: T42UXB_20240925T061631
- Acquisition date: 2024-09-25
- Cloud coverage: 0% (completely cloud-free!)
- File size: 8.13 MB
- Bands: 6 (B2, B3, B4, B8A, B11, B12) in correct order
- Dimensions: 530 × 335 pixels

## Troubleshooting

### Authentication Issues
```bash
# Re-authenticate with Earth Engine
earthengine authenticate
```

### Import Errors
```bash
# Ensure you're in the project directory
cd skyterra-hls-download

# Verify installation
uv run python -c "from src import download_images_end_to_end; print('Success!')"
```

### Memory Issues
If processing large images:
- Use `output_path=None` to avoid disk I/O
- Crop images before merging temporally
- Process in smaller batches

### Download Size Errors
If hitting Earth Engine's 50MB limit:
```python
# Reduce area
buffer_distance=5000  # instead of 10000

# Or reduce resolution
scale=60  # instead of 30
```

## Use Cases

### Change Detection
```python
# Download images from two time periods
images_before = download_images_end_to_end(...)  # 2023 data
images_after = download_images_end_to_end(...)   # 2024 data

# Merge separately
stack_before = merge_temporal_images(images_before, 'before.tif')
stack_after = merge_temporal_images(images_after, 'after.tif')

# Compare bands for change detection
```

### Vegetation Monitoring
```python
# Download seasonal images
images = download_images_end_to_end(
    date_range=('2024-01-01', '2024-12-31'),
    coordinates=farm_location,
    number_of_images=12,  # Monthly
    cloud_coverage=20
)

# Merge for time series analysis
temporal_stack = merge_temporal_images(images, 'annual.tif')

# Calculate NDVI over time using NIR (B8A) and Red (B4) bands
```

### Training Data Extraction
```python
# Download large area
large_image = download_images_end_to_end(
    coordinates=region,
    buffer_distance=20000
)

# Extract multiple training chips
for location in training_locations:
    crop_image(
        input_path=large_image[0],
        coordinates=location,
        output_path=f'train_{location.id}.tif',
        coordinate_type='point_offset'
    )
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `uv run pytest tests/ -v`
5. Submit a pull request

## Citation

If you use this software in your research, please cite:

```
SkyTerra HLS Downloader
NASA Harmonized Landsat Sentinel-2 (HLS) data access and processing tool
https://github.com/your-org/skyterra-hls-download
```

## License

[Add your license here]

## Support

For issues or questions:
- Open an issue on GitHub
- Check Earth Engine documentation: https://developers.google.com/earth-engine
- HLS product guide: https://lpdaac.usgs.gov/products/hlss30v002/

## Acknowledgments

- NASA's HLS project for providing harmonized satellite imagery
- Google Earth Engine for the cloud platform
- USGS and ESA for Landsat and Sentinel-2 data
