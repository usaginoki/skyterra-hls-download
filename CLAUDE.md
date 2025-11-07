# Development Documentation

This document describes the development process, architectural decisions, and challenges encountered during the creation of the SkyTerra HLS Downloader library.

## Project Overview

**Purpose:** Create a Python library for searching, downloading, and processing NASA's Harmonized Landsat Sentinel-2 (HLS) satellite imagery using Google Earth Engine.

**Development Period:** October 2024 - Present

**Key Requirements:**
1. Search and download HLS imagery from Google Earth Engine
2. Support flexible coordinate inputs (point, polygon, Earth Engine geometry)
3. Download single or multiple temporally-spaced images
4. Process multi-band GeoTIFF imagery
5. Merge temporal images into multi-band stacks
6. Crop spatial regions of interest
7. Extract RGB visualizations from multi-temporal data

## Development Timeline

### Phase 1: Core Download Functionality

**Initial Request:** "I need a collection of Python functions which uses Earth Engine to find and download images based on different parameters"

**Implementation:**
- Created `src/hls_downloader.py` with four main functions:
  - `find_single_image()` - Find least cloudy image matching criteria
  - `find_multiple_images()` - Find temporally-spaced images
  - `batch_download()` - Download multiple images
  - `download_images_end_to_end()` - Complete workflow wrapper

**Helper Functions:**
- `_parse_coordinates()` - Convert various coordinate formats to ee.Geometry
- `_extract_metadata()` - Extract image metadata for downloads

**Design Decisions:**
- Default bands: B2, B3, B4, B8A, B11, B12 (standard HLS optical/SWIR bands)
- Default collection: NASA/HLS/HLSS30/v002 (Sentinel-2 harmonized)
- Default cloud coverage threshold: 10%
- Default buffer distance: 10km for point coordinates

### Phase 2: Multi-Band Issue Discovery

**Problem:** Downloaded images only contained 1 band instead of 6

**Root Cause:** Earth Engine returns ZIP files with separate .tif files for each band (B2.tif, B3.tif, B4.tif, B8A.tif, B11.tif, B12.tif). The initial implementation only extracted the first file.

**Solution:**
- Updated `batch_download()` to detect multiple .tif files in ZIP
- Extract all files to temporary directory
- Use rasterio to merge into single multi-band GeoTIFF
- Write merged file with proper band organization

**Verification:**
- File size increased from 0.30 MB (single band) to 8.13 MB (6 bands)
- Confirmed all 6 bands present in correct order

### Phase 3: Band Ordering Issue

**Problem:** Bands were sorted alphabetically (B11, B12, B2, B3, B4, B8A) instead of spectral order

**Requirement:** "Make sure the order of the bands upon merge is as indicated (B2, B3, B4, B8A, B11, B12)"

**Solution:**
- Created `_sort_band_files()` helper function
- Implemented custom sorting with band priority list
- Applied sorting before merging band files

```python
def _sort_band_files(tif_files: List[str]) -> List[str]:
    band_order = ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12']
    def get_band_priority(filename: str) -> int:
        parts = filename.split('.')
        for part in parts:
            if part in band_order:
                return band_order.index(part)
        return len(band_order)
    return sorted(tif_files, key=get_band_priority)
```

**Verification:**
- Created test script to verify band descriptions
- Confirmed proper spectral order: B2 → B3 → B4 → B8A → B11 → B12

### Phase 4: Processing Utilities

**Request:** "Create a function which can merge indicated files into a single tif file, and create a function which cuts out a rectangle based on the indicated image and provided coordinates"

**Implementation:** Created `src/image_utils.py` with two main functions:

#### `merge_temporal_images()`
- Stacks multiple temporal images into single multi-band GeoTIFF
- Creates temporal band labels: t0_B2, t0_B3, ..., t1_B2, t1_B3, ...
- Validates all images have matching dimensions and CRS
- Preserves acquisition dates in metadata

**Key Features:**
- Band descriptions with temporal prefixes
- Metadata tags for temporal information
- Support for arbitrary number of input images
- Optional output (file path or numpy array)

#### `crop_image()`
- Extracts spatial regions from GeoTIFF images
- Two coordinate modes:
  - `bbox`: Direct bounding box [min_lon, min_lat, max_lon, max_lat]
  - `point_offset`: Center point with meter distances ((lon, lat), x_distance_m, y_distance_m)
- Preserves all bands and metadata
- Works with both single-temporal and multi-temporal images

**Design Decisions:**
- Point-offset uses approximate degree conversion (111,320 meters/degree)
- Conversion accounts for latitude using cosine correction
- All bands preserved during cropping
- Band descriptions maintained in output

### Phase 5: Documentation and Testing

**Request:** "Remove all unnecessary files, create a single README"

**Actions:**
- Consolidated all documentation into comprehensive README.md (684 lines)
- Removed scattered test scripts
- Added 5 quick start examples
- Documented all API functions with parameters and return values
- Included band information and troubleshooting sections

**Testing Implementation:**
- Created `tests/test_hls_downloader.py` (21 tests)
  - Coordinate parsing validation
  - Metadata extraction
  - Image search functionality
  - Batch download with band merging

- Created `tests/test_image_utils.py` (20 tests)
  - Band sorting validation
  - Temporal merging (2-5 images)
  - Spatial cropping (both modes)
  - Error handling
  - Integration workflows

**Total Test Coverage:** 41 tests passing

### Phase 6: Real-World Validation

**Test Location:** 51°04'50"N 70°29'56"E (near Karaganda, Kazakhstan)

**Test Parameters:**
- Date range: April - September 2025
- Number of images: 3
- Area: 2km × 2km (1km to each side from center)
- Process: Download → Merge → Crop

**Results:**
- Successfully downloaded 3 cloud-free images (0% cloud coverage)
- Dates: May 5, June 17, September 25, 2025
- Merged file: 18 bands (3 temporal steps × 6 bands)
- Proper temporal labeling: t0_B2 through t2_B12
- Cropped to exact ROI: 212 × 133 pixels
- File sizes: Individual (2.94 MB), Merged (8.82 MB), Cropped (3.88 MB)

### Phase 7: RGB Extraction Feature

**Request:** "Add new util function: RGB extractor. The function gets an image, analyzes number of bands, extracts B2 B3 B4 for each temporal step and creates an RGB, and saves the images to a new folder"

**Implementation:** Added `extract_rgb_images()` to `src/image_utils.py`

**Key Features:**
- Automatic temporal step detection from band count
- Extracts B2 (Blue), B3 (Green), B4 (Red) for each time period
- Configurable brightness scaling (default: 3.0)
- Outputs PNG files for easy visualization
- Smart naming: t0_rgb.png, t1_rgb.png, t2_rgb.png

**Technical Details:**
```python
def extract_rgb_images(
    input_path: str,
    output_directory: str,
    scale_factor: float = 3.0,
    bands_per_temporal: int = 6
) -> List[str]
```

**Band Extraction Logic:**
- For each temporal step: base_idx = temporal_idx × bands_per_temporal
- Blue: base_idx + 1 (B2)
- Green: base_idx + 2 (B3)
- Red: base_idx + 3 (B4)

**Scaling Process:**
1. Read raw reflectance values (typically 0-0.3 range)
2. Multiply by scale_factor (default 3.0)
3. Clip to 0-1 range
4. Convert to 8-bit (0-255) for PNG export

**Dependencies Added:**
- Pillow (PIL) for PNG image creation

**Validation:**
- Tested with 18-band merged image (3 temporal steps)
- Created 3 RGB visualizations successfully
- File sizes: ~27-29 KB per PNG
- Proper color balance confirmed

## Architecture Decisions

### Module Organization

**Two-module design:**
1. `hls_downloader.py` - Earth Engine interaction and downloads
2. `image_utils.py` - Post-processing and analysis

**Rationale:** Separation of concerns - downloading vs. processing

### Coordinate Flexibility

**Supported Formats:**
- Simple point: `(lon, lat)`
- Polygon: `[[lon1, lat1], [lon2, lat2], ...]`
- Earth Engine geometry: `ee.Geometry.Rectangle(...)`

**Rationale:** Different use cases require different input formats

### Band Ordering Standard

**Spectral Order:** B2, B3, B4, B8A, B11, B12

**Rationale:**
- Follows increasing wavelength (visible → NIR → SWIR)
- Common convention in remote sensing
- Simplifies RGB extraction (B2, B3, B4 always first three bands)

### Temporal Labeling Convention

**Format:** `t{idx}_{band_name}`

**Examples:** t0_B2, t1_B2, t2_B2

**Rationale:**
- Clear temporal sequence identification
- Preserves band identity
- Easy to parse programmatically

## Challenges and Solutions

### Challenge 1: Earth Engine ZIP Structure

**Problem:** Earth Engine returns separate files for each band in ZIP archive

**Impact:** Initial implementation only saved first band

**Solution:**
- Detect multiple .tif files in ZIP
- Extract all to temporary directory
- Merge using rasterio.stack() into multi-band GeoTIFF
- Clean up temporary files

**Code Pattern:**
```python
with ZipFile(zip_path) as z:
    tif_files = [f for f in z.namelist() if f.endswith('.tif')]
    if len(tif_files) > 1:
        # Extract and merge
        sorted_files = _sort_band_files(tif_files)
        # Merge bands in correct order
```

### Challenge 2: Band Ordering Consistency

**Problem:** Alphabetical sorting (B11, B12, B2, B3, B4, B8A) not spectral order

**Impact:** Wrong bands used for RGB extraction, confusing analysis

**Solution:** Custom sorting function with explicit band priority

**Key Learning:** Always validate assumptions about data ordering

### Challenge 3: Earth Engine Download Limits

**Problem:** 50MB download limit from Earth Engine

**Impact:** Limiting factor for large areas or many bands

**Solutions Implemented:**
- Document buffer_distance parameter for area control
- Suggest scale parameter for resolution adjustment
- Recommend crop-after-download workflow for large regions

**Documentation:** Added troubleshooting section with concrete examples

### Challenge 4: Coordinate System Conversions

**Problem:** Converting meter distances to degrees for cropping

**Solution:** Approximate conversion with latitude correction
```python
meters_per_degree = 111320
lon_offset = x_distance / (meters_per_degree * np.cos(np.radians(lat)))
lat_offset = y_distance / meters_per_degree
```

**Limitation:** Approximation only, not suitable for high-precision geodesy

**Future Improvement:** Could use pyproj for exact conversions

### Challenge 5: RGB Brightness Scaling

**Problem:** Raw HLS reflectance values (0-0.3 range) too dark for visualization

**Solution:** Configurable scale_factor parameter (default 3.0)

**Rationale:**
- Surface reflectance typically 0-30% (0.0-0.3)
- Multiply by 3.0 brings to 0-90% of display range
- User-adjustable for different scenes (darker/brighter)

## Testing Strategy

### Test Categories

**1. Unit Tests**
- Individual function validation
- Parameter handling
- Error conditions
- Edge cases

**2. Integration Tests**
- Multi-step workflows
- Data passing between functions
- End-to-end processes

**3. Real-World Tests**
- Actual Earth Engine downloads
- Real coordinate locations
- Current year data (2025)

### Test Coverage

**Download Functions (21 tests):**
- Coordinate parsing (point, polygon, geometry)
- Metadata extraction
- Single image search
- Multiple image search
- Batch download with merging
- Error handling

**Processing Functions (20 tests):**
- Band sorting validation
- Temporal merging (2, 3, 4, 5 images)
- Spatial cropping (bbox and point_offset)
- RGB extraction
- Dimension/CRS validation
- Integration workflows

**Workflow Test:**
- Complete pipeline: Download → Merge → Crop → Extract RGB
- Real location (51°04'50"N 70°29'56"E)
- Real data (2025 HLS imagery)
- Verifies all outputs

## Key Metrics

**Code Statistics:**
- `hls_downloader.py`: 499 lines
- `image_utils.py`: 479 lines
- Total tests: 41 passing
- Documentation: 761 lines (README + CLAUDE.md)

**Performance:**
- Download speed: ~2-3 MB/image
- Merge speed: <1 second for 3 images
- Crop speed: <0.5 seconds
- RGB extraction: <1 second per temporal step

**File Sizes (typical):**
- Single HLS image (6 bands, 30m resolution, 5km area): ~3 MB
- Merged 3-temporal stack: ~9 MB
- Cropped ROI (2km × 2km): ~4 MB
- RGB PNG: ~25-30 KB per temporal step

## Future Enhancements

### Potential Improvements

1. **Exact Coordinate Transformations**
   - Replace approximate degree conversion with pyproj
   - Support for multiple CRS systems
   - High-precision geodetic calculations

2. **Advanced Band Indices**
   - NDVI calculation (Normalized Difference Vegetation Index)
   - NDWI (Water index)
   - Other spectral indices
   - Automatic calculation and storage

3. **Cloud Masking**
   - Use HLS quality bands
   - Automatic cloud pixel removal
   - Generate cloud-free composites

4. **Time Series Analysis**
   - Temporal interpolation
   - Anomaly detection
   - Trend analysis

5. **Parallel Processing**
   - Multi-threaded downloads
   - Parallel image processing
   - GPU-accelerated computations

6. **Additional Visualizations**
   - False color composites (NIR-Red-Green)
   - Color-infrared
   - Custom band combinations
   - Animated GIFs for temporal sequences

7. **Export Formats**
   - Cloud-optimized GeoTIFF (COG)
   - NetCDF for time series
   - Zarr for cloud storage
   - JPEG2000 for compression

## Lessons Learned

### Technical Insights

1. **Always validate Earth Engine outputs**
   - Don't assume band ordering
   - Check for multi-file ZIP archives
   - Verify band counts

2. **Design for flexibility**
   - Multiple coordinate input formats
   - Optional return types (file vs array)
   - Configurable parameters with sensible defaults

3. **Document limitations clearly**
   - Earth Engine 50MB limit
   - Approximate coordinate conversions
   - Cloud coverage as filter, not mask

4. **Test with real data early**
   - Unit tests catch logic errors
   - Real-world tests catch assumption errors
   - Integration tests catch workflow issues

### Development Process

1. **Iterative refinement works**
   - Started with basic functionality
   - Added features based on discovered needs
   - Fixed issues as they arose

2. **User feedback is invaluable**
   - Band ordering issue found through testing
   - RGB extraction added based on visualization need
   - Documentation improved through usage questions

3. **Comprehensive documentation matters**
   - 5 quick start examples cover common use cases
   - API reference enables self-service
   - Troubleshooting section prevents repeated questions

## Acknowledgments

**Data Sources:**
- NASA HLS Project - Harmonized Landsat Sentinel-2 dataset
- Google Earth Engine - Cloud processing platform
- USGS - Landsat data
- ESA - Sentinel-2 data

**Technologies:**
- Python 3.11+
- earthengine-api - Earth Engine Python SDK
- rasterio - GeoTIFF processing
- Pillow - Image creation
- pytest - Testing framework

---

*This document maintained as development log and architectural reference for the SkyTerra HLS Downloader project.*
