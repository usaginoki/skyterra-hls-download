import ee
import requests
import zipfile
import io
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Union, Optional

# Authenticate and initialize the Earth Engine API.
# You'll need to run `earthengine authenticate` in your terminal first.
try:
    ee.Initialize(project='global-harmony-430907-b1')
except Exception as e:
    print("Please authenticate the Earth Engine API by running 'earthengine authenticate' in your terminal.")
    print(e)
    exit()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _sort_band_files(tif_files: List[str]) -> List[str]:
    """
    Sort band files according to standard band order: B2, B3, B4, B8A, B11, B12.

    Parameters:
    -----------
    tif_files : List[str]
        List of .tif filenames from Earth Engine download

    Returns:
    --------
    List[str] : Sorted list of filenames
    """
    # Define the standard band order
    band_order = ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12']

    def get_band_priority(filename: str) -> int:
        """Extract band name and return its priority order"""
        # Extract band name from filename (e.g., "image.B2.tif" -> "B2")
        parts = filename.split('.')
        for part in parts:
            if part in band_order:
                return band_order.index(part)
        # If band not in standard order, put it at the end
        return len(band_order)

    return sorted(tif_files, key=get_band_priority)


def _parse_coordinates(
    coordinates: Union[Tuple[float, float], List[List[float]], ee.Geometry],
    buffer_distance: int = 10000
) -> ee.Geometry:
    """
    Convert coordinates to ee.Geometry.

    Parameters:
    -----------
    coordinates : Union[Tuple[float, float], List[List[float]], ee.Geometry]
        Either:
        - Point: (lon, lat) tuple, will be buffered by buffer_distance
        - Polygon: List of [lon, lat] coordinate pairs
        - ee.Geometry: Pass through directly
    buffer_distance : int
        Buffer distance in meters for Point coordinates (default: 10000)

    Returns:
    --------
    ee.Geometry : Earth Engine Geometry object
    """
    # Check if it's a Point (lon, lat)
    if isinstance(coordinates, tuple) and len(coordinates) == 2:
        lon, lat = coordinates
        return ee.Geometry.Point([lon, lat]).buffer(buffer_distance)

    # Check if it's a Polygon (list of coordinate pairs)
    if isinstance(coordinates, list) and len(coordinates) > 0:
        if isinstance(coordinates[0], (list, tuple)) and len(coordinates[0]) == 2:
            return ee.Geometry.Polygon(coordinates)

    # Check if it's already an ee.Geometry object (by checking for common methods)
    if hasattr(coordinates, 'getInfo') or hasattr(coordinates, 'bounds'):
        return coordinates

    raise ValueError(
        "Coordinates must be either a (lon, lat) tuple, "
        "a list of [lon, lat] pairs for a polygon, "
        "or an ee.Geometry object"
    )


def _extract_metadata(image: ee.Image, region: ee.Geometry) -> Dict:
    """
    Extract metadata from an ee.Image.

    Parameters:
    -----------
    image : ee.Image
        Earth Engine Image object
    region : ee.Geometry
        Region of interest

    Returns:
    --------
    Dict : Dictionary containing date, cloud_coverage, and image_id
    """
    properties = image.getInfo()['properties']

    # Extract date (format varies by collection, try common formats)
    date_str = properties.get('system:time_start')
    if date_str:
        date = datetime.fromtimestamp(int(date_str) / 1000).strftime('%Y-%m-%d')
    else:
        date = "unknown"

    # Extract cloud coverage
    cloud_coverage = properties.get('CLOUD_COVERAGE', properties.get('CLOUDY_PIXEL_PERCENTAGE', 0))

    # Extract image ID
    image_id = properties.get('system:index', 'unknown')

    return {
        'date': date,
        'cloud_coverage': cloud_coverage,
        'image_id': image_id,
        'region': region
    }


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def find_single_image(
    date_range: Tuple[str, str],
    coordinates: Union[Tuple[float, float], List[List[float]], ee.Geometry],
    image_collection: str = "NASA/HLS/HLSS30/v002",
    bands: Optional[List[str]] = None,
    cloud_coverage: float = 10.0,
    buffer_distance: int = 10000
) -> Optional[Dict]:
    """
    Find a single image with the least cloud coverage matching the criteria.

    Parameters:
    -----------
    date_range : Tuple[str, str]
        (start_date, end_date) in 'YYYY-MM-DD' format
    coordinates : Union[Tuple[float, float], List[List[float]], ee.Geometry]
        Either (lon, lat), polygon coordinates, or ee.Geometry
    image_collection : str
        Earth Engine image collection ID (default: "NASA/HLS/HLSS30/v002")
    bands : Optional[List[str]]
        List of band names (default: ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12'])
    cloud_coverage : float
        Maximum cloud coverage percentage (default: 10.0)
    buffer_distance : int
        Buffer distance for point coordinates in meters (default: 10000)

    Returns:
    --------
    Optional[Dict] : Dictionary with 'image' (ee.Image) and 'metadata' (Dict),
                     or None if no image found
    """
    if bands is None:
        bands = ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12']

    # Parse coordinates to ee.Geometry
    region = _parse_coordinates(coordinates, buffer_distance)

    # Build the image collection with filters
    start_date, end_date = date_range
    collection = (
        ee.ImageCollection(image_collection)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUD_COVERAGE', cloud_coverage))
        .filterBounds(region)
        .select(bands)
        .sort('CLOUD_COVERAGE')  # Sort by cloud coverage (ascending)
    )

    # Get the first image (least cloudy)
    size = collection.size().getInfo()
    if size == 0:
        return None

    image = collection.first()
    metadata = _extract_metadata(image, region)

    return {
        'image': image,
        'metadata': metadata
    }


def find_multiple_images(
    date_range: Tuple[str, str],
    coordinates: Union[Tuple[float, float], List[List[float]], ee.Geometry],
    image_collection: str = "NASA/HLS/HLSS30/v002",
    bands: Optional[List[str]] = None,
    cloud_coverage: float = 10.0,
    number_of_images: int = 3,
    buffer_distance: int = 10000,
    max_expansion_days: int = 30
) -> List[Dict]:
    """
    Find multiple images evenly spaced across the date range.

    Parameters:
    -----------
    date_range : Tuple[str, str]
        (start_date, end_date) in 'YYYY-MM-DD' format
    coordinates : Union[Tuple[float, float], List[List[float]], ee.Geometry]
        Either (lon, lat), polygon coordinates, or ee.Geometry
    image_collection : str
        Earth Engine image collection ID (default: "NASA/HLS/HLSS30/v002")
    bands : Optional[List[str]]
        List of band names (default: ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12'])
    cloud_coverage : float
        Maximum cloud coverage percentage (default: 10.0)
    number_of_images : int
        Number of images to find (default: 3)
    buffer_distance : int
        Buffer distance for point coordinates in meters (default: 10000)
    max_expansion_days : int
        Maximum days to expand search window if images not found (default: 30)

    Returns:
    --------
    List[Dict] : List of dictionaries with 'image' and 'metadata'
    """
    if bands is None:
        bands = ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12']

    # Parse dates
    start_date = datetime.strptime(date_range[0], '%Y-%m-%d')
    end_date = datetime.strptime(date_range[1], '%Y-%m-%d')
    total_days = (end_date - start_date).days

    if total_days <= 0:
        raise ValueError("end_date must be after start_date")

    # Calculate target dates (evenly spaced)
    interval_days = total_days / number_of_images
    target_dates = [
        start_date + timedelta(days=interval_days * (i + 0.5))
        for i in range(number_of_images)
    ]

    found_images = []

    # Initial search window (start with 3 days around each target)
    initial_window_days = max(3, total_days // (number_of_images * 2))

    for target_date in target_dates:
        image_found = False
        expansion_days = 0

        # Gradually expand the search window until we find an image
        while not image_found and expansion_days <= max_expansion_days:
            window_days = initial_window_days + expansion_days

            search_start = max(start_date, target_date - timedelta(days=window_days))
            search_end = min(end_date, target_date + timedelta(days=window_days))

            result = find_single_image(
                date_range=(search_start.strftime('%Y-%m-%d'), search_end.strftime('%Y-%m-%d')),
                coordinates=coordinates,
                image_collection=image_collection,
                bands=bands,
                cloud_coverage=cloud_coverage,
                buffer_distance=buffer_distance
            )

            if result:
                # Check if we already have this image (by ID)
                if not any(img['metadata']['image_id'] == result['metadata']['image_id']
                          for img in found_images):
                    found_images.append(result)
                    image_found = True
                    print(f"Found image for target date {target_date.strftime('%Y-%m-%d')}: "
                          f"{result['metadata']['image_id']} (date: {result['metadata']['date']}, "
                          f"cloud: {result['metadata']['cloud_coverage']:.1f}%)")
                else:
                    # Duplicate found, expand window to find a different image
                    expansion_days += 3
            else:
                # No image found, expand window
                expansion_days += 3

        if not image_found:
            print(f"Warning: Could not find unique image for target date "
                  f"{target_date.strftime('%Y-%m-%d')} within {max_expansion_days} day expansion")

    # Sort by date
    found_images.sort(key=lambda x: x['metadata']['date'])

    print(f"\nFound {len(found_images)} out of {number_of_images} requested images")
    return found_images


def batch_download(
    image_objects: List[Dict],
    output_directory: str,
    scale: int = 30,
    crs: str = 'EPSG:4326'
) -> List[str]:
    """
    Download multiple images to a directory.

    Parameters:
    -----------
    image_objects : List[Dict]
        List of dictionaries containing 'image' and 'metadata' from find functions
    output_directory : str
        Directory path to save downloaded images
    scale : int
        Resolution in meters (default: 30)
    crs : str
        Coordinate reference system (default: 'EPSG:4326')

    Returns:
    --------
    List[str] : List of file paths for successfully downloaded images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    downloaded_files = []

    for idx, img_obj in enumerate(image_objects, 1):
        image = img_obj['image']
        metadata = img_obj['metadata']
        region = metadata['region']

        # Generate filename
        filename = f"{metadata['image_id']}_{metadata['date']}.tif"
        filepath = os.path.join(output_directory, filename)

        print(f"[{idx}/{len(image_objects)}] Downloading {filename}...")

        try:
            # Get download URL
            url = image.getDownloadURL({
                'scale': scale,
                'crs': crs,
                'region': region.getInfo()['coordinates']
            })

            # Download the file
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            # Earth Engine returns a zip file with separate .tif for each band
            # We need to merge them into a single multi-band GeoTIFF
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                tif_files = _sort_band_files([name for name in z.namelist() if name.endswith('.tif')])

                if not tif_files:
                    print(f"  ✗ No .tif files found in downloaded zip for {filename}")
                    continue

                if len(tif_files) == 1:
                    # Single band, just extract it
                    with open(filepath, 'wb') as f:
                        f.write(z.read(tif_files[0]))
                    downloaded_files.append(filepath)
                    print(f"  ✓ Saved single-band image to {filepath}")
                else:
                    # Multiple bands, merge into single multi-band GeoTIFF
                    import rasterio
                    import tempfile
                    import shutil

                    temp_dir = tempfile.mkdtemp()
                    try:
                        # Extract all band files to temp directory
                        band_files = []
                        for tif_name in tif_files:
                            temp_path = os.path.join(temp_dir, tif_name)
                            with open(temp_path, 'wb') as f:
                                f.write(z.read(tif_name))
                            band_files.append(temp_path)

                        # Read first band to get metadata
                        with rasterio.open(band_files[0]) as src0:
                            meta = src0.meta.copy()
                            meta.update(count=len(band_files))

                        # Create multi-band GeoTIFF
                        with rasterio.open(filepath, 'w', **meta) as dst:
                            for idx, band_file in enumerate(band_files, 1):
                                with rasterio.open(band_file) as src:
                                    dst.write(src.read(1), idx)
                                    # Extract band name from filename (e.g., B2, B3, etc.)
                                    band_name = os.path.basename(band_file).split('.')[-2]
                                    dst.set_band_description(idx, band_name)

                        downloaded_files.append(filepath)
                        print(f"  ✓ Saved {len(band_files)}-band image to {filepath}")

                    finally:
                        # Clean up temp directory
                        shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            print(f"  ✗ Error downloading {filename}: {str(e)}")

    print(f"\nSuccessfully downloaded {len(downloaded_files)}/{len(image_objects)} images")
    return downloaded_files


def download_images_end_to_end(
    date_range: Tuple[str, str],
    coordinates: Union[Tuple[float, float], List[List[float]], ee.Geometry],
    output_directory: str,
    image_collection: str = "NASA/HLS/HLSS30/v002",
    bands: Optional[List[str]] = None,
    cloud_coverage: float = 10.0,
    number_of_images: int = 3,
    buffer_distance: int = 5000,
    scale: int = 30,
    crs: str = 'EPSG:4326',
    max_expansion_days: int = 30
) -> List[str]:
    """
    End-to-end function to find and download images.

    Parameters:
    -----------
    date_range : Tuple[str, str]
        (start_date, end_date) in 'YYYY-MM-DD' format
    coordinates : Union[Tuple[float, float], List[List[float]], ee.Geometry]
        Either (lon, lat), polygon coordinates, or ee.Geometry
    output_directory : str
        Directory path to save downloaded images
    image_collection : str
        Earth Engine image collection ID (default: "NASA/HLS/HLSS30/v002")
    bands : Optional[List[str]]
        List of band names (default: ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12'])
    cloud_coverage : float
        Maximum cloud coverage percentage (default: 10.0)
    number_of_images : int
        Number of images to find (default: 3)
    buffer_distance : int
        Buffer distance for point coordinates in meters (default: 10000)
    scale : int
        Resolution in meters (default: 30)
    crs : str
        Coordinate reference system (default: 'EPSG:4326')
    max_expansion_days : int
        Maximum days to expand search window (default: 30)

    Returns:
    --------
    List[str] : List of file paths for successfully downloaded images
    """
    print("="*70)
    print("EARTH ENGINE IMAGE DOWNLOAD")
    print("="*70)
    print(f"Collection: {image_collection}")
    print(f"Date range: {date_range[0]} to {date_range[1]}")
    print(f"Bands: {bands or ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12']}")
    print(f"Max cloud coverage: {cloud_coverage}%")
    print(f"Number of images: {number_of_images}")
    print(f"Output directory: {output_directory}")
    print("="*70)
    print()

    # Find images
    print("STEP 1: Finding images...")
    print("-"*70)
    image_objects = find_multiple_images(
        date_range=date_range,
        coordinates=coordinates,
        image_collection=image_collection,
        bands=bands,
        cloud_coverage=cloud_coverage,
        number_of_images=number_of_images,
        buffer_distance=buffer_distance,
        max_expansion_days=max_expansion_days
    )

    if not image_objects:
        print("\n✗ No images found matching criteria")
        return []

    # Download images
    print("\nSTEP 2: Downloading images...")
    print("-"*70)
    downloaded_files = batch_download(
        image_objects=image_objects,
        output_directory=output_directory,
        scale=scale,
        crs=crs
    )

    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)

    return downloaded_files


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: Find a single image
    print("Example 1: Find a single image")
    result = find_single_image(
        date_range=('2024-04-25', '2024-04-26'),
        coordinates=(-109.53, 29.19),  # Point coordinates
        cloud_coverage=30
    )
    if result:
        print(f"Found image: {result['metadata']}")
    else:
        print("No image found")

    print("\n" + "="*70 + "\n")

    # Example 2: Find and download multiple images
    print("Example 2: Find and download multiple images")
    downloaded = download_images_end_to_end(
        date_range=('2024-04-01', '2024-04-30'),
        coordinates=(-109.53, 29.19),
        output_directory='./downloads',
        number_of_images=3,
        cloud_coverage=30
    )

    print(f"\nDownloaded files:")
    for filepath in downloaded:
        print(f"  - {filepath}")
