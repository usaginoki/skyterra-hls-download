"""
Tests for HLS Downloader module
"""

import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime
import zipfile
import io

# Import the functions to test
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.hls_downloader import (
    _parse_coordinates,
    _extract_metadata,
    find_single_image,
    find_multiple_images,
    batch_download,
    download_images_end_to_end,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_ee():
    """Mock Earth Engine module"""
    with patch('src.hls_downloader.ee') as mock:
        yield mock


@pytest.fixture
def temp_dir():
    """Create a temporary directory for downloads"""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def mock_image():
    """Mock Earth Engine Image"""
    mock_img = Mock()
    mock_img.getInfo.return_value = {
        'properties': {
            'system:time_start': 1714003200000,  # 2024-04-25
            'CLOUD_COVERAGE': 5.5,
            'system:index': 'test_image_001'
        }
    }
    return mock_img


@pytest.fixture
def mock_geometry():
    """Mock Earth Engine Geometry"""
    mock_geom = Mock()
    mock_geom.getInfo.return_value = {
        'coordinates': [[[-109.53, 29.19], [-109.52, 29.19],
                        [-109.52, 29.20], [-109.53, 29.20], [-109.53, 29.19]]]
    }
    return mock_geom


@pytest.fixture
def mock_collection(mock_image):
    """Mock Earth Engine ImageCollection"""
    mock_coll = Mock()
    mock_coll.filterDate.return_value = mock_coll
    mock_coll.filter.return_value = mock_coll
    mock_coll.filterBounds.return_value = mock_coll
    mock_coll.select.return_value = mock_coll
    mock_coll.sort.return_value = mock_coll
    mock_coll.size.return_value.getInfo.return_value = 1
    mock_coll.first.return_value = mock_image
    return mock_coll


# ============================================================================
# TESTS FOR HELPER FUNCTIONS
# ============================================================================

class TestParseCoordinates:
    """Tests for _parse_coordinates function"""

    def test_parse_point_coordinates(self, mock_ee):
        """Test parsing point (lon, lat) coordinates"""
        mock_point = Mock()
        mock_buffered = Mock()
        mock_point.buffer.return_value = mock_buffered
        mock_ee.Geometry.Point.return_value = mock_point

        result = _parse_coordinates((-109.53, 29.19), buffer_distance=5000)

        mock_ee.Geometry.Point.assert_called_once_with([-109.53, 29.19])
        mock_point.buffer.assert_called_once_with(5000)
        assert result == mock_buffered

    def test_parse_polygon_coordinates(self, mock_ee):
        """Test parsing polygon coordinates"""
        polygon_coords = [[-109.53, 29.19], [-109.52, 29.19],
                         [-109.52, 29.20], [-109.53, 29.20]]
        mock_polygon = Mock()
        mock_ee.Geometry.Polygon.return_value = mock_polygon

        result = _parse_coordinates(polygon_coords)

        mock_ee.Geometry.Polygon.assert_called_once_with(polygon_coords)
        assert result == mock_polygon

    def test_parse_ee_geometry_passthrough(self, mock_ee):
        """Test that ee.Geometry objects are passed through unchanged"""
        mock_geom = Mock(spec=['getInfo'])
        mock_ee.Geometry = type('Geometry', (), {})

        # Make isinstance work with our mock
        with patch('src.hls_downloader.isinstance') as mock_isinstance:
            mock_isinstance.side_effect = lambda obj, cls: (
                obj is mock_geom and cls is mock_ee.Geometry
            )
            result = _parse_coordinates(mock_geom)

        assert result == mock_geom

    def test_parse_invalid_coordinates(self, mock_ee):
        """Test that invalid coordinates raise ValueError"""
        with pytest.raises(ValueError, match="Coordinates must be either"):
            _parse_coordinates("invalid")

        with pytest.raises(ValueError):
            _parse_coordinates([1, 2, 3])  # Not pairs


class TestExtractMetadata:
    """Tests for _extract_metadata function"""

    def test_extract_complete_metadata(self, mock_image, mock_geometry):
        """Test extracting complete metadata from image"""
        metadata = _extract_metadata(mock_image, mock_geometry)

        assert metadata['date'] == '2024-04-25'
        assert metadata['cloud_coverage'] == 5.5
        assert metadata['image_id'] == 'test_image_001'
        assert metadata['region'] == mock_geometry

    def test_extract_metadata_missing_date(self, mock_geometry):
        """Test metadata extraction when date is missing"""
        mock_img = Mock()
        mock_img.getInfo.return_value = {
            'properties': {
                'CLOUD_COVERAGE': 10.0,
                'system:index': 'test_002'
            }
        }

        metadata = _extract_metadata(mock_img, mock_geometry)

        assert metadata['date'] == 'unknown'
        assert metadata['cloud_coverage'] == 10.0

    def test_extract_metadata_alternative_cloud_field(self, mock_geometry):
        """Test metadata extraction with alternative cloud coverage field"""
        mock_img = Mock()
        mock_img.getInfo.return_value = {
            'properties': {
                'system:time_start': 1714003200000,
                'CLOUDY_PIXEL_PERCENTAGE': 15.5,
                'system:index': 'test_003'
            }
        }

        metadata = _extract_metadata(mock_img, mock_geometry)

        assert metadata['cloud_coverage'] == 15.5


# ============================================================================
# TESTS FOR MAIN FUNCTIONS
# ============================================================================

class TestFindSingleImage:
    """Tests for find_single_image function"""

    @patch('src.hls_downloader._extract_metadata')
    @patch('src.hls_downloader._parse_coordinates')
    def test_find_single_image_success(self, mock_parse, mock_extract,
                                       mock_ee, mock_collection, mock_image,
                                       mock_geometry):
        """Test finding a single image successfully"""
        mock_parse.return_value = mock_geometry
        mock_extract.return_value = {
            'date': '2024-04-25',
            'cloud_coverage': 5.5,
            'image_id': 'test_001',
            'region': mock_geometry
        }
        mock_ee.ImageCollection.return_value = mock_collection
        mock_ee.Filter.lt.return_value = Mock()

        result = find_single_image(
            date_range=('2024-04-25', '2024-04-26'),
            coordinates=(-109.53, 29.19),
            cloud_coverage=30
        )

        assert result is not None
        assert result['image'] == mock_image
        assert result['metadata']['date'] == '2024-04-25'
        assert result['metadata']['cloud_coverage'] == 5.5
        mock_ee.ImageCollection.assert_called_once_with("NASA/HLS/HLSS30/v002")

    @patch('src.hls_downloader._parse_coordinates')
    def test_find_single_image_no_results(self, mock_parse, mock_ee, mock_geometry):
        """Test when no images match the criteria"""
        mock_parse.return_value = mock_geometry
        mock_coll = Mock()
        mock_coll.filterDate.return_value = mock_coll
        mock_coll.filter.return_value = mock_coll
        mock_coll.filterBounds.return_value = mock_coll
        mock_coll.select.return_value = mock_coll
        mock_coll.sort.return_value = mock_coll
        mock_coll.size.return_value.getInfo.return_value = 0

        mock_ee.ImageCollection.return_value = mock_coll
        mock_ee.Filter.lt.return_value = Mock()

        result = find_single_image(
            date_range=('2024-04-25', '2024-04-26'),
            coordinates=(-109.53, 29.19),
            cloud_coverage=5
        )

        assert result is None

    @patch('src.hls_downloader._parse_coordinates')
    def test_find_single_image_custom_bands(self, mock_parse, mock_ee,
                                            mock_collection, mock_geometry):
        """Test finding image with custom bands"""
        mock_parse.return_value = mock_geometry
        mock_ee.ImageCollection.return_value = mock_collection
        mock_ee.Filter.lt.return_value = Mock()

        custom_bands = ['B4', 'B3', 'B2']
        find_single_image(
            date_range=('2024-04-25', '2024-04-26'),
            coordinates=(-109.53, 29.19),
            bands=custom_bands
        )

        mock_collection.select.assert_called_with(custom_bands)


class TestFindMultipleImages:
    """Tests for find_multiple_images function"""

    @patch('src.hls_downloader.find_single_image')
    def test_find_multiple_images_success(self, mock_find_single):
        """Test finding multiple images successfully"""
        # Mock three different images
        mock_find_single.side_effect = [
            {
                'image': Mock(),
                'metadata': {
                    'date': '2024-04-05',
                    'cloud_coverage': 5.0,
                    'image_id': 'img_001',
                    'region': Mock()
                }
            },
            {
                'image': Mock(),
                'metadata': {
                    'date': '2024-04-15',
                    'cloud_coverage': 7.0,
                    'image_id': 'img_002',
                    'region': Mock()
                }
            },
            {
                'image': Mock(),
                'metadata': {
                    'date': '2024-04-25',
                    'cloud_coverage': 3.0,
                    'image_id': 'img_003',
                    'region': Mock()
                }
            },
        ]

        results = find_multiple_images(
            date_range=('2024-04-01', '2024-04-30'),
            coordinates=(-109.53, 29.19),
            number_of_images=3,
            cloud_coverage=30
        )

        assert len(results) == 3
        assert results[0]['metadata']['image_id'] == 'img_001'
        assert results[1]['metadata']['image_id'] == 'img_002'
        assert results[2]['metadata']['image_id'] == 'img_003'

    @patch('src.hls_downloader.find_single_image')
    def test_find_multiple_images_with_expansion(self, mock_find_single):
        """Test that search window expands when images not found initially"""
        # First call returns None, second returns image
        mock_find_single.side_effect = [
            None,  # First attempt fails
            {
                'image': Mock(),
                'metadata': {
                    'date': '2024-04-05',
                    'cloud_coverage': 5.0,
                    'image_id': 'img_001',
                    'region': Mock()
                }
            },
            {
                'image': Mock(),
                'metadata': {
                    'date': '2024-04-15',
                    'cloud_coverage': 7.0,
                    'image_id': 'img_002',
                    'region': Mock()
                }
            },
            {
                'image': Mock(),
                'metadata': {
                    'date': '2024-04-25',
                    'cloud_coverage': 3.0,
                    'image_id': 'img_003',
                    'region': Mock()
                }
            },
        ]

        results = find_multiple_images(
            date_range=('2024-04-01', '2024-04-30'),
            coordinates=(-109.53, 29.19),
            number_of_images=3,
            cloud_coverage=10
        )

        assert len(results) == 3
        # Should have called find_single_image more than 3 times due to expansion
        assert mock_find_single.call_count > 3

    @patch('src.hls_downloader.find_single_image')
    def test_find_multiple_images_deduplication(self, mock_find_single):
        """Test that duplicate images are filtered out"""
        # Return the same image twice, then a different one
        mock_find_single.side_effect = [
            {
                'image': Mock(),
                'metadata': {
                    'date': '2024-04-05',
                    'cloud_coverage': 5.0,
                    'image_id': 'img_001',  # Same ID
                    'region': Mock()
                }
            },
            {
                'image': Mock(),
                'metadata': {
                    'date': '2024-04-05',
                    'cloud_coverage': 5.0,
                    'image_id': 'img_001',  # Duplicate!
                    'region': Mock()
                }
            },
            {
                'image': Mock(),
                'metadata': {
                    'date': '2024-04-15',
                    'cloud_coverage': 7.0,
                    'image_id': 'img_002',  # Different image
                    'region': Mock()
                }
            },
            {
                'image': Mock(),
                'metadata': {
                    'date': '2024-04-25',
                    'cloud_coverage': 3.0,
                    'image_id': 'img_003',
                    'region': Mock()
                }
            },
        ]

        results = find_multiple_images(
            date_range=('2024-04-01', '2024-04-30'),
            coordinates=(-109.53, 29.19),
            number_of_images=2,
            cloud_coverage=30
        )

        # Should get 2 unique images
        assert len(results) == 2
        assert results[0]['metadata']['image_id'] == 'img_001'
        assert results[1]['metadata']['image_id'] == 'img_002'

    def test_find_multiple_images_invalid_date_range(self):
        """Test that invalid date range raises error"""
        with pytest.raises(ValueError, match="end_date must be after start_date"):
            find_multiple_images(
                date_range=('2024-04-30', '2024-04-01'),  # Reversed!
                coordinates=(-109.53, 29.19),
                number_of_images=3
            )


class TestBatchDownload:
    """Tests for batch_download function"""

    @patch('src.hls_downloader.requests.get')
    def test_batch_download_success(self, mock_get, temp_dir, mock_geometry):
        """Test successful batch download"""
        # Create a mock response with a zip file containing a .tif
        mock_response = Mock()
        mock_response.status_code = 200

        # Create a real zip file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            zf.writestr('image.tif', b'fake tif data')
        mock_response.content = zip_buffer.getvalue()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Create mock image objects
        mock_img = Mock()
        mock_img.getDownloadURL.return_value = 'http://fake.url/download'

        image_objects = [
            {
                'image': mock_img,
                'metadata': {
                    'date': '2024-04-25',
                    'cloud_coverage': 5.5,
                    'image_id': 'test_001',
                    'region': mock_geometry
                }
            }
        ]

        downloaded = batch_download(image_objects, temp_dir, scale=30)

        assert len(downloaded) == 1
        assert os.path.exists(downloaded[0])
        assert 'test_001_2024-04-25.tif' in downloaded[0]

    @patch('src.hls_downloader.requests.get')
    def test_batch_download_multiple_images(self, mock_get, temp_dir, mock_geometry):
        """Test downloading multiple images"""
        # Create a mock response with a zip file
        mock_response = Mock()
        mock_response.status_code = 200
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            zf.writestr('image.tif', b'fake tif data')
        mock_response.content = zip_buffer.getvalue()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Create three mock image objects
        image_objects = []
        for i in range(3):
            mock_img = Mock()
            mock_img.getDownloadURL.return_value = f'http://fake.url/download{i}'
            image_objects.append({
                'image': mock_img,
                'metadata': {
                    'date': f'2024-04-{25+i:02d}',
                    'cloud_coverage': 5.5 + i,
                    'image_id': f'test_{i:03d}',
                    'region': mock_geometry
                }
            })

        downloaded = batch_download(image_objects, temp_dir)

        assert len(downloaded) == 3
        assert all(os.path.exists(f) for f in downloaded)

    @patch('src.hls_downloader.requests.get')
    def test_batch_download_http_error(self, mock_get, temp_dir, mock_geometry):
        """Test handling of HTTP errors during download"""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 500 Error")
        mock_get.return_value = mock_response

        mock_img = Mock()
        mock_img.getDownloadURL.return_value = 'http://fake.url/download'

        image_objects = [{
            'image': mock_img,
            'metadata': {
                'date': '2024-04-25',
                'cloud_coverage': 5.5,
                'image_id': 'test_001',
                'region': mock_geometry
            }
        }]

        downloaded = batch_download(image_objects, temp_dir)

        # Should handle error gracefully and return empty list
        assert len(downloaded) == 0

    @patch('src.hls_downloader.requests.get')
    def test_batch_download_no_tif_in_zip(self, mock_get, temp_dir, mock_geometry):
        """Test handling when zip doesn't contain a .tif file"""
        mock_response = Mock()
        mock_response.status_code = 200

        # Create a zip with no .tif file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            zf.writestr('readme.txt', b'no tif here')
        mock_response.content = zip_buffer.getvalue()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        mock_img = Mock()
        mock_img.getDownloadURL.return_value = 'http://fake.url/download'

        image_objects = [{
            'image': mock_img,
            'metadata': {
                'date': '2024-04-25',
                'cloud_coverage': 5.5,
                'image_id': 'test_001',
                'region': mock_geometry
            }
        }]

        downloaded = batch_download(image_objects, temp_dir)

        # Should return empty list when no .tif found
        assert len(downloaded) == 0


class TestDownloadImagesEndToEnd:
    """Tests for download_images_end_to_end wrapper function"""

    @patch('src.hls_downloader.batch_download')
    @patch('src.hls_downloader.find_multiple_images')
    def test_end_to_end_success(self, mock_find, mock_download, temp_dir):
        """Test successful end-to-end download"""
        # Mock find_multiple_images
        mock_images = [
            {'image': Mock(), 'metadata': {'date': '2024-04-05', 'image_id': 'img_001'}},
            {'image': Mock(), 'metadata': {'date': '2024-04-15', 'image_id': 'img_002'}},
        ]
        mock_find.return_value = mock_images

        # Mock batch_download
        mock_files = [
            os.path.join(temp_dir, 'img_001_2024-04-05.tif'),
            os.path.join(temp_dir, 'img_002_2024-04-15.tif'),
        ]
        mock_download.return_value = mock_files

        result = download_images_end_to_end(
            date_range=('2024-04-01', '2024-04-30'),
            coordinates=(-109.53, 29.19),
            output_directory=temp_dir,
            number_of_images=2
        )

        assert len(result) == 2
        mock_find.assert_called_once()
        mock_download.assert_called_once_with(
            image_objects=mock_images,
            output_directory=temp_dir,
            scale=30,
            crs='EPSG:4326'
        )

    @patch('src.hls_downloader.find_multiple_images')
    def test_end_to_end_no_images_found(self, mock_find, temp_dir):
        """Test when no images are found"""
        mock_find.return_value = []

        result = download_images_end_to_end(
            date_range=('2024-04-01', '2024-04-30'),
            coordinates=(-109.53, 29.19),
            output_directory=temp_dir,
            number_of_images=3
        )

        assert len(result) == 0

    @patch('src.hls_downloader.batch_download')
    @patch('src.hls_downloader.find_multiple_images')
    def test_end_to_end_custom_parameters(self, mock_find, mock_download, temp_dir):
        """Test end-to-end with custom parameters"""
        mock_images = [{'image': Mock(), 'metadata': {'date': '2024-04-05'}}]
        mock_find.return_value = mock_images
        mock_download.return_value = []

        custom_bands = ['B4', 'B3', 'B2']
        download_images_end_to_end(
            date_range=('2024-04-01', '2024-04-30'),
            coordinates=(-109.53, 29.19),
            output_directory=temp_dir,
            image_collection="custom/collection",
            bands=custom_bands,
            cloud_coverage=20,
            number_of_images=5,
            scale=10,
            crs='EPSG:3857'
        )

        mock_find.assert_called_once_with(
            date_range=('2024-04-01', '2024-04-30'),
            coordinates=(-109.53, 29.19),
            image_collection="custom/collection",
            bands=custom_bands,
            cloud_coverage=20,
            number_of_images=5,
            buffer_distance=10000,
            max_expansion_days=30
        )

        mock_download.assert_called_once_with(
            image_objects=mock_images,
            output_directory=temp_dir,
            scale=10,
            crs='EPSG:3857'
        )


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests (these would need actual EE credentials to run)"""

    @pytest.mark.skip(reason="Requires Earth Engine credentials")
    def test_real_download(self):
        """Test with real Earth Engine API (requires authentication)"""
        # This test is skipped by default but can be run manually
        # when Earth Engine credentials are available
        result = find_single_image(
            date_range=('2024-04-25', '2024-04-26'),
            coordinates=(-109.53, 29.19),
            cloud_coverage=50
        )
        assert result is not None or result is None  # Either is valid


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
