"""
Tests for image processing utilities
"""

import pytest
import numpy as np
import tempfile
import shutil
import os
from unittest.mock import Mock, patch
import rasterio
from rasterio.transform import from_bounds

# Import functions to test
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.image_utils import merge_temporal_images, crop_image
from src.hls_downloader import _sort_band_files


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def create_test_geotiff():
    """Factory fixture to create test GeoTIFF files"""
    def _create(filepath, width=100, height=100, num_bands=6, band_descriptions=None):
        """Create a test GeoTIFF file with random data"""
        # Create random data
        data = np.random.rand(num_bands, height, width).astype(np.float32)

        # Define metadata
        transform = from_bounds(71.0, 51.0, 72.0, 52.0, width, height)
        meta = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': num_bands,
            'dtype': 'float32',
            'crs': 'EPSG:4326',
            'transform': transform
        }

        # Write GeoTIFF
        with rasterio.open(filepath, 'w', **meta) as dst:
            dst.write(data)
            # Set band descriptions
            if band_descriptions:
                for idx, desc in enumerate(band_descriptions, 1):
                    dst.set_band_description(idx, desc)

        return filepath

    return _create


# ============================================================================
# TESTS FOR BAND SORTING
# ============================================================================

class TestBandSorting:
    """Tests for _sort_band_files function"""

    def test_sort_standard_bands(self):
        """Test sorting of standard HLS bands"""
        files = [
            'image.B11.tif',
            'image.B2.tif',
            'image.B12.tif',
            'image.B3.tif',
            'image.B8A.tif',
            'image.B4.tif'
        ]

        sorted_files = _sort_band_files(files)

        expected = [
            'image.B2.tif',
            'image.B3.tif',
            'image.B4.tif',
            'image.B8A.tif',
            'image.B11.tif',
            'image.B12.tif'
        ]

        assert sorted_files == expected

    def test_sort_preserves_order_for_unknown_bands(self):
        """Test that unknown bands are placed at the end"""
        files = [
            'image.B2.tif',
            'image.BQA.tif',
            'image.B4.tif',
            'image.BOTHER.tif'
        ]

        sorted_files = _sort_band_files(files)

        # Known bands should come first, in order
        assert sorted_files[:2] == ['image.B2.tif', 'image.B4.tif']
        # Unknown bands at the end
        assert set(sorted_files[2:]) == {'image.BQA.tif', 'image.BOTHER.tif'}

    def test_sort_empty_list(self):
        """Test sorting empty list"""
        assert _sort_band_files([]) == []


# ============================================================================
# TESTS FOR TEMPORAL MERGE
# ============================================================================

class TestMergeTemporalImages:
    """Tests for merge_temporal_images function"""

    def test_merge_two_images(self, temp_dir, create_test_geotiff):
        """Test merging two temporal images"""
        # Create two test images
        img1 = os.path.join(temp_dir, 'image_2024-01-01.tif')
        img2 = os.path.join(temp_dir, 'image_2024-02-01.tif')

        band_desc = ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12']
        create_test_geotiff(img1, num_bands=6, band_descriptions=band_desc)
        create_test_geotiff(img2, num_bands=6, band_descriptions=band_desc)

        # Merge images
        output = os.path.join(temp_dir, 'merged.tif')
        result = merge_temporal_images([img1, img2], output)

        assert result == output
        assert os.path.exists(output)

        # Verify merged image
        with rasterio.open(output) as src:
            assert src.count == 12  # 6 bands × 2 images
            assert src.width == 100
            assert src.height == 100

            # Check band descriptions
            descriptions = src.descriptions
            assert descriptions[0] == 't0_B2'
            assert descriptions[5] == 't0_B12'
            assert descriptions[6] == 't1_B2'
            assert descriptions[11] == 't1_B12'

    def test_merge_three_images(self, temp_dir, create_test_geotiff):
        """Test merging three temporal images"""
        images = []
        for i in range(3):
            img_path = os.path.join(temp_dir, f'image_t{i}_2024-0{i+1}-01.tif')
            create_test_geotiff(img_path, num_bands=4, band_descriptions=['B2', 'B3', 'B4', 'B8A'])
            images.append(img_path)

        output = os.path.join(temp_dir, 'merged.tif')
        result = merge_temporal_images(images, output)

        assert os.path.exists(output)

        with rasterio.open(output) as src:
            assert src.count == 12  # 4 bands × 3 images
            assert src.descriptions[0] == 't0_B2'
            assert src.descriptions[4] == 't1_B2'
            assert src.descriptions[8] == 't2_B2'

    def test_merge_return_array_without_saving(self, temp_dir, create_test_geotiff):
        """Test merging without saving to file"""
        img1 = os.path.join(temp_dir, 'image1.tif')
        img2 = os.path.join(temp_dir, 'image2.tif')

        create_test_geotiff(img1, num_bands=3)
        create_test_geotiff(img2, num_bands=3)

        # Merge without output path
        result = merge_temporal_images([img1, img2], output_path=None)

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 6  # 3 bands × 2 images
        assert result.shape[1] == 100  # height
        assert result.shape[2] == 100  # width

    def test_merge_empty_list_raises_error(self):
        """Test that empty image list raises ValueError"""
        with pytest.raises(ValueError, match="image_paths cannot be empty"):
            merge_temporal_images([], 'output.tif')

    def test_merge_nonexistent_file_raises_error(self):
        """Test that nonexistent file raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            merge_temporal_images(['nonexistent.tif'], 'output.tif')

    def test_merge_mismatched_dimensions_raises_error(self, temp_dir, create_test_geotiff):
        """Test that mismatched dimensions raise ValueError"""
        img1 = os.path.join(temp_dir, 'image1.tif')
        img2 = os.path.join(temp_dir, 'image2.tif')

        create_test_geotiff(img1, width=100, height=100)
        create_test_geotiff(img2, width=200, height=200)  # Different size!

        output = os.path.join(temp_dir, 'merged.tif')

        with pytest.raises(ValueError, match="mismatched dimensions"):
            merge_temporal_images([img1, img2], output)

    def test_merge_preserves_metadata(self, temp_dir, create_test_geotiff):
        """Test that merge preserves CRS and transform from first image"""
        img1 = os.path.join(temp_dir, 'image1.tif')
        img2 = os.path.join(temp_dir, 'image2.tif')

        create_test_geotiff(img1, num_bands=2)
        create_test_geotiff(img2, num_bands=2)

        output = os.path.join(temp_dir, 'merged.tif')
        merge_temporal_images([img1, img2], output)

        # Check that CRS and bounds match first image
        with rasterio.open(img1) as src1:
            with rasterio.open(output) as src_merged:
                assert src1.crs == src_merged.crs
                assert src1.bounds == src_merged.bounds
                assert src1.transform == src_merged.transform


# ============================================================================
# TESTS FOR CROP IMAGE
# ============================================================================

class TestCropImage:
    """Tests for crop_image function"""

    def test_crop_with_bounding_box(self, temp_dir, create_test_geotiff):
        """Test cropping with bounding box coordinates"""
        input_img = os.path.join(temp_dir, 'input.tif')
        create_test_geotiff(input_img, width=200, height=200, num_bands=6)

        output_img = os.path.join(temp_dir, 'cropped.tif')

        # Crop a smaller region
        bbox = [71.2, 51.2, 71.8, 51.8]
        result = crop_image(input_img, bbox, output_img, coordinate_type='bbox')

        assert result == output_img
        assert os.path.exists(output_img)

        # Verify cropped image
        with rasterio.open(output_img) as src:
            assert src.count == 6  # All bands preserved
            # Cropped dimensions should be smaller
            assert src.width < 200
            assert src.height < 200

    def test_crop_with_point_offset(self, temp_dir, create_test_geotiff):
        """Test cropping with point and offset coordinates"""
        input_img = os.path.join(temp_dir, 'input.tif')
        create_test_geotiff(input_img, width=200, height=200, num_bands=4)

        output_img = os.path.join(temp_dir, 'cropped.tif')

        # Crop around a point with 5km offset
        coords = ((71.5, 51.5), 5000, 5000)
        result = crop_image(input_img, coords, output_img, coordinate_type='point_offset')

        assert os.path.exists(output_img)

        with rasterio.open(output_img) as src:
            assert src.count == 4
            # Should have created a cropped region
            assert src.width > 0
            assert src.height > 0

    def test_crop_return_array_without_saving(self, temp_dir, create_test_geotiff):
        """Test cropping without saving to file"""
        input_img = os.path.join(temp_dir, 'input.tif')
        create_test_geotiff(input_img, width=200, height=200, num_bands=3)

        bbox = [71.2, 51.2, 71.8, 51.8]
        result = crop_image(input_img, bbox, output_path=None, coordinate_type='bbox')

        assert isinstance(result, tuple)
        arr, transform, crs = result

        assert isinstance(arr, np.ndarray)
        assert arr.shape[0] == 3  # 3 bands
        assert arr.shape[1] > 0  # height
        assert arr.shape[2] > 0  # width

    def test_crop_preserves_band_descriptions(self, temp_dir, create_test_geotiff):
        """Test that cropping preserves band descriptions"""
        input_img = os.path.join(temp_dir, 'input.tif')
        band_desc = ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12']
        create_test_geotiff(input_img, width=200, height=200, num_bands=6, band_descriptions=band_desc)

        output_img = os.path.join(temp_dir, 'cropped.tif')
        bbox = [71.2, 51.2, 71.8, 51.8]
        crop_image(input_img, bbox, output_img, coordinate_type='bbox')

        # Check band descriptions are preserved
        with rasterio.open(output_img) as src:
            descriptions = src.descriptions
            assert descriptions[0] == 'B2'
            assert descriptions[5] == 'B12'

    def test_crop_multitemporal_image(self, temp_dir, create_test_geotiff):
        """Test cropping a multi-temporal merged image"""
        # Create a multi-temporal image (12 bands = 2 time steps × 6 bands)
        input_img = os.path.join(temp_dir, 'multitemporal.tif')
        band_desc = [f't{i//6}_{b}' for i in range(12) for b in ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12'][:1]]
        create_test_geotiff(input_img, width=200, height=200, num_bands=12, band_descriptions=band_desc[:12])

        output_img = os.path.join(temp_dir, 'cropped.tif')
        bbox = [71.2, 51.2, 71.8, 51.8]
        crop_image(input_img, bbox, output_img, coordinate_type='bbox')

        # Verify all 12 bands are preserved
        with rasterio.open(output_img) as src:
            assert src.count == 12

    def test_crop_invalid_bbox_raises_error(self, temp_dir, create_test_geotiff):
        """Test that invalid bounding box raises ValueError"""
        input_img = os.path.join(temp_dir, 'input.tif')
        create_test_geotiff(input_img)

        with pytest.raises(ValueError, match="Bounding box coordinates must be"):
            crop_image(input_img, [71.0, 51.0], coordinate_type='bbox')

    def test_crop_invalid_point_offset_raises_error(self, temp_dir, create_test_geotiff):
        """Test that invalid point offset raises ValueError"""
        input_img = os.path.join(temp_dir, 'input.tif')
        create_test_geotiff(input_img)

        with pytest.raises(ValueError, match="Point offset coordinates must be"):
            crop_image(input_img, ((71.0, 51.0),), coordinate_type='point_offset')

    def test_crop_unknown_coordinate_type_raises_error(self, temp_dir, create_test_geotiff):
        """Test that unknown coordinate type raises ValueError"""
        input_img = os.path.join(temp_dir, 'input.tif')
        create_test_geotiff(input_img)

        with pytest.raises(ValueError, match="Unknown coordinate_type"):
            crop_image(input_img, [71.0, 51.0, 72.0, 52.0], coordinate_type='invalid')

    def test_crop_nonexistent_file_raises_error(self):
        """Test that nonexistent file raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            crop_image('nonexistent.tif', [71.0, 51.0, 72.0, 52.0])


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for combined workflows"""

    def test_merge_then_crop_workflow(self, temp_dir, create_test_geotiff):
        """Test complete workflow: merge temporal images then crop"""
        # Create 3 temporal images
        images = []
        for i in range(3):
            img_path = os.path.join(temp_dir, f'image_{i}_2024-0{i+1}-01.tif')
            create_test_geotiff(img_path, width=200, height=200, num_bands=6,
                              band_descriptions=['B2', 'B3', 'B4', 'B8A', 'B11', 'B12'])
            images.append(img_path)

        # Step 1: Merge temporal images
        merged_path = os.path.join(temp_dir, 'merged.tif')
        merge_temporal_images(images, merged_path)

        # Verify merged
        with rasterio.open(merged_path) as src:
            assert src.count == 18  # 6 bands × 3 images

        # Step 2: Crop the merged image
        cropped_path = os.path.join(temp_dir, 'cropped.tif')
        bbox = [71.2, 51.2, 71.8, 51.8]
        crop_image(merged_path, bbox, cropped_path, coordinate_type='bbox')

        # Verify cropped
        with rasterio.open(cropped_path) as src:
            assert src.count == 18  # All bands preserved
            assert src.width < 200  # Cropped
            assert src.height < 200  # Cropped

            # Check band descriptions are preserved
            assert 't0_B2' in src.descriptions[0]
            assert 't1_B2' in src.descriptions[6]
            assert 't2_B2' in src.descriptions[12]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
