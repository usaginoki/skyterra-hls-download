"""
Test complete workflow: Download, Merge, and Crop temporal images
Location: 51°04'50"N 70°29'56"E (near Karaganda, Kazakhstan)
Area: 1 km to each side (2km x 2km)
Period: April to September 2025
Process: Download → Merge → Crop
"""

from src.hls_downloader import download_images_end_to_end
from src.image_utils import merge_temporal_images, crop_image

# Convert coordinates from DMS to decimal degrees
# 51°04'50"N = 51 + 4/60 + 50/3600 = 51.0805556°N
# 70°29'56"E = 70 + 29/60 + 56/3600 = 70.4988889°E
longitude = 70 + 29/60 + 56/3600  # 70.4988889°E
latitude = 51 + 4/60 + 50/3600     # 51.0805556°N

print("=" * 70)
print("TESTING: Complete Workflow (Download → Merge → Crop)")
print("=" * 70)
print(f"Location: 51°04'50\"N 70°29'56\"E")
print(f"Decimal: {latitude:.6f}°N, {longitude:.6f}°E")
print(f"Area: 1 km to each side (2km × 2km)")
print(f"Period: April - September 2025")
print(f"Number of images: 3")
print("=" * 70)
print()

# Step 1: Download 3 temporal images
print("STEP 1: Downloading 3 temporal images...")
print("-" * 70)

images = download_images_end_to_end(
    date_range=('2025-04-01', '2025-09-30'),  # April to September 2025
    coordinates=(longitude, latitude),
    output_directory='./test_output',
    number_of_images=3,
    cloud_coverage=30,
    buffer_distance=1000  # 1km to each side
)

print(f"\nDownloaded {len(images)} images:")
for idx, img in enumerate(images, 1):
    import os
    size_mb = os.path.getsize(img) / (1024 * 1024)
    print(f"  {idx}. {os.path.basename(img)} ({size_mb:.2f} MB)")

# Step 2: Merge temporal images
if images:
    print("\n" + "=" * 70)
    print("STEP 2: Merging temporal images...")
    print("-" * 70)

    merged_file = merge_temporal_images(
        image_paths=images,
        output_path='./test_output/merged_temporal.tif'
    )

    # Verify the merged file
    import rasterio
    with rasterio.open(merged_file) as src:
        print(f"\nMerged file: {merged_file}")
        print(f"  Dimensions: {src.width} × {src.height} pixels")
        print(f"  Total bands: {src.count}")
        print(f"  Expected: {len(images) * 6} bands (3 images × 6 bands)")
        print(f"  File size: {os.path.getsize(merged_file) / (1024 * 1024):.2f} MB")
        print(f"\n  Band descriptions:")
        for idx in range(1, min(7, src.count + 1)):  # Show first 6 bands
            desc = src.descriptions[idx - 1] if src.descriptions[idx - 1] else f"Band {idx}"
            print(f"    Band {idx}: {desc}")
        if src.count > 6:
            print(f"    ...")
            for idx in range(src.count - 2, src.count + 1):
                desc = src.descriptions[idx - 1] if src.descriptions[idx - 1] else f"Band {idx}"
                print(f"    Band {idx}: {desc}")

    # Step 3: Crop the merged image to exact region of interest
    print("\n" + "=" * 70)
    print("STEP 3: Cropping merged image to ROI...")
    print("-" * 70)

    # Crop to exactly 1km to each side from center point
    cropped_file = crop_image(
        input_path=merged_file,
        coordinates=((longitude, latitude), 1000, 1000),  # Center point, 1km x and y offsets
        output_path='./test_output/final_cropped.tif',
        coordinate_type='point_offset'
    )

    # Verify the cropped file
    with rasterio.open(cropped_file) as src:
        print(f"\nCropped file: {cropped_file}")
        print(f"  Dimensions: {src.width} × {src.height} pixels")
        print(f"  Total bands: {src.count} (all bands preserved)")
        print(f"  File size: {os.path.getsize(cropped_file) / (1024 * 1024):.2f} MB")
        print(f"  Coverage: ~{(src.bounds.right - src.bounds.left) * 111:.1f}km × {(src.bounds.top - src.bounds.bottom) * 111:.1f}km")
        print(f"  Bounds:")
        print(f"    Center: {(src.bounds.left + src.bounds.right)/2:.6f}°E, {(src.bounds.bottom + src.bounds.top)/2:.6f}°N")

    print("\n" + "=" * 70)
    print("SUCCESS! Complete workflow finished.")
    print("=" * 70)
    print(f"\nFiles created:")
    print(f"  1. Individual images: {len(images)} files in ./test_output/")
    print(f"  2. Merged file: ./test_output/merged_temporal.tif ({os.path.getsize(merged_file) / (1024 * 1024):.2f} MB)")
    print(f"  3. Cropped file: ./test_output/final_cropped.tif ({os.path.getsize(cropped_file) / (1024 * 1024):.2f} MB)")
    print(f"\nWorkflow: Download → Merge → Crop ✓")
    print(f"Final file ready for analysis: {cropped_file}")
else:
    print("\n" + "=" * 70)
    print("WARNING: No images found for the specified criteria")
    print("=" * 70)
    print("\nTry adjusting:")
    print("  - Increase cloud_coverage threshold")
    print("  - Expand date range")
    print("  - Change location")
