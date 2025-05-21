#!/usr/bin/env python3
# download_sentinel.py
import argparse
import os
from src import data
from src import features

# Parse bbox argument
parser = argparse.ArgumentParser(description='Download Sentinel-2 data and calculate soil index for a given AOI.')
parser.add_argument('--bbox', type=float, nargs=4, required=True, help='Bounding box coordinates: lon_min lat_min lon_max lat_max')
args = parser.parse_args()
AOI = args.bbox

# Define output paths
OUTPUT_BASE_DIR = 'data_output'
SENTINEL2_BANDS_DIR = os.path.join(OUTPUT_BASE_DIR, 'sentinel2_bands')
FEATURE_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, 'features')
bbox_str = '_'.join([f"{coord:.2f}" for coord in AOI])
SOIL_INDEX_RASTER_PATH = os.path.join(FEATURE_OUTPUT_DIR, f"soil_index_bbox_{bbox_str}.tif")

# Define Sentinel-2 bands needed for soil index
# Using the correct band names from the STAC catalog
SENTINEL2_BANDS_NEEDED = ['red', 'nir']  # Red (B4) and NIR (B8)

# Define a reasonable date range for Sentinel-2 imagery to reduce cloud cover
SENTINEL2_DATE_RANGE = ("2023-06-01", "2023-08-31")  # Dry season for Amazon

def main():
    print("Starting Sentinel-2 data download and soil index calculation...")
    
    # Create output directories
    os.makedirs(SENTINEL2_BANDS_DIR, exist_ok=True)
    os.makedirs(FEATURE_OUTPUT_DIR, exist_ok=True)
    
    # Check if bbox-specific soil index raster exists
    if os.path.exists(SOIL_INDEX_RASTER_PATH):
        print(f"Soil index raster for bbox already exists: {SOIL_INDEX_RASTER_PATH}")
        print("Reusing existing data.")
        return
    
    # Step 1: Download Sentinel-2 bands
    print("\nDownloading Sentinel-2 bands...")
    downloaded_bands = data.download_sentinel2_bands(
        aoi=AOI,
        bands=SENTINEL2_BANDS_NEEDED,
        output_dir=SENTINEL2_BANDS_DIR,
        date_range=SENTINEL2_DATE_RANGE
    )
    
    if not downloaded_bands:
        print("Failed to download Sentinel-2 bands. Exiting.")
        return
    
    print("\nSuccessfully downloaded bands:")
    for band, path in downloaded_bands.items():
        print(f"  {band}: {path}")
    
    # Step 2: Calculate soil index
    print("\nCalculating soil index...")
    soil_index_path = features.calculate_soil_index(
        band04_path=downloaded_bands['red'],  # Using 'red' instead of 'B04'
        band08_path=downloaded_bands['nir'],  # Using 'nir' instead of 'B08'
        output_raster_path=SOIL_INDEX_RASTER_PATH
    )
    
    if not soil_index_path:
        print("Failed to calculate soil index. Exiting.")
        return
    
    print(f"\nSoil index calculated and saved to: {soil_index_path}")
    print("\nYou can now run find_sites.py to create the anomaly raster.")

if __name__ == "__main__":
    main() 