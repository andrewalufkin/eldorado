#!/usr/bin/env python3
# download_gedi.py

import os
from src import data

# AOI around a known GEDI shot: lat -0.29, lon -68.65
AOI = [-68.66, -0.30, -68.64, -0.29]  # lon_min, lat_min, lon_max, lat_max

# Define output paths
OUTPUT_BASE_DIR = 'data_output'
GEDI_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, 'gedi')
GEDI_PARQUET_FILE = os.path.join(GEDI_OUTPUT_DIR, 'gedi_l2a_aoi_data.parquet')
TEMP_GEDI_DOWNLOAD_DIR = os.path.join(OUTPUT_BASE_DIR, 'temp_gedi_downloads')

def main():
    print(f"Starting GEDI data download and processing for AOI: {AOI}")
    print("Using a tiny AOI around a known GEDI shot location")
    
    # Create output directories
    os.makedirs(GEDI_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_GEDI_DOWNLOAD_DIR, exist_ok=True)
    
    # Download and process GEDI data
    print("\nDownloading and processing GEDI L2A data...")
    result = data.download_and_process_gedi_l2a(
        aoi=AOI,
        output_parquet_path=GEDI_PARQUET_FILE,
        temp_download_dir=TEMP_GEDI_DOWNLOAD_DIR
    )
    
    if result:
        print(f"\nSuccess! GEDI data saved to: {result}")
        print(f"Number of shots processed: {len(data.pd.read_parquet(result))}")
    else:
        print("\nFailed to download and process GEDI data.")

if __name__ == "__main__":
    main() 