# src/data.py
import earthaccess
import h5py
import pandas as pd
import os
import pystac_client
import rasterio
from rasterio.session import AWSSession
import boto3 # For AWS session if needed for requester pays or specific regions
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
# from shapely.geometry import box # If you want to define AOI as a polygon

# Define your AOI (should be consistent with find_sites.py)
# AOI = [lon_min, lat_min, lon_max, lat_max]
# Example: AOI = [-63.5, -8.2, -62.9, -7.6]

# Define the output directory for downloaded data and parquet file
# Suggestion: create a 'data' directory in your project root,
# or a temporary directory.
# For checkpoint, you might want to store the parquet file in a
# predictable location referenced by your manifest.
# e.g., os.makedirs('data/gedi', exist_ok=True)
# GEDI_OUTPUT_DIR = 'data/gedi'
# GEDI_PARQUET_FILE = os.path.join(GEDI_OUTPUT_DIR, 'gedi_l2a_data.parquet')

SENTINEL2_L2A_AWS_CATALOG = "https://earth-search.aws.element84.com/v1"
# Or use the newer planetary computer STAC endpoint if preferred, though element84 is common for S2 on AWS
# SENTINEL2_PC_CATALOG = "https://planetarycomputer.microsoft.com/api/stac/v1"

def get_bbox_filename(bbox, base_dir):
    """
    Generate a unique filename based on bbox coordinates.
    Format: gedi_l2a_bbox_{lon_min}_{lat_min}_{lon_max}_{lat_max}.parquet
    """
    # Round coordinates to 2 decimal places for cleaner filenames
    bbox_str = '_'.join([f"{coord:.2f}" for coord in bbox])
    return os.path.join(base_dir, f"gedi_l2a_bbox_{bbox_str}.parquet")

def download_and_process_gedi_l2a(aoi, output_parquet_path, temp_download_dir='temp_gedi_data', chunk_size=10):
    """
    Downloads GEDI L2A data for a given AOI, processes it in chunks, and saves relevant fields to a Parquet file.
    
    Args:
        aoi (list): [lon_min, lat_min, lon_max, lat_max]
        output_parquet_path (str): Path to save the final parquet file
        temp_download_dir (str): Directory for temporary downloads
        chunk_size (int): Number of granules to process at once
    """
    print("\n=== GEDI Data Processing Started ===")
    print(f"AOI: {aoi}")
    print(f"Output path: {output_parquet_path}")
    print(f"Temporary directory: {temp_download_dir}")
    print("===================================\n")

    # Check if we already have data for this exact bbox
    bbox_specific_path = get_bbox_filename(aoi, os.path.dirname(output_parquet_path))
    if os.path.exists(bbox_specific_path):
        print(f"Found existing GEDI data for this bbox at: {bbox_specific_path}")
        print("Using existing data instead of downloading.")
        return bbox_specific_path

    os.makedirs(temp_download_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)

    print("Authenticating with NASA Earthdata...")
    try:
        auth = earthaccess.login(strategy="netrc")
        if not auth.authenticated:
            print("Authentication failed. Please ensure your .netrc file is configured or try another auth strategy.")
            return None
    except Exception as e:
        print(f"Earthdata login failed: {e}")
        print("Please ensure you have a NASA Earthdata account and .netrc file configured.")
        print("See: https://www.nasa.gov/earthdata/s3-access")
        return None

    print(f"Searching for GEDI L2A granules in AOI: {aoi}...")
    try:
        granules = earthaccess.search_data(
            short_name="GEDI02_A",
            version="002",
            bounding_box=(aoi[0], aoi[1], aoi[2], aoi[3]),
            count=-1
        )
    except Exception as e:
        print(f"Error searching for GEDI granules: {e}")
        return None

    if not granules:
        print("No GEDI L2A granules found for the specified AOI.")
        return None

    print(f"Found {len(granules)} GEDI L2A granules. Processing in chunks of {chunk_size}...")
    
    # Process granules in chunks
    all_shot_data = []
    for i in range(0, len(granules), chunk_size):
        chunk = granules[i:i + chunk_size]
        print(f"\nProcessing chunk {i//chunk_size + 1} of {(len(granules) + chunk_size - 1)//chunk_size}...")
        
        # Download chunk
        chunk_download_paths = []
        for granule in chunk:
            try:
                file_path = earthaccess.download(granule, local_path=temp_download_dir)
                if isinstance(file_path, list):
                    chunk_download_paths.extend(file_path)
                elif file_path:
                    chunk_download_paths.append(file_path)
            except Exception as e:
                print(f"Could not download {granule.get('meta', {}).get('native-id', 'unknown granule')}: {e}")
        
        if not chunk_download_paths:
            print("No files were successfully downloaded in this chunk.")
            continue
            
        # Process downloaded files
        processed_files = [p for p in chunk_download_paths if p and os.path.exists(p) and p.endswith(('.h5', '.HDF5'))]
        
        for hdf_file_path in processed_files:
            print(f"Processing {hdf_file_path}...")
            try:
                with h5py.File(hdf_file_path, 'r') as hdf_file:
                    for beam_name in [name for name in hdf_file if name.startswith('BEAM')]:
                        beam = hdf_file[beam_name]
                        
                        # Check if geolocation group exists
                        if 'geolocation' not in beam:
                            continue
                            
                        # Find the algorithm-specific dataset paths
                        lat_paths = [k for k in beam['geolocation'].keys() if k.startswith('lat_lowestmode')]
                        lon_paths = [k for k in beam['geolocation'].keys() if k.startswith('lon_lowestmode')]
                        
                        if not lat_paths or not lon_paths:
                            continue
                            
                        # Use the first available algorithm setting
                        lat_path = f"geolocation/{lat_paths[0]}"
                        lon_path = f"geolocation/{lon_paths[0]}"
                        
                        # Check for RH and quality flag datasets
                        if 'rh' not in beam:
                            continue
                            
                        quality_paths = [k for k in beam.keys() if k.startswith('quality_flag')]
                        if not quality_paths:
                            continue
                        quality_path = quality_paths[0]
                        
                        # Read the datasets
                        try:
                            lats = beam[lat_path][:]
                            lons = beam[lon_path][:]
                            rh_metrics = beam['rh'][:]
                            quality_flags = beam[quality_path][:]
                            
                            for i in range(len(lats)):
                                lat, lon = lats[i], lons[i]
                                qf = quality_flags[i]
                                
                                # Filter by quality and AOI
                                if qf >= 1 and (aoi[0] <= lon <= aoi[2]) and (aoi[1] <= lat <= aoi[3]):
                                    if rh_metrics.shape[1] > 95:
                                        rh95 = rh_metrics[i][95]
                                        all_shot_data.append({
                                            'latitude': lat,
                                            'longitude': lon,
                                            'rh95': rh95,
                                            'quality_flag': qf
                                        })
                                        
                        except Exception as e:
                            print(f"  Error reading datasets from {beam_name}: {e}")
                            continue
                            
            except Exception as e:
                print(f"Error processing file {hdf_file_path}: {e}")
                continue
                
        # Clean up chunk files
        print("Cleaning up chunk files...")
        for p in processed_files:
            try:
                os.remove(p)
                print(f"  Removed: {p}")
            except OSError as e:
                print(f"  Error deleting temporary file {p}: {e}")
                
        # Save intermediate results to avoid losing data
        if all_shot_data:
            temp_parquet = os.path.join(temp_download_dir, f"temp_gedi_data_chunk_{i//chunk_size}.parquet")
            pd.DataFrame(all_shot_data).to_parquet(temp_parquet)
            print(f"Saved intermediate results to {temp_parquet}")

    # Combine all chunks and save final result
    if all_shot_data:
        print(f"\nConverting {len(all_shot_data)} shots to DataFrame...")
        gedi_df = pd.DataFrame(all_shot_data)
        
        # Save to bbox-specific path
        print(f"Saving GEDI data to {bbox_specific_path}...")
        gedi_df.to_parquet(bbox_specific_path)
        print("GEDI data processing complete.")
        
        # Clean up temporary parquet files
        for f in os.listdir(temp_download_dir):
            if f.startswith("temp_gedi_data_chunk_") and f.endswith(".parquet"):
                try:
                    os.remove(os.path.join(temp_download_dir, f))
                except OSError as e:
                    print(f"Error deleting temporary parquet file {f}: {e}")
        
        # Remove temp directory if empty
        try:
            if os.path.exists(temp_download_dir) and not os.listdir(temp_download_dir):
                os.rmdir(temp_download_dir)
                print(f"Removed empty directory: {temp_download_dir}")
        except OSError as e:
            print(f"Error deleting temporary directory {temp_download_dir}: {e}")
            
        return bbox_specific_path
    else:
        print("No valid GEDI shots found after processing and filtering.")
        return None

# Sentinel-2 data download functionality
def download_sentinel2_bands(aoi, bands, output_dir, date_range=("2023-01-01", "2023-12-31")):
    """
    Searches for Sentinel-2 L2A data for a given AOI and date range,
    and downloads specified bands as COGs (or parts of them).

    Args:
        aoi (list): [lon_min, lat_min, lon_max, lat_max]
        bands (list): List of band names, e.g., ['B04', 'B08'] for Red and NIR.
        output_dir (str): Directory to save downloaded band files.
        date_range (tuple): (start_date, end_date) in 'YYYY-MM-DD' format.

    Returns:
        dict: A dictionary where keys are band names and values are paths to downloaded band files.
              Returns None on failure.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Searching Sentinel-2 L2A data for AOI: {aoi} and date range: {date_range}")

    try:
        # Using earth-search/element84 STAC catalog for Sentinel-2 on AWS
        catalog = pystac_client.Client.open(SENTINEL2_L2A_AWS_CATALOG)

        # Define search parameters
        # Note: pystac_client expects bbox in [lon_min, lat_min, lon_max, lat_max]
        # and intersects expects GeoJSON-like geometry or a WKT string.
        # For simplicity with bbox, ensure it matches what STAC API expects.
        # Some STAC APIs use [minx, miny, maxx, maxy]
        search_bbox = [aoi[0], aoi[1], aoi[2], aoi[3]]
        
        # Construct a datetime string for the STAC API
        # Example: "2023-01-01T00:00:00Z/2023-12-31T23:59:59Z"
        search_datetime = f"{date_range[0]}T00:00:00Z/{date_range[1]}T23:59:59Z"

        search = catalog.search(
            collections=["sentinel-2-l2a"], # Sentinel-2 L2A collection
            bbox=search_bbox,
            datetime=search_datetime,
            # Add cloud cover filter if desired, e.g., query={"eo:cloud_cover": {"lt": 20}}
            # For this small AOI, it might be less critical, but good for larger areas.
            # Using 'query' for specific properties if 'filter' is not available or for complex queries.
            # Check pystac_client documentation for the best way to filter.
            # Alternatively, filter results after getting them.
            max_items=10 # Get a few items and pick the best one (e.g., least cloudy)
        )
        
        items = list(search.items()) # Convert iterator to list
        if not items:
            print("No Sentinel-2 L2A items found for the specified AOI and date range.")
            return None
        
        print(f"Found {len(items)} Sentinel-2 items. Attempting to use the first one.")
        # For simplicity, using the first item.
        # In a more robust pipeline, you'd sort by cloud cover or date.
        # Example: items.sort(key=lambda item: item.properties.get('eo:cloud_cover', 101))
        # item_to_use = items[0]

        # A more robust way: find least cloudy scene
        items.sort(key=lambda item: item.properties.get('eo:cloud_cover', 101) if item.properties else 101)
        item_to_use = items[0]
        cloud_cover = item_to_use.properties.get('eo:cloud_cover', 'N/A')
        print(f"Selected item {item_to_use.id} with cloud cover: {cloud_cover}%")


    except Exception as e:
        print(f"Error searching Sentinel-2 STAC catalog: {e}")
        return None

    downloaded_band_paths = {}
    # AWS S3 session for rasterio, may not always be needed if files are public,
    # but good practice for COG access on AWS.
    # aws_session = boto3.Session() # Configure with your AWS profile if needed
    # rio_env = rasterio.Env(aws_session=aws_session) # Deprecated way
    # Instead, use AWSSession directly if accessing requester-pays buckets or needing specific AWS settings.
    # For public Sentinel-2 COGs on AWS, often direct access via HTTPS URLs works.
    
    # The assets dictionary in a STAC item contains links to the actual data files (bands)
    # Common band names in Sentinel-2 STAC items are 'B04' (Red), 'B08' (NIR)
    for band_name in bands:
        if band_name in item_to_use.assets:
            asset_href = item_to_use.assets[band_name].href
            output_filename = os.path.join(output_dir, f"{item_to_use.id}_{band_name}.tif")
            print(f"Downloading (or accessing part of) {band_name} from {asset_href} to {output_filename}...")
            try:
                with rasterio.Env(AWS_NO_SIGN_REQUEST='YES'):
                    with rasterio.open(asset_href) as src:
                        print(f"Source CRS: {src.crs}")
                        print(f"Source Transform: {src.transform}")
                        # Transform AOI to raster's CRS
                        transformed_aoi_bounds = transform_bounds(
                            'EPSG:4326',  # AOI is WGS84
                            src.crs,
                            *aoi
                        )
                        print(f"Transformed AOI bounds: {transformed_aoi_bounds}")
                        window = from_bounds(*transformed_aoi_bounds, transform=src.transform)
                        print(f"Calculated window: col_off={window.col_off}, row_off={window.row_off}, width={window.width}, height={window.height}")
                        if window.width == 0 or window.height == 0:
                            print(f"WARNING: Calculated window for {band_name} is 0x0 pixels. Skipping this band.")
                            continue
                        data = src.read(1, window=window)
                        out_transform = rasterio.windows.transform(window, src.transform)
                        out_profile = src.profile.copy()
                        out_profile.update({
                            'height': data.shape[0],
                            'width': data.shape[1],
                            'transform': out_transform,
                            'compress': 'lzw'
                        })
                        with rasterio.open(output_filename, 'w', **out_profile) as dst:
                            dst.write(data, 1)
                downloaded_band_paths[band_name] = output_filename
                print(f"Successfully saved {band_name} for AOI to {output_filename}")
            except Exception as e:
                print(f"Error processing band {band_name} from {asset_href}: {e}")
        else:
            print(f"Band {band_name} not found in assets for item {item_to_use.id}")

    if len(downloaded_band_paths) == len(bands):
        return downloaded_band_paths
    else:
        print("Failed to retrieve all requested Sentinel-2 bands.")
        return None

# Example of how you might call this from find_sites.py:
# if __name__ == '__main__':
#     # This is for testing src/data.py directly
#     AOI_TEST = [-63.5, -8.2, -62.9, -7.6] # Defined in find_sites.py
#     OUTPUT_DIR = '../data/gedi' # Adjust path as needed relative to src/data.py
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     GEDI_PARQUET_FILE_TEST = os.path.join(OUTPUT_DIR, 'gedi_l2a_test_data.parquet')
#     TEMP_DOWNLOAD_DIR_TEST = '../temp_gedi_data' # Adjust path
    
#     print(f"Starting GEDI download and processing for AOI: {AOI_TEST}")
#     result_path = download_and_process_gedi_l2a(AOI_TEST, GEDI_PARQUET_FILE_TEST, TEMP_DOWNLOAD_DIR_TEST)
#     if result_path:
#         print(f"GEDI data successfully saved to: {result_path}")
#         # Load and print some info to verify
#         df = pd.read_parquet(result_path)
#         print(f"DataFrame head:\n{df.head()}")
#         print(f"DataFrame shape: {df.shape}")
#     else:
#         print("GEDI data processing failed.")