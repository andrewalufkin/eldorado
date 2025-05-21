#!/usr/bin/env python3
# openai_to_z/find_sites.py
import argparse
import os
import rasterio
from src import data
from src import features
from src import utils
import numpy as np
import json
import subprocess
import sys
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors

# Step 1: Hard-code one small AOI
AOI = [-68.66, -0.30, -68.64, -0.29]  # lon_min, lat_min, lon_max, lat_max (using the same AOI as download_gedi.py)
# Note: this is very small so there won't be many anomalies. GEDI data is dense so for this script I'm running 
# on my local machine I'm only using a tiny AOI. Obviously the ultimate area will be much larger.

# Define parameters for clustering
TOP_N_ABSOLUTE_PIXELS = 2000  # Number of top pixels to consider for clustering
DBSCAN_EPS_METERS = 100  # DBSCAN epsilon in meters
DBSCAN_MIN_SAMPLES = 10  # Minimum samples for DBSCAN
NUM_LARGEST_CLUSTERS = 5  # Number of largest clusters to find
TARGET_PROJECTED_CRS = "EPSG:32720"  # UTM Zone 20S for the AOI
OUTPUT_CENTROIDS_CRS = "EPSG:4326"  # For WKT in manifest

# Define resolution for the target 30m grid (in degrees for EPSG:4326)
TARGET_RESOLUTION_DEGREES = 0.00027  # Approximate 30m in degrees for EPSG:4326

def parse_args():
    parser = argparse.ArgumentParser(description='Find potential archaeological sites using GEDI and Sentinel-2 data.')
    parser.add_argument('--bbox', type=float, nargs=4, required=True,
                      help='Bounding box coordinates in format: lon_min lat_min lon_max lat_max')
    parser.add_argument('--output-dir', type=str, default='data_output',
                      help='Base output directory (default: data_output)')
    parser.add_argument('--skip-downloads', action='store_true',
                      help='Skip downloading GEDI and Sentinel-2 data if they already exist')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility (default: 42)')
    return parser.parse_args()

def run_download_script(script_name, aoi):
    """Run a download script with the given AOI."""
    print(f"\nRunning {script_name}...")
    try:
        # Create a temporary script that sets the AOI
        temp_script = f"temp_{script_name}"
        with open(script_name, 'r') as f:
            script_content = f.read()
        
        # Replace the AOI definition (for backward compatibility)
        new_aoi_line = f"AOI = {aoi}  # lon_min, lat_min, lon_max, lat_max\n"
        script_content = script_content.replace("AOI = [-68.66, -0.30, -68.64, -0.29]", new_aoi_line)
        
        with open(temp_script, 'w') as f:
            f.write(script_content)
        
        # Make the script executable
        os.chmod(temp_script, 0o755)
        
        # Run the script with --bbox argument if it's a sentinel or gedi download
        if script_name in ['download_sentinel.py', 'download_gedi.py']:
            bbox_args = ["--bbox"] + [str(x) for x in aoi]
            result = subprocess.run([f"./{temp_script}"] + bbox_args, capture_output=True, text=True)
        else:
            result = subprocess.run([f"./{temp_script}"], capture_output=True, text=True)
        
        # Clean up
        os.remove(temp_script)
        
        if result.returncode != 0:
            print(f"Error running {script_name}:")
            print(result.stderr)
            return False
        
        print(result.stdout)
        return True
    except Exception as e:
        print(f"Error running {script_name}: {str(e)}")
        return False

def setup_output_paths(base_output_dir, aoi):
    """Setup all output paths based on the base output directory and AOI."""
    # Define output paths
    GEDI_OUTPUT_DIR = os.path.join(base_output_dir, 'gedi')
    # Generate bbox-specific filename
    bbox_str = '_'.join([f"{coord:.2f}" for coord in aoi])
    GEDI_PARQUET_FILE = os.path.join(GEDI_OUTPUT_DIR, f"gedi_l2a_bbox_{bbox_str}.parquet")
    TEMP_GEDI_DOWNLOAD_DIR = os.path.join(base_output_dir, 'temp_gedi_downloads')

    SENTINEL2_BANDS_DIR = os.path.join(base_output_dir, 'sentinel2_bands')
    SENTINEL2_BANDS_NEEDED = ['B04', 'B08'] # Red (B4) and NIR (B8)
    SENTINEL2_DATE_RANGE = ("2023-06-01", "2023-08-31") 

    FEATURE_OUTPUT_DIR = os.path.join(base_output_dir, 'features')
    # BBOX-specific soil index raster path
    SOIL_INDEX_RASTER_PATH = os.path.join(FEATURE_OUTPUT_DIR, f"soil_index_bbox_{bbox_str}.tif")

    GEDI_F1_RASTER_PATH = os.path.join(FEATURE_OUTPUT_DIR, 'gedi_f1_canopy_depression.tif')

    # Paths for aligned/processed features (will be 30m)
    ALIGNED_F1_RASTER_PATH = os.path.join(FEATURE_OUTPUT_DIR, 'f1_canopy_depression_30m.tif')
    ALIGNED_F2_RASTER_PATH = os.path.join(FEATURE_OUTPUT_DIR, 'f2_soil_index_30m.tif')

    # Paths for Z-scored features and anomaly raster
    ZSCORED_F1_PATH = os.path.join(FEATURE_OUTPUT_DIR, 'f1_zscored_30m.tif')
    ZSCORED_F2_PATH = os.path.join(FEATURE_OUTPUT_DIR, 'f2_zscored_30m.tif')
    ANOMALY_RASTER_PATH = os.path.join(FEATURE_OUTPUT_DIR, 'anomaly_raster_30m.tif')

    # Path for GPT ranking output
    GPT_RANKING_JSON_PATH = os.path.join(base_output_dir, 'gpt_site_ranking.json')
    MANIFEST_FILE_PATH = os.path.join(base_output_dir, 'manifest.json')

    return {
        'GEDI_OUTPUT_DIR': GEDI_OUTPUT_DIR,
        'GEDI_PARQUET_FILE': GEDI_PARQUET_FILE,
        'TEMP_GEDI_DOWNLOAD_DIR': TEMP_GEDI_DOWNLOAD_DIR,
        'SENTINEL2_BANDS_DIR': SENTINEL2_BANDS_DIR,
        'SENTINEL2_BANDS_NEEDED': SENTINEL2_BANDS_NEEDED,
        'SENTINEL2_DATE_RANGE': SENTINEL2_DATE_RANGE,
        'FEATURE_OUTPUT_DIR': FEATURE_OUTPUT_DIR,
        'SOIL_INDEX_RASTER_PATH': SOIL_INDEX_RASTER_PATH,
        'GEDI_F1_RASTER_PATH': GEDI_F1_RASTER_PATH,
        'ALIGNED_F1_RASTER_PATH': ALIGNED_F1_RASTER_PATH,
        'ALIGNED_F2_RASTER_PATH': ALIGNED_F2_RASTER_PATH,
        'ZSCORED_F1_PATH': ZSCORED_F1_PATH,
        'ZSCORED_F2_PATH': ZSCORED_F2_PATH,
        'ANOMALY_RASTER_PATH': ANOMALY_RASTER_PATH,
        'GPT_RANKING_JSON_PATH': GPT_RANKING_JSON_PATH,
        'MANIFEST_FILE_PATH': MANIFEST_FILE_PATH
    }

def analyze_point_distribution(coords_list, output_dir):
    """
    Analyze and visualize the distribution of points before clustering.
    
    Args:
        coords_list: List of (lon, lat) tuples
        output_dir: Directory to save diagnostic plots
    """
    print("\nAnalyzing point distribution...")
    
    # Convert to numpy array for easier manipulation
    coords = np.array(coords_list)
    
    # 1. Calculate nearest neighbor distances
    nbrs = NearestNeighbors(n_neighbors=2).fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    neighbor_distances = distances[:, 1]  # Distance to nearest neighbor
    
    print(f"Nearest neighbor statistics:")
    print(f"  Mean distance: {np.mean(neighbor_distances):.2f} degrees")
    print(f"  Median distance: {np.median(neighbor_distances):.2f} degrees")
    print(f"  Min distance: {np.min(neighbor_distances):.2f} degrees")
    print(f"  Max distance: {np.max(neighbor_distances):.2f} degrees")
    
    # 2. Create diagnostic plots
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Point distribution
    plt.figure(figsize=(10, 10))
    plt.scatter(coords[:, 0], coords[:, 1], c='blue', alpha=0.5, s=1)
    plt.title('Spatial Distribution of Top 2000 Points')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(os.path.join(output_dir, 'point_distribution.png'))
    plt.close()
    
    # Plot 2: Nearest neighbor distance histogram
    plt.figure(figsize=(10, 6))
    plt.hist(neighbor_distances, bins=50)
    plt.title('Distribution of Nearest Neighbor Distances')
    plt.xlabel('Distance (degrees)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'neighbor_distances.png'))
    plt.close()
    
    return {
        'mean_distance': float(np.mean(neighbor_distances)),
        'median_distance': float(np.median(neighbor_distances)),
        'min_distance': float(np.min(neighbor_distances)),
        'max_distance': float(np.max(neighbor_distances))
    }

def main(args_dict):
    seed = args_dict['seed']
    base_output_dir = args_dict['base_output_dir']
    aoi = args_dict['aoi']
    skip_downloads = args_dict['skip_downloads']
    
    print(f"Running find_sites.py with seed: {seed}")
    print(f"Using AOI: {aoi}")
    print(f"Outputting data to base directory: {base_output_dir}")
    print(f"Target 30m grid resolution (approx degrees for EPSG:4326): {TARGET_RESOLUTION_DEGREES}")

    # Setup all output paths
    paths = setup_output_paths(base_output_dir, aoi)

    # Create output directories
    os.makedirs(paths['GEDI_OUTPUT_DIR'], exist_ok=True)
    os.makedirs(paths['TEMP_GEDI_DOWNLOAD_DIR'], exist_ok=True)
    os.makedirs(paths['SENTINEL2_BANDS_DIR'], exist_ok=True)
    os.makedirs(paths['FEATURE_OUTPUT_DIR'], exist_ok=True)

    # Check if GEDI parquet file exists and download if needed
    if not os.path.exists(paths['GEDI_PARQUET_FILE']) and not skip_downloads:
        print("GEDI data not found. Downloading...")
        if not run_download_script('download_gedi.py', aoi):
            print("Failed to download GEDI data. Exiting.")
            return
    elif not os.path.exists(paths['GEDI_PARQUET_FILE']):
        print(f"GEDI parquet file {paths['GEDI_PARQUET_FILE']} not found and downloads are skipped. Exiting.")
        return

    # Check if Sentinel-2 data exists and download if needed
    if not os.path.exists(paths['SOIL_INDEX_RASTER_PATH']) and not skip_downloads:
        print("Sentinel-2 data not found. Downloading...")
        if not run_download_script('download_sentinel.py', aoi):
            print("Failed to download Sentinel-2 data. Exiting.")
            return
    elif not os.path.exists(paths['SOIL_INDEX_RASTER_PATH']):
        print(f"Sentinel-2 soil index raster {paths['SOIL_INDEX_RASTER_PATH']} not found and downloads are skipped. Exiting.")
        return

    # Define Target 30m Grid Profile
    print("\nDefining target 30m grid profile...")
    target_profile_30m, target_transform_30m = features.create_target_grid_profile(
        aoi_bounds=aoi,
        resolution=TARGET_RESOLUTION_DEGREES,
        crs='EPSG:4326'
    )
    print(f"Target 30m grid - Profile: {target_profile_30m['width']}x{target_profile_30m['height']} pixels, CRS: {target_profile_30m['crs']}")

    # Derive f1 = (RH95 < 30 m) from GEDI
    print("\nDeriving GEDI feature f1 (canopy depression)...")
    f1_raster_path = features.create_gedi_canopy_depression_feature(
        gedi_parquet_path=paths['GEDI_PARQUET_FILE'],
        target_profile=target_profile_30m,
        aoi_transform=target_transform_30m,
        rh_threshold=30.0,
        output_raster_path=paths['GEDI_F1_RASTER_PATH']
    )
    if not f1_raster_path or not os.path.exists(f1_raster_path):
        print("Failed to create GEDI f1 feature raster. Exiting.")
        return
    print(f"GEDI f1 (canopy depression) raster created at: {f1_raster_path}")
    current_f1_aligned_path = f1_raster_path

    # --- Step 4 (partial): Re-grid Sentinel-2 f2 (soil index) to target 30m grid ---
    print("\nRe-gridding Sentinel-2 f2 (soil index) to target 30m grid...")
    current_f2_aligned_path = features.align_raster_to_target(
        source_raster_path=paths['SOIL_INDEX_RASTER_PATH'],
        target_profile=target_profile_30m,
        output_raster_path=paths['ALIGNED_F2_RASTER_PATH'],
        resampling_method=rasterio.enums.Resampling.bilinear
    )
    if not current_f2_aligned_path or not os.path.exists(current_f2_aligned_path):
        print("Failed to align Sentinel-2 f2 feature raster. Exiting.")
        return
    print(f"Sentinel-2 f2 (soil index) aligned to 30m grid at: {current_f2_aligned_path}")

    # --- Step 4 (continued): Z-score each feature ---
    print("\nZ-scoring aligned features...")
    zscored_f1_path = features.zscore_raster(
        raster_path=current_f1_aligned_path,
        output_raster_path=paths['ZSCORED_F1_PATH'],
    )
    zscored_f2_path = features.zscore_raster(
        raster_path=current_f2_aligned_path,
        output_raster_path=paths['ZSCORED_F2_PATH'],
    )

    if not (zscored_f1_path and os.path.exists(zscored_f1_path) and \
            zscored_f2_path and os.path.exists(zscored_f2_path)):
        print("Failed to Z-score one or both features. Exiting.")
        return
    print(f"Z-scored f1 saved to: {zscored_f1_path}")
    print(f"Z-scored f2 saved to: {zscored_f2_path}")

    # --- Step 4 (continued): Sum Z-scored features → "anomaly" raster ---
    print("\nSumming Z-scored features to create anomaly raster...")
    anomaly_raster_path = features.sum_rasters(
        raster_path_list=[zscored_f1_path, zscored_f2_path],
        output_raster_path=paths['ANOMALY_RASTER_PATH'],
        target_profile=target_profile_30m
    )

    if not anomaly_raster_path or not os.path.exists(anomaly_raster_path):
        print("Failed to create anomaly raster. Exiting.")
        return
    print(f"Anomaly raster created at: {anomaly_raster_path}")

    # --- Step 4 (continued): Take top N pixels, cluster with DBSCAN, Grab centroids ---
    print(f"\nExtracting top {TOP_N_ABSOLUTE_PIXELS} pixels from anomaly raster...")
    top_pixel_coords_values_epsg4326 = utils.get_top_n_pixel_coords(
        raster_path=anomaly_raster_path,
        top_n_absolute=TOP_N_ABSOLUTE_PIXELS
    )

    if not top_pixel_coords_values_epsg4326:
        print("No top pixels found or error in extraction. Cannot proceed with clustering. Exiting.")
        return
    print(f"Extracted {len(top_pixel_coords_values_epsg4326)} top pixel coordinates (EPSG:4326).")

    # Add diagnostic analysis
    diagnostic_dir = os.path.join(base_output_dir, 'diagnostics')
    print('DEBUG: First top pixel dict:', top_pixel_coords_values_epsg4326[0])
    coords_only = [(d['longitude'], d['latitude']) for d in top_pixel_coords_values_epsg4326]
    point_stats = analyze_point_distribution(
        coords_only,
        diagnostic_dir
    )
    
    # Set DBSCAN parameters for this test
    adjusted_eps = 1000  # meters
    adjusted_min_samples = 2
    print(f"\nOverriding DBSCAN parameters for test:")
    print(f"  eps: {adjusted_eps}m")
    print(f"  min_samples: {adjusted_min_samples}")

    # Visualize anomaly values for the top 2000 points
    anomaly_values = [d['value'] for d in top_pixel_coords_values_epsg4326]
    plt.figure(figsize=(10, 6))
    plt.hist(anomaly_values, bins=50)
    plt.title('Distribution of Anomaly Values (Top 2000 Pixels)')
    plt.xlabel('Anomaly Value')
    plt.ylabel('Count')
    plt.savefig(os.path.join(diagnostic_dir, 'anomaly_value_histogram.png'))
    plt.close()

    print(f"\nReprojecting top pixel coordinates to {TARGET_PROJECTED_CRS} for DBSCAN...")
    top_pixel_coords_projected = utils.reproject_coordinates(
        coords_list=top_pixel_coords_values_epsg4326,
        source_crs="EPSG:4326",
        target_crs=TARGET_PROJECTED_CRS
    )

    if not top_pixel_coords_projected:
        print("Failed to reproject coordinates. Cannot proceed with clustering. Exiting.")
        return
    print(f"Reprojected {len(top_pixel_coords_projected)} coordinates to {TARGET_PROJECTED_CRS}.")
    
    print("\nClustering coordinates with DBSCAN and extracting centroids...")
    found_centroids = utils.cluster_coords_dbscan_get_centroids(
        coords_with_proj_list=top_pixel_coords_projected,
        eps_meters=adjusted_eps,
        min_samples=adjusted_min_samples,
        num_largest_clusters_to_find=NUM_LARGEST_CLUSTERS,
        output_centroids_crs=OUTPUT_CENTROIDS_CRS
    )

    if not found_centroids:
        print("DBSCAN did not return any centroids. Check data and parameters. Exiting.")
        return
    
    if len(found_centroids) < NUM_LARGEST_CLUSTERS:
        print(f"Warning: DBSCAN found {len(found_centroids)} clusters, which is less than the requested {NUM_LARGEST_CLUSTERS}.")
        
    print(f"\n--- {len(found_centroids)} Largest Cluster Centroids ({OUTPUT_CENTROIDS_CRS}) ---")
    for i, centroid_info in enumerate(found_centroids):
        print(f"Centroid {i+1}: Lon={centroid_info['centroid_lon']:.5f}, Lat={centroid_info['centroid_lat']:.5f}, "
              f"ClusterSize={centroid_info['size']}, ClusterLabel={centroid_info['cluster_label']}")

    # --- Step 4 (continued): Enhance centroids with feature values ---
    print("\nEnhancing centroids with feature values...")
    centroid_coords_for_sampling = [(c['centroid_lon'], c['centroid_lat']) for c in found_centroids]

    # Sample f1 values (Canopy Depression)
    f1_values_at_centroids = utils.sample_raster_at_points(
        raster_path=paths['GEDI_F1_RASTER_PATH'],
        points_coords_epsg4326=centroid_coords_for_sampling
    )

    # Sample f2 values (Soil Index)
    f2_values_at_centroids = utils.sample_raster_at_points(
        raster_path=paths['ALIGNED_F2_RASTER_PATH'],
        points_coords_epsg4326=centroid_coords_for_sampling
    )

    # Add these feature values to the found_centroids dictionaries
    for i, centroid_info in enumerate(found_centroids):
        centroid_info['f1_value'] = f1_values_at_centroids[i]
        centroid_info['f2_value'] = f2_values_at_centroids[i]

    # --- Step 5: Create a deterministic manifest ---
    print("\nCreating deterministic manifest...")
    manifest_data = {
        "identified_anomaly_footprints_details": []
    }

    for i, centroid_info in enumerate(found_centroids):
        footprint_info = {
            "rank_in_script": i + 1,
            "centroid_wkt_epsg4326": f"POINT ({centroid_info['centroid_lon']} {centroid_info['centroid_lat']})",
            "cluster_size": centroid_info['size'],
            "cluster_label": centroid_info['cluster_label'],
            "f1_value": float(centroid_info['f1_value']),
            "f2_value": float(centroid_info['f2_value'])
        }
        manifest_data["identified_anomaly_footprints_details"].append(footprint_info)

    # Save manifest
    with open(paths['MANIFEST_FILE_PATH'], 'w') as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Manifest saved to {paths['MANIFEST_FILE_PATH']}")

if __name__ == "__main__":
    args = parse_args()
    main({
        'seed': args.seed,
        'base_output_dir': args.output_dir,
        'aoi': args.bbox,
        'skip_downloads': args.skip_downloads
    })