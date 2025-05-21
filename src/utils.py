import rasterio
from rasterio.transform import xy
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from pyproj import Transformer
import os
import json
import subprocess
from shapely.geometry import Point
import openai

def get_top_n_pixel_coords(raster_path, top_n_percent=None, top_n_absolute=None):
    """
    Gets the values and geographic coordinates of the top N or top N percent of pixels
    from a raster.

    Args:
        raster_path (str): Path to the input raster.
        top_n_percent (float, optional): Percentage of top pixels to retrieve (e.g., 0.01 for 1%).
        top_n_absolute (int, optional): Absolute number of top pixels to retrieve.
                                        If both are None, or raster is small, might return all valid pixels.
                                        If both are provided, top_n_absolute takes precedence.
    Returns:
        list of tuples: [(longitude, latitude, value), ...] for the top pixels.
                        Returns None on failure or if no valid pixels.
    """
    print(f"Getting top pixel coordinates from {raster_path}...")
    try:
        with rasterio.open(raster_path) as src:
            array = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs

            if crs.to_string() != 'EPSG:4326':
                print(f"Warning: Raster CRS is {crs.to_string()}, expected EPSG:4326 for direct lon/lat conversion.")

            valid_pixels_mask = ~np.isnan(array)
            if not np.any(valid_pixels_mask):
                print("No valid (non-NaN) pixels found in the raster.")
                return None
            
            pixel_values = array[valid_pixels_mask]
            row_indices, col_indices = np.where(valid_pixels_mask)

            if top_n_absolute is not None:
                num_top_pixels = min(top_n_absolute, len(pixel_values))
            elif top_n_percent is not None:
                num_top_pixels = min(int(len(pixel_values) * top_n_percent), len(pixel_values))
            else:
                print("No top_n specified, considering all valid pixels for sorting.")
                num_top_pixels = len(pixel_values)
                if num_top_pixels == 0:
                    return []

            indices_sorted_desc = np.argsort(pixel_values)[::-1]
            top_indices = indices_sorted_desc[:num_top_pixels]

            top_pixel_coords_values = []
            for i in top_indices:
                r, c = row_indices[i], col_indices[i]
                val = pixel_values[i]
                lon, lat = xy(transform, r + 0.5, c + 0.5, offset='center')
                top_pixel_coords_values.append({'longitude': lon, 'latitude': lat, 'value': val})
            
            print(f"Retrieved {len(top_pixel_coords_values)} top pixel coordinates.")
            return top_pixel_coords_values

    except Exception as e:
        print(f"Error getting top N pixel coords: {e}")
        import traceback
        traceback.print_exc()
        return None

def reproject_coordinates(coords_list, source_crs="EPSG:4326", target_crs="EPSG:32720"):
    """
    Reprojects a list of coordinate dictionaries.

    Args:
        coords_list (list of dict): [{'longitude': lon, 'latitude': lat, ...}, ...].
        source_crs (str): Source CRS string (e.g., "EPSG:4326").
        target_crs (str): Target CRS string (e.g., "EPSG:32720" for UTM 20S).

    Returns:
        list of dict: Reprojected coordinates [{'x_proj': x, 'y_proj': y, ...original_fields}].
    """
    if not coords_list:
        return []
    print(f"Reprojecting {len(coords_list)} coordinates from {source_crs} to {target_crs}...")
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    reprojected_coords = []
    for coord_entry in coords_list:
        lon, lat = coord_entry['longitude'], coord_entry['latitude']
        x_proj, y_proj = transformer.transform(lon, lat)
        reprojected_entry = coord_entry.copy()
        reprojected_entry['x_proj'] = x_proj
        reprojected_entry['y_proj'] = y_proj
        reprojected_coords.append(reprojected_entry)
    return reprojected_coords

def cluster_coords_dbscan_get_centroids(
    coords_with_proj_list,
    eps_meters=100, 
    min_samples=5, 
    num_largest_clusters_to_find=5,
    output_centroids_crs="EPSG:4326"
    ):
    """
    Clusters coordinates using DBSCAN and returns centroids of the N largest clusters.

    Args:
        coords_with_proj_list (list of dict): Must contain 'x_proj', 'y_proj' in a projected CRS (meters).
                                           Can also contain original 'longitude', 'latitude' for reference.
        eps_meters (float): DBSCAN epsilon in meters.
        min_samples (int): DBSCAN min_samples.
        num_largest_clusters_to_find (int): Number of largest cluster centroids to return.
        output_centroids_crs (str): The CRS to convert centroids to (e.g. "EPSG:4326").

    Returns:
        list of dict: [{'centroid_lon': lon, 'centroid_lat': lat, 'cluster_label': label, 'size': size}, ...]
                      or None on failure.
    """
    if not coords_with_proj_list:
        print("No coordinates provided for clustering.")
        return []
        
    points_for_dbscan = np.array([[entry['x_proj'], entry['y_proj']] for entry in coords_with_proj_list])
    
    if points_for_dbscan.shape[0] < min_samples:
        print(f"Number of points ({points_for_dbscan.shape[0]}) is less than min_samples ({min_samples}). DBSCAN may not find clusters.")
        if points_for_dbscan.shape[0] == 0:
            return []

    print(f"Running DBSCAN with eps={eps_meters}m, min_samples={min_samples} on {len(points_for_dbscan)} points...")
    db = DBSCAN(eps=eps_meters, min_samples=min_samples).fit(points_for_dbscan)
    
    cluster_labels = db.labels_
    unique_labels = set(cluster_labels)
    num_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
    print(f"DBSCAN found {num_clusters_found} clusters (excluding noise).")

    if num_clusters_found == 0:
        print("No clusters found by DBSCAN.")
        return []

    cluster_info = []
    for label in unique_labels:
        if label == -1:
            continue
        points_in_cluster_mask = (cluster_labels == label)
        cluster_size = np.sum(points_in_cluster_mask)
        cluster_info.append({'label': label, 'size': cluster_size, 'mask': points_in_cluster_mask})

    cluster_info.sort(key=lambda x: x['size'], reverse=True)

    top_cluster_centroids = []
    projected_crs_of_inputs = "EPSG:32720"

    transformer_to_output_crs = None
    if output_centroids_crs and output_centroids_crs != projected_crs_of_inputs:
        transformer_to_output_crs = Transformer.from_crs(projected_crs_of_inputs, output_centroids_crs, always_xy=True)

    for i in range(min(num_largest_clusters_to_find, len(cluster_info))):
        cluster = cluster_info[i]
        label = cluster['label']
        size = cluster['size']
        mask = cluster['mask']
        
        points_of_this_cluster = points_for_dbscan[mask]
        centroid_proj_x, centroid_proj_y = np.mean(points_of_this_cluster, axis=0)
        
        centroid_out_lon, centroid_out_lat = centroid_proj_x, centroid_proj_y
        if transformer_to_output_crs:
            centroid_out_lon, centroid_out_lat = transformer_to_output_crs.transform(centroid_proj_x, centroid_proj_y)

        top_cluster_centroids.append({
            'centroid_lon': centroid_out_lon,
            'centroid_lat': centroid_out_lat,
            'cluster_label': int(label),
            'size': int(size),
            'centroid_x_proj_original_crs': centroid_proj_x,
            'centroid_y_proj_original_crs': centroid_proj_y
        })
        print(f"  Cluster {label}: size={size}, centroid (proj)={centroid_proj_x:.2f},{centroid_proj_y:.2f} -> (out CRS) {centroid_out_lon:.5f},{centroid_out_lat:.5f}")

    return top_cluster_centroids

def sample_raster_at_points(raster_path, points_coords_epsg4326):
    """
    Samples raster values at a list of point coordinates (EPSG:4326).
    Assumes the input raster is also in EPSG:4326 or can be handled by rasterio
    if points are transformed to its CRS. For simplicity, assumes raster is EPSG:4326.

    Args:
        raster_path (str): Path to the raster file.
        points_coords_epsg4326 (list of tuples): [(lon1, lat1), (lon2, lat2), ...].

    Returns:
        list: Sampled values corresponding to each point. List contains None for points outside raster extent.
    """
    sampled_values = []
    try:
        with rasterio.open(raster_path) as src:
            if src.crs.to_string().upper() != 'EPSG:4326':
                print(f"Warning: Raster {raster_path} CRS is {src.crs}. Expected EPSG:4326 for direct sampling.")

            for lon, lat in points_coords_epsg4326:
                try:
                    py, px = src.index(lon, lat)
                    if 0 <= px < src.width and 0 <= py < src.height:
                        value = src.read(1, window=rasterio.windows.Window(px, py, 1, 1))[0, 0]
                        sampled_values.append(value.item())
                    else:
                        sampled_values.append(None)
                except Exception:
                    sampled_values.append(None)
            
    except Exception as e:
        print(f"Error sampling raster {raster_path}: {e}")
        return [None] * len(points_coords_epsg4326)
    return sampled_values

def get_git_sha():
    """Gets the current git commit SHA."""
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        return sha
    except Exception as e:
        print(f"Could not get git SHA: {e}. Ensure git is installed and in a git repository.")
        return None

def create_and_print_manifest(
    centroids_data,
    seed,
    input_gedi_path,
    input_sentinel_soil_index_path,
    anomaly_raster_path,
    gpt_ranking_json_path
):
    """
    Creates a structured manifest dictionary and prints it as JSON.
    """
    git_sha = get_git_sha()

    manifest_footprints = []
    for i, centroid_info in enumerate(centroids_data):
        wkt_point = Point(centroid_info['centroid_lon'], centroid_info['centroid_lat']).wkt
        
        footprint_entry = {
            "rank_in_script": i + 1,
            "centroid_wkt_epsg4326": wkt_point,
            "cluster_label_from_dbscan": centroid_info.get('cluster_label', 'N/A'),
            "cluster_size_in_pixels": centroid_info.get('size', 0),
            "feature_f1_canopy_depression_flag": centroid_info.get('feature_f1_value'),
            "feature_f2_soil_index_value": round(centroid_info['feature_f2_value'], 5) if centroid_info.get('feature_f2_value') is not None else None
        }
        manifest_footprints.append(footprint_entry)

    manifest = {
        "git_sha": git_sha if git_sha else "N/A (git error or not a repo)",
        "seed_used": seed,
        "input_data_sources": {
            "gedi_l2a_processed_shots_file": input_gedi_path,
            "sentinel2_derived_soil_index_file_original_res": input_sentinel_soil_index_path,
        },
        "derived_data_products": {
            "anomaly_raster_30m_epsg4326": anomaly_raster_path,
            "gpt_site_ranking_output_json": gpt_ranking_json_path
        },
        "identified_anomaly_footprints_details": manifest_footprints
    }

    manifest_json_string = json.dumps(manifest, indent=2)
    print("\n--- BEGIN DETERMINISTIC MANIFEST ---")
    print(manifest_json_string)
    print("--- END DETERMINISTIC MANIFEST ---\n")
    
    return manifest

def get_openai_api_key():
    """Retrieves the OpenAI API key from the environment variable."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
    return api_key

def reprompt_gpt_and_save_ranking(
    centroids_data_for_gpt,
    output_json_path,
    model_name="gpt-4o-mini",
    api_seed=42
):
    """
    Constructs a prompt with site candidates and their features,
    queries the OpenAI API, and saves the ranking JSON response.
    """
    print(f"\n=== OpenAI API Request Details ===")
    print(f"Model: {model_name}")
    print(f"API Seed: {api_seed}")
    print(f"Number of centroids to rank: {len(centroids_data_for_gpt)}")
    print(f"Output path: {output_json_path}")
    print("================================\n")

    # 1. Construct the table_string for the prompt
    table_header = "| Centroid (WKT)                     | Canopy Feature (f1) | Soil Index (f2) |\n"
    table_separator = "|------------------------------------|---------------------|-----------------|\n"
    table_rows = []

    if not centroids_data_for_gpt:
        print("No centroid data provided for GPT prompt. Skipping GPT re-prompt.")
        return None

    for i, centroid in enumerate(centroids_data_for_gpt):
        wkt_point = Point(centroid['centroid_lon'], centroid['centroid_lat']).wkt
        
        # Represent f1 (Canopy Depression RH95 < 30m)
        f1_desc = "Unknown"
        if centroid.get('feature_f1_value') == 1.0:
            f1_desc = "Canopy < 30m"
        elif centroid.get('feature_f1_value') == 0.0:
            f1_desc = "Canopy >= 30m"
        elif centroid.get('feature_f1_value') is None:
            f1_desc = "GEDI data N/A"
            
        f2_val = centroid.get('feature_f2_value')
        f2_str = f"{f2_val:.4f}" if f2_val is not None else "N/A"
        
        table_rows.append(f"| {wkt_point.ljust(34)} | {f1_desc.ljust(19)} | {f2_str.ljust(15)} |\n")

    if not table_rows:
        print("Could not generate any rows for the GPT prompt table. Skipping.")
        return None

    table_string = table_header + table_separator + "".join(table_rows)
    print("Constructed table for GPT prompt:\n" + table_string)

    # 2. Set up OpenAI client and make the API call
    api_key = get_openai_api_key()
    if not api_key:
        print("Cannot proceed with GPT re-prompt due to missing API key.")
        return None
        
    try:
        client = openai.OpenAI(api_key=api_key)

        system_message_content = "You are a geomorphology analyst."
        user_message_content = f"""
Below are {len(centroids_data_for_gpt)} candidate archaeological footprints from the Amazon.
For each I list: its WKT centroid, a GEDI-derived canopy height feature, and a Sentinel-2 derived soil redness index.

{table_string}

Rank them 1–{len(centroids_data_for_gpt)} by likelihood of being anthropogenic earthworks.
Justify each rank in ≤25 words.
Return JSON:
[{{
  "rank": 1,
  "centroid_wkt": "POINT(...)",
  "confidence": 0.83,  // Your estimated confidence 0.0-1.0
  "rationale": "..."
}}, ... // one entry for each of the {len(centroids_data_for_gpt)} input sites
]
"""
        print("\nSending request to OpenAI API...")

        response = client.chat.completions.create(
            model=model_name,
            seed=api_seed,
            messages=[
                {"role": "system", "content": system_message_content},
                {"role": "user", "content": user_message_content}
            ],
            response_format={"type": "json_object"}
        )
        
        gpt_response_content = response.choices[0].message.content
        print("\n=== OpenAI API Response ===")
        print(f"Model used: {response.model}")
        print(f"Response ID: {response.id}")
        print(f"Created at: {response.created}")
        print("Response content:")
        print(gpt_response_content)
        print("===========================\n")

        # 3. Save the response to a file
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f:
            try:
                parsed_json = json.loads(gpt_response_content)
                json.dump(parsed_json, f, indent=2)
            except json.JSONDecodeError:
                print("Warning: GPT response was not valid JSON. Saving raw response.")
                f.write(gpt_response_content)
                
        print(f"GPT ranking response saved to {output_json_path}")
        return gpt_response_content

    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during GPT re-prompt: {e}")
        import traceback
        traceback.print_exc()
    
    return None
