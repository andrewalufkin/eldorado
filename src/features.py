# src/features.py
import rasterio
from rasterio import features as rio_features # Corrected import alias
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling # Ensure Resampling is imported
import numpy as np
import pandas as pd
import os

def calculate_soil_index(band04_path, band08_path, output_raster_path):
    """
    Calculates a soil index (B04-B08)/(B04+B08) from Sentinel-2 bands.
    Note: Standard NDVI is (B08-B04)/(B08+B04). This is -(NDVI).

    Args:
        band04_path (str): Path to the B04 (Red) GeoTIFF.
        band08_path (str): Path to the B08 (NIR) GeoTIFF.
        output_raster_path (str): Path to save the calculated index GeoTIFF.

    Returns:
        str: Path to the output index raster, or None on failure.
    """
    print(f"Calculating soil index from {band04_path} and {band08_path}")
    try:
        with rasterio.open(band04_path) as b04_src, rasterio.open(band08_path) as b08_src:
            # Read data as float32 to allow for NaN and float results
            b04 = b04_src.read(1).astype(np.float32)
            b08 = b08_src.read(1).astype(np.float32)

            # Ensure arrays have the same shape (should be true if downloaded correctly for AOI)
            if b04.shape != b08.shape:
                # This might require resampling/aligning if they are not perfectly matched
                # For now, assume they are aligned from the download step.
                raise ValueError("Band shapes do not match. Ensure they are aligned.")

            # Handle potential division by zero or invalid values
            # Set numerator and denominator
            numerator = b04 - b08
            denominator = b04 + b08

            # Initialize index with NaNs
            soil_index = np.full(b04.shape, np.nan, dtype=np.float32)
            
            # Calculate index only where denominator is not zero
            # Avoid division by zero warnings and NaN propagation issues
            valid_mask = (denominator != 0)
            soil_index[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
            
            # Optional: Clip values to a certain range, e.g., -1 to 1, though this index might vary
            # soil_index = np.clip(soil_index, -1, 1)


            # Update profile for the output raster
            profile = b04_src.profile
            profile.update(dtype=rasterio.float32, count=1, compress='lzw') # Save as float32

            os.makedirs(os.path.dirname(output_raster_path), exist_ok=True)
            with rasterio.open(output_raster_path, 'w', **profile) as dst:
                dst.write(soil_index.astype(rasterio.float32), 1)
            
            print(f"Soil index raster saved to {output_raster_path}")
            return output_raster_path

    except Exception as e:
        print(f"Error calculating soil index: {e}")
        return None

def create_target_grid_profile(aoi_bounds, resolution, crs='EPSG:4326'):
    """
    Creates a rasterio profile for a target grid.

    Args:
        aoi_bounds (list): [lon_min, lat_min, lon_max, lat_max]
        resolution (float): Target resolution in the units of the CRS (e.g., degrees for EPSG:4326, meters for UTM).
                           For 30m, you'd typically use a UTM projection.
                           If using EPSG:4326 (lat/lon), resolution is in degrees.
                           ~30m in degrees is approx 0.00027 degrees, but this varies with latitude.
                           It's often better to work in a projected CRS for fixed meter resolutions.
                           For this checkpoint, we'll stick to EPSG:4326 for simplicity with the given AOI,
                           and assume the 30m is a nominal target.
                           The plan says "re-grid both features to 30m". If Sentinel was 10m,
                           and GEDI is points, we need a common grid. Let's use the AOI bounds
                           and calculate width/height based on a nominal 30m in degrees for now.
                           A better approach uses a projected CRS.
        crs (str): Coordinate Reference System.

    Returns:
        dict: A rasterio profile.
        affine.Affine: The affine transform for the grid.
    """
    lon_min, lat_min, lon_max, lat_max = aoi_bounds

    # For EPSG:4326, resolution is in degrees.
    # Approximate 30m in degrees: 30 / 111320 (meters per degree at equator) ~ 0.000269...
    # Let's use a slightly rounded version or one that divides the AOI extent well.
    # The plan just says "30m", not specifying CRS. We should aim for consistency.
    # If Sentinel bands were at 10m (projected), their output index is also ~10m (projected).
    # If we use EPSG:4326 for GEDI rasterization, then Sentinel also needs to be reprojected to this.

    # Let's assume for now the 30m target grid should align with the AOI in EPSG:4326
    # This simplifies things if the input Sentinel data was also processed/clipped to this AOI in EPSG:4326.
    # If Sentinel soil index is already a GeoTIFF, its CRS and transform should be used as a reference
    # or both should be reprojected to a common projected CRS (like a UTM zone).

    # Given the AOI [-63.5, -8.2, -62.9, -7.6], the extent is:
    # Lon extent: -62.9 - (-63.5) = 0.6 degrees
    # Lat extent: -7.6 - (-8.2) = 0.6 degrees

    # If we aim for roughly 30m cells using degree-based resolution:
    pixel_size_deg = resolution # e.g., 0.00027 for ~30m

    width = int(np.ceil((lon_max - lon_min) / pixel_size_deg))
    height = int(np.ceil((lat_max - lat_min) / pixel_size_deg)) # lat is typically inverted in transform

    # Create an affine transform: from_bounds(west, south, east, north, width, height)
    # Note: rasterio typically uses north-up, so lat_max is the 'top'
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)

    profile = {
        'driver': 'GTiff',
        'dtype': rasterio.float32, # Or int for binary, but float32 is flexible
        'nodata': None, # Or a specific nodata value like -9999
        'width': width,
        'height': height,
        'count': 1,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw'
    }
    return profile, transform


def create_gedi_canopy_depression_feature(gedi_parquet_path, target_profile, aoi_transform, rh_threshold=30.0, output_raster_path=None):
    """
    Creates a binary canopy depression feature raster from GEDI data.
    Cells with GEDI RH95 < rh_threshold get value 1, others 0.

    Args:
        gedi_parquet_path (str): Path to the GEDI data in Parquet format.
        target_profile (dict): Rasterio profile for the output raster (defines grid).
        aoi_transform (affine.Affine): Affine transform of the target grid.
        rh_threshold (float): RH95 threshold for canopy depression.
        output_raster_path (str): Path to save the output GeoTIFF.

    Returns:
        str: Path to the output feature raster, or None on failure.
    """
    print(f"Creating GEDI canopy depression feature (RH95 < {rh_threshold}m)...")
    try:
        gedi_df = pd.read_parquet(gedi_parquet_path)
        if gedi_df.empty:
            print("GEDI data is empty. Cannot create feature.")
            # Create an empty or zero raster if this happens, as per requirements
            # For now, returning None.
            # Consider creating a raster of all zeros if no points meet criteria or df is empty.
            # This ensures downstream steps always have a file.
            # For now, we'll handle this by checking the return value in the main script.
            if output_raster_path:
                os.makedirs(os.path.dirname(output_raster_path), exist_ok=True)
                with rasterio.open(output_raster_path, 'w', **target_profile) as dst:
                    # Write array of zeros
                    dst.write(np.zeros((target_profile['height'], target_profile['width']), dtype=target_profile['dtype']), 1)
                print(f"Empty GEDI data; created empty raster at {output_raster_path}")
                return output_raster_path
            return None


        # Filter for relevant GEDI shots
        # Assuming columns are 'latitude', 'longitude', 'rh95'
        depressed_canopy_shots = gedi_df[gedi_df['rh95'] < rh_threshold]

        if depressed_canopy_shots.empty:
            print(f"No GEDI shots found with RH95 < {rh_threshold}m.")
            # Create a raster of all zeros
            if output_raster_path:
                os.makedirs(os.path.dirname(output_raster_path), exist_ok=True)
                with rasterio.open(output_raster_path, 'w', **target_profile) as dst:
                    dst.write(np.zeros((target_profile['height'], target_profile['width']), dtype=target_profile['dtype']), 1)
                print(f"No GEDI shots met criteria; created zero raster at {output_raster_path}")
                return output_raster_path
            return None # Or handle differently

        # Create an empty array for the raster
        raster_array = np.zeros((target_profile['height'], target_profile['width']), dtype=target_profile['dtype'])

        # Get coordinates of the depressed canopy shots
        # Create geometries (Points) for rasterio.features.rasterize
        # Need (geometry, value) pairs
        shapes = []
        for _, shot in depressed_canopy_shots.iterrows():
            # Create a point geometry. Note: pystac uses [lon, lat] for bbox,
            # shapely and geojson often use (lon, lat) for Point.
            # Check consistency. For rasterio pixel mapping, it expects (x, y) i.e. (lon, lat)
            shapes.append({'geometry': {'type': 'Point', 'coordinates': (shot['longitude'], shot['latitude'])}, 'value': 1})
        
        if not shapes: # Should be redundant due to earlier check but good practice
             print("No shapes to rasterize (this should not happen if depressed_canopy_shots was not empty).")
             # Create a raster of all zeros as above
             if output_raster_path:
                os.makedirs(os.path.dirname(output_raster_path), exist_ok=True)
                with rasterio.open(output_raster_path, 'w', **target_profile) as dst:
                    dst.write(np.zeros((target_profile['height'], target_profile['width']), dtype=target_profile['dtype']), 1)
                print(f"No shapes; created zero raster at {output_raster_path}")
                return output_raster_path
             return None


        # Rasterize the points. This will burn 'value' (1) into the raster_array
        # where the points fall.
        # The `features.rasterize` function expects an iterable of (geometry, value) pairs.
        # Geometries must be GeoJSON-like dicts or objects that provide __geo_interface__.
        # `all_touched=True` can be useful if points might fall on cell edges.
        # `merge_alg=rasterio.enums.MergeAlg.replace` or `add` can be used. For binary, replace is fine.
        # Default fill is 0, which is what we initialized raster_array with.
        # Transform should be the one for the target grid.
        
        # Convert (geom_dict, value) tuples for rasterize
        geom_value_pairs = [(shape['geometry'], shape['value']) for shape in shapes]

        # Burn features
        # Note: using rio_features.rasterize
        burned_raster = rio_features.rasterize(
            shapes=geom_value_pairs,
            out_shape=(target_profile['height'], target_profile['width']),
            transform=aoi_transform, # Use the transform from the target profile
            fill=0, # Background value
            dtype=target_profile['dtype'] # Ensure output dtype matches profile
            # merge_alg=rasterio.enums.MergeAlg.add # If you want to count points per pixel
            # For binary (presence/absence), the default overwrite or setting value is fine.
        )
        # The `shapes` argument expects geometries that are 'iterable objects that implement the geo_interface'.
        # A list of GeoJSON-like dicts is common.

        # The above rasterize call creates a new array. Let's assign it or ensure values are set.
        # If shapes have value 1, and fill is 0, burned_raster will have 1s and 0s.
        raster_array = burned_raster # Use the result of rasterize


        if output_raster_path:
            os.makedirs(os.path.dirname(output_raster_path), exist_ok=True)
            with rasterio.open(output_raster_path, 'w', **target_profile) as dst:
                dst.write(raster_array, 1)
            print(f"GEDI canopy depression feature raster saved to {output_raster_path}")
            return output_raster_path
        else:
            # If no output path, maybe return the array directly (not for this script's flow)
            return None # Or raise error if path is mandatory

    except Exception as e:
        print(f"Error creating GEDI canopy depression feature: {e}")
        import traceback
        traceback.print_exc()
        return None

def align_raster_to_target(
    source_raster_path,
    target_profile, # The full profile dict of the target grid
    output_raster_path,
    resampling_method=Resampling.bilinear # e.g., Resampling.bilinear, Resampling.nearest
):
    """
    Aligns a source raster to a target grid definition (CRS, transform, dimensions).

    Args:
        source_raster_path (str): Path to the source raster to be aligned.
        target_profile (dict): Rasterio profile of the target grid.
                               Must contain 'crs', 'transform', 'width', 'height', 'dtype'.
        output_raster_path (str): Path to save the aligned GeoTIFF.
        resampling_method (rasterio.enums.Resampling): Resampling method to use.

    Returns:
        str: Path to the output aligned raster, or None on failure.
    """
    print(f"Aligning raster {source_raster_path} to target grid...")
    print(f"Target grid: {target_profile['width']}x{target_profile['height']}, CRS: {target_profile['crs']}")
    try:
        with rasterio.open(source_raster_path) as src:
            # Determine the transform and dimensions needed for reprojecting
            # from source CRS to target CRS, matching the target grid.
            # We use the target_profile's transform, crs, width, and height directly.
            dst_transform = target_profile['transform']
            dst_crs = target_profile['crs']
            dst_height = target_profile['height']
            dst_width = target_profile['width']
            dst_dtype = target_profile.get('dtype', src.dtypes[0]) # Use target dtype or source's
            
            # Create an empty destination array
            destination_array = np.empty((dst_height, dst_width), dtype=dst_dtype)

            # Perform the reprojection
            reproject(
                source=rasterio.band(src, 1), # Assuming single-band source
                destination=destination_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=resampling_method,
                # It's good to specify nodata if your source has it and you want to preserve it
                # src_nodata=src.nodata, # Uncomment and set if applicable
                # dst_nodata=target_profile.get('nodata', None) # Uncomment and set if applicable
            )
            
            # Update the profile for the output raster based on the target profile
            out_profile = target_profile.copy()
            out_profile['dtype'] = dst_dtype # Ensure dtype matches the reprojected array

            os.makedirs(os.path.dirname(output_raster_path), exist_ok=True)
            with rasterio.open(output_raster_path, 'w', **out_profile) as dst:
                dst.write(destination_array, 1)
            
            print(f"Successfully aligned raster saved to {output_raster_path}")
            return output_raster_path

    except Exception as e:
        print(f"Error aligning raster: {e}")
        import traceback
        traceback.print_exc()
        return None

def zscore_raster(raster_path, output_raster_path, nodata_value=None):
    """
    Calculates the Z-score for each pixel in a raster.
    Z = (value - mean) / std_dev

    Args:
        raster_path (str): Path to the input raster.
        output_raster_path (str): Path to save the Z-scored raster.
        nodata_value (float, optional): If specified, this value will be ignored in mean/std calculation
                                       and preserved as nodata in the output.

    Returns:
        str: Path to the output Z-scored raster, or None on failure.
    """
    print(f"Calculating Z-score for raster: {raster_path}")
    try:
        with rasterio.open(raster_path) as src:
            array = src.read(1).astype(np.float32)
            profile = src.profile.copy()

            # Handle nodata if specified
            if nodata_value is not None:
                mask = (array == nodata_value)
                valid_pixels = array[~mask]
            elif src.nodata is not None: # Check if source has internal nodata
                nodata_value = src.nodata
                mask = (array == nodata_value) # or np.isclose(array, nodata_value) for floats
                valid_pixels = array[~mask]
            else: # No nodata specified or found
                mask = np.zeros_like(array, dtype=bool) # All pixels are valid
                valid_pixels = array.flatten()
            
            if valid_pixels.size == 0:
                print("No valid pixels to calculate mean and std (all nodata or empty). Saving as is or with zeros.")
                # Decide behavior: save as original, or all zeros/nodata
                # For now, let's try to save a raster of nodata/zeros
                zscore_array = np.full_like(array, nodata_value if nodata_value is not None else 0, dtype=np.float32)
            else:
                mean = np.mean(valid_pixels)
                std_dev = np.std(valid_pixels)
                print(f"  Mean: {mean}, Std Dev: {std_dev}")

                if std_dev == 0:
                    # If std is 0, all valid pixels are the same (equal to mean).
                    # Z-score is technically undefined or could be set to 0.
                    print("  Standard deviation is 0. Setting Z-scores of valid pixels to 0.")
                    zscore_array = np.zeros_like(array, dtype=np.float32)
                    if nodata_value is not None: # Put back nodata values
                        zscore_array[mask] = nodata_value
                else:
                    zscore_array = (array - mean) / std_dev
                    if nodata_value is not None: # Put back nodata values
                        zscore_array[mask] = nodata_value # Or a specific Z-score nodata if preferred

            profile.update(dtype=rasterio.float32)
            if nodata_value is not None: # Ensure output profile reflects nodata if used
                 profile['nodata'] = nodata_value


            os.makedirs(os.path.dirname(output_raster_path), exist_ok=True)
            with rasterio.open(output_raster_path, 'w', **profile) as dst:
                dst.write(zscore_array, 1)
            
            print(f"Z-scored raster saved to {output_raster_path}")
            return output_raster_path

    except Exception as e:
        print(f"Error calculating Z-score for {raster_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def sum_rasters(raster_path_list, output_raster_path, target_profile):
    """
    Sums multiple rasters that are assumed to be aligned (same grid, CRS, dimensions).

    Args:
        raster_path_list (list of str): List of paths to input rasters.
        output_raster_path (str): Path to save the summed raster.
        target_profile (dict): The profile that the output summed raster should have.
                               (All input rasters should already conform to this).

    Returns:
        str: Path to the output summed raster, or None on failure.
    """
    if not raster_path_list:
        print("No rasters provided to sum.")
        return None
    
    print(f"Summing {len(raster_path_list)} rasters...")
    try:
        # Initialize sum_array with zeros, based on the target profile's dimensions
        sum_array = np.zeros((target_profile['height'], target_profile['width']), dtype=np.float32)
        
        # Keep track if any raster had actual nodata values that should propagate
        # For simplicity, we assume if a pixel is nodata in ANY input, it's nodata in output,
        # unless a more complex rule is needed (e.g. np.nansum).
        # For Z-scored data, nodata might have been handled, or might be NaNs.
        # If inputs can have NaNs, use np.nansum.
        
        # For Z-scores, they might not have explicit integer nodata but could have NaNs if std_dev was 0
        # or if original data had NaNs not handled by zscore_raster's nodata_value.
        # Let's assume zscore_raster outputs float32 where nodata might be np.nan if not specified otherwise.

        current_sum_is_nan = np.zeros_like(sum_array, dtype=bool)

        for i, raster_path in enumerate(raster_path_list):
            print(f"  Adding raster: {raster_path}")
            with rasterio.open(raster_path) as src:
                array = src.read(1).astype(np.float32)
                
                # Basic check for shape conformity
                if array.shape != (target_profile['height'], target_profile['width']):
                    raise ValueError(f"Raster {raster_path} shape {array.shape} does not match target shape {(target_profile['height'], target_profile['width'])}")

                # Handle NaNs properly if they exist (e.g., from Z-score of flat regions or original nodata)
                # If array contains NaN, simple addition propagates NaN.
                # If we want to treat NaN as 0 for sum, use np.nan_to_num(array, nan=0.0)
                # For an anomaly score, if one feature is "no data" (NaN), the sum might be misleading.
                # Let's sum directly. If a Z-score was NaN, the sum will be NaN.
                
                # Update sum_array, pixels that become NaN will stay NaN
                sum_array = np.nansum(np.stack([sum_array, array]), axis=0) if i > 0 or np.isnan(array).any() else sum_array + array
                if i == 0 and not np.isnan(array).any(): # First raster, no NaNs yet from sum_array
                    sum_array = array.copy()
                
                # A simpler robust way for direct sum, allowing NaNs to propagate:
                # if i == 0:
                #     sum_array = array.copy()
                # else:
                #     sum_array += array # This will propagate NaNs: x + NaN = NaN

        # Final profile for the summed raster
        out_profile = target_profile.copy()
        out_profile['dtype'] = 'float32' # Sum is float
        # If inputs had a common nodata value that became NaN and you want to preserve it:
        # out_profile['nodata'] = np.nan # Or specific nodata if desired

        os.makedirs(os.path.dirname(output_raster_path), exist_ok=True)
        with rasterio.open(output_raster_path, 'w', **out_profile) as dst:
            dst.write(sum_array, 1)
            
        print(f"Summed anomaly raster saved to {output_raster_path}")
        return output_raster_path

    except Exception as e:
        print(f"Error summing rasters: {e}")
        import traceback
        traceback.print_exc()
        return None

# Example for testing zscore and sum (requires dummy aligned rasters)
# if __name__ == '__main__':
#     # --- Setup for testing zscore and sum ---
#     FEATURE_OUTPUT_DIR_TEST = '../data_output/features' # Adjust path
#     # Assume ALIGNED_F1_RASTER_PATH and ALIGNED_F2_RASTER_PATH exist from previous steps
#     # For testing, create simple dummy aligned files if they don't exist.
#     dummy_aligned_f1_path = os.path.join(FEATURE_OUTPUT_DIR_TEST, 'f1_canopy_depression_30m.tif') # Should exist
#     dummy_aligned_f2_path = os.path.join(FEATURE_OUTPUT_DIR_TEST, 'f2_soil_index_30m.tif')       # Should exist

#     # Use the target profile from the main script context for consistency
#     AOI_TEST_MAIN = [-63.5, -8.2, -62.9, -7.6]
#     TARGET_RESOLUTION_DEG_TEST = 0.00027
#     target_prof_test, _ = create_target_grid_profile(AOI_TEST_MAIN, TARGET_RESOLUTION_DEG_TEST, crs='EPSG:4326')


#     if not (os.path.exists(dummy_aligned_f1_path) and os.path.exists(dummy_aligned_f2_path)):
#         print(f"Ensure dummy aligned files exist for testing: {dummy_aligned_f1_path}, {dummy_aligned_f2_path}")
#     else:
#         print("\nTesting Z-scoring...")
#         zscored_f1_test_path = zscore_raster(dummy_aligned_f1_path, os.path.join(FEATURE_OUTPUT_DIR_TEST, 'test_f1_zscored.tif'))
#         zscored_f2_test_path = zscore_raster(dummy_aligned_f2_path, os.path.join(FEATURE_OUTPUT_DIR_TEST, 'test_f2_zscored.tif'))

#         if zscored_f1_test_path and zscored_f2_test_path:
#             print("Z-scoring test successful.")
#             print("\nTesting summing rasters...")
#             summed_test_path = sum_rasters(
#                 [zscored_f1_test_path, zscored_f2_test_path],
#                 os.path.join(FEATURE_OUTPUT_DIR_TEST, 'test_anomaly_summed.tif'),
#                 target_prof_test # Pass the target profile for the output sum raster
#             )
#             if summed_test_path:
#                 print("Summing rasters test successful.")
#             else:
#                 print("Summing rasters test failed.")
#         else:
#             print("Z-scoring test failed for one or both rasters.")