import subprocess
import json
import os
import shutil
from pathlib import Path
from shapely.wkt import loads as wkt_loads
from pyproj import Transformer # For coordinate transformation
import pytest # Pytest itself
import numpy as np # For checking centroid values if needed

# --- Configuration for the Test ---
# Path to the main script to be tested
# Assumes this test file is in 'openai_to_z/tests/' and find_sites.py is in 'openai_to_z/'
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPT_TO_TEST = PROJECT_ROOT / "find_sites.py"

# Base directory for test outputs (will be created under 'openai_to_z/tests/')
TEST_OUTPUT_BASE_DIR = Path(__file__).parent / "test_run_temp_outputs"
MANIFEST_FILENAME = "manifest.json" # Expected name of the manifest file

FIXED_SEED_FOR_TEST = 42  # The seed to use for both runs
MAX_ALLOWED_DISTANCE_METERS = 50.0  # Max allowed distance between centroids
TARGET_PROJECTED_CRS_FOR_DISTANCE = "EPSG:32720"  # UTM Zone 20S (consistent with DBSCAN eps)
NUM_CLUSTERS_TO_COMPARE = 5 # We expect up to 5 centroids

# Input files to copy for each run
gedi_parquet_src = PROJECT_ROOT / "data_output" / "gedi" / "gedi_l2a_aoi_data.parquet"
soil_index_src = PROJECT_ROOT / "data_output" / "features" / "soil_index.tif"

# --- Helper Functions ---
def run_find_sites_script_and_get_manifest(seed, run_specific_output_base_dir):
    """
    Runs find_sites.py with a given seed and output directory.
    Returns the parsed manifest data.
    """
    # Ensure the specific output directory for this run is clean/exists
    if run_specific_output_base_dir.exists():
        shutil.rmtree(run_specific_output_base_dir)
    run_specific_output_base_dir.mkdir(parents=True, exist_ok=True)

    # Copy required input files
    gedi_dst = run_specific_output_base_dir / "gedi"
    gedi_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(gedi_parquet_src, gedi_dst / "gedi_l2a_aoi_data.parquet")

    features_dst = run_specific_output_base_dir / "features"
    features_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(soil_index_src, features_dst / "soil_index.tif")

    cmd = [
        "python", str(SCRIPT_TO_TEST),
        "--seed", str(seed),
        "--base_output_dir", str(run_specific_output_base_dir) # Direct output here
    ]

    print(f"\nExecuting command: {' '.join(cmd)}")
    process_result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if process_result.returncode != 0:
        print("STDOUT of failed run:")
        print(process_result.stdout)
        print("STDERR of failed run:")
        print(process_result.stderr)
        pytest.fail(
            f"find_sites.py (seed {seed}, output to {run_specific_output_base_dir}) "
            f"failed with return code {process_result.returncode}",
            pytrace=False
        )

    # The manifest is saved inside the base_output_dir by find_sites.py
    manifest_path = run_specific_output_base_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        print("STDOUT of (apparently) successful run (but no manifest):")
        print(process_result.stdout)
        print("STDERR of (apparently) successful run (but no manifest):")
        print(process_result.stderr)
        pytest.fail(f"Manifest file {manifest_path} not found after running script.", pytrace=False)

    with open(manifest_path, 'r') as f:
        manifest_data = json.load(f)
    return manifest_data

def get_centroids_from_manifest(manifest_data):
    """Extracts centroid WKT strings from manifest data, sorted by rank."""
    footprints = manifest_data.get("identified_anomaly_footprints_details", [])
    # Sort by 'rank_in_script' to ensure consistent order for comparison.
    # The manifest creation step should ideally ensure this order.
    footprints.sort(key=lambda x: x.get("rank_in_script", float('inf')))
    return [fp["centroid_wkt_epsg4326"] for fp in footprints]

def calculate_distance_meters_between_wkt_points(wkt1_epsg4326, wkt2_epsg4326, projected_crs_for_metric):
    """
    Calculates the distance in meters between two WKT points (assumed EPSG:4326).
    """
    point1_geom = wkt_loads(wkt1_epsg4326)  # shapely geometry object
    point2_geom = wkt_loads(wkt2_epsg4326)

    # Transformer from EPSG:4326 (lon/lat) to the projected CRS (meters)
    transformer = Transformer.from_crs("EPSG:4326", projected_crs_for_metric, always_xy=True)

    x1_proj, y1_proj = transformer.transform(point1_geom.x, point1_geom.y)
    x2_proj, y2_proj = transformer.transform(point2_geom.x, point2_geom.y)

    # Euclidean distance in the projected CRS (should be in meters)
    distance = ((x2_proj - x1_proj)**2 + (y2_proj - y1_proj)**2)**0.5
    return distance

# --- Pytest Fixture for Test Directory Setup/Teardown ---
@pytest.fixture(scope="module") # Run setup/teardown once per test module
def test_output_dirs_setup_teardown():
    """Manages the creation and cleanup of the base test output directory."""
    # Cleanup before starting, in case of debris from a previous failed run
    if TEST_OUTPUT_BASE_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_BASE_DIR)
    TEST_OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Define specific directories for each run
    run1_dir = TEST_OUTPUT_BASE_DIR / "run1_outputs"
    run2_dir = TEST_OUTPUT_BASE_DIR / "run2_outputs"

    yield run1_dir, run2_dir  # Provide these to the test function

    # Teardown: Clean up the base test output directory after all tests in module are done
    # Comment this out if you want to inspect the output files after a test run.
    # print(f"\nCleaning up test output directory: {TEST_OUTPUT_BASE_DIR}")
    # shutil.rmtree(TEST_OUTPUT_BASE_DIR)


# --- The Actual Test Function ---
def test_centroid_determinism(test_output_dirs_setup_teardown):
    """
    Tests that running find_sites.py twice with the same seed produces
    centroids within the allowed distance tolerance (±50m).
    """
    run1_output_dir, run2_output_dir = test_output_dirs_setup_teardown

    print(f"\n--- Test Run 1: find_sites.py with seed {FIXED_SEED_FOR_TEST} ---")
    manifest1 = run_find_sites_script_and_get_manifest(FIXED_SEED_FOR_TEST, run1_output_dir)
    centroids1_wkt_list = get_centroids_from_manifest(manifest1)

    print(f"\n--- Test Run 2: find_sites.py with seed {FIXED_SEED_FOR_TEST} ---")
    manifest2 = run_find_sites_script_and_get_manifest(FIXED_SEED_FOR_TEST, run2_output_dir)
    centroids2_wkt_list = get_centroids_from_manifest(manifest2)

    # --- Assertions ---
    num_centroids_run1 = len(centroids1_wkt_list)
    num_centroids_run2 = len(centroids2_wkt_list)

    print(f"\nNumber of centroids from Run 1: {num_centroids_run1}")
    print(f"Number of centroids from Run 2: {num_centroids_run2}")

    assert num_centroids_run1 > 0, "First run did not produce any centroids according to the manifest."
    assert num_centroids_run2 > 0, "Second run did not produce any centroids according to the manifest."

    # Check if the number of found centroids is consistent.
    # The plan aims for 5, but the script might find fewer. Consistency is key.
    assert num_centroids_run1 == num_centroids_run2, \
        f"Number of centroids differs between runs: Run 1 found {num_centroids_run1}, Run 2 found {num_centroids_run2}."

    # Determine how many centroids to actually compare (up to NUM_CLUSTERS_TO_COMPARE)
    num_to_compare = min(num_centroids_run1, NUM_CLUSTERS_TO_COMPARE)
    if num_to_compare == 0: # Should have been caught by earlier asserts if this happens
        pytest.fail("No centroids available for comparison, though runs seemed to produce some.", pytrace=False)
    
    print(f"\nComparing the (up to) top {num_to_compare} centroids from each run...")
    distances_ok = True
    for i in range(num_to_compare):
        wkt1 = centroids1_wkt_list[i]
        wkt2 = centroids2_wkt_list[i]

        print(f"  Comparing Centroid Pair {i+1}:")
        print(f"    Run 1 Centroid: {wkt1}")
        print(f"    Run 2 Centroid: {wkt2}")

        distance = calculate_distance_meters_between_wkt_points(
            wkt1, wkt2, TARGET_PROJECTED_CRS_FOR_DISTANCE
        )
        print(f"    Calculated distance: {distance:.2f} meters")

        if distance > MAX_ALLOWED_DISTANCE_METERS:
            distances_ok = False
            print(f"    ERROR: Centroid pair {i+1} distance ({distance:.2f}m) EXCEEDS tolerance ({MAX_ALLOWED_DISTANCE_METERS}m).")
        else:
            print(f"    OK: Distance ({distance:.2f}m) is within tolerance ({MAX_ALLOWED_DISTANCE_METERS}m).")
    
    assert distances_ok, "One or more centroid pairs exceeded the distance tolerance."
    print("\nCentroid determinism test passed successfully.")
