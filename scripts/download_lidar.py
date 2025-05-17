#!/usr/bin/env python3

import subprocess
import pathlib
import os # Added for creating directories

# --- Configuration ---
# !! IMPORTANT: Update this BASE_URL if your tiles come from a different dataset !!
# Example for "LiDAR_Forest_Inventory_Brazil" dataset:
BASE_URL = "https://daac.ornl.gov/daacdata/cms/LiDAR_Forest_Inventory_Brazil/data/"
# Example for a hypothetical different dataset (you'd need to find the correct one):
# BASE_URL = "https://e4ftl01.cr.usgs.gov/SOMEDATASET/PATH/"

TILE_LIST_FILE = "tile_list.txt" # File containing one tile name per line
COOKIES_FILE = pathlib.Path.home() / ".urs_cookies"
DOWNLOAD_SUBDIR = "downloaded_lidar_data" # Name of the subdirectory for downloads

# --- IMPORTANT: NASA Earthdata Login (URS) Authentication ---
# (Keeping the URS authentication message as it's crucial)
# This script relies on `wget` being able to authenticate with NASA Earthdata.
# To enable this, you MUST have a ~/.netrc file in your home directory
# with your URS credentials.
#
# 1. Create or edit ~/.netrc (e.g., /Users/andrewadams/.netrc):
#    machine urs.earthdata.nasa.gov
#        login YOUR_USERNAME
#        password YOUR_PASSWORD
#
# 2. Replace YOUR_USERNAME and YOUR_PASSWORD with your actual Earthdata credentials.
#
# 3. Set strict permissions for this file:
#    chmod 600 ~/.netrc
#
# If this file is not set up correctly, downloads requiring authentication will fail.
# ---

def download_tile(tile_name, base_url, cookies_file_path, output_directory):
    """
    Downloads a single tile using wget with URS authentication into the specified directory.
    """
    if not tile_name:
        print("Skipping empty tile name.")
        return

    url = base_url + tile_name
    
    # Ensure the output directory exists
    pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Command for wget:
    # -P <prefix_directory>: Save files to this directory.
    cmd = [
        "wget",
        "--load-cookies", str(cookies_file_path),
        "--save-cookies", str(cookies_file_path),
        "--keep-session-cookies",
        "-c",              # Continue partial downloads
        "-P", str(output_directory), # Specify the output directory
        # "-N",            # Optional: timestamping, only download if remote is newer
        url
    ]

    print(f"Attempting to download: {tile_name} to {output_directory}")
    print(f"Executing: {' '.join(cmd)}")

    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Successfully downloaded {tile_name} to {pathlib.Path(output_directory) / tile_name}.")
        # print("wget stdout:\n", process.stdout) # Uncomment for more verbose output
        # print("wget stderr:\n", process.stderr) # Uncomment for more verbose output

    except subprocess.CalledProcessError as e:
        print(f"Error downloading {tile_name}.")
        print(f"Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}.")
        print("wget stdout:\n", e.stdout)
        print("wget stderr:\n", e.stderr)
        if "401" in e.stderr or "403" in e.stderr or "Authentication" in e.stderr or "urs.earthdata.nasa.gov" in e.stderr:
            print("Authentication error suspected. Please ensure your ~/.netrc file is correctly configured for urs.earthdata.nasa.gov.")
        elif "404 Not Found" in e.stderr:
            print(f"404 Not Found: The file {tile_name} might not exist at {url}. Please verify the BASE_URL and tile name.")
        else:
            print("Please check wget output above for more details.")
    except FileNotFoundError:
        print("Error: wget command not found. Please ensure wget is installed and in your PATH.")
        print("On macOS, you can install it with: brew install wget")
    except Exception as e:
        print(f"An unexpected error occurred while trying to download {tile_name}: {e}")

def main():
    print(f"Using cookies file: {COOKIES_FILE}")
    
    # Define the full path to the download directory
    # If DOWNLOAD_SUBDIR is an absolute path, this will use it.
    # If it's relative, it will be relative to the script's CWD.
    download_dir_path = pathlib.Path(DOWNLOAD_SUBDIR)
    
    # Create the download directory if it doesn't exist.
    # This is also done in download_tile, but doing it here once can be clearer.
    download_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Files will be downloaded to: {download_dir_path.resolve()}")


    try:
        with open(TILE_LIST_FILE, "r") as f:
            tile_names = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Tile list file '{TILE_LIST_FILE}' not found in the current directory.")
        print("Please create it and add one tile filename per line.")
        return

    if not tile_names:
        print(f"No tile names found in '{TILE_LIST_FILE}'. Nothing to download.")
        return

    print(f"Found {len(tile_names)} tile(s) to download from '{TILE_LIST_FILE}'.")

    for name in tile_names:
        download_tile(name, BASE_URL, COOKIES_FILE, download_dir_path)
        print("-" * 30)

if __name__ == "__main__":
    main()