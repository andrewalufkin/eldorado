#!/usr/bin/env python3

import os
import subprocess
from pathlib import Path
import glob
import re
import json

def get_tile_coordinates(filename):
    """Extract tile coordinates from filename (assuming format like 'tile_123_456.laz')"""
    match = re.search(r'tile_(\d+)_(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None

def find_adjacent_tiles(tile_coords, all_tiles):
    """Find tiles that are adjacent to the given tile coordinates"""
    x, y = tile_coords
    adjacent = []
    for tile in all_tiles:
        coords = get_tile_coordinates(tile)
        if coords:
            tx, ty = coords
            if (abs(tx - x) == 1 and ty == y) or (abs(ty - y) == 1 and tx == x):
                adjacent.append(tile)
    return adjacent

def process_laz_file(laz_file):
    """Process a single .laz file using PDAL"""
    output_dir = Path("processed_dtm")
    output_dir.mkdir(exist_ok=True)
    
    # Create output filename
    base_name = Path(laz_file).stem
    output_file = output_dir / f"{base_name}_dtm.tif"
    
    # Read the pipeline template
    with open("pipelines/dtm.json", 'r') as f:
        pipeline = json.load(f)
    
    # Update the pipeline with the correct filenames
    pipeline[0]["filename"] = str(laz_file)
    pipeline[3]["filename"] = str(output_file)
    
    # Write the temporary pipeline file
    temp_pipeline = output_dir / f"{base_name}_pipeline.json"
    with open(temp_pipeline, 'w') as f:
        json.dump(pipeline, f, indent=4)
    
    # Run PDAL pipeline
    cmd = ["pdal", "pipeline", str(temp_pipeline)]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully processed {laz_file}")
        # Clean up temporary pipeline file
        temp_pipeline.unlink()
        return str(output_file)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {laz_file}: {e}")
        # Clean up temporary pipeline file
        if temp_pipeline.exists():
            temp_pipeline.unlink()
        return None

def merge_tiles(tile_files, output_file):
    """Merge multiple tiles using gdal_merge.py"""
    cmd = ["gdal_merge.py", "-o", output_file] + tile_files
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully merged tiles into {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error merging tiles: {e}")
        return False

def main():
    # Get all .laz files
    laz_files = glob.glob("downloaded_lidar_data/*.laz")
    processed_files = []
    
    # Process each .laz file
    for laz_file in laz_files:
        processed_file = process_laz_file(laz_file)
        if processed_file:
            processed_files.append(processed_file)
    
    # Find and merge adjacent tiles
    merged_dir = Path("merged_dtm")
    merged_dir.mkdir(exist_ok=True)
    
    processed_files = [Path(f) for f in processed_files]
    processed_files.sort()
    
    # Group adjacent tiles
    merged_groups = []
    used_files = set()
    
    for file in processed_files:
        if file in used_files:
            continue
            
        coords = get_tile_coordinates(file.name)
        if not coords:
            continue
            
        adjacent = find_adjacent_tiles(coords, [f.name for f in processed_files])
        if adjacent:
            group = [file]
            for adj_file in adjacent:
                adj_path = next((f for f in processed_files if f.name == adj_file), None)
                if adj_path and adj_path not in used_files:
                    group.append(adj_path)
                    used_files.add(adj_path)
            
            if len(group) > 1:
                merged_groups.append(group)
                used_files.update(group)
    
    # Merge each group of adjacent tiles
    for i, group in enumerate(merged_groups):
        output_file = merged_dir / f"merged_dtm_{i+1}.tif"
        merge_tiles([str(f) for f in group], str(output_file))

if __name__ == "__main__":
    main() 