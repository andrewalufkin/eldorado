#!/usr/bin/env python3

import os
import sys
import json
from pathlib import Path
import subprocess
import tempfile
import base64
from openai import OpenAI
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
from dotenv import load_dotenv

def get_dtm_stats(dtm_file):
    """Get basic statistics about the DTM file"""
    dataset = gdal.Open(str(dtm_file))
    if dataset is None:
        raise ValueError(f"Could not open DTM file: {dtm_file}")
    
    band = dataset.GetRasterBand(1)
    stats = band.GetStatistics(True, True)
    
    return {
        "min_elevation": stats[0],
        "max_elevation": stats[1],
        "mean_elevation": stats[2],
        "std_dev": stats[3],
        "width": dataset.RasterXSize,
        "height": dataset.RasterYSize,
        "projection": dataset.GetProjection()
    }

def create_dtm_visualization(dtm_file, output_file):
    """Create a visualization of the DTM"""
    dataset = gdal.Open(str(dtm_file))
    if dataset is None:
        raise ValueError(f"Could not open DTM file: {dtm_file}")
    
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray()
    
    # Create a figure with a specific size
    plt.figure(figsize=(10, 10))
    
    # Create the visualization
    plt.imshow(data, cmap='terrain')
    plt.colorbar(label='Elevation (m)')
    plt.title('DTM Visualization')
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def encode_image_to_base64(image_path):
    """Encode an image file to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_dtm_with_gpt4(dtm_file, api_key):
    """Analyze DTM features using GPT-4 Vision"""
    # Get DTM statistics
    stats = get_dtm_stats(dtm_file)
    
    # Create visualization
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        viz_file = tmp.name
    
    create_dtm_visualization(dtm_file, viz_file)
    
    # Encode the visualization
    base64_image = encode_image_to_base64(viz_file)
    
    # Prepare the prompt with statistics
    prompt = f"""Please analyze this Digital Terrain Model (DTM) visualization and describe the surface features in plain English.
    
Key statistics about the DTM:
- Elevation range: {stats['min_elevation']:.2f}m to {stats['max_elevation']:.2f}m
- Mean elevation: {stats['mean_elevation']:.2f}m
- Standard deviation: {stats['std_dev']:.2f}m
- Resolution: {stats['width']}x{stats['height']} pixels

Please describe:
1. The overall terrain characteristics
2. Any notable features (hills, valleys, flat areas, etc.)
3. The general slope patterns
4. Any potential areas of interest for archaeological investigation

Focus on describing the surface features in clear, non-technical language."""

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    try:
        # Call GPT-4 Vision API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        # Clean up temporary file
        os.unlink(viz_file)
        
        return response.choices[0].message.content
        
    except Exception as e:
        # Clean up temporary file in case of error
        if os.path.exists(viz_file):
            os.unlink(viz_file)
        raise e

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_dtm.py <dtm_file>")
        sys.exit(1)
    
    # Load environment variables from .env file
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env file")
        sys.exit(1)
    
    dtm_file = Path(sys.argv[1])
    if not dtm_file.exists():
        print(f"Error: DTM file {dtm_file} does not exist")
        sys.exit(1)
    
    try:
        analysis = analyze_dtm_with_gpt4(dtm_file, api_key)
        print("\nDTM Analysis:")
        print("=" * 80)
        print(analysis)
        print("=" * 80)
        
    except Exception as e:
        print(f"Error analyzing DTM: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 