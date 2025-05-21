# El Dorado - Geospatial Site Analysis Tool

A Python-based tool for analyzing satellite data to identify potential sites of interest using GEDI and Sentinel-2 data.

## Overview

This project processes and analyzes satellite data from multiple sources to identify potential sites of interest based on various geospatial features. It combines GEDI (Global Ecosystem Dynamics Investigation) data with Sentinel-2 imagery to create composite analyses.

## Features

- GEDI data processing and analysis
- Sentinel-2 imagery processing
- Feature extraction and analysis
- Automated site identification
- Data validation and quality checks
- Support for custom Area of Interest (AOI)

## Prerequisites

- Python 3.x
- GDAL
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd El-Dorado
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `src/` - Core source code modules
- `tests/` - Test files
- `data_output/` - Output directory for processed data
- `find_sites.py` - Main script for site identification
- `download_sentinel.py` - Sentinel-2 data download utility
- `download_gedi.py` - GEDI data download utility
- `check_raster_validity.py` - Raster data validation tool
- `source_catalog.csv` - Data source catalog

## Dependencies

The project relies on several key Python packages:
- earthaccess - NASA Earth data access
- h5py - HDF5 file handling
- pandas - Data manipulation
- pyarrow - Data processing
- gdal - Geospatial data processing
- rasterio - Raster data handling
- scikit-image - Image processing
- openai - AI integration
- boto3 - AWS integration
- pystac_client - STAC catalog access
- numpy - Numerical computing
- shapely - Geospatial operations
- pytest - Testing framework
- pyproj - Projection handling

## Usage

1. Download required data:
```bash
python download_gedi.py
python download_sentinel.py
```

2. Run the main analysis:
```bash
python find_sites.py
```

## Configuration

The project uses several configurable parameters:
- AOI (Area of Interest) coordinates
- Target resolution (default: 30m)
- Clustering parameters
- Feature thresholds

## Output

The analysis generates several outputs in the `data_output/` directory:
- Processed GEDI data
- Sentinel-2 band data
- Feature rasters
- Site rankings
- Analysis manifests
