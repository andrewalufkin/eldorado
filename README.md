# Andrew Adams
# Org ID: org-Lcsgy5jd3kya3ejhOrv2zyKd
# El Dorado - Geospatial Site Analysis Tool

A Python-based tool for analyzing satellite data to identify potential sites of interest using GEDI and Sentinel-2 data. This project meets the requirements for Checkpoint 1 - An Early Explorer by demonstrating multi-source data analysis, anomaly detection, and reproducible results.

## Overview

This project processes and analyzes satellite data from multiple sources to identify potential sites of interest based on various geospatial features. It combines GEDI (Global Ecosystem Dynamics Investigation) data with Sentinel-2 imagery to create composite analyses. The tool is designed to be reproducible and includes AI-powered site ranking capabilities.

## Key Features

### Data Integration
- GEDI L2A data processing and analysis (NASA Earthdata)
- Sentinel-2 L2A imagery processing (AWS Element84)
- Automated data download and preprocessing
- Quality control and validation checks

### Anomaly Detection
- Canopy height analysis using GEDI RH95 metrics
- Soil index calculation from Sentinel-2 bands
- DBSCAN clustering for site identification
- Production of 5 candidate anomaly footprints
- WKT-formatted output with centroids and radii

### Reproducibility
- Deterministic processing pipeline
- Fixed random seed (42) for reproducibility
- Comprehensive manifest generation
- Automated verification tests
- Dataset ID tracking and logging

### AI Integration
- OpenAI GPT-4 integration for site ranking
- Structured prompt engineering
- Confidence scoring for each site
- JSON-formatted ranking output
- Re-prompting capability for future analysis

## Prerequisites

- Python 3.x
- GDAL
- NASA Earthdata account (for GEDI data access)
- OpenAI API key (for site ranking)
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

4. Configure credentials:
```bash
# Add NASA Earthdata credentials to ~/.netrc
# Set OpenAI API key
export OPENAI_API_KEY='your-api-key'
```

## Project Structure

- `src/` - Core source code modules
  - `data.py` - Data loading and preprocessing
  - `features.py` - Feature extraction and analysis
  - `utils.py` - Utility functions and AI integration
- `tests/` - Test files
  - `test_determinism.py` - Reproducibility verification
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
python download_gedi.py --bbox <lon_min> <lat_min> <lon_max> <lat_max>
python download_sentinel.py --bbox <lon_min> <lat_min> <lon_max> <lat_max>
```

2. Run the main analysis:
```bash
python find_sites.py --bbox <lon_min> <lat_min> <lon_max> <lat_max> --seed 42
```

3. Verify reproducibility:
```bash
pytest tests/test_determinism.py
```

## Configuration

The project uses several configurable parameters:
- AOI (Area of Interest) coordinates
- Target resolution (default: 30m)
- Clustering parameters:
  - DBSCAN EPS: 100 meters
  - Min samples: 10
  - Target clusters: 5
- Feature thresholds:
  - GEDI RH95: 30m
  - Soil index normalization

## Output

The analysis generates several outputs in the `data_output/` directory:
- Processed GEDI data (Parquet format)
- Sentinel-2 band data (GeoTIFF)
- Feature rasters (30m resolution)
- Site rankings (JSON)
- Analysis manifests (JSON)
  - Dataset IDs
  - Processing parameters
  - Site footprints
  - GPT rankings
  - Confidence scores

## Reproducibility

The project ensures reproducibility through:
- Fixed random seed (42)
- Comprehensive manifest generation
- Dataset ID tracking
- Automated verification tests
- Consistent coordinate systems
- Deterministic processing pipeline

Results are verified to be reproducible within 50 meters between runs, meeting the checkpoint requirements for consistency in site identification.

## Example Run

A successful run has been completed with the following results:

### Identified Sites
The analysis identified 5 candidate sites in the Amazon region, with the following characteristics:

1. Primary Site (Cluster 2)
   - Location: -62.9348°W, -7.6879°S
   - Cluster Size: 1,074 pixels
   - Soil Index: -0.8443 (significant soil anomaly)
   - Canopy Height: Normal (f1_value: 0.0)

2. Secondary Site (Cluster 1)
   - Location: -63.3098°W, -8.1800°S
   - Cluster Size: 366 pixels
   - Soil Index: -0.8251 (significant soil anomaly)
   - Canopy Height: Normal (f1_value: 0.0)

3. Tertiary Site (Cluster 4)
   - Location: -62.9401°W, -8.0688°S
   - Cluster Size: 252 pixels
   - Soil Index: 0.3581 (moderate soil anomaly)
   - Canopy Height: Normal (f1_value: 0.0)

Additional sites were identified with smaller cluster sizes (54-74 pixels) and varying soil index values.

### Diagnostic Visualizations
The analysis generated several diagnostic plots in `data_output/diagnostics/`:
- `anomaly_value_histogram.png`: Distribution of anomaly scores
- `neighbor_distances.png`: Spatial distribution of identified points
- `point_distribution.png`: Geographic distribution of candidate sites

### Reproducibility
The results have been verified to be reproducible within 50 meters between runs, meeting the checkpoint requirements. The manifest file (`data_output/manifest.json`) contains the complete details of the identified sites and can be used to verify reproducibility.
