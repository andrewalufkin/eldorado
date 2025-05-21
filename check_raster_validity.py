import rasterio
import numpy as np
import os

raster_path = 'data_output/features/gedi_f1_canopy_depression.tif'

if not os.path.exists(raster_path):
    print(f"Raster file not found: {raster_path}")
    exit(1)

with rasterio.open(raster_path) as src:
    print(f"Raster opened: {raster_path}")
    print(f"  CRS: {src.crs}")
    print(f"  Shape: {src.width} x {src.height}")
    print(f"  Data type: {src.dtypes[0]}")
    arr = src.read(1)
    print(f"  Min value: {np.nanmin(arr)}")
    print(f"  Max value: {np.nanmax(arr)}")
    print(f"  Unique values: {np.unique(arr)}")
    print(f"  Sample (top-left 5x5):\n{arr[:5,:5]}") 