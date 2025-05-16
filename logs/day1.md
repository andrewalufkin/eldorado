# Day 1 Summary: Data Plumbing & First GPT Vision Call

**Date:** 2025-05-16
**Objective:** Establish a working pipeline to download a Digital Elevation Model (DEM), process it into a hillshade PNG, send it to GPT-4o for analysis, and log the results. Test the core mechanics before focusing on specific Amazonian datasets.

## 1. DEM & LiDAR Sources

*   **Initial Attempts (Sierra Nevada & Gabilan Mesa Samples):**
    *   The initial `curl` command for the Sierra Nevada sample (`OT.042013.26911.2`) resulted in a very small file (713 bytes), likely an error page, indicating the link or dataset ID was outdated.
    *   A subsequent attempt with a Gabilan Mesa sample URL also resulted in a 404 error.
*   **Successful DEM Download (SRTMGL1 via OpenTopography API):**
    *   Realized that direct API access often requires an API key.
    *   Obtained an OpenTopography API key (`OT_API_KEY`).
    *   Successfully downloaded a sample DTM using the OpenTopography Global DEM API:
        *   **Dataset Type:** `SRTMGL1` (Global 1-arcsecond, ~30m resolution)
        *   **Coordinates for Sample:** `south=37.0&north=37.05&west=-122.0&east=-121.95` (California)
        *   **Output Format:** GeoTIFF (`.tif`)
        *   **File Name:** `data/raw/lidar/sample_dtm.tif`
        *   **Size:** ~40KB
    *   This SRTM sample was used to test the rest of the pipeline, with the understanding that an Amazonian LiDAR dataset will be sourced next.

## 2. GPT-4o Vision Interaction

*   **Model Version:** `gpt-4o-2024-05-13`
*   **Prompt Text:**
    ```
    You are an archaeological scout.
    Return JSON: {"label": "yes"|"no"|"maybe", "confidence": 0-1}
    Criteria: convex or rectilinear shape 70-400 m across; embankment rim visible.
    ```
*   **Image Sent:** `data/raw/lidar/hs_512.png` (A 512x512 PNG derived from the SRTMGL1 DTM hillshade).
*   **Example Response Received:**
    ```json
    {
      "label": "maybe",
      "confidence": 0.6
    }
    ```
    This was considered a reasonable response for a generic landscape sample not expected to contain the target archaeological features.

## 3. Cost Per Call (Sanity Check)

*   **Estimated Input Tokens:**
    *   Prompt: ~150 tokens
    *   512px Image: ~85 tokens
    *   Total Input: ~235 tokens
*   **Estimated Output Tokens (for the "maybe" response):** ~15-25 tokens (e.g., `{"label": "maybe", "confidence": 0.6}` is fairly short).
*   **Pricing (Example - User to verify current pricing for `gpt-4o-2024-05-13`):**
    *   Input: $X per 1M tokens
    *   Output: $Y per 1M tokens
*   **Calculated Cost per call:** Approximately `(235/1000000 * $X) + (20/1000000 * $Y)`.
    *   Based on the original example figures of $2.50/1M input and $10/1M output, this would be roughly $0.0005875 + $0.0002 = $0.0007875.
    *   I checked the console and it jsut said < 0.01 cents
*   **Conclusion:** Cost per call is very low, suitable for bulk processing.

## 4. Known Issues & Challenges Encountered / Resolutions

*   **Imports** I had to use conda/mamba to get everything working right.
*   **Outdated OpenTopography Links:** Initial `curl` links for sample DTMs were non-functional.
    *   **Resolution:** Switched to using the OpenTopography Global DEM API with an API key.
*   **Colab Secret Management for `curl`:** Difficulty passing the `OT_API_KEY` from Python to the `curl` shell command. The placeholder "OT_API_KEY" was being sent.
    *   **Resolution:** Used `google.colab.userdata.get('OT_API_KEY')` to reliably fetch the secret in Python and then used an f-string to correctly embed the key into the `curl` command string executed by `get_ipython().system()`.
*   **GDAL `PROJ` Warnings:** `Warning 1: PROJ: proj_create_from_database: Open of /usr/local/share/proj failed` encountered during `gdaldem` and `gdalinfo` operations in Colab.
    *   **Status:** These warnings did not prevent successful file generation (`sample_hs.tif`, `hs_512.png`) or metadata reading. The DTM had well-defined CRS information.
*   **SQLite Database Creation Error:** `OperationalError: unable to open database file` when trying to `sqlite3.connect('logs/day1.sqlite')`.
    *   **Resolution:** The `logs/` directory did not exist. Created it using `!mkdir -p logs` before running the Python script, or by adding `os.makedirs('logs', exist_ok=True)` in the `log_result` function.
*   **Verification of DTM Download:** Initial small file sizes (e.g., 173 bytes, 713 bytes) indicated download failures (error messages from server).
    *   **Resolution:** Used `!gdalinfo` and `!cat` (or `!head`) on the downloaded `.tif` file to confirm it was a valid GeoTIFF and not an error message. A successful download was ~40KB.

## 5. Key Outputs Created

*   `data/raw/lidar/sample_dtm.tif` (SRTMGL1 DEM)
*   `data/raw/lidar/sample_hs.tif` (Hillshade of the DEM)
*   `data/raw/lidar/hs_512.png` (512x512 PNG version of the hillshade for GPT-4o)
*   `logs/day1.sqlite` (SQLite database logging the API call timestamp, image MD5, and GPT-4o response)

## 6. Next Steps

*   Source actual LiDAR DTMs from the Amazon basin.
*   Adapt the download process if necessary for different data sources/formats (e.g., LAZ files requiring local conversion to DTM).
*   Iterate on the GPT-4o prompt based on results from actual Amazonian data. 
*   Test against known ancient cities in the Amazon.
*   Continue building out the pipeline for batch processing.