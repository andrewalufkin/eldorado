# Day 2 Summary: Dataset Selection & Initial Testing

**Date:** 2025-05-17
**Objective:** Source and test archaeological datasets to validate our approach, focusing on known archaeological sites and control samples.

## 1. Dataset Selection & Acquisition

* **NASA Dataset Exploration:**
    * Identified and downloaded sample .laz files from known archaeological sites
    * Acquired control samples for comparison and validation
    * Files selected to represent diverse archaeological features and terrain types

## 2. Research & Strategy Development

* **Consultation & Planning:**
    * Engaged in discussions with deep research, o3, and 4.1 teams
    * Analyzed archaeological study norms and best practices
    * Developed comprehensive strategy for identifying key data sets
    * Established criteria for data selection and validation

## 3. Initial Testing

* **Prompt Validation:**
    * Processed .laz file from a known archaeological site
    * Tested GPT Vision prompt against verified archaeological features
    * Initial results show promise for feature detection

## 4. Grid System Development

* **Mask Implementation:**
    * Created exclusion mask for non-forest tiles
    * Implemented river and water proximity filters
    * Added elevation threshold to exclude flood-prone areas
    * Developed scoring system for candidate ranking

* **Anomaly Detection Strategy:**
    1. Apply exclusion mask to filter unreasonable tiles
    2. Mark tiles in regions connected to historical sources
    3. Document tiles in regions with known archaeological digs
    4. Process LiDAR data using Hough transform and segmentation
    5. Analyze marked images using 4o for feature detection
    6. Use Sentinel-2 for terra preta/vegetation scarring detection
    7. Process promising candidates through o3 for detailed analysis

## 5. Known Issues & Challenges Encountered / Resolutions

* **Amazon River Exclusion:**
    * **Issue:** Current elevation criteria may exclude historically significant low-elevation areas near major rivers
    * **Proposed Solutions:**
        * Implement Relative Elevation Models (HAND data)
        * Use distance-based elevation buffering near major rivers
        * Incorporate flood frequency masks

* **Urban Area Exclusion:**
    * **Issue:** Need to exclude currently populated areas
    * **Proposed Solutions:**
        * Utilize Global Human Settlement Layer (GHSL)
        * Implement WorldPop population density data
        * Consider VIIRS/DMSP nighttime lights data

## 6. Grid Scoring System Development

* **High-Weight Adjustments:**
    * Implement scoring for regions mentioned in key colonial writings
    * Weight based on text reliability and uniqueness
    * Exclude known archaeological sites

* **Low-Weight Adjustments:**
    * Apply smaller positive weights to speculative sources
    * Implement nuanced likelihood scoring

* **Negative Weight Implementation:**
    * Decrease scores for documented archaeological sites
    * Create buffer zones around known sites
    * Apply strong negative weights to exclude areas

## 7. Next Steps

1. **Refine Amazon River Exclusion:**
    * Implement relative elevation models
    * Develop distance-based elevation buffering
    * Integrate flood frequency masks

2. **Update Urban Area Exclusion:**
    * Implement GHSL data integration
    * Add population density filters
    * Incorporate nighttime lights data

3. **Grid Scoring Implementation:**
    * Develop georeferencing system for historical texts
    * Create weighted scoring system
    * Implement spatial joins for score updates

4. **Technical Implementation:**
    * Load scored_grid into GeoPandas
    * Create geometry overlays
    * Implement spatial joins
    * Update score columns based on intersections