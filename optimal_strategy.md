Here’s a detailed but actionable strategy summary for maximizing your chances in the OpenAI to Z Amazon rainforest challenge—drawing from your research, the starter pack, and my own analysis. This combines technical tactics, data source prioritization, creative prompt engineering, and practical advice on workflow, all with the contest’s incentives and checkpoints in mind.

1. Strategic Overview: How to Win

Your Goal:
Algorithmically discover and document previously unknown (or at least “un-catalogued”) archaeological sites in the Amazon rainforest, using open-source geospatial data and OpenAI models. Your evidence should stack: geospatial anomaly + historical text cross-reference + comparison to known features.

Contest Meta-Strategy:

Hit early deadlines: First 5 days are critical—early, polished submissions get you bonus API credits ($100, possibly $1000 if you’re top-5 early birds).
Prioritize reproducibility: Judges demand that your workflow and “discoveries” are scriptable, logged, and not just manual one-offs.
Leverage your strengths: Patience and willingness to iterate with LLMs are an edge here. You can outwork teams that try to brute-force with code but miss nuance or overlook context.
2. Core Technical Workflow (Optimal Sequence)

A. Initial Setup & Sanity Checks (Day 0–1)
Clone the starter repo and load the PDFs.
Smoke test:
Download a LiDAR tile (OpenTopography or a Sentinel-2 scene).
Run a basic GPT-4o/o3 prompt (“describe the surface features”).
Print/log model version and dataset ID ([Checkpoint 1 requirement][9]).
Automate prompt logging and dataset tracking to a CSV or simple database for reproducibility.
B. Data Source Prioritization
1. LiDAR (High Priority)

GEDI footprints: Good for 25m-resolution canopy/ground height, but sparse—use for recon, not mapping.
Airborne LiDAR:
OpenTopography for dense swaths, especially Manaus region.
Zenodo EBA/Embrapa for Brazilian swaths (point clouds, DTMs).
NASA G-LiHT/LVIS: Some strips in Peru/Brazil.
SRTM: Don’t use for canopy-penetrating analysis—it only maps the treetops.
2. Multispectral & Radar (Support)

Sentinel-2: 10m optical, good for detecting soil scars/vegetation differences after partial clearing.
NICFI/PlanetScope: 3m cubes, monthly mosaics, great for finding recent disturbances and clearings.
Sentinel-1 SAR: Can see through clouds, but limited in sub-canopy feature detection; use as a backup or for seasonal monitoring.
3. Historical & Textual Data

Internet Archive / Library of Congress:
Public domain expedition diaries, mile-by-mile logs, and indigenous village references.
Useful for extracting place names, coordinates, and anecdotal evidence (checkpoint 2 requirement).
4. Academic References

Use journal supplements for site coordinates, methodological insights, and—sometimes—actual raw geospatial data ([starter-pack list][10]).
C. Feature Detection Pipeline (Days 2–3)
1. Recon

Pull GEDI ground-elevation footprints.
Use a grid or interpolate to make an “anomaly” heatmap (look for sharp canopy dips, platform-like flat areas, or linear embankments).
Overlay NICFI or Sentinel-2 for surface color changes or geometric patterns.
2. Candidate Generation

Algorithmically scan LiDAR tiles (or interpolated GEDI) for geometric anomalies:
Use Hough transform or a simple segmentation model to pick up circles, rectangles, straight ditches (features ≥80m, per the starter-pack sample prompt).
LLM post-processing:
Use GPT-4o to rate anthropogenic likelihood (“Is this shape likely man-made?” 0–1).
Save at least five top candidates, including coordinates, confidence, and logs ([Checkpoint 1][9]).
3. Negative Evidence

Cross-reference with published geoglyph/earthwork databases to avoid “rediscovering” known sites (per the starter-pack and contest FAQ).
Log all prompt/model/dataset combinations for deterministic reruns.
D. Historical Cross-Reference (Day 4)
Extract relevant expedition diary snippets:
Download public domain PDFs (LoC/Internet Archive).
Use regex to isolate paragraphs with place names, coordinates, distances, “ruins,” etc.
Pass promising paragraphs to GPT (“Extract every place name, distance, or coordinate + two lines of context”).
Link diary/village references to your candidate anomaly (ideally within 30km tolerance for “village drift”).
Screenshot and archive the actual page image for review.
E. Synthesis & Comparison (Day 5)
Compare your anomaly to a catalogued feature (e.g., similar geometry, size, context).
Draft your rationale:
“LiDAR shows concentric 120m ditch + raised platform; GEDI canopy dip and 1920 expedition diary waypoint align within 250m; Sentinel-2 soil scar confirms anthropogenic earthwork.”
Package as a notebook with reproducible code, logs, and cited sources.
3. Creative Prompt Engineering

Use plain English prompts, then refine:
“Scan this LiDAR raster for geometric shapes (rectangles, circles, straight ditches). Return center coordinates for anything ≥80m.”
“Given a coordinate and the matching Sentinel-2 scene, tell me in plain English whether the surface patterns look man-made or natural. Include a 0–1 confidence.”
“Read this diary and extract every sentence that mentions a river, compass direction, or distance travelled.”
Don’t over-tune prompts: Start wide, only add constraints as noise overwhelms signal ([starter-pack tip][10]).
4. Practical and Contest-Specific Advice

Keep a notebook/log of everything: File names, dates, prompt versions, and code—judges will want to reproduce your pipeline.
Record a video of your “aha!” moment: They want to see discoveries live.
Don’t waste time on teams unless someone brings a truly complementary skill—solo is faster for week one.
Ask for feedback from the organizers early; visibility helps.
Prioritize submitting for Checkpoint 1 within the first 24–48 hours for the $100/$1000 API credit bonuses.
Focus on a single strong candidate for Checkpoint 2: Don’t spread yourself thin.
Prepare your narrative/story early: This clarifies gaps in your reasoning and gives you a head start for the final presentation.
5. Summary Table: Data Source Priorities

Source	What For	How to Use
OpenTopography	Airborne LiDAR (high-res DEMs)	Find geometric earthworks, mounds, roads, canals.
GEDI	Canopy/ground profile (spotty)	Initial anomaly screening; interpolate to make heatmaps.
NICFI/Sentinel-2	Multispectral imagery	Confirm surface scars, detect recent clearing, soil color anomalies.
Library of Congress	Expedition diaries	Historical cross-references, river/village positions.
Published papers	Known sites, methods, data links	Avoid rediscoveries; grab new coordinates/methods.
GPT-4o family	Prompt processing, extraction	Summarize, classify, and connect textual/geospatial clues.
Bottom Line

Move fast and log everything.
Prioritize LiDAR, then multispectral, then text cross-reference.
Use LLMs to automate boring triage (entity extraction, anomaly ranking).
Submit early and polish your story.
Don’t chase myths—argue for real, evidence-backed human landscape engineering (that’s what actually wins these contests).
If you want concrete scripts, code templates, or walkthroughs for any specific step (e.g., how to regex diary PDFs or process GEDI shots in GEE), just ask and I’ll deliver. Otherwise, the above is your “map” to gold in the Amazon and in the OpenAI to Z challenge.