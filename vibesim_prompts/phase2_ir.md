# Vibe Sim Prompt Sequence — Hackathon_IR (Phase 2, Week 4+)

IR has no pre-supplied .npy distribution — build one from the Duality IR reference
set first (see `scripts/analyze_pot_distribution.py --labels_dir` mode), then mirror
the same custom-distribution workflow used for EO.

## Batch A — Sanity check
```
Capture data from various random map locations
Set the number of dataset images to 15
```

## Batch B — Distribution-matched main run (after building your custom .npy from
the Duality IR reference labels)
```
Change the pixels on target distribution to Custom
Load the custom distribution from duality_ir_pixels_on_target.npy
Set the number of dataset images to 300
Capture data from various random map locations
```

## Batch C — Viewing angle sweep (thermal signature changes with aspect)
```
Set the maximum altitude angle to 60.0 and minimum altitude angle to 5.0
Set the number of dataset images to 150
```
Also vary drone X/Y/Z rotation via USD edits per Section 5.1 to get front, side,
top-down, and banking silhouettes — thermal hotspots (motors/battery) are most
visible at different aspects than in EO.

## Batch D — Range sweep
```
Capture data from various random map locations
Set the number of dataset images to 150
```
(Range variation should fall out of random map location sampling combined with
altitude sweep — verify actual pixel-size spread afterward with
`dataset_analysis.ipynb`.)

## Batch E — Background temperature / environmental variation
```
Set the number of dataset images to 120
Capture data from various random map locations
```
If the IR scenario exposes explicit background-temperature or time-of-day
controls, ask the agent directly: "What background temperature or environmental
settings are available for this scenario?" and sweep across them — this is the
single most IR-specific gap vs. the EO pipeline.

## Batch F — Clutter
```
Enable birds
Enable manned aircrafts
Set the number of dataset images to 120
```
Birds and aircraft have very different thermal signatures than drones (body heat
vs. motor/battery heat vs. engine heat) — useful negative examples for
distinguishing false positives in thermal.

## Fixed-pattern noise
The problem statement notes Vibe Sim's IR pipeline models fixed-pattern noise
characteristic of real IR sensors. Ask the agent: "Is fixed pattern noise enabled
for this IR scenario?" and enable it if not on by default — omitting it is a known
sim-to-real gap for thermal imagery specifically.

## Verify
```
What is the current pixels on target distribution?
What is the current sample size?
```
