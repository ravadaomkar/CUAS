# Vibe Sim Prompt Sequence — Hackathon_RuralCUAS

Run these in order in the Vibe Sim agent chat pane. Export images+labels after each
batch into `data/rural/<batch_name>/`.

## Batch A — Sanity check (10-20 images)
```
Capture data from various random map locations
Set the number of dataset images to 15
```
Inspect output before scaling up.

## Batch B — Distribution-matched main run
```
Change the pixels on target distribution to Custom
Load the custom distribution from ieee_pixels_on_target.npy
Set the number of dataset images to 300
Capture data from various random map locations
```

## Batch C — Altitude/background sweep
```
Set the maximum altitude angle to 60.0 and minimum altitude angle to 5.0
Set the number of dataset images to 150
Capture data from various random map locations
```

## Batch D — Sensor realism
```
Change the fstop to 5.6
Add film grain noise to RGB outputs
Set the number of dataset images to 100
```

## Batch E — Placement diversity (run 3x, one per placement band)
```
Set the Vertical Range Upper to 0.35 and the Vertical Range Lower to 0.0
Set the number of dataset images to 80
```
```
Set the Vertical Range Upper to 0.65 and the Vertical Range Lower to 0.35
Set the number of dataset images to 80
```
```
Set the Vertical Range Upper to 1.0 and the Vertical Range Lower to 0.65
Set the number of dataset images to 80
```

## Batch F — Clutter
```
Enable birds
Enable manned aircrafts
Set the number of dataset images to 120
Capture data from various random map locations
```

## Verify
```
What is the current pixels on target distribution?
What are the current properties of the Gaussian Distribution?
```
Then run `dataset_analysis.ipynb` on the exported batch to confirm the size
distribution looks right before moving to the urban scenario.
