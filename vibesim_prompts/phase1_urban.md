# Vibe Sim Prompt Sequence — Hackathon_UrbanCUAS_ChangingWeather

Same overall approach as the rural scenario, plus weather sweeps once available.
Export each batch into `data/urban/<batch_name>/`.

## Batch A — Sanity check
```
Capture data from various random map locations
Set the number of dataset images to 15
```

## Batch B — Distribution-matched main run
```
Change the pixels on target distribution to Custom
Load the custom distribution from ieee_pixels_on_target.npy
Set the number of dataset images to 300
Capture data from various random map locations
```

## Batch C — Weather sweep (repeat once weather is available; one batch per condition)
```
Set weather condition to clear
Set the number of dataset images to 80
```
```
Set weather condition to overcast
Set the number of dataset images to 80
```
```
Set weather condition to fog
Set the number of dataset images to 80
```
```
Set weather condition to rain
Set the number of dataset images to 80
```
(If exact weather-condition prompt phrasing differs in your Vibe Sim build, check
the Falcon documentation page linked in Section 5.4 — the agent should confirm
available options if you ask: "What weather conditions are available?")

## Batch D — Altitude/background sweep
```
Set the maximum altitude angle to 45.0 and minimum altitude angle to 10.0
Set the number of dataset images to 150
Capture data from various random map locations
```

## Batch E — Clutter (urban has more manned aircraft relevance — near airports/flight paths)
```
Enable birds
Enable manned aircrafts
Set the number of dataset images to 150
Capture data from various random map locations
```

## Verify
```
What is the current pixels on target distribution?
What is the current sample size?
```
Run `dataset_analysis.ipynb` to confirm distribution match before merging with
the rural set.
