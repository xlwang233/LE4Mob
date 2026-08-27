# London pretrained checkpoints

This directory separates shared London coordinate encoders from artifacts used only by Gowalla-London next-location baselines.

```text
london/
├── calliper/
│   └── location_encoder.pt
├── le4mob/
│   └── location_encoder.pt
└── gowalla_london_pretrained_baseline/
```

`calliper/` and `le4mob/` contain geography-level coordinate encoders shared by:

- conventional and inductive Gowalla-London next-location prediction; and
- London commuter-flow generation.

There is one directory per method because both downstream settings intentionally use the same full-London encoder. Both encoders accept projected `x-y` coordinates in EPSG:27700 (metres) and return 128-dimensional embeddings.

`location_encoder.pt` is the inference artifact consumed by downstream tasks. Full pretraining checkpoints and machine-specific metadata are not versioned.

`gowalla_london_pretrained_baseline/` contains the setting-specific Skip-gram, POI2Vec, TALE, and TEASER artifacts used only by next-location prediction. See its own README for details.
