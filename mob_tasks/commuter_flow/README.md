# Commuter flow generation

The London commuter flow experiment is implemented as an exploratory notebook: [`commuter_flow_generation.ipynb`](commuter_flow_generation.ipynb). The notebook exposes the complete training and evaluation pipeline; only the DeepGravity and Bilinear model definitions and configuration validation live in Python modules.

## Data

Runtime data belong under:

```text
data/commuter_flow/london/
├── representation/
│   ├── london_oa_embeddings_le4mob_dim128_distpres0.1.csv.gz
│   └── ... baseline embedding tables
└── flow/
    ├── train_data_{0..4}.csv.gz
    ├── val_data_{0..4}.csv.gz
    └── test_data_{0..4}.csv.gz
```

Every flow row contains an origin Output Area and aligned lists named `destinations`, `counts`, `distances`, and `distance_normed`. All seven representation tables cover the same 26,369 London Output Areas. The configuration maps each representation name to its exact filename and dimension.

## Run

The first experiment cell controls the representation, downstream model, seed protocol, debug mode, and an optional device override. The committed defaults use LE4Mob with DeepGravity and execute a small debug run.

For the full paper protocol, set:

```python
REPRESENTATION = "le4mob"
MODEL_TYPE = "deepgravity"  # or "bilinear"
USE_MULTIPLE_MODEL_SEEDS = True
DEBUG = False
```

The five geographic splits are always `0, 1, 2, 3, 4`. When repeated model seeds are enabled, every split is trained with seeds `10, 20, 30, 40, 50`. When disabled, every split is trained once with the fixed seed `10`.

Generated histories, predictions, metrics, and downstream checkpoints are written below `outputs/commuter_flow/london/` and are ignored by Git. The embedding filename and SHA-256 checksum are saved inside each downstream checkpoint's metadata.
