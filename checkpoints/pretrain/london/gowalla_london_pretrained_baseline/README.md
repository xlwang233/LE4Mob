# Gowalla-London legacy next-location pretraining artifacts

These directories port the exact pretraining artifacts referenced by the archived Gowalla-London conventional and inductive next-location experiment configurations.
Vanilla-E2E has no artifact because it learns its location lookup table with the downstream model.

## Mobility baselines

| Destination | Archived `pretrained_new` source |
|---|---|
| `skipgram_conventional` | `gowalla_london_skipgram_20250516_114729` |
| `skipgram_inductive` | `gowalla_london_skipgram_inductive10pct_20250517_094235` |
| `poi2vec_conventional` | `gowalla_london_poi2vec_20250516_135112` |
| `poi2vec_inductive` | `gowalla_london_poi2vec_inductive10pct_20250521_162049` |
| `teaser_conventional` | `gowalla_london_teaser_20250516_140414` |
| `teaser_inductive` | `gowalla_london_teaser_inductive10pct_20250522_155652` |
| `tale_conventional` | `gowalla_london_tale_20250516_125039` |
| `tale_inductive` | `gowalla_london_tale_inductive10pct_20250522_160838` |

Every copied file is byte-identical to its archived source. Embedding matrices have shape `(958, 128)`. Conventional directories contain run `0`; inductive directories contain runs `0` through `4`.

## Coordinate encoders

| Destination | Legacy source | Legacy SHA-256 |
|---|---|---|
| `calliper` | `epoch=29-val_loss=6.6173.ckpt` | `da526114b574949c4e6abac41d0853ded97efff1dcb079682af07b6564ea72d3` |
| `le4mob` | `epoch=67-val_loss=6.6712.ckpt` | `fca95885f3e9a93aeb9f097f212e8f30ea2ad503ebbf2ea2e3ad10090a991409` |

The archived inductive CaLLiPer and LE4Mob experiments deliberately reuse the same full-data coordinate encoders as their conventional counterparts; there are no separate inductively pretrained coordinate encoders for these experiments. We therefore keep one shared directory per method and omit the `_conventional` and `_inductive` postfixes.
LE4Mob metadata records `setting: conventional` and `heldout_policy: none`, accurately describing how the shared encoder was pretrained. The other mobility baselines have setting-specific directory names because their conventional and inductive artifacts differ.

All coordinate encoders use projected `x-y` inputs in EPSG:27700, metre units, and produce 128-dimensional embeddings.
