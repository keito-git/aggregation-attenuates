# Data

No dataset files are redistributed here. Both datasets are public; download them as described
below. The SHA-256 checksums are those of the files used for the paper.

## GoEmotions

Source: https://github.com/google-research/google-research/tree/master/goemotions

Place these files in `data/goemotions/`:

| file | where | SHA-256 |
|---|---|---|
| `goemotions_1.csv` | raw annotations, `https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv` | `cac049036bad5d68d1081f72b65f2cc51e4df82af05e3e22cfa747051cac1af3` |
| `goemotions_2.csv` | same location, `goemotions_2.csv` | `f699ecc5aa425c1720c1d02475f1e41815244b680bd75b282eb770d2c76cd84d` |
| `goemotions_3.csv` | same location, `goemotions_3.csv` | `467f1e7191af00f2e76cc7f425885c2dc304bea8aff284b10e8c460d22f2e1af` |
| `test.tsv` | `goemotions/data/test.tsv` in the repository | `0587b2dd8b27b97352adbfc3fb083d46005c8946657fdc2b1ca8b1cc7f1f8be4` |
| `emotions.txt` | `goemotions/data/emotions.txt` in the repository | `45c3ef86782d2a4d7fedcd6d8c111aa0d0e94720689bd164fac94fefb4495a89` |

Then build the prepared file and the 1,200-item reference:

```bash
python3 data/prepare_goemotions.py --raw_dir data/goemotions --out data/goe_test_prepared.json
python3 analysis/build_goe_reference.py --data data/goe_test_prepared.json \
    --subset ref/C_sc_probs.npz --out data/goe_reference_1200.npz
```

`goe_test_prepared.json` holds the 5,375 test examples with at least three annotators, their human soft
labels (fraction of annotators selecting each of the 28 labels), the mean binary entropy `H_human`, and
the entropy stratum. Expected SHA-256 of the output:
`c992a969552422c42ac3cb3cfebc88bc91245c7e8776ff07e1cd42cb5c5dab1f`.

The evaluation subset (400 items from each non-empty entropy stratum) is defined by the item ids stored
in `ref/C_sc_probs.npz`.

## LeWiDi-MP

The MultiPICo (MP) irony-detection test split of the Learning With Disagreements (LeWiDi) 2025 shared
task, in the shared task's JSON format (one entry per item with `text.post` and the official
`soft_label`). Obtain it from the official shared-task data release and save it as `data/MP_test.json`
(3,756 items). SHA-256 of the file used:
`da313236d8b6b1cb056c3965751d56c33fffd3d635f53546a924948246ed0ad6`.

## Files in `ref/`

`ref/C_sc_probs.npz` contains `ids` (the 1,200 GoEmotions example ids of the evaluation subset) and
`p_sc` (1,200 x 28 soft labels from an earlier self-consistency run of this study; not used by any script). It contains
model outputs and item ids only (no text and no human annotations). The generation scripts read only
`ids` from it.
