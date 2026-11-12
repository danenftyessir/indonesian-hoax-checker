# data/

Datasets are not committed (see `.gitignore`); only this README and `.gitkeep` files are tracked.
Check each dataset's license before sharing.

| Folder | Content |
|---|---|
| `raw/` | Original datasets, never modified (Komdigi refuted claims, NERGrit, NERP, classifier data, IndoNLI) |
| `interim/` | Cleaning, deduplication, normalization, label mapping results |
| `processed/` | Validated, model-ready datasets |
| `annotations/` | Manual annotations (CLAIM, TIME, claim verification); tracked in Git |

Download instructions: `configs/data_sources.yaml` lists every source and its local path; see `docs/dataset_documentation.md` for licenses and audit notes. Task 3 (Komdigi claims, IndoNLI, BRIN) pipeline:

```
kaggle datasets download -d ireddragonicy/indonesian-hoax-news-dataset -p data/raw/refuted_claims --unzip
python -m indonesian_hoax_checker.retrieval.corpus data/raw/refuted_claims/komdigi_hoaks.csv data/interim/refuted_claims/claims_raw.jsonl
python -m indonesian_hoax_checker.evaluation.splits data/interim/refuted_claims/claims_raw.jsonl --kaggle-version <N>
python -m indonesian_hoax_checker.stance.dataset indonli data/raw/stance/indonli/train.jsonl data/processed/stance/indonli_train.jsonl --split-name train
```
