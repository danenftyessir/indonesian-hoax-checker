# indonesian-hoax-checker

NLP-based Indonesian hoax detection and claim verification. Research-oriented repository covering four workstreams:

- **NER** (`src/indonesian_hoax_checker/ner/`): entities, CLAIM, TIME
- **Hoax classification** (`classification/`): TF-IDF + LR baseline, IndoBERT
- **Refuted-claims retrieval** (`retrieval/`): BM25 and sentence embeddings over the Komdigi clarification base
- **Stance detection** (`stance/`): supports / refutes / unrelated via NLI

`pipeline/` connects them; `configs/experiments/rq1..rq5` map to the research questions.

## Setup

```bash
python -m venv .venv
pip install -e . -r requirements/dev.txt
pytest
```

Model training runs in Colab notebooks under `notebooks/` (NER: `notebooks/02_ner/train_ner_colab.ipynb`); the notebooks only call code in `src/`.

See `docs/` for architecture, dataset documentation, and experiment protocol.
