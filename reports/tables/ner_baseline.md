Tahap 1: NERGrit test (209 kalimat), label PER/ORG/LOC. IndoNLU melaporkan 69,87 untuk IndoBERT-base pada split NERGrit mereka (split kami: dev dipisah dari train).

| Model | F1 | PER | ORG | LOC |
|---|---|---|---|---|
| BiLSTM-CRF | 0.676 ± 0.004 | 0.676 ± 0.006 | 0.469 ± 0.075 | 0.750 ± 0.024 |
| IndoBERT-base-p1 | 0.783 ± 0.012 | 0.817 ± 0.017 | 0.581 ± 0.054 | 0.841 ± 0.011 |
