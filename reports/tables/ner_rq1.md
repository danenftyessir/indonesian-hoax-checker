RQ1: base = tanpa perlakuan, norm = normalisasi teks informal, aug = +1 salinan informal data latih, aug3 = +3 salinan. Varian dipilih dari dev. Uji sintetis dibuat dengan generator yang sama dengan augmentasi (probabilitas dan seed berbeda), jadi sebagian dalam-distribusi.

| Layer | Varian | Dev | Test formal | Tulisan tangan (30) | Informal sintetis | NERGrit test |
|---|---|---|---|---|---|---|
| CLAIM | base | 0.752 ± 0.019 | 0.630 ± 0.024 | 0.101 ± 0.021 | 0.348 ± 0.015 | — |
| CLAIM | norm | 0.775 ± 0.015 | 0.629 ± 0.017 | 0.089 ± 0.005 | 0.351 ± 0.018 | — |
| CLAIM | aug | 0.761 ± 0.024 | 0.647 ± 0.029 | 0.091 ± 0.003 | 0.624 ± 0.029 | — |
| CLAIM | aug3 | 0.802 ± 0.013 | 0.654 ± 0.034 | 0.106 ± 0.052 | 0.638 ± 0.053 | — |
| Entitas | base | 0.778 ± 0.003 | 0.803 ± 0.012 | 0.616 ± 0.087 | 0.717 ± 0.019 | 0.705 ± 0.025 |
| Entitas | norm | 0.782 ± 0.007 | 0.804 ± 0.007 | 0.600 ± 0.093 | 0.726 ± 0.024 | 0.712 ± 0.027 |
| Entitas | aug | 0.776 ± 0.001 | 0.792 ± 0.007 | 0.630 ± 0.072 | 0.770 ± 0.010 | 0.720 ± 0.036 |
