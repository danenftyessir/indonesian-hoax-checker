# Audit dataset umn-cyber/mafindo-2024

import re, numpy as np, pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ds = load_dataset("umn-cyber/mafindo-2024")
tr, te = ds["train"].to_pandas(), ds["test"].to_pandas()
for n, d in [("train", tr), ("test", te)]:
    print(n, len(d), "hoax share", round(d.is_hoax.mean(), 3))

def feats(d):
    t, c = d.title.fillna(""), d.content.fillna("")
    return pd.DataFrame({
        "title_trailing_quote": t.str.contains(r'[”"]\s*$'),
        "title_label_prefix": t.str.match(r'^\s*[\[\(]\s*(SALAH|HOAKS|HOAX|DISINFORMASI|PENIPUAN)', case=False),
        "outlet_name": c.str.contains(r'detikcom|kompas\.com|tempo\.co|antara|cnn indonesia', case=False),
        "has_url": c.str.contains(r'https?://|www\.', case=False),
        "hashtag": c.str.contains(r'#\w+'),
        "allcaps_title": t.apply(lambda s: len(s) > 8 and s.isupper()),
        "short_content": c.str.split().str.len() < 30,
    }).astype(int)

Ftr, Fte = feats(tr), feats(te)
print(Ftr.groupby(tr.is_hoax.values).mean().T.round(3))      

norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
print("dup titles in train:", tr.title.map(norm).duplicated().sum(),
      "| train∩test titles:", len(set(tr.title.map(norm)) & set(te.title.map(norm))))

def run(name, Xtr, Xte):
    m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, tr.is_hoax)
    print(f"{name:28s} macro-F1", round(f1_score(te.is_hoax, m.predict(Xte), average="macro"), 3))

run("artifact flags only", Ftr, Fte)
v = TfidfVectorizer(sublinear_tf=True, min_df=3, max_features=50000)
run("content TF-IDF", v.fit_transform(tr.content.fillna("")), v.transform(te.content.fillna("")))
strip = lambda s: re.sub(r"detikcom|kompas\.com|tempo\.co|antara|[\[\(]\s*salah\s*[\]\)]|https?://\S+|#\w+", " ", s, flags=re.I).lower()
v2 = TfidfVectorizer(sublinear_tf=True, min_df=3, max_features=50000)
run("content, artifacts stripped", v2.fit_transform(tr.content.fillna("").map(strip)), v2.transform(te.content.fillna("").map(strip)))
