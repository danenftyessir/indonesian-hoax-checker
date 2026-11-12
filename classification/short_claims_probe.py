# Probe klaim pendek: 300 klaim (150 hoaks, 150 valid) untuk menguji classifier pada input sependek isi CLI

import argparse, hashlib, json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_dataset as cd 

N_PER_CLASS = 150
PREFIX_RE = re.compile(r"^\s*[\[\(]\s*(?:HOAKS|HOAX|DISINFORMASI|PENIPUAN|SALAH)\s*[\]\)]\s*[:\-–—]?\s*", re.I)
OUTLET_RE = re.compile(r"detik|kompas|tempo|antara|cnn|kumparan|liputan6|tribun|republika|merdeka|viva", re.I)
DATELINE_RE = re.compile(r"^[A-Z][A-Z .,/]{2,40}\s*[-–—]") 
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()  


def first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]


def build(komdigi: pd.DataFrame, tr: pd.DataFrame, te: pd.DataFrame, seed: int):
    # return DataFrame klaim: text, label (hoax/valid), source, orig_row
    rng = np.random.default_rng(seed)
    # hoaks: judul Komdigi berprefiks, prefiks dibuang, unik
    t = komdigi.title.fillna("").astype(str)
    has_prefix = t.map(lambda s: bool(PREFIX_RE.match(s)))
    cand = pd.DataFrame({"text": t.map(lambda s: PREFIX_RE.sub("", s).strip()), "orig_row": komdigi.index})[has_prefix]
    cand["n"] = cand.text.str.split().str.len()
    cand = cand[cand.n.between(5, 35)]
    cand = cand[~cand.text.map(norm).duplicated()]
    if len(cand) < N_PER_CLASS:
        sys.exit(f"Hanya {len(cand)} judul Komdigi memenuhi syarat; butuh {N_PER_CLASS}")
    h = cand.iloc[rng.choice(len(cand), N_PER_CLASS, replace=False)]
    hoax = pd.DataFrame({"text": h.text.to_numpy(), "label": "hoax", "source": "komdigi_title", "orig_row": h.orig_row.to_numpy()})

    # valid: kalimat pertama artikel valid di test Mafindo; judul tidak ada di train
    train_titles = set(tr.title.fillna("").map(norm))
    v = te[(te.is_hoax == 0) & ~te.title.fillna("").map(norm).isin(train_titles)].copy()
    v["text"] = v.content.fillna("").astype(str).map(first_sentence)
    v["n"] = v.text.str.split().str.len()
    ok = (v.n.between(8, 35) & v.text.str.contains(r"[.!?][\"”']?$") & ~v.text.str.contains(OUTLET_RE)
          & ~v.text.str.contains(DATELINE_RE))
    v = v[ok]
    v = v[~v.text.map(norm).duplicated()]
    if len(v) < N_PER_CLASS:
        sys.exit(f"Hanya {len(v)} kalimat valid memenuhi syarat; butuh {N_PER_CLASS}")
    v = v.iloc[rng.choice(len(v), N_PER_CLASS, replace=False)]
    valid = pd.DataFrame({"text": v.text.to_numpy(), "label": "valid", "source": "mafindo_test_first_sentence",
                          "orig_row": v.index.to_numpy()})
    out = pd.concat([hoax, valid], ignore_index=True)
    out.insert(0, "id", [f"sc{i:03d}" for i in range(len(out))])
    out["n_words"] = out.text.str.split().str.len()
    out["text_sha1"] = out.text.map(lambda s: hashlib.sha1(s.strip().encode("utf-8")).hexdigest())
    return out


def load_mafindo_frames():
    from datasets import load_dataset
    ds = load_dataset("umn-cyber/mafindo-2024")
    return ds["train"].to_pandas(), ds["test"].to_pandas()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--komdigi-csv", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--claims-out", type=Path, default=Path("data/interim/classifier/short_claims.jsonl"))
    ap.add_argument("--manifest-out", type=Path, default=Path("data/manifests/short_claims_manifest.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/short_claims_probe.json"))
    a = ap.parse_args()

    komdigi = pd.read_csv(a.komdigi_csv)
    tr, te = load_mafindo_frames()
    claims = build(komdigi, tr, te, a.seed)
    for p in (a.claims_out, a.manifest_out, a.out):
        p.parent.mkdir(parents=True, exist_ok=True)
    claims[["id", "text", "label", "source", "orig_row", "n_words", "text_sha1"]].to_json(
        a.claims_out, orient="records", lines=True, force_ascii=False)
    claims.drop(columns="text").to_csv(a.manifest_out, index=False)

    y = (claims.label == "hoax").astype(int).to_numpy()
    clf = cd.make_model().fit(tr.content.fillna("").astype(str).to_numpy(dtype=object), tr.is_hoax.astype(int).to_numpy())
    score = clf.predict_proba(claims.text.to_numpy(dtype=object))[:, 1]
    res = {"mafindo_tfidf_lr_on_short_claims": cd.evaluate(y, score),
           "length_only_auroc": round(float(roc_auc_score(y, claims.n_words)), 3),
           "words_mean_by_class": claims.groupby("label").n_words.mean().round(1).to_dict()}
    print(f"{len(claims)} klaim ({int(y.sum())} hoaks) | komdigi rows {len(komdigi)} | seed {a.seed}")
    r = res["mafindo_tfidf_lr_on_short_claims"]
    print(f"mafindo TF-IDF+LR -> klaim pendek: macroF1 {r['macro_f1']:.3f} CI{r['macro_f1_ci95']} | AUROC {r['auroc']:.3f} "
          f"CI{r['auroc_ci95']} | hoax P {r['hoax_precision']:.2f} R {r['hoax_recall']:.2f} | pred/true hoax "
          f"{r['pred_hoax_rate']:.2f}/{r['true_hoax_rate']:.2f}")
    print(f"AUROC panjang kata saja: {res['length_only_auroc']} | rata-rata kata per kelas: {res['words_mean_by_class']}")
    print("\nContoh (3 per kelas) untuk cek mata:")
    for lab in ("hoax", "valid"):
        for s in claims[claims.label == lab].text.head(3):
            print(f"  [{lab}] {s}")
    import sklearn
    a.out.write_text(json.dumps({"meta": {"seed": a.seed, "sklearn": sklearn.__version__, "komdigi_rows": len(komdigi),
                                          "n_per_class": N_PER_CLASS}, "results": res}, indent=2, ensure_ascii=False))
    print(f"\nklaim -> {a.claims_out}\nmanifest -> {a.manifest_out}\nhasil -> {a.out}")


if __name__ == "__main__":
    main()
