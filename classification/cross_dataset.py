# Skrip Uji cross-dataset Mafindo-2024 & Mendeley dengan TF-IDF + Logistic Regression

import argparse, json, re, sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_audit import mendeley_audit as ma  # noqa: E402

STRIP_RE = re.compile(r"detikcom|kompas\.com|tempo\.co|antara|[\[\(]\s*salah\s*[\]\)]|https?://\S+|#\w+", re.I)


def strip_artifacts(s: str) -> str:
    return STRIP_RE.sub(" ", s).lower()


def make_model():
    return make_pipeline(TfidfVectorizer(sublinear_tf=True, min_df=3, max_features=50000),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))


def load_mafindo():
    from datasets import load_dataset
    ds = load_dataset("umn-cyber/mafindo-2024")
    out = []
    for split in ("train", "test"):
        d = ds[split].to_pandas()
        out.append((d.content.fillna("").astype(str).to_numpy(dtype=object), d.is_hoax.astype(int).to_numpy()))
    return out


def evaluate(y, score, B=1000, seed=0):
    # Prediksi threshold 0,5
    p = (score >= 0.5).astype(int)
    rng = np.random.default_rng(seed)
    n = len(y)
    f1s, aucs = [], []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(set(y[i])) < 2:
            continue
        f1s.append(f1_score(y[i], p[i], average="macro"))
        aucs.append(roc_auc_score(y[i], score[i]))
    ci = lambda v: [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])]  # noqa: E731
    return {
        "n": int(n), "true_hoax_rate": round(float(y.mean()), 3), "pred_hoax_rate": round(float(p.mean()), 3),
        "macro_f1": round(float(f1_score(y, p, average="macro")), 3), "macro_f1_ci95": ci(f1s),
        "auroc": round(float(roc_auc_score(y, score)), 3), "auroc_ci95": ci(aucs),
        "hoax_precision": round(float(precision_score(y, p, zero_division=0)), 3),
        "hoax_recall": round(float(recall_score(y, p)), 3), "accuracy": round(float(accuracy_score(y, p)), 3),
    }


def per_topic(df, score):
    rows = {}
    for t, g in df.groupby("topic"):
        idx = g.index.to_numpy()
        y, p = g.y.to_numpy(), (score[idx] >= 0.5).astype(int)
        rows[f"{t} {g.topic_name.iloc[0]}"] = {
            "n": int(len(g)), "hoax_rate": round(float(y.mean()), 3), "pred_hoax_rate": round(float(p.mean()), 3),
            "macro_f1": round(float(f1_score(y, p, average="macro")), 3) if len(set(y)) > 1 else None}
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True, help="folder Mendeley mentah (CSV + xlsx)")
    ap.add_argument("--out", type=Path, default=Path("results/cross_dataset.json"))
    a = ap.parse_args()

    df, _ = ma.build_manifest(a.data_dir)
    m_txt, m_y = df.text.to_numpy(dtype=object), df.y.to_numpy()
    (f_txt, f_y), (t_txt, t_y) = load_mafindo()
    print(f"Mafindo train {len(f_y)} (hoax {f_y.mean():.3f}) | test {len(t_y)} (hoax {t_y.mean():.3f}) | "
          f"Mendeley {len(m_y)} (hoax {m_y.mean():.3f}; tanpa bantahan {int((~df.debunk_marker).sum())})")

    res = {}
    clf = make_model().fit(f_txt, f_y)
    s_m = clf.predict_proba(m_txt)[:, 1]
    res["A_mafindo_to_mendeley"] = evaluate(m_y, s_m)
    keep = (~df.debunk_marker).to_numpy()
    res["B_mafindo_to_mendeley_no_debunk"] = evaluate(m_y[keep], s_m[keep])
    res["B_mafindo_to_mendeley_debunk_only"] = evaluate(m_y[~keep], s_m[~keep])

    clf_s = make_model().fit([strip_artifacts(t) for t in f_txt], f_y)
    res["D_stripped_mafindo_to_stripped_mendeley"] = evaluate(
        m_y, clf_s.predict_proba([strip_artifacts(t) for t in m_txt])[:, 1])

    clf_m = make_model().fit(m_txt, m_y)
    res["C_mendeley_to_mafindo_test"] = evaluate(t_y, clf_m.predict_proba(t_txt)[:, 1])
    res["in_domain_mafindo_test_reference"] = evaluate(t_y, clf.predict_proba(t_txt)[:, 1])
    res["A_per_topic"] = per_topic(df, s_m)

    for k, v in res.items():
        if k.endswith("per_topic"):
            continue
        print(f"{k:42s} macroF1 {v['macro_f1']:.3f} CI{v['macro_f1_ci95']} | AUROC {v['auroc']:.3f} CI{v['auroc_ci95']} | "
              f"hoax P {v['hoax_precision']:.2f} R {v['hoax_recall']:.2f} | pred/true hoax {v['pred_hoax_rate']:.2f}/"
              f"{v['true_hoax_rate']:.2f} | n={v['n']}")
    print("\nA per topik (n kecil, ~50; bising):")
    for k, v in res["A_per_topic"].items():
        print(f"  {k:55s} n={v['n']:3d} hoax {v['hoax_rate']:.2f} pred {v['pred_hoax_rate']:.2f} macroF1 {v['macro_f1']}")

    import sklearn
    out = {"meta": {"sklearn": sklearn.__version__, "threshold": 0.5, "model": "tfidf(word,min_df=3,50k)+LR balanced",
                    "text_field": "content"}, "results": res}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nresults -> {a.out}")


if __name__ == "__main__":
    main()
