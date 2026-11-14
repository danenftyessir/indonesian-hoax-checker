# Kalibrasi dan analisis error skor IndoBERT pada Mendeley dari hasil skor yang disimpan finetune_indobert.py


import argparse, glob, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def ece(y, p, bins=10):
    # Expected calibration error untuk probabilitas kelas positif (hoax)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def reliability(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append({"bin": f"{b / bins:.1f}-{(b + 1) / bins:.1f}", "n": int(m.sum()),
                     "mean_pred": round(float(p[m].mean()), 3) if m.any() else None,
                     "frac_hoax": round(float(y[m].mean()), 3) if m.any() else None})
    return rows


def fit_temperature(z, y):
    def nll(T):
        p = np.clip(sigmoid(z / T), EPS, 1 - EPS)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(minimize_scalar(nll, bounds=(0.05, 50.0), method="bounded").x)


def loto_calibrate(z, y, topics):
    # Temperature dan Platt dipasang pada topik lain, diterapkan ke topik yang ditahan
    p_temp, p_platt, temps = np.zeros(len(z)), np.zeros(len(z)), {}
    for t in np.unique(topics):
        te, tr = topics == t, topics != t
        T = fit_temperature(z[tr], y[tr]); temps[str(t)] = round(T, 3)
        p_temp[te] = sigmoid(z[te] / T)
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(z[tr].reshape(-1, 1), y[tr])
        p_platt[te] = lr.predict_proba(z[te].reshape(-1, 1))[:, 1]
    return p_temp, p_platt, temps


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {"ece": round(ece(y, p), 3), "brier": round(float(np.mean((p - y) ** 2)), 3),
            "macro_f1": round(float(f1_score(y, pred, average="macro")), 3),
            "auroc": round(float(roc_auc_score(y, p)), 3), "pred_hoax_rate": round(float(pred.mean()), 3)}


def breakdown(df, col, score):
    # Jumlah dan galat per nilai kategori. FN = hoaks diprediksi valid, FP = valid diprediksi hoaks
    pred = (score >= 0.5).astype(int)
    rows = []
    for v, g in df.groupby(col, observed=True):
        i = g.index.to_numpy()
        y = g.y.to_numpy()
        rows.append({"category": str(v), "n": int(len(g)), "true_hoax": int(y.sum()),
                     "pred_hoax": int(pred[i].sum()), "FN": int(((y == 1) & (pred[i] == 0)).sum()),
                     "FP": int(((y == 0) & (pred[i] == 1)).sum()),
                     "error_rate": round(float((pred[i] != y).mean()), 3)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=Path("data/manifests/mendeley_manifest.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/indobert_calibration_error_analysis.json"))
    a = ap.parse_args()

    files = sorted(glob.glob(str(a.scores_dir / "scores_seed*.npz")))
    if not files:
        raise SystemExit(f"Tidak ada scores_seed*.npz di {a.scores_dir}")
    df = pd.read_csv(a.manifest)
    y, topics = df.y.to_numpy(), df.topic.to_numpy()
    scores = {Path(f).stem.replace("scores_", ""): np.load(f)["mendeley"] for f in files}
    for k, s in scores.items():
        if len(s) != len(df):
            raise SystemExit(f"{k}: {len(s)} skor, manifest {len(df)} baris; urutan tidak bisa dipastikan")
    print(f"{len(scores)} seed: {list(scores)} | Mendeley {len(df)} item (hoaks {int(y.sum())})")

    # --- Kalibrasi per seed
    per_seed, temps_all = {}, {}
    for k, s in scores.items():
        z = logit(s)
        p_t, p_p, temps = loto_calibrate(z, y, topics)
        per_seed[k] = {"raw": metrics(y, s), "temperature_loto": metrics(y, p_t), "platt_loto": metrics(y, p_p)}
        temps_all[k] = temps
    agg = {}
    for method in ("raw", "temperature_loto", "platt_loto"):
        agg[method] = {m: {"mean": round(float(np.mean([per_seed[k][method][m] for k in scores])), 3),
                           "sd": round(float(np.std([per_seed[k][method][m] for k in scores], ddof=1)), 3) if len(scores) > 1 else None}
                       for m in ("ece", "brier", "macro_f1", "auroc", "pred_hoax_rate")}
    print("\nKalibrasi Mendeley (rata-rata antar seed; LOTO = dipasang di dalam CV leave-one-topic-out):")
    for method, ms in agg.items():
        print(f"  {method:18s} " + " | ".join(f"{m} {v['mean']:.3f}" for m, v in ms.items()))

    # --- Analisis galat pada skor rata-rata antar seed
    mean_score = np.mean(list(scores.values()), axis=0)
    df = df.reset_index(drop=True)
    df["len_bin"] = pd.qcut(df.n_words, 4, duplicates="drop").astype(str)
    cats = {"topic": "topic_name", "source_tier": "source_tier", "debunk_marker": "debunk_marker",
            "unanimous": "unanimous", "length_quartile": "len_bin", "true_label": "label"}
    errors = {name: breakdown(df, col, mean_score) for name, col in cats.items()}
    pred = (mean_score >= 0.5).astype(int)
    cm = {"TP": int(((y == 1) & (pred == 1)).sum()), "FN": int(((y == 1) & (pred == 0)).sum()),
          "FP": int(((y == 0) & (pred == 1)).sum()), "TN": int(((y == 0) & (pred == 0)).sum())}
    rho = spearmanr(mean_score, df.n_words)
    shortcut = {"spearman_score_vs_n_words": round(float(rho.statistic), 3),
                "auroc_n_words_for_hoax": round(float(roc_auc_score(y, df.n_words)), 3),
                "auroc_ensemble_score": round(float(roc_auc_score(y, mean_score)), 3)}
    print(f"\nEnsemble rata-rata seed: {cm} | {shortcut}")
    for name, rows in errors.items():
        print(f"\n[{name}]")
        for r in rows:
            print(f"  {r['category'][:44]:44s} n={r['n']:3d} hoaks={r['true_hoax']:3d} prediksi={r['pred_hoax']:3d} "
                  f"FN={r['FN']:3d} FP={r['FP']:3d} galat={r['error_rate']:.2f}")

    out = {"seeds": list(scores), "calibration": {"aggregate": agg, "per_seed": per_seed, "temperature_by_heldout_topic": temps_all},
           "reliability_raw_ensemble": reliability(y, mean_score), "confusion_ensemble": cm,
           "shortcut_checks": shortcut, "error_analysis": errors}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nhasil -> {a.out}")


if __name__ == "__main__":
    main()
