#!/usr/bin/env python3
"""Baseline TF-IDF + LR untuk Mendeley 600, dengan protokol split yang diulang.

Memakai ulang fungsi dari data_audit/mendeley_audit.py (manifest, model, report). Perbedaan dari skrip audit:
split acak dan split terkelompok-klaster diulang R kali; yang dilaporkan adalah rata-rata dan sebaran macro-F1.
Alasan: pada split terkelompok, klaster berukuran sama saling seri, sehingga penempatan ke fold bergantung pada
urutan id klaster. Satu kali jalan memberi selisih sekitar +-0,02 yang bukan sinyal.

Protokol
  random_5fold             StratifiedKFold(5, shuffle) diulang dengan seed berbeda
  cluster_grouped_5fold    klaster near-duplicate tidak dipecah; urutan klaster diacak per ulangan
  leave_one_topic_out      deterministik (12 topik), satu kali
  domain_only_*            baseline nama domain saja (cek shortcut sumber)

Pemakaian
  python classification/baseline.py --data-dir data/raw/mendeley [--repeats 20] [--out results/mendeley_baseline_repeated.json]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_audit import mendeley_audit as ma  # noqa: E402  (dipakai ulang, tidak disalin)


def grouped_folds(groups, k, seed):
    """Greedy: klaster terbesar dulu ke fold terisi paling sedikit. Urutan klaster yang seri diacak oleh seed."""
    uniq, inv = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inv)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    order = perm[np.argsort(-sizes[perm], kind="stable")]
    load, fold_of = np.zeros(k), np.zeros(len(uniq), dtype=int)
    for g in order:
        f = int(load.argmin())
        fold_of[g] = f
        load[f] += sizes[g]
    fold = fold_of[inv]
    return [(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(k)]


def summarize(y, preds):
    """preds: daftar prediksi out-of-fold (satu per ulangan)."""
    f1 = np.array([f1_score(y, p, average="macro") for p in preds])
    out = {
        "n_repeats": len(preds),
        "macro_f1_mean": round(float(f1.mean()), 3),
        "macro_f1_sd": round(float(f1.std(ddof=1)), 3) if len(f1) > 1 else None,
        "macro_f1_min": round(float(f1.min()), 3),
        "macro_f1_max": round(float(f1.max()), 3),
        "hoax_precision_mean": round(float(np.mean([precision_score(y, p, zero_division=0) for p in preds])), 3),
        "hoax_recall_mean": round(float(np.mean([recall_score(y, p) for p in preds])), 3),
        # CI bootstrap atas 600 item, hanya dari ulangan pertama (ketidakpastian sampel, bukan sebaran split)
        "bootstrap_ci95_repeat0": ma.bootstrap_ci(y, preds[0]),
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("results/mendeley_baseline_repeated.json"))
    a = ap.parse_args()

    df, _ = ma.build_manifest(a.data_dir)
    y = df.y.to_numpy()
    txt, topics, doms = (df[c].to_numpy(dtype=object) for c in ("text", "topic", "domain"))
    clusters = df.nearidup_cluster.to_numpy()
    seeds = range(a.repeats)

    res = {"majority_all_valid": summarize(y, [np.zeros_like(y)])}
    res["tfidf_lr__random_5fold"] = summarize(y, [cross_val_predict(
        ma.lr_text(), txt, y, cv=StratifiedKFold(5, shuffle=True, random_state=s)) for s in seeds])
    res["tfidf_lr__cluster_grouped_5fold"] = summarize(y, [cross_val_predict(
        ma.lr_text(), txt, y, cv=grouped_folds(clusters, 5, s)) for s in seeds])
    res["tfidf_lr__leave_one_topic_out"] = summarize(y, [cross_val_predict(
        ma.lr_text(), txt, y, cv=LeaveOneGroupOut(), groups=topics)])
    res["domain_only__random_5fold"] = summarize(y, [cross_val_predict(
        ma.lr_dom(), doms, y, cv=StratifiedKFold(5, shuffle=True, random_state=s)) for s in seeds])
    res["domain_only__leave_one_topic_out"] = summarize(y, [cross_val_predict(
        ma.lr_dom(), doms, y, cv=LeaveOneGroupOut(), groups=topics)])

    for k, v in res.items():
        sd = "n/a" if v["macro_f1_sd"] is None else f"{v['macro_f1_sd']:.3f}"
        print(f"{k:36s} macroF1 {v['macro_f1_mean']:.3f} sd {sd} range [{v['macro_f1_min']:.3f}, "
              f"{v['macro_f1_max']:.3f}] n={v['n_repeats']} | boot CI {v['bootstrap_ci95_repeat0']} | "
              f"hoax P {v['hoax_precision_mean']:.2f} R {v['hoax_recall_mean']:.2f}")
    out = {"meta": {"repeats": a.repeats, "seeds": list(seeds), "sklearn": sklearn.__version__, "n": len(df)},
           "baselines": res}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nresults -> {a.out}")


if __name__ == "__main__":
    main()
