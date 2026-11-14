# Skrip Evaluasi apakah skor IndoBERT memisahkan hoaks dari valid di DALAM rentang panjang yang sama

import argparse, glob, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

MIN_PER_CLASS = 20 # Batas jumlah contoh per kelas utk menghitung AUROC

def boot_auc(y, s, B=500, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            vals.append(roc_auc_score(y[i], s[i]))
    return [round(float(x), 3) for x in np.percentile(vals, [2.5, 97.5])]


def per_bin(y, scores, n_words, bins, labels):
    # Mengembalikan baris per rentang panjang
    ens = np.mean(list(scores.values()), axis=0)
    cat = pd.cut(n_words, bins, labels=labels)
    rows = []
    for lab in labels:
        m = np.asarray(cat == lab)
        yy, ss = y[m], ens[m]
        n_h, n_v = int(yy.sum()), int((yy == 0).sum())
        row = {"bin": str(lab), "n": int(m.sum()), "hoax": n_h, "valid": n_v,
               "true_hoax_rate": round(float(yy.mean()), 3) if m.sum() else None,
               "pred_hoax_rate": round(float((ss >= 0.5).mean()), 3) if m.sum() else None}
        if min(n_h, n_v) >= MIN_PER_CLASS:
            row["auroc"] = round(float(roc_auc_score(yy, ss)), 3)
            row["auroc_ci95"] = boot_auc(yy, ss)
            row["auroc_per_seed"] = [round(float(roc_auc_score(yy, s[m])), 3) for s in scores.values()]
            row["macro_f1"] = round(float(f1_score(yy, (ss >= 0.5).astype(int), average="macro")), 3)
        else:
            row["auroc"] = None
            row["note"] = f"kelas terkecil < {MIN_PER_CLASS}; tidak dihitung"
        rows.append(row)
    return rows


def show(title, rows, overall_auc, len_auc):
    print(f"\n{title}\n  AUROC keseluruhan {overall_auc:.3f} | AUROC jumlah kata saja {len_auc:.3f}")
    print(f"  {'rentang':10s} {'n':>5s} {'hoaks':>6s} {'valid':>6s} {'hoaks asli':>10s} {'hoaks pred':>10s}  AUROC dalam rentang")
    for r in rows:
        a = "n/a" if r["auroc"] is None else f"{r['auroc']:.3f} {r['auroc_ci95']}"
        print(f"  {r['bin']:10s} {r['n']:5d} {r['hoax']:6d} {r['valid']:6d} {str(r['true_hoax_rate']):>10s} "
              f"{str(r['pred_hoax_rate']):>10s}  {a}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=Path("data/manifests/mendeley_manifest.csv"))
    ap.add_argument("--mafindo", action="store_true", help="juga Mafindo test (unduh dari Hugging Face)")
    ap.add_argument("--out", type=Path, default=Path("results/length_controlled_eval.json"))
    a = ap.parse_args()

    files = sorted(glob.glob(str(a.scores_dir / "scores_seed*.npz")))
    if not files:
        raise SystemExit(f"Tidak ada scores_seed*.npz di {a.scores_dir}")
    npz = {Path(f).stem.replace("scores_", ""): np.load(f) for f in files}
    out = {"seeds": list(npz)}

    # --- Mendeley
    df = pd.read_csv(a.manifest)
    y, nw = df.y.to_numpy(), df.n_words.to_numpy()
    sc = {k: v["mendeley"] for k, v in npz.items()}
    assert all(len(s) == len(df) for s in sc.values()), "jumlah skor != baris manifest"
    q = np.quantile(nw, [0, .25, .5, .75, 1])
    edges = [0, *q[1:-1], 10 ** 7]
    labs = [f"{int(edges[i]) + 1}-{int(edges[i + 1])}" if i < 3 else f">{int(edges[3])}" for i in range(4)]
    ens = np.mean(list(sc.values()), axis=0)
    rows = per_bin(y, sc, nw, edges, labs)
    out["mendeley"] = {"overall_auroc": round(float(roc_auc_score(y, ens)), 3),
                       "wordcount_only_auroc": round(float(roc_auc_score(y, nw)), 3), "bins": rows}
    show("Mendeley 600 (kuartil jumlah kata)", rows, out["mendeley"]["overall_auroc"], out["mendeley"]["wordcount_only_auroc"])

    # --- Mafindo test
    if a.mafindo:
        from datasets import load_dataset
        te = load_dataset("umn-cyber/mafindo-2024")["test"].to_pandas()
        yt, nt = te.is_hoax.astype(int).to_numpy(), te.content.fillna("").str.split().str.len().to_numpy()
        st = {k: v["mafindo_test"] for k, v in npz.items()}
        assert all(len(s) == len(te) for s in st.values()), "jumlah skor != baris Mafindo test"
        bins = [0, 29, 59, 99, 199, 399, 10 ** 7]
        labs = ["<30", "30-59", "60-99", "100-199", "200-399", "400+"]
        ens_t = np.mean(list(st.values()), axis=0)
        rows = per_bin(yt, st, nt, bins, labs)
        out["mafindo_test"] = {"overall_auroc": round(float(roc_auc_score(yt, ens_t)), 3),
                               "wordcount_only_auroc": round(float(roc_auc_score(yt, -nt)), 3), "bins": rows}
        show("Mafindo test (rentang jumlah kata)", rows, out["mafindo_test"]["overall_auroc"],
             out["mafindo_test"]["wordcount_only_auroc"])
    else:
        print("\n(Mafindo dilewati; tambahkan --mafindo di Colab untuk melengkapi)")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nhasil -> {a.out}")


if __name__ == "__main__":
    main()
