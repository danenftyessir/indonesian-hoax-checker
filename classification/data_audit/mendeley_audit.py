# Audit dataset Mendeley "Indonesian Hoax News Detection Dataset"

import argparse, hashlib, json, re, sys, zipfile
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import openpyxl
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, GroupKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline

CSV_NAME = "600 news with valid hoax label.csv"
XLSX_NAME = "600 news labelling process.xlsx"
NEARDUP_THRESHOLD = 0.8
TOPIC_NAMES = {
    "KW1": "ikan lele mengandung sel kanker", "KW2": "tusuk jari bantu pasien stroke",
    "KW3": "iPhone 6 mudah melengkung", "KW4": "Reog Ponorogo dibakar di Filipina",
    "KW5": "peserta aksi 212 tak bisa masuk Istiqlal", "KW6": "sikat gigi dari bulu babi",
    "KW7": "permen dot berbahaya", "KW8": "Pokemon GO = 'Aku Yahudi'",
    "KW9": "pelemparan Al-Qur'an di Mako Brimob", "KW10": "CTO Traveloka walk out saat Anies pidato",
    "KW11": "Facebook akan tutup di Indonesia", "KW12": "gaji Presiden naik",
}

# Heuristics
DEBUNK_RE = re.compile(r"\bhoa?ks?\b|\bhoax\b|klarifikasi|bantah|dibantah|tidak benar|bukan fakta|faktanya|mitos|isu bahwa|kabar bohong|menepis|menegaskan")
MAINSTREAM = ["kompas", "detik", "tempo", "viva", "republika", "tribunnews", "merdeka", "metrotvnews", "liputan6",
              "okezone", "cnnindonesia", "antaranews", "sindonews", "jawapos", "suara.com", "bisnis.com",
              "kumparan", "tirto", "beritasatu", "inilah.com"]


def find_file(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if hits:
        return hits[0]
    for z in list(root.rglob("*.zip")):            
        out = z.parent / f"_extracted_{z.stem}"
        if not out.exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(out)
        hits = list(out.rglob(name))
        if hits:
            return hits[0]
        for z2 in out.rglob("*.zip"):
            out2 = z2.parent / f"_extracted_{z2.stem}"
            if not out2.exists():
                with zipfile.ZipFile(z2) as zf:
                    zf.extractall(out2)
            hits = list(out2.rglob(name))
            if hits:
                return hits[0]
    sys.exit(f"Cannot find '{name}' under {root}")


def rating(x):
    if x is None:
        return None
    s = str(x).strip().lower()
    return "hoax" if s.startswith("h") else "valid" if s.startswith("v") else None


def load_votes(path: Path) -> pd.DataFrame:
    wb = openpyxl.load_workbook(path, data_only=True)
    rows = []
    for ws in wb.worksheets:                                  
        for r in ws.iter_rows(values_only=True):
            if len(r) < 7 or r[2] is None:
                continue
            r1, r2, r3, vote = (rating(r[i]) for i in (3, 4, 5, 6))
            if None in (r1, r2, r3, vote):                    
                continue
            rows.append(dict(topic=ws.title, url=str(r[2]).strip(), R1=r1, R2=r2, R3=r3, vote_workbook=vote))
    return pd.DataFrame(rows)


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", quotechar='"', encoding="cp1252", engine="python")
    df.columns = ["text", "label_csv"]
    df["label_csv"] = df.label_csv.str.strip().str.lower()
    df["text"] = df.text.astype(str)
    return df


def build_manifest(data_dir: Path):
    csv, xlsx = find_file(data_dir, CSV_NAME), find_file(data_dir, XLSX_NAME)
    txt, votes = load_csv(csv), load_votes(xlsx)
    if len(txt) != len(votes):
        sys.exit(f"Row count mismatch: csv={len(txt)} workbook={len(votes)}; alignment by row order is not safe")
    df = pd.concat([votes.reset_index(drop=True), txt], axis=1)
    df.insert(0, "row_id", range(len(df)))
    df["topic_name"] = df.topic.map(TOPIC_NAMES)
    df["n_hoax"] = (df[["R1", "R2", "R3"]] == "hoax").sum(axis=1)
    df["unanimous"] = df.n_hoax.isin([0, 3])
    df["label"] = np.where(df.n_hoax >= 2, "hoax", "valid")       
    df["y"] = (df.label == "hoax").astype(int)
    df["csv_label_mismatch"] = df.label_csv != df.label
    df["vote_differs_from_majority"] = df.vote_workbook != df.label
    df["domain"] = df.url.map(lambda u: urlparse(u if u.startswith("http") else "http://" + u).netloc.lower().removeprefix("www."))
    df["source_tier"] = df.domain.map(lambda d: "mainstream" if any(m in d for m in MAINSTREAM) else "other")
    df["url_year"] = df.url.str.extract(r"[/\-_.](20(?:1[0-9]))[/\-_.]?")[0]
    dup = df.url.map(df.url.value_counts()) > 1
    df["url_dup_group"] = -1
    df.loc[dup, "url_dup_group"] = df[dup].groupby("url").ngroup()
    df["n_words"] = df.text.str.split().str.len()
    df["debunk_marker"] = df.text.str.lower().str.contains(DEBUNK_RE)
    df["text_sha1"] = df.text.map(lambda t: hashlib.sha1(t.strip().encode("utf-8")).hexdigest())
    # near-duplicate clusters: connected components over same-topic pairs with cosine >= threshold
    X = TfidfVectorizer(sublinear_tf=True).fit_transform(df.text)
    S = cosine_similarity(X)
    np.fill_diagonal(S, 0)
    tp = df.topic.to_numpy(dtype=object)
    same_topic = tp[:, None] == tp[None, :]
    _, comp = connected_components(csr_matrix((S >= NEARDUP_THRESHOLD) & same_topic), directed=False)
    df["nearidup_cluster"] = comp
    df["nearidup_cluster_size"] = df.nearidup_cluster.map(df.nearidup_cluster.value_counts())
    df["has_neardup"] = (S.max(1) >= NEARDUP_THRESHOLD)
    audit = {
        "n": len(df), "label_counts_majority": df.label.value_counts().to_dict(),
        "label_counts_csv": df.label_csv.value_counts().to_dict(),
        "csv_rows_contradicting_majority": int(df.csv_label_mismatch.sum()),
        "csv_rows_contradicting_unanimous_vote": int((df.csv_label_mismatch & df.unanimous).sum()),
        "workbook_vote_differs_from_majority": int(df.vote_differs_from_majority.sum()),
        "unanimous_share": round(float(df.unanimous.mean()), 3),
        "urls_listed_more_than_once": int(df[dup].url.nunique()),
        "articles_with_neardup_ge_0.8": int(df.has_neardup.sum()),
        "nearidup_clusters": int(df.nearidup_cluster.nunique()),
        "hoax_rate_by_source_tier": df.groupby("source_tier").y.mean().round(3).to_dict(),
        "debunk_marker_rate_by_class": df.groupby("label").debunk_marker.mean().round(3).to_dict(),
        "domains": int(df.domain.nunique()),
    }
    cnt = np.stack([3 - df.n_hoax, df.n_hoax], 1)
    p = cnt.sum(0) / (len(cnt) * 3)
    P = ((cnt ** 2).sum(1) - 3) / 6
    audit["fleiss_kappa"] = round(float((P.mean() - (p ** 2).sum()) / (1 - (p ** 2).sum())), 3)
    return df, audit


def bootstrap_ci(y, p, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    v = [f1_score(y[i], p[i], average="macro") for i in (rng.integers(0, n, n) for _ in range(B))]
    return [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])]


def report(y, p):
    return {"macro_f1": round(float(f1_score(y, p, average="macro")), 3), "ci95": bootstrap_ci(y, p),
            "hoax_precision": round(float(precision_score(y, p, zero_division=0)), 3),
            "hoax_recall": round(float(recall_score(y, p)), 3), "accuracy": round(float(accuracy_score(y, p)), 3)}


def lr_text():
    # TF-IDF (word 1-2gram) + logistic regression
    return make_pipeline(TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))


def lr_dom():
    # char 3-5gram TF-IDF + logistic regression
    return make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))


def run_experiments(df: pd.DataFrame):
    y = df.y.to_numpy()
    txt, topics, doms = (df[c].to_numpy(dtype=object) for c in ("text", "topic", "domain"))
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    gkf = GroupKFold(5)
    res = {"majority_all_valid": report(y, np.zeros_like(y))}
    res["tfidf_lr__random_5fold"] = report(y, cross_val_predict(lr_text(), txt, y, cv=skf))
    res["tfidf_lr__cluster_grouped_5fold"] = report(
        y, cross_val_predict(lr_text(), txt, y, cv=gkf, groups=df.nearidup_cluster.to_numpy()))
    res["tfidf_lr__leave_one_topic_out"] = report(
        y, cross_val_predict(lr_text(), txt, y, cv=LeaveOneGroupOut(), groups=topics))
    res["domain_only__random_5fold"] = report(y, cross_val_predict(lr_dom(), doms, y, cv=skf))
    res["domain_only__leave_one_topic_out"] = report(
        y, cross_val_predict(lr_dom(), doms, y, cv=LeaveOneGroupOut(), groups=topics))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-manifest", type=Path, default=Path("data/manifests/mendeley_manifest.csv"))
    ap.add_argument("--out-results", type=Path, default=Path("results/mendeley_baseline.json"))
    ap.add_argument("--skip-experiments", action="store_true")
    a = ap.parse_args()

    df, audit = build_manifest(a.data_dir)
    cols = [c for c in df.columns if c != "text"]                  
    a.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(a.out_manifest, index=False)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    out = {"audit": audit}
    if not a.skip_experiments:
        out["baselines"] = run_experiments(df)
        for k, v in out["baselines"].items():
            print(f"{k:36s} macroF1 {v['macro_f1']:.3f} CI{v['ci95']} | hoax P {v['hoax_precision']:.2f} R {v['hoax_recall']:.2f}")
    a.out_results.parent.mkdir(parents=True, exist_ok=True)
    a.out_results.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nmanifest -> {a.out_manifest}\nresults  -> {a.out_results}")


if __name__ == "__main__":
    main()
