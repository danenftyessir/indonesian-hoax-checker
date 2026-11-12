# Cek apakah kueri `no_match` tidak punya referensi dari klaim Komdigi

import argparse, json, re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PREFIX_RE = re.compile(r"^\s*[\[\(]\s*(?:HOAKS|HOAX|DISINFORMASI|PENIPUAN|SALAH)\s*[\]\)]\s*[:\-–—]?\s*", re.I)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--komdigi-csv", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=Path("results/no_match_check.json"))
    a = ap.parse_args()

    q = [json.loads(l) for l in a.queries.read_text(encoding="utf-8").splitlines() if l.strip()]
    titles = pd.read_csv(a.komdigi_csv).title.fillna("").astype(str).map(lambda s: PREFIX_RE.sub("", s).strip())
    titles = titles[titles.str.len() > 0].reset_index(drop=True)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(list(titles) + [r["text"] for r in q])
    S = cosine_similarity(vec.transform([r["text"] for r in q]), vec.transform(titles))
    best, arg = S.max(1), S.argmax(1)

    flagged = [(q[i]["id"], round(float(best[i]), 3)) for i in np.argsort(-best) if best[i] >= a.threshold]
    print(f"{len(q)} kueri vs {len(titles)} judul Komdigi | cosine maks: median {np.median(best):.3f}, "
          f"p95 {np.percentile(best, 95):.3f}, maks {best.max():.3f}")
    print(f"Ditandai (>= {a.threshold}): {len(flagged)}")
    for qi in np.argsort(-best)[:10]:
        print(f"  {q[qi]['id']} {best[qi]:.3f}\n    kueri : {q[qi]['text'][:110]}\n    judul : {titles[arg[qi]][:110]}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"threshold": a.threshold, "n_queries": len(q), "n_titles": int(len(titles)),
                                 "flagged": flagged, "max_cosine_by_id": {q[i]["id"]: round(float(best[i]), 3) for i in range(len(q))}},
                                indent=2, ensure_ascii=False))
    print(f"\nhasil -> {a.out}")


if __name__ == "__main__":
    main()
