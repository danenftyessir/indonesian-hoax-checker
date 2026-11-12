# Near-duplicate clustering dan split waktu untuk basis klaim Komdigi
# (docs/decisions/0008-near-duplicate-dan-pembekuan-test-set.md). Pemilik: Danendra.
#
#   python -m indonesian_hoax_checker.evaluation.splits \
#       data/interim/refuted_claims/claims_raw.jsonl \
#       --output data/processed/refuted_claims/claims.jsonl \
#       --review-dir data/interim/refuted_claims \
#       --splits-dir data/splits --kaggle-version 396
#
# Aturan:
# - cosine TF-IDF char_wb 3–5-gram antar `claim` (judul tanpa prefiks); pasangan >= 0,8 digabung
#   (union-find) menjadi satu klaster;
# - seluruh klaster masuk split yang sama, ditentukan tanggal TERBARU anggotanya;
# - test = tanggal >= test_start (1 Sep 2026); dev = [dev_start, test_start); sisanya train.
#   dev_start adalah usulan (default 1 Jul 2026), belum ada di lembar keputusan.

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from indonesian_hoax_checker.retrieval.corpus import load_claims, write_jsonl

DEFAULT_THRESHOLD = 0.8
DEFAULT_TEST_START = "2026-09-01"
DEFAULT_DEV_START = "2026-07-01"
NGRAM_RANGE = (3, 5)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def char_tfidf_similarity(texts: list[str]) -> np.ndarray:
    """Matriks cosine (n x n) TF-IDF char_wb 3–5-gram. n ~ 3.000 masih muat di memori (~70 MB)."""
    if not texts:
        return np.zeros((0, 0))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=NGRAM_RANGE, lowercase=True)
    x = vec.fit_transform(texts)
    sim = (x @ x.T).toarray()
    np.fill_diagonal(sim, 0.0)
    return sim


def similar_pairs(sim: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    """Pasangan (i < j) dengan cosine >= threshold, terurut skor menurun."""
    iu, ju = np.triu_indices(sim.shape[0], k=1)
    mask = sim[iu, ju] >= threshold
    pairs = [(int(i), int(j), float(sim[i, j])) for i, j in zip(iu[mask], ju[mask])]
    pairs.sort(key=lambda p: -p[2])
    return pairs


def cluster_near_duplicates(
    texts: list[str], threshold: float = DEFAULT_THRESHOLD
) -> tuple[list[int], list[tuple[int, int, float]], np.ndarray]:
    """Kembalikan (cluster_id per teks, pasangan >= threshold, matriks cosine).
    cluster_id = indeks terkecil anggota klaster, lalu dipadatkan menjadi 0..k-1 urut kemunculan."""
    sim = char_tfidf_similarity(texts)
    pairs = similar_pairs(sim, threshold)
    uf = UnionFind(len(texts))
    for i, j, _ in pairs:
        uf.union(i, j)
    roots = [uf.find(i) for i in range(len(texts))]
    compact: dict[int, int] = {}
    cluster_ids = []
    for r in roots:
        if r not in compact:
            compact[r] = len(compact)
        cluster_ids.append(compact[r])
    return cluster_ids, pairs, sim


def sample_pairs_for_review(
    sim: np.ndarray,
    texts: list[str],
    ids: list[str],
    threshold: float = DEFAULT_THRESHOLD,
    n_above: int = 50,
    n_below: int = 20,
    seed: int = 13,
) -> list[dict[str, Any]]:
    """Sampel pasangan untuk pemeriksaan manual ambang (0008):
    n_above pasangan >= threshold (separuh dari zona batas [threshold, threshold+0.1)),
    n_below pasangan di [threshold-0.1, threshold) untuk mengecek duplikat yang lolos."""
    rng = random.Random(seed)
    iu, ju = np.triu_indices(sim.shape[0], k=1)
    scores = sim[iu, ju]
    taken: set[int] = set()

    def pick(mask: np.ndarray, n: int, zone: str) -> list[dict[str, Any]]:
        idx = [k for k in np.flatnonzero(mask).tolist() if k not in taken]
        rng.shuffle(idx)
        taken.update(idx[:n])
        return [
            {
                "zone": zone,
                "similarity": round(float(scores[k]), 4),
                "id_a": ids[iu[k]],
                "claim_a": texts[iu[k]],
                "id_b": ids[ju[k]],
                "claim_b": texts[ju[k]],
                "is_duplicate": "",  # diisi manual: y / n
                "note": "",
            }
            for k in idx[:n]
        ]

    border = (scores >= threshold) & (scores < threshold + 0.1)
    high = scores >= threshold + 0.1
    below = (scores >= threshold - 0.1) & (scores < threshold)
    half = n_above // 2
    rows = pick(border, half, "borderline") + pick(high, n_above - half, "high")
    if len(rows) < n_above:  # salah satu zona kurang, isi dari zona lain
        rows += pick(scores >= threshold, n_above - len(rows), "fill")
    rows += pick(below, n_below, "below")
    rows.sort(key=lambda r: -r["similarity"])
    return rows


def _parse_date(value: str) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def assign_time_split(
    claims: list[dict[str, Any]],
    cluster_ids: list[int],
    test_start: str = DEFAULT_TEST_START,
    dev_start: str = DEFAULT_DEV_START,
) -> list[str]:
    """Split per klaster berdasarkan tanggal terbaru anggotanya. Klaim tanpa tanggal ikut train."""
    t_test = datetime.fromisoformat(test_start).replace(tzinfo=timezone.utc)
    t_dev = datetime.fromisoformat(dev_start).replace(tzinfo=timezone.utc)
    if t_dev >= t_test:
        raise ValueError("dev_start harus lebih awal dari test_start")

    newest: dict[int, datetime] = {}
    for c, cid in zip(claims, cluster_ids):
        dt = _parse_date(c.get("published_at", ""))
        if dt is not None and (cid not in newest or dt > newest[cid]):
            newest[cid] = dt

    splits = []
    for cid in cluster_ids:
        dt = newest.get(cid)
        if dt is None or dt < t_dev:
            splits.append("train")
        elif dt < t_test:
            splits.append("dev")
        else:
            splits.append("test")
    return splits


def sha256_of(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    files: list[str | Path], out: str | Path, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    entries = []
    for f in files:
        f = Path(f)
        with f.open("rb") as fh:
            n_lines = sum(1 for _ in fh)
        entries.append(
            {"file": f.as_posix(), "lines": n_lines, "sha256": sha256_of(f), "bytes": f.stat().st_size}
        )
    manifest = {
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": entries,
        **(extra or {}),
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": .gitattributes memaksa eol=lf, dan hash di manifest harus sama di semua OS
    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Selalu LF (bukan CRLF bawaan csv di Windows): file ini di-hash di manifest.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8", newline="\n")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Klaster near-duplicate + split waktu basis klaim.")
    p.add_argument("claims", help="claims_raw.jsonl dari retrieval.corpus")
    p.add_argument("--output", default="data/processed/refuted_claims/claims.jsonl")
    p.add_argument("--review-dir", default="data/interim/refuted_claims")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--test-start", default=DEFAULT_TEST_START)
    p.add_argument("--dev-start", default=DEFAULT_DEV_START)
    p.add_argument("--kaggle-version", default="", help="nomor versi dataset Kaggle saat unduh")
    args = p.parse_args(argv)

    claims = load_claims(args.claims)
    texts = [c["claim"] for c in claims]
    ids = [c["claim_id"] for c in claims]
    cluster_ids, pairs, sim = cluster_near_duplicates(texts, args.threshold)
    splits = assign_time_split(claims, cluster_ids, args.test_start, args.dev_start)
    for c, cid, sp in zip(claims, cluster_ids, splits):
        c["cluster_id"] = cid
        c["split"] = sp

    write_jsonl(claims, args.output)

    review_dir = Path(args.review_dir)
    _write_csv(
        [
            {"id_a": ids[i], "id_b": ids[j], "similarity": round(s, 4), "claim_a": texts[i], "claim_b": texts[j]}
            for i, j, s in pairs
        ],
        review_dir / "near_duplicate_pairs.csv",
    )
    _write_csv(
        sample_pairs_for_review(sim, texts, ids, args.threshold),
        review_dir / "near_duplicate_review.csv",
    )

    splits_dir = Path(args.splits_dir)
    split_csv = splits_dir / "refuted_claims_split.csv"
    _write_csv(
        [
            {"claim_id": c["claim_id"], "cluster_id": c["cluster_id"], "split": c["split"], "published_at": c["published_at"]}
            for c in claims
        ],
        split_csv,
    )

    sizes: dict[int, int] = {}
    for cid in cluster_ids:
        sizes[cid] = sizes.get(cid, 0) + 1
    n_clusters = len(sizes)
    multi = sum(1 for n in sizes.values() if n > 1)
    largest = sorted(sizes.values(), reverse=True)[:5]
    stats = {
        "n_claims": len(claims),
        "threshold": args.threshold,
        "n_pairs_above_threshold": len(pairs),
        "n_clusters": n_clusters,
        "n_clusters_multi": multi,
        "n_claims_in_multi_clusters": len(claims) - (n_clusters - multi),
        # klaster raksasa = tanda ambang terlalu rendah atau rantai transitif; cek saat review 50 pasangan
        "largest_cluster_sizes": largest,
        "split_counts": {s: splits.count(s) for s in ("train", "dev", "test")},
        "test_start": args.test_start,
        "dev_start": args.dev_start,
        "kaggle_version": args.kaggle_version,
        "source_claims": Path(args.claims).as_posix(),
    }
    write_manifest([args.output, split_csv], splits_dir / "manifest.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
