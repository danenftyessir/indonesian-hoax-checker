# Template anotasi task 3 (keputusan 0007): 300 pasangan stance dan 300 kueri retrieval,
# dibangkitkan dari basis klaim yang sudah di-split, lalu diisi/diverifikasi manual.
#
#   python -m indonesian_hoax_checker.data.claim_verification_templates \
#       data/processed/refuted_claims/claims.jsonl \
#       --out-dir data/annotations/claim_verification \
#       [--valid-claims data/short_claims.jsonl]
#
# Keluaran (JSONL, satu baris per item, kolom kosong = diisi manusia):
#   stance_pairs.jsonl       100 supports (hypothesis parafrasa, kosong), 100 refutes (draf dari
#                            fact_sentence, wajib diedit), 100 unrelated (otomatis, diverifikasi)
#   retrieval_queries.jsonl  100 original (judul asli, gold terisi), 100 paraphrase (query kosong),
#                            100 no_match (dari --valid-claims bila ada, selain itu kosong)
# Semua premise/gold diambil dari split `test` supaya evaluasi akhir tidak bocor ke dev/train;
# ganti dengan --split dev untuk set tuning.

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from indonesian_hoax_checker.retrieval.corpus import load_claims, read_jsonl, write_jsonl


def _by_cluster(claims: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for c in claims:
        out.setdefault(c.get("cluster_id", -1), []).append(c)
    return out


def sample_one_per_cluster(claims: list[dict[str, Any]], n: int, rng: random.Random) -> list[dict[str, Any]]:
    """Ambil maksimal satu klaim per klaster supaya item anotasi tidak saling duplikat."""
    clusters = list(_by_cluster(claims).values())
    rng.shuffle(clusters)
    return [rng.choice(members) for members in clusters[:n]]


def _shares_token(a: str, b: str) -> bool:
    ta = {t for t in a.lower().split() if len(t) > 4}
    tb = {t for t in b.lower().split() if len(t) > 4}
    return bool(ta & tb)


def build_stance_pairs(
    claims: list[dict[str, Any]], n_per_label: int = 100, seed: int = 13
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    base = sample_one_per_cluster(claims, 3 * n_per_label, rng)
    if len(base) < 3 * n_per_label:
        raise ValueError(f"butuh {3 * n_per_label} klaster berbeda, hanya ada {len(base)}")
    sup, ref, unr = base[:n_per_label], base[n_per_label : 2 * n_per_label], base[2 * n_per_label :]
    rows = []
    for i, c in enumerate(sup):
        rows.append(_pair_row(f"team-sup-{i:03d}", c, "", "supports",
                              "Tulis parafrasa klaim (kata berbeda, isi sama)."))
    for i, c in enumerate(ref):
        rows.append(_pair_row(f"team-ref-{i:03d}", c, c.get("fact_sentence") or "", "refutes",
                              "Draf dari kalimat 'Faktanya'; ubah menjadi kalimat yang menyangkal klaim."))
    # unrelated: pasangkan dengan klaim dari klaster lain; utamakan yang berbagi satu kata panjang
    # (entitas/topik sama) supaya kasusnya tidak terlalu mudah
    pool = [c for c in claims if c["cluster_id"] not in {x["cluster_id"] for x in unr}]
    for i, c in enumerate(unr):
        hard = [p for p in pool if _shares_token(c["claim"], p["claim"])]
        other = rng.choice(hard) if hard and rng.random() < 0.7 else rng.choice(pool)
        rows.append(_pair_row(f"team-unr-{i:03d}", c, other["claim"], "unrelated",
                              f"Otomatis dari klaim {other['claim_id']}; cek bukan parafrasa/sangkalan.",
                              hypothesis_claim_id=other["claim_id"]))
    return rows


def _pair_row(pair_id: str, c: dict[str, Any], hypothesis: str, label: str, note: str, **extra) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "premise": c["claim"],
        "premise_claim_id": c["claim_id"],
        "premise_url": c.get("url", ""),
        "hypothesis": hypothesis,
        "label": label,            # label target; kosongkan bila annotator menilai label lain
        "annotators": [],          # isi kode anggota, mis. ["danendra", "dzaky"]
        "status": "draft",         # draft -> annotated -> verified
        "instruction": note,
        **extra,
    }


def build_retrieval_queries(
    claims: list[dict[str, Any]],
    valid_claims: list[dict[str, Any]] | None,
    n_per_type: int = 100,
    seed: int = 13,
    prefer_short: bool = False,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    clusters = _by_cluster(claims)
    base = sample_one_per_cluster(claims, 2 * n_per_type, rng)
    if len(base) < 2 * n_per_type:
        raise ValueError(f"butuh {2 * n_per_type} klaster berbeda, hanya ada {len(base)}")
    rows = []
    for i, c in enumerate(base[:n_per_type]):
        rows.append({
            "query_id": f"q-orig-{i:03d}", "query": c["claim"], "type": "original",
            "gold_claim_ids": [m["claim_id"] for m in clusters[c["cluster_id"]]],
            "source_claim_id": c["claim_id"], "status": "auto",
        })
    for i, c in enumerate(base[n_per_type:]):
        rows.append({
            "query_id": f"q-para-{i:03d}", "query": "", "type": "paraphrase",
            "gold_claim_ids": [m["claim_id"] for m in clusters[c["cluster_id"]]],
            "source_claim_id": c["claim_id"], "source_claim": c["claim"], "status": "draft",
            "instruction": "Tulis ulang source_claim dengan kata berbeda (gaya pesan berantai boleh).",
        })
    valid = [v for v in (valid_claims or []) if v.get("label", "valid") == "valid"]
    if prefer_short:
        # Kueri no_match jauh lebih panjang dari judul Komdigi (median 19 vs 8 kata); skor BM25 naik
        # dengan panjang kueri, jadi ambil yang terpendek supaya ambang tolak tidak belajar panjang.
        valid.sort(key=lambda v: (v.get("n_words") or len(v["text"].split()), str(v.get("id", ""))))
    else:
        rng.shuffle(valid)
    for i in range(n_per_type):
        v = valid[i] if i < len(valid) else None
        rows.append({
            "query_id": f"q-nomatch-{i:03d}", "query": " ".join(v["text"].split()) if v else "",
            "type": "no_match",
            "gold_claim_ids": [], "source_id": v.get("id") if v else None,
            "status": "auto" if v else "draft",
            "instruction": "Klaim/berita yang TIDAK ada di basis Komdigi; cek manual dengan BM25 top-10.",
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Template anotasi stance dan kueri retrieval.")
    p.add_argument("claims", help="data/processed/refuted_claims/claims.jsonl (sudah ada split)")
    p.add_argument("--out-dir", default="data/annotations/claim_verification")
    p.add_argument("--split", default="test")
    p.add_argument("--valid-claims", default=None, help="data/short_claims.jsonl untuk kueri no_match")
    p.add_argument("--n-per-label", type=int, default=100)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--overwrite", action="store_true", help="timpa file yang sudah ada (anotasi hilang!)")
    p.add_argument("--prefer-short-no-match", action="store_true",
                   help="ambil kueri no_match terpendek (n_words) alih-alih acak; mengurangi confound panjang")
    args = p.parse_args(argv)

    claims = [c for c in load_claims(args.claims) if c.get("split") == args.split]
    valid = list(read_jsonl(args.valid_claims)) if args.valid_claims else None
    out = Path(args.out_dir)
    targets = {
        out / "stance_pairs.jsonl": lambda: build_stance_pairs(claims, args.n_per_label, args.seed),
        out / "retrieval_queries.jsonl": lambda: build_retrieval_queries(
            claims, valid, args.n_per_label, args.seed, args.prefer_short_no_match),
    }
    written = {}
    for path, fn in targets.items():
        if path.exists() and not args.overwrite:
            print(f"lewati {path}: sudah ada (pakai --overwrite untuk menimpa)")
            continue
        written[path.as_posix()] = write_jsonl(fn(), path)
    print(json.dumps({"split": args.split, "n_claims_in_split": len(claims), "written": written},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
