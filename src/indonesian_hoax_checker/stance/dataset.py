# Konversi data stance ke format docs/data_dictionary.md "Stance":
#   pair_id, premise, hypothesis, label, source, (annotators), has_negation
#
#   python -m indonesian_hoax_checker.stance.dataset indonli \
#       data/raw/stance/indonli/train.jsonl data/processed/stance/indonli_train.jsonl
#   python -m indonesian_hoax_checker.stance.dataset brin \
#       data/raw/stance/brin_fakenews_mafindo.csv data/processed/stance/brin_refutes.jsonl
#
# Catatan bias negasi IndoNLI: 53% hipotesis contradiction memuat tidak/bukan/belum. Kolom
# has_negation disimpan supaya metrik bisa dipisah (docs/experiment_protocol.md task 3b).

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from indonesian_hoax_checker.retrieval.corpus import (
    normalize_whitespace,
    read_jsonl,
    strip_prefix,
    write_jsonl,
)
from indonesian_hoax_checker.stance.labels import STANCE_LABELS, to_stance

NEGATION_RE = re.compile(r"\b(?:tidak|bukan|belum|tak|nggak|gak|enggak|tanpa)\b", re.IGNORECASE)


def has_negation(text: str) -> bool:
    return bool(NEGATION_RE.search(text or ""))


def make_pair(
    pair_id: str,
    premise: str,
    hypothesis: str,
    label: str,
    source: str,
    annotators: list[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    premise = normalize_whitespace(premise)
    hypothesis = normalize_whitespace(hypothesis)
    row = {
        "pair_id": pair_id,
        "premise": premise,
        "hypothesis": hypothesis,
        "label": to_stance(label),
        "source": source,
        "has_negation": has_negation(hypothesis),
    }
    if annotators is not None:
        row["annotators"] = annotators
    row.update(extra)
    return row


def convert_indonli(path: str | Path, split_name: str = "") -> list[dict[str, Any]]:
    """IndoNLI JSONL (premise, hypothesis, label e/n/c, annotator_type, sentence_size, pair_id?)."""
    rows = []
    for i, r in enumerate(read_jsonl(path)):
        pid = str(r.get("pair_id", r.get("id", i)))
        rows.append(
            make_pair(
                f"indonli-{split_name or 'x'}-{pid}",
                r["premise"],
                r["hypothesis"],
                r["label"],
                "indonli",
                annotator_type=r.get("annotator_type"),
            )
        )
    return rows


def convert_brin(path: str | Path, min_fact_chars: int = 20) -> list[dict[str, Any]]:
    """nlp-brin-id/fakenews-mafindo: (Title, Fact) berlabel `refutes` untuk baris HOAX.
    Title dibersihkan dari prefiks [SALAH]/[HOAKS]. Baris NON-HOAX dan Fact kosong dilewati.
    Pasangan `supports` (parafrasa judul) TIDAK dibuat di sini: sumber parafrasa belum diputuskan."""
    path = Path(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False) if path.suffix == ".csv" else pd.read_json(
        path, lines=path.suffix == ".jsonl", dtype=str
    )
    cols = {c.lower(): c for c in df.columns}
    need = ["title", "fact"]
    missing = [c for c in need if c not in cols]
    if missing:
        raise ValueError(f"Kolom {missing} tidak ada di {path}; kolom: {list(df.columns)}")
    label_col = cols.get("label") or cols.get("classification_binary") or cols.get("is_hoax")
    rows = []
    for i, r in enumerate(df.to_dict("records")):
        if label_col and str(r[label_col]).strip() in {"0", "NON-HOAX", "non-hoax", "False"}:
            continue
        title, _ = strip_prefix(str(r[cols["title"]]))
        fact = normalize_whitespace(str(r[cols["fact"]]))
        if not title or len(fact) < min_fact_chars:
            continue
        rows.append(
            make_pair(
                f"brin-{r.get(cols.get('id', ''), i) or i}",
                title,
                fact,
                "refutes",
                "brin_fact",
                hoax_type=r.get(cols.get("classification", ""), "") or None,
            )
        )
    return rows


def convert_team(path: str | Path) -> list[dict[str, Any]]:
    """Anotasi tim (data/annotations/claim_verification/stance_pairs.jsonl): baris yang sudah
    berlabel final (`label` terisi) saja; validasi label dan dua annotator."""
    rows = []
    for r in read_jsonl(path):
        if not r.get("label"):
            continue
        ann = r.get("annotators") or []
        if len(ann) < 2:
            raise ValueError(f"{r.get('pair_id')}: butuh dua annotator, ada {ann}")
        rows.append(
            make_pair(r["pair_id"], r["premise"], r["hypothesis"], r["label"], "team", ann,
                      premise_claim_id=r.get("premise_claim_id"))
        )
    return rows


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {lab: sum(1 for r in rows if r["label"] == lab) for lab in STANCE_LABELS}


def negation_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    out = {}
    for lab in STANCE_LABELS:
        sub = [r for r in rows if r["label"] == lab]
        out[lab] = round(sum(r["has_negation"] for r in sub) / len(sub), 3) if sub else float("nan")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Konversi data stance ke JSONL proyek.")
    p.add_argument("source", choices=["indonli", "brin", "team"])
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--split-name", default="", help="untuk indonli: train/val/test_lay/test_expert")
    args = p.parse_args(argv)
    if args.source == "indonli":
        rows = convert_indonli(args.input, args.split_name)
    elif args.source == "brin":
        rows = convert_brin(args.input)
    else:
        rows = convert_team(args.input)
    n = write_jsonl(rows, args.output)
    print(json.dumps({"written": n, "labels": label_counts(rows), "negation_rate": negation_stats(rows)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
