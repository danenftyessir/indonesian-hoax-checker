# Basis klaim terbantah: dari CSV Kaggle (ireddragonicy/indonesian-hoax-news-dataset) ke
# claims JSONL sesuai docs/data_dictionary.md. Pemilik: Danendra (task 3a).
#
#   python -m indonesian_hoax_checker.retrieval.corpus \
#       data/raw/refuted_claims/komdigi_hoaks.csv data/interim/refuted_claims/claims_raw.jsonl
#
# Tahap ini belum memberi cluster_id dan split; itu dikerjakan evaluation/splits.py.

from __future__ import annotations

import argparse
import html
import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = ("id", "title", "published_at")

# "[HOAKS] ...", "[DISINFORMASI] ...", "[HOAKS/DISINFORMASI] ...", "[SALAH] ..." dan varian lain
_PREFIX_RE = re.compile(r"^\s*(?:\[([^\]]{1,40})\]\s*[:\-–]?\s*)+")
_TAG_RE = re.compile(r"\[([^\]]{1,40})\]")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“\[(])")

# Kalimat klaim: "Beredar ...", "Telah beredar ...", "Viral di media sosial ...",
# "Di media sosial beredar ...", "Pada tanggal 9 Januari 2025, telah beredar ...". Kata kunci harus
# muncul di 60 karakter pertama kalimat supaya kalimat klarifikasi yang kebetulan memuat "beredar"
# di tengah tidak terpilih. "bereda" = typo yang ada di data Komdigi.
_CLAIM_SENT_RE = re.compile(r"^.{0,60}?\b(?:beredar|bereda|viral|tersebar)\b", re.IGNORECASE)
_CLAIM_FALLBACK_RE = re.compile(
    r"\b(?:mengeklaim|mengklaim|menyebutkan|menarasikan|diklaim)\b", re.IGNORECASE
)
# Kalimat bantahan: "Faktanya, ...", "Setelah ditelusuri, ...", "Penjelasan: ..."
_FACT_SENT_RE = re.compile(
    r"^(?:faktanya|setelah ditelusuri|setelah dilakukan penelusuran|berdasarkan penelusuran)",
    re.IGNORECASE,
)


def normalize_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def clean_html(raw: str | None) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    # pisahkan blok supaya kalimat tidak menempel setelah tag dibuang
    text = re.sub(r"</?(?:p|div|br|li|h\d|tr)[^>]*>", "\n", raw, flags=re.IGNORECASE)
    text = _HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)  # "berantai ." -> "berantai." setelah tag inline dibuang
    return normalize_whitespace(text)


def strip_prefix(title: str) -> tuple[str, list[str]]:
    """'[HOAKS] Pemerintah ...' -> ('Pemerintah ...', ['HOAKS']). Tag dikembalikan huruf besar."""
    title = normalize_whitespace(html.unescape(title or ""))
    m = _PREFIX_RE.match(title)
    if not m:
        return title, []
    tags = [t.strip().upper() for t in _TAG_RE.findall(m.group(0))]
    return title[m.end() :].strip(" :-–"), tags


# "Penjelasan:", "Penjelasan :", "KATEGORI: HOAKS Penjelasan : ..." di awal kalimat
_SECTION_LABEL_RE = re.compile(
    r"^(?:(?:kategori\s*:\s*(?:hoaks|disinformasi|misinformasi)?\s*)?(?:penjelasan|narasi|klarifikasi)\s*:\s*)+",
    re.IGNORECASE,
)


def split_sentences(text: str) -> list[str]:
    """Pisah kalimat; label bagian artikel Komdigi ('Penjelasan:', 'Narasi:') di awal kalimat dibuang."""
    text = normalize_whitespace(text)
    if not text:
        return []
    out = []
    for s in _SENT_SPLIT_RE.split(text):
        s = _SECTION_LABEL_RE.sub("", s.strip())
        if s:
            out.append(s)
    return out


def extract_claim_sentence(body: str) -> str | None:
    """Kalimat 'Beredar ... yang mengeklaim ...' pertama; fallback kalimat pertama yang memuat
    'mengeklaim'/'menyebutkan'; None bila tidak ada."""
    sentences = split_sentences(body)
    for s in sentences:
        if _CLAIM_SENT_RE.match(s):
            return s
    for s in sentences:
        if _CLAIM_FALLBACK_RE.search(s):
            return s
    return None


def extract_fact_sentence(body: str) -> str | None:
    """Kalimat bantahan pertama ('Faktanya, ...'); dipakai sebagai draf hypothesis `refutes`."""
    for s in split_sentences(body):
        if _FACT_SENT_RE.match(s):
            return s
    return None


def _parse_tags(*values: Any) -> list[str]:
    out: list[str] = []
    for v in values:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                try:
                    parsed = json.loads(v)
                    v = parsed if isinstance(parsed, list) else [v]
                except json.JSONDecodeError:
                    v = [p.strip(" '\"") for p in v.strip("[]").split(",")]
            else:
                v = re.split(r"[;,|]", v)
        for t in v:
            t = str(t).strip().lower()
            if t and t not in out:
                out.append(t)
    return out


def _iso_date(value: Any) -> str:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return ""
    return ts.isoformat()


def load_kaggle_csv(path: str | Path) -> pd.DataFrame:
    # utf-8-sig: file Kaggle diawali BOM, tanpa ini kolom pertama terbaca "﻿id"
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ada di {path}: {missing}. Kolom tersedia: {list(df.columns)}"
        )
    return df


def row_to_claim(row: dict[str, Any]) -> dict[str, Any]:
    claim, tags = strip_prefix(row.get("title", ""))
    body = normalize_whitespace(row.get("body_text", "") or "") or clean_html(
        row.get("body_html", "")
    )
    return {
        "claim_id": str(row["id"]),
        "claim": claim,
        "title_tags": tags,
        "claim_sentence": extract_claim_sentence(body),
        "fact_sentence": extract_fact_sentence(body),
        "clarification": body,
        "url": row.get("url", "") or "",
        "published_at": _iso_date(row.get("published_at")),
        "tags": _parse_tags(row.get("tags"), row.get("topics")),
    }


def build_claims(df: pd.DataFrame) -> list[dict[str, Any]]:
    claims = [row_to_claim(r) for r in df.to_dict("records")]
    # buang judul kosong (setelah prefiks dibuang) dan id ganda; id ganda = unduhan Kaggle bermasalah
    seen: set[str] = set()
    out = []
    for c in claims:
        if not c["claim"] or c["claim_id"] in seen:
            continue
        seen.add(c["claim_id"])
        out.append(c)
    return out


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_claims(path: str | Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def summarize(claims: list[dict[str, Any]]) -> dict[str, Any]:
    dates = sorted(c["published_at"] for c in claims if c["published_at"])
    return {
        "n_claims": len(claims),
        "with_claim_sentence": sum(1 for c in claims if c["claim_sentence"]),
        "with_fact_sentence": sum(1 for c in claims if c["fact_sentence"]),
        "with_tags": sum(1 for c in claims if c["tags"]),
        "empty_clarification": sum(1 for c in claims if not c["clarification"]),
        "date_min": dates[0] if dates else None,
        "date_max": dates[-1] if dates else None,
        "title_tags": _count(t for c in claims for t in c["title_tags"]),
    }


def _count(items: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="CSV Kaggle Komdigi -> claims JSONL (tanpa cluster/split)."
    )
    p.add_argument("csv", help="data/raw/refuted_claims/komdigi_hoaks.csv")
    p.add_argument("output", help="data/interim/refuted_claims/claims_raw.jsonl")
    args = p.parse_args(argv)

    df = load_kaggle_csv(args.csv)
    claims = build_claims(df)
    n = write_jsonl(claims, args.output)
    print(
        json.dumps(
            {"input_rows": len(df), "written": n, **summarize(claims)}, ensure_ascii=False, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
