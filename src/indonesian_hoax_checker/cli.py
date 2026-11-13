# CLI demo utk memanggil pipeline/orchestrator.py

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from indonesian_hoax_checker import __version__
from indonesian_hoax_checker.pipeline.orchestrator import TASKS, PipelineResult, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hoax-checker",
        description=(
            "Pemeriksa hoaks bahasa Indonesia. Alat bantu penyaringan awal, bukan pemutus akhir"
        ),
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Klaim atau artikel yang diperiksa. Kosongkan untuk mode interaktif.",
    )
    parser.add_argument(
        "--task",
        choices=TASKS,
        default="all",
        help="Jalankan seluruh pipeline (all) atau satu task saja",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Tampilkan keluaran tiap tahap (entitas, label dan skor, bukti dan stance)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def format_result(res: PipelineResult, verbose: bool) -> str:
    lines: list[str] = []
    stub = lambda name: " [STUB]" if name in res.stubs else "" 

    if res.verdict is not None:
        v = res.verdict
        lines.append(f"Indikasi: {v['indication']}{' [STUB]' if res.stubs else ''}")
        lines.append(f"  {v['explanation']}")
    want = lambda t: verbose or res.task == t  
    if res.ner is not None and want("ner"):
        ents = ", ".join(f"{e['text']}/{e['label']}" for e in res.ner["entities"]) or "(tidak ada)"
        lines.append(f"NER{stub('ner')}: entitas={ents}; klaim inti={res.ner['core_claim']!r}")
    if res.classification is not None and want("classify"):
        c = res.classification
        lines.append(f"Klasifikasi{stub('classify')}: {c['label']} (skor {c['confidence']:.2f})")
    if res.retrieval is not None and want("retrieve"):
        r = res.retrieval
        if r["matched_claim"] is None:
            lines.append(f"Retrieval{stub('retrieve')}: tidak ada klaim terbantah yang mirip")
        else:
            lines.append(
                f"Retrieval{stub('retrieve')}: {r['matched_claim']!r} "
                f"(skor {r['similarity_score']:.2f}, {r.get('scorer', '?')}); klarifikasi: {r['clarification']}"
            )
    if res.task in ("all", "stance") and want("stance"):
        if res.stance is None:
            lines.append(f"Stance{stub('stance')}: dilewati (tidak ada klaim terbantah untuk dibandingkan)")
        else:
            lines.append(
                f"Stance{stub('stance')}: {res.stance['label']} (skor {res.stance['confidence']:.2f})"
            )
    if res.stubs:
        lines.append(f"Catatan: tahap {', '.join(res.stubs)} adalah hasil bukan prediksi model")
    return "\n".join(lines)


def run(text: str, task: str, verbose: bool) -> int:
    try:
        res = run_pipeline(text, task)
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 2
    print(format_result(res, verbose))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.text is not None:
        return run(args.text, args.task, args.verbose)

    # Mode interaktif: satu input per baris, baris kosong atau Ctrl+D untuk keluar
    status = 0
    while True:
        try:
            line = input("klaim> ").strip()
        except EOFError:
            break
        if not line:
            break
        status = run(line, args.task, args.verbose)
    return status
