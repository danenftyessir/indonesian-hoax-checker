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
        epilog=(
            'Teks yang diawali tanda "-" harus didahului "--", mis. hoax-checker -- "-5 derajat di Bandung". '
            "Tanpa argumen teks: mode interaktif (satu input per baris; baris kosong atau Ctrl+D/Ctrl+Z untuk keluar)."
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


CLARIFICATION_CHARS = 300  # klarifikasi Komdigi bisa satu paragraf penuh; teks lengkap ada di URL


def _shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " ..."


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
    if res.classification is not None and want("classify") and res.classifier_skipped:
        lines.append("Klasifikasi: dilewati (teks terlalu pendek untuk classifier)")
    elif res.classification is not None and want("classify"):
        c = res.classification
        lines.append(f"Klasifikasi{stub('classify')}: {c['label']} (skor {c['confidence']:.2f}"
                         f"{'' if 'classify' in res.stubs else ', belum dikalibrasi'})")
    if res.retrieval is not None and want("retrieve"):
        r = res.retrieval
        if r["matched_claim"] is None:
            lines.append(f"Retrieval{stub('retrieve')}: tidak ada klaim terbantah yang mirip")
        else:
            lines.append(
                f"Retrieval{stub('retrieve')}: {r['matched_claim']!r} "
                f"(skor {r['similarity_score']:.2f}, {r.get('scorer', '?')}); "
                f"klarifikasi: {_shorten(r['clarification'] or '', CLARIFICATION_CHARS)}"
            )
            url = (r.get("candidates") or [{}])[0].get("url")
            if url:
                lines.append(f"  bantahan resmi: {url}")
    if res.task in ("all", "stance") and want("stance"):
        if res.stance is None:
            lines.append(f"Stance{stub('stance')}: dilewati (tidak ada klaim terbantah untuk dibandingkan)")
        else:
            lines.append(
                f"Stance{stub('stance')}: {res.stance['label']} (skor {res.stance['confidence']:.2f})"
            )
            segment = res.stance.get("segment")
            if segment and segment != res.text:
                lines.append(f"  bagian input yang dinilai: {_shorten(segment, CLARIFICATION_CHARS)!r}")
            if res.stance.get("note"):
                lines.append(f"  catatan: {res.stance['note']}")
    if res.stubs:
        lines.append(f"Catatan: tahap {', '.join(res.stubs)} adalah hasil bukan prediksi model")
    return "\n".join(lines)


def harden_streams() -> None:
    """Jangan sampai CLI berhenti karena encoding konsol: di Windows, keluaran yang di-pipe atau
    dialihkan ke file memakai cp1252 dan gagal pada emoji atau aksara non-Latin. Karakter yang tidak
    bisa ditulis/dibaca diganti, bukan memicu UnicodeEncodeError/UnicodeDecodeError."""
    for stream in (sys.stdout, sys.stderr, sys.stdin):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


def run(text: str, task: str, verbose: bool) -> int:
    try:
        res = run_pipeline(text, task)
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 2
    except Exception as err:  # noqa: BLE001 - satu input bermasalah tidak boleh mematikan mode interaktif
        print(f"Error: input tidak bisa diproses ({type(err).__name__}: {err})", file=sys.stderr)
        return 1
    print(format_result(res, verbose))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    harden_streams()
    args = build_parser().parse_args(argv)
    if args.text is not None:
        return run(args.text, args.task, args.verbose)

    # Mode interaktif: satu input per baris, baris kosong atau Ctrl+D untuk keluar
    status = 0
    while True:
        try:
            # prompt hanya untuk terminal; saat input di-pipe, prompt mengotori keluaran
            line = input("klaim> " if sys.stdin.isatty() else "").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        status = run(line, args.task, args.verbose)
    return status
