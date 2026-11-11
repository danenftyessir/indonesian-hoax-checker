# CLI demo. Untuk sekarang hanya membaca argumen; pipeline dipasang di fase 3.
# Jangan import model di sini supaya --help tetap cepat.

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from indonesian_hoax_checker import __version__

TASKS = ("all", "ner", "classify", "retrieve", "stance")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hoax-checker",
        description=(
            "Pemeriksa hoaks bahasa Indonesia. Alat bantu penyaringan awal, bukan pemutus akhir."
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
        help="Jalankan seluruh pipeline (all) atau satu task saja.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Tampilkan keluaran tiap tahap (entitas, label dan skor, bukti dan stance).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def run(text: str, task: str, verbose: bool) -> int:
    # Sementara: nanti diganti dengan panggilan ke pipeline/orchestrator.py
    print(f"[{task}] pipeline belum diimplementasikan. Input diterima: {text!r}", file=sys.stderr)
    return 1


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
