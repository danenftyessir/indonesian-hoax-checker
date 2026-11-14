# Jalankan NER (model terpasang di configs/pipeline.yaml) pada data/demo_cases.txt dan tulis tabel hasilnya.
#   python -m indonesian_hoax_checker.ner.demo
# Keluaran: reports/tables/ner_demo_cases.md (entitas dan core_claim per kasus, untuk error analysis).

import argparse
from pathlib import Path

import yaml


# Kembalikan daftar (kategori, teks); "\n" jadi ganti paragraf, "<kosong>" jadi string kosong
def read_demo_cases(path) -> list[tuple[str, str]]:
    cases, category = [], ""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            category = line[3:].strip()
        elif line.strip() and not line.startswith("#"):
            text = "" if line.strip() == "<kosong>" else line.replace("\\n", "\n")
            cases.append((category, text))
    return cases


def cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " / ") or "(kosong)"


def main() -> None:
    from indonesian_hoax_checker.ner.predict import NERPredictor

    parser = argparse.ArgumentParser(description="NER pada kasus demo.")
    parser.add_argument("--cases", default="data/demo_cases.txt")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--output", default="reports/tables/ner_demo_cases.md")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())["ner"]
    predictor = NERPredictor(cfg["checkpoint"], cfg.get("device", "auto"), cfg.get("claim_checkpoint"))
    cases = read_demo_cases(args.cases)
    results = predictor.predict_batch([text for _, text in cases])

    lines = [f"NER pada `{args.cases}` (entitas: {Path(cfg['checkpoint']).name}, "
             f"CLAIM: {Path(cfg['claim_checkpoint']).name}, dengan pasca-proses CLAIM).", "",
             "| Kategori | Teks | Entitas | core_claim |", "|---|---|---|---|"]
    for (category, text), result in zip(cases, results):
        ents = ", ".join(f"{e['text']}/{e['label']}" for e in result["entities"] if e["label"] != "CLAIM")
        lines.append(f"| {category.split('.')[0]} | {cell(text)} | {cell(ents) if ents else '-'} | "
                     f"{cell(result['core_claim']) if result['core_claim'] else '-'} |")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
