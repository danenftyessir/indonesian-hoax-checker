# Tabel dan grafik NER untuk laporan, dibuat dari metrik hasil training (folder outputs/metrics tiap run Drive).
#
#   python -m indonesian_hoax_checker.ner.report \
#       --stage1 <ner_20261112-1801>/outputs/metrics \
#       --rq1 <rq1_ner_20261113-*>/outputs/metrics \
#       --claim-predictions <ner_20261114-*>/outputs/predictions
#
# Keluaran: reports/tables/ner_*.md dan .csv, reports/figures/ner_*.png.
# Angka F1 = entity-level seqeval strict, rata-rata ± simpangan baku 3 seed (13, 42, 2026).

import argparse
import csv
import json
from pathlib import Path

from indonesian_hoax_checker.ner.error_analysis import read_predictions
from indonesian_hoax_checker.ner.labels import get_spans

TABLES = Path("reports/tables")
FIGURES = Path("reports/figures")
# Palet kategorikal tervalidasi (CVD aman); kontras rendah diimbangi label angka di tiap batang
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"

TEST_NAMES = {"test": "Seed test formal", "seed_informal_test": "Tulisan tangan (30)",
              "seed_sim_informal_test": "Informal sintetis", "nergrit_test": "NERGrit test"}


def load(folder: Path, name: str) -> dict:
    return json.loads((folder / f"{name}_summary.json").read_text())


def fmt(v: dict | None) -> str:
    return f"{v['mean']:.3f} ± {v['std']:.3f}" if v else "—"


def score(summary: dict, key: str) -> dict | None:
    if key == "test":
        return summary["test_f1"]
    if key == "dev":
        return summary["dev_f1"]
    return summary.get("extra_f1", {}).get(key) or summary["test_f1_per_type"].get(key)


def write_table(name: str, header: list[str], rows: list[list[str]], note: str) -> None:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    (TABLES / f"{name}.md").write_text(note + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    with open(TABLES / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header] + rows)
    print("tabel", TABLES / f"{name}.md")


def setup_axes(ax, ylabel: str) -> None:
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)


def grouped_bars(path: Path, title: str, groups: list[str], series: list[tuple[str, list[dict | None]]],
                 ylabel: str = "F1 (strict)") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    width = 0.8 / len(series)
    for k, (label, values) in enumerate(series):
        xs = [i - 0.4 + width * (k + 0.5) for i in range(len(groups))]
        means = [v["mean"] if v else 0 for v in values]
        errs = [v["std"] if v else 0 for v in values]
        bars = ax.bar(xs, means, width * 0.92, yerr=errs if any(errs) else None, color=COLORS[k], label=label,
                      error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2})
        for bar, m, e, v in zip(bars, means, errs, values):
            if v:
                # Label di atas garis simpangan baku supaya tidak bertumpuk
                ax.text(bar.get_x() + bar.get_width() / 2, m + e + 0.02, f"{m:.2f}", ha="center", fontsize=7,
                        color=INK)
    ax.set_xticks(range(len(groups)), groups, color=INK)
    setup_axes(ax, ylabel)
    # Legenda di bawah judul, di luar area batang
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(series))
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=24)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("grafik", path)


def strict_overlap(path: Path) -> tuple[float, float]:
    gold_n = pred_n = exact = hit_g = hit_p = 0
    for _, gold, pred in read_predictions(path):
        g, p = get_spans(gold), get_spans(pred)
        gold_n, pred_n = gold_n + len(g), pred_n + len(p)
        exact += len(set(g) & set(p))
        hit_g += sum(any(a[1] < b[2] and b[1] < a[2] for b in p) for a in g)
        hit_p += sum(any(a[1] < b[2] and b[1] < a[2] for b in g) for a in p)

    def f1(prec, rec):
        return 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    return f1(exact / pred_n, exact / gold_n), f1(hit_p / pred_n, hit_g / gold_n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tabel dan grafik NER untuk laporan.")
    parser.add_argument("--stage1", required=True, help="outputs/metrics run baseline + tahap 2 (ner_20261112-1801)")
    parser.add_argument("--rq1", required=True, help="outputs/metrics run RQ1")
    parser.add_argument("--claim-predictions", help="outputs/predictions run model CLAIM terpasang (ner_20261114-*)")
    args = parser.parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    s1, rq1 = Path(args.stage1), Path(args.rq1)

    # 1. Baseline tahap 1 di NERGrit test
    base = {"BiLSTM-CRF": load(s1, "ner_bilstm_crf"), "IndoBERT-base-p1": load(s1, "ner_indobert")}
    types = ["PER", "ORG", "LOC"]
    write_table("ner_baseline", ["Model", "F1", *types],
                [[m, fmt(s["test_f1"]), *(fmt(s["test_f1_per_type"][t]) for t in types)] for m, s in base.items()],
                "Tahap 1: NERGrit test (209 kalimat), label PER/ORG/LOC. IndoNLU melaporkan 69,87 untuk IndoBERT-base "
                "pada split NERGrit mereka (split kami: dev dipisah dari train).")
    grouped_bars(FIGURES / "ner_baseline.png", "Tahap 1: F1 per tipe di NERGrit test",
                 ["Semua", *types], [(m, [s["test_f1"], *(s["test_f1_per_type"][t] for t in types)])
                                     for m, s in base.items()])

    # 2. Tahap 2: model gabungan vs dua layer
    stage2 = {"BiLSTM-CRF gabungan": "ner_bilstm_crf_claim_time", "IndoBERT gabungan": "ner_indobert_claim_time",
              "IndoBERT layer entitas": "ner_indobert_entity", "IndoBERT layer CLAIM": "ner_indobert_claim"}
    cols = ["PER", "ORG", "LOC", "TIME", "CLAIM", "nergrit_test", "seed_informal_test"]
    rows = []
    for label, name in stage2.items():
        s = load(s1, name)
        rows.append([label, *(fmt(score(s, c)) for c in cols)])
    write_table("ner_stage2", ["Model", *cols], rows,
                "Tahap 2: seed set (test 100 kalimat formal, diperiksa manusia). Model gabungan memakai satu layer "
                "(CLAIM menimpa entitas di dalamnya); dua layer = keputusan 0012. nergrit_test dan seed_informal_test "
                "adalah F1 keseluruhan.")

    # 3. RQ1: normalisasi dan augmentasi
    tests = ["dev", "test", "seed_informal_test", "seed_sim_informal_test"]
    rows, claim_series, entity_series = [], [], []
    for layer, variants in (("CLAIM", ["base", "norm", "aug", "aug3"]), ("Entitas", ["base", "norm", "aug"])):
        for v in variants:
            name = f"rq1_{'claim' if layer == 'CLAIM' else 'entity'}_{v}"
            s = load(rq1, name)
            rows.append([layer, v, *(fmt(score(s, t)) for t in tests), fmt(score(s, "nergrit_test"))])
            target = claim_series if layer == "CLAIM" else entity_series
            target.append((v, [score(s, t) for t in tests[1:]]))
    write_table("ner_rq1", ["Layer", "Varian", "Dev", "Test formal", "Tulisan tangan (30)", "Informal sintetis",
                            "NERGrit test"], rows,
                "RQ1: base = tanpa perlakuan, norm = normalisasi teks informal, aug = +1 salinan informal data latih, "
                "aug3 = +3 salinan. Varian dipilih dari dev. Uji sintetis dibuat dengan generator yang sama dengan "
                "augmentasi (probabilitas dan seed berbeda), jadi sebagian dalam-distribusi.")
    groups = [TEST_NAMES[t] for t in tests[1:]]
    grouped_bars(FIGURES / "ner_rq1_claim.png", "RQ1 layer CLAIM: normalisasi dan augmentasi", groups, claim_series)
    grouped_bars(FIGURES / "ner_rq1_entity.png", "RQ1 layer entitas: normalisasi dan augmentasi", groups,
                 entity_series)

    # 4. Model CLAIM terpasang: F1 strict vs tumpang tindih (sebelum pasca-proses)
    if args.claim_predictions:
        pred_dir = Path(args.claim_predictions)
        files = {"Seed test formal": "ner_indobert_claim_seed42.conll",
                 "Informal sintetis": "ner_indobert_claim_seed42_seed_sim_informal_test.conll",
                 "Tulisan tangan (30)": "ner_indobert_claim_seed42_seed_informal_test.conll"}
        rows, strict, overlap = [], [], []
        for label, fname in files.items():
            st, ov = strict_overlap(pred_dir / fname)
            rows.append([label, f"{st:.3f}", f"{ov:.3f}"])
            strict.append({"mean": st, "std": 0})
            overlap.append({"mean": ov, "std": 0})
        write_table("ner_claim_strict_vs_overlap", ["Uji", "F1 strict", "F1 tumpang tindih"], rows,
                    "Model CLAIM terpasang (aug3, seed 42), sebelum pasca-proses, label tulisan tangan sebelum "
                    "diperbaiki. Tumpang tindih = span prediksi beririsan dengan span gold.")
        grouped_bars(FIGURES / "ner_claim_strict_vs_overlap.png",
                     "CLAIM: letak klaim ditemukan, batas span yang meleset", list(files),
                     [("F1 strict", strict), ("F1 tumpang tindih", overlap)], ylabel="F1")


if __name__ == "__main__":
    main()
