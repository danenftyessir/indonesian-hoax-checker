# Label stance dan pemetaannya (configs/labels.yaml, docs/label_mapping.md, keputusan 0002).
# premise = klaim yang sudah dibantah, hypothesis = klaim input.

from __future__ import annotations

from pathlib import Path

import yaml

from indonesian_hoax_checker.common.schemas import StanceLabel

STANCE_LABELS: tuple[StanceLabel, ...] = ("supports", "refutes", "unrelated")
LABEL2ID: dict[str, int] = {lab: i for i, lab in enumerate(STANCE_LABELS)}
ID2LABEL: dict[int, str] = {i: lab for lab, i in LABEL2ID.items()}

# IndoNLI memakai 'e'/'n'/'c' di JSONL dan nama penuh di beberapa turunan HF.
INDONLI_TO_STANCE: dict[str, StanceLabel] = {
    "entailment": "supports",
    "contradiction": "refutes",
    "neutral": "unrelated",
    "e": "supports",
    "c": "refutes",
    "n": "unrelated",
}

# Nama label NLI umum pada model HF (id2label) -> stance. Dipakai NLIStanceModel zero-shot.
NLI_NAME_TO_STANCE: dict[str, StanceLabel] = {
    "entailment": "supports",
    "entail": "supports",
    "contradiction": "refutes",
    "contradict": "refutes",
    "neutral": "unrelated",
    # model yang sudah di-fine-tune dengan label proyek
    "supports": "supports",
    "refutes": "refutes",
    "unrelated": "unrelated",
}

_LABELS_YAML = Path(__file__).resolve().parents[3] / "configs" / "labels.yaml"


def load_from_config(path: str | Path = _LABELS_YAML) -> dict:
    """Baca bagian `stance` dari configs/labels.yaml dan pastikan konsisten dengan modul ini."""
    with Path(path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["stance"]
    if tuple(cfg["labels"]) != STANCE_LABELS:
        raise ValueError(f"labels.yaml {cfg['labels']} != {STANCE_LABELS}")
    for nli, st in cfg["indonli_mapping"].items():
        if INDONLI_TO_STANCE[nli] != st:
            raise ValueError(f"indonli_mapping {nli}->{st} berbeda dari {INDONLI_TO_STANCE[nli]}")
    return cfg


def to_stance(label: str) -> StanceLabel:
    """Normalisasi label apa pun (IndoNLI, nama NLI, atau label proyek) ke label stance."""
    key = label.strip().lower()
    if key in STANCE_LABELS:
        return key  # type: ignore[return-value]
    if key in INDONLI_TO_STANCE:
        return INDONLI_TO_STANCE[key]
    if key in NLI_NAME_TO_STANCE:
        return NLI_NAME_TO_STANCE[key]
    raise ValueError(f"label tidak dikenal: {label!r}")
