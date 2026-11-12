# Tahap-tahap pipeline. Model dimuat saat pertama dipakai, supaya import dan --help tetap cepat.
# Tahap yang modelnya belum ada tetap mengembalikan nilai netral bentuk common/schemas.py,
# sehingga pipeline bisa dijalankan sebagian (daftar tahap stub ada di STUB_STAGES).
#
# NER: NERStage membaca checkpoint dari configs/pipeline.yaml. Bila folder checkpoint belum ada
# (belum mengekstrak ner_results.zip dari Drive hasil_training), run_ner memakai stub.

from __future__ import annotations

import re
from pathlib import Path

import yaml

from indonesian_hoax_checker.common.schemas import (
    ClassificationResult,
    NERResult,
    RetrievalResult,
    StanceResult,
)

PIPELINE_CONFIG = "configs/pipeline.yaml"

STUB_STAGES: set[str] = {"ner", "classify", "retrieve", "stance"}


class NERStage:
    name = "ner"

    # checkpoint=None -> mode stub (selalu entitas kosong), sesuai docs/decisions/0005
    def __init__(self, checkpoint: str | None, device: str = "auto"):
        self.checkpoint = checkpoint
        self.device = device
        self._predictor = None

    @classmethod
    def from_config(cls, path: str = PIPELINE_CONFIG) -> "NERStage":
        cfg = yaml.safe_load(Path(path).read_text()).get("ner", {})
        return cls(cfg.get("checkpoint"), cfg.get("device", "auto"))

    @property
    def is_stub(self) -> bool:
        return self.checkpoint is None

    def _load(self):
        if self._predictor is None:
            if not Path(self.checkpoint).exists():
                raise FileNotFoundError(
                    f"Checkpoint NER {self.checkpoint} belum ada. Unduh ner_results.zip dari Drive hasil_training "
                    "dan ekstrak di root repo, atau set ner.checkpoint: null di configs/pipeline.yaml untuk mode stub."
                )
            from indonesian_hoax_checker.ner.predict import NERPredictor

            self._predictor = NERPredictor(self.checkpoint, self.device)
        return self._predictor

    def __call__(self, text: str) -> NERResult:
        return self.run_batch([text])[0]

    def run_batch(self, texts: list[str]) -> list[NERResult]:
        if self.is_stub:
            return [{"entities": [], "core_claim": ""} for _ in texts]
        return self._load().predict_batch(texts)


_ner_stage: NERStage | None = None


def _get_ner_stage() -> NERStage | None:
    """NERStage dari config bila checkpoint-nya ada di disk; None (= stub) bila tidak."""
    global _ner_stage
    if _ner_stage is None:
        try:
            stage = NERStage.from_config()
        except (OSError, yaml.YAMLError):
            return None
        if stage.is_stub or not Path(stage.checkpoint).exists():
            return None
        _ner_stage = stage
    return _ner_stage


def run_ner(text: str) -> NERResult:
    stage = _get_ner_stage()
    if stage is not None:
        STUB_STAGES.discard("ner")
        return stage(text)
    # Stub: tanpa entitas; klaim inti = kalimat pertama
    first = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    return {"entities": [], "core_claim": first}


def run_classify(text: str, ner: NERResult | None = None) -> ClassificationResult:
    # Stub: 0.5 = tidak tahu. Skor asli harus terkalibrasi (docs/experiment_protocol.md)
    return {"label": "valid", "confidence": 0.5}


def run_retrieve(text: str, ner: NERResult) -> RetrievalResult:
    # Stub: tidak ada klaim terbantah yang cocok
    return {"matched_claim": None, "clarification": None, "similarity_score": None}


def run_stance(text: str, retrieval: RetrievalResult) -> StanceResult | None:
    # premise = klaim terbantah, hypothesis = klaim input (docs/decisions/0002)
    # Tanpa klaim cocok tidak ada pasangan yang bisa dinilai
    if retrieval["matched_claim"] is None:
        return None
    return {"label": "unrelated", "confidence": 0.5}
