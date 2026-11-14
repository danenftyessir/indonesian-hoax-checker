# Skrip tahapan dalam pipeline

# NER: NERStage membaca checkpoint dari configs/pipeline.yaml, Bila folder checkpoint belum ada

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

    # checkpoint=None -> entitas kosong (mode percobaan), sesuai docs/decisions/0005
    # claim_checkpoint (opsional): model CLAIM terpisah, lihat NERPredictor
    def __init__(self, checkpoint: str | None, device: str = "auto", claim_checkpoint: str | None = None):
        self.checkpoint = checkpoint
        self.claim_checkpoint = claim_checkpoint
        self.device = device
        self._predictor = None

    @classmethod
    def from_config(cls, path: str = PIPELINE_CONFIG) -> "NERStage":
        cfg = yaml.safe_load(Path(path).read_text()).get("ner", {})
        return cls(cfg.get("checkpoint"), cfg.get("device", "auto"), cfg.get("claim_checkpoint"))

    @property
    def is_stub(self) -> bool:
        return self.checkpoint is None

    def _load(self):
        if self._predictor is None:
            if not Path(self.checkpoint).exists():
                raise FileNotFoundError(
                    f"Checkpoint NER {self.checkpoint} belum ada. Unduh ner_results.zip dari Drive hasil_training "
                    "dan ekstrak di root repo, atau set ner.checkpoint: null di configs/pipeline.yaml untuk mode percobaan"
                )
            from indonesian_hoax_checker.ner.predict import NERPredictor

            if self.claim_checkpoint and not Path(self.claim_checkpoint).exists():
                raise FileNotFoundError(f"Checkpoint CLAIM {self.claim_checkpoint} belum ada.")
            self._predictor = NERPredictor(self.checkpoint, self.device, self.claim_checkpoint)
        return self._predictor

    def __call__(self, text: str) -> NERResult:
        return self.run_batch([text])[0]

    def run_batch(self, texts: list[str]) -> list[NERResult]:
        if self.is_stub:
            return [{"entities": [], "core_claim": ""} for _ in texts]
        return self._load().predict_batch(texts)


_ner_stage: NERStage | None = None


def _get_ner_stage() -> NERStage | None:
    # NERStage dari config bila checkpoint-nya ada
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


_classifier = None


def _get_classifier():
    # HoaxClassifier dari config bila checkpoint-nya ada
    global _classifier
    if _classifier is None:
        try:
            cfg = yaml.safe_load(Path(PIPELINE_CONFIG).read_text()).get("classification", {})
        except (OSError, yaml.YAMLError):
            return None
        checkpoint = cfg.get("checkpoint")
        if not checkpoint or not Path(checkpoint).exists():
            return None
        from indonesian_hoax_checker.classification.predict import HoaxClassifier

        _classifier = HoaxClassifier(checkpoint, cfg.get("max_length", 256), cfg.get("device", "auto"))
    return _classifier


def run_classify(text: str, ner: NERResult | None = None) -> ClassificationResult:
    clf = _get_classifier()
    if clf is not None:
        STUB_STAGES.discard("classify")
        return clf(text)
    return {"label": "valid", "confidence": 0.5}


def run_retrieve(text: str, ner: NERResult) -> RetrievalResult:
    # Tidak ada klaim terbantah yang cocok
    return {"matched_claim": None, "clarification": None, "similarity_score": None}


def run_stance(text: str, retrieval: RetrievalResult) -> StanceResult | None:
    # premise = klaim terbantah, hypothesis = klaim input
    # Tanpa klaim cocok tidak ada pasangan yang bisa dinilai
    if retrieval["matched_claim"] is None:
        return None
    return {"label": "unrelated", "confidence": 0.5}
