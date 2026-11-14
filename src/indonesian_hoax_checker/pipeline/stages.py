# Skrip tahapan dalam pipeline

# NER: NERStage membaca checkpoint dari configs/pipeline.yaml, Bila folder checkpoint belum ada

from __future__ import annotations

import re
import sys
import threading
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
                    f"Checkpoint NER {self.checkpoint} belum ada. Unduh ner_checkpoints_seed42.zip dari Drive "
                    "hasil_training/ner_20261112-1801 lalu zip CLAIM hasil_training/ner_20261114-1153, ekstrak "
                    "berurutan ke artifacts/checkpoints/ner/, atau set ner.checkpoint: null di configs/pipeline.yaml "
                    "untuk mode percobaan"
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


def classifier_min_words() -> int:
    # Batas kata minimum untuk classifier (classification.min_words di configs/pipeline.yaml); 0 = tanpa batas
    # model dilatih pada artikel Mafindo; pada teks pendek ia menjawab "hoax" hampir selalu
    try:
        cfg = yaml.safe_load(Path(PIPELINE_CONFIG).read_text()).get("classification", {})
    except (OSError, yaml.YAMLError):
        return 0
    return int(cfg.get("min_words", 0) or 0)


def classifier_too_short(text: str) -> bool:
    n = classifier_min_words()
    return bool(n) and len(text.split()) < n


def run_classify(text: str, ner: NERResult | None = None) -> ClassificationResult:
    clf = _get_classifier()
    if clf is not None:
        STUB_STAGES.discard("classify")
        if classifier_too_short(text):
            return {"label": "valid", "confidence": 0.5}  # tidak tahu; diteruskan ke tinjauan manusia
        return clf(text)
    return {"label": "valid", "confidence": 0.5}


def _pipeline_section(name: str) -> dict:
    # Bagian `name` dari configs/pipeline.yaml; {} bila file/bagian tidak ada
    try:
        return yaml.safe_load(Path(PIPELINE_CONFIG).read_text(encoding="utf-8")).get(name) or {}
    except (OSError, yaml.YAMLError, AttributeError):
        return {}


def _warn(msg: str) -> None:
    print(f"Peringatan: {msg}", file=sys.stderr)


# Retrieval dan stance dimuat sekali per proses. Orchestrator memanggil retrieval di thread terpisah,
# jadi pemuatan dikunci supaya mode interaktif tidak membangun indeks dua kali.
_load_lock = threading.Lock()
_retriever = None
_retriever_failed = False


def _get_retriever():
    # Retriever dari retrieval.config (configs/models/retrieval_*.yaml) + override di configs/pipeline.yaml.
    # Basis klaim tidak ada atau model embedding gagal dimuat -> None (mode stub).
    global _retriever, _retriever_failed
    with _load_lock:
        if _retriever is not None or _retriever_failed:
            return _retriever
        cfg = _pipeline_section("retrieval")
        if not cfg.get("config"):
            _retriever_failed = True
            return None
        from indonesian_hoax_checker.retrieval.search import build_retriever, load_config

        try:
            model_cfg = load_config(cfg["config"])
        except (OSError, yaml.YAMLError) as err:
            _warn(f"config retrieval {cfg['config']} tidak terbaca ({err}); retrieval memakai stub")
            _retriever_failed = True
            return None
        for key in ("min_score", "top_k", "entity_filter"):
            if key in cfg:
                model_cfg[key] = cfg[key]
        if not Path(model_cfg.get("claims_path", "")).exists():
            _warn(f"basis klaim {model_cfg.get('claims_path')} belum ada; retrieval memakai stub")
            _retriever_failed = True
            return None
        try:
            _retriever = build_retriever(model_cfg)
        except OSError as err:  # mis. model embedding tidak bisa diunduh
            _warn(f"retriever gagal dimuat ({err}); retrieval memakai stub")
            _retriever_failed = True
        return _retriever


def run_retrieve(text: str, ner: NERResult | None) -> RetrievalResult:
    retriever = _get_retriever()
    if retriever is None:
        # Tidak ada klaim terbantah yang cocok
        return {"matched_claim": None, "clarification": None, "similarity_score": None,
                "candidates": [], "scorer": "bm25"}
    STUB_STAGES.discard("retrieve")
    entities = ner["entities"] if ner else None
    return retriever.retrieve(text, entities)


_stance_model = None
_stance_failed = False


def _get_stance_model():
    # stance.model di configs/pipeline.yaml: "similarity", path checkpoint lokal, atau nama model HF.
    # Path lokal yang belum ada / model HF yang gagal diunduh -> None (mode stub).
    global _stance_model, _stance_failed
    with _load_lock:
        if _stance_model is not None or _stance_failed:
            return _stance_model
        cfg = _pipeline_section("stance")
        name = cfg.get("model")
        if not name:
            _stance_failed = True
            return None
        if not _stance_configured():
            _warn(f"checkpoint stance {name} belum ada; stance memakai stub")
            _stance_failed = True
            return None
        from indonesian_hoax_checker.stance.predict import load_model

        device = cfg.get("device", "auto")
        try:
            _stance_model = load_model(name, float(cfg.get("threshold", 0.5)), None if device == "auto" else device)
        except (OSError, ValueError) as err:
            _warn(f"model stance {name} gagal dimuat ({err}); stance memakai stub")
            _stance_failed = True
        return _stance_model


def _stance_configured() -> bool:
    # True bila stance.model terisi dan, untuk path lokal, foldernya ada. Tidak memuat model.
    name = _pipeline_section("stance").get("model")
    if not name:
        return False
    local = name != "similarity" and (Path(name).exists() or name.startswith(("artifacts/", "./", "/")))
    return not local or Path(name).exists()


def run_stance(text: str, retrieval: RetrievalResult | None) -> StanceResult | None:
    # premise = klaim terbantah teratas, hypothesis = klaim input
    # Tanpa klaim cocok tidak ada pasangan yang bisa dinilai
    if retrieval is None or retrieval["matched_claim"] is None:
        # "Dilewati" adalah hasil nyata bila retrieval nyata dan model stance tersedia; model tidak
        # perlu dimuat hanya untuk melepas tanda stub
        if "retrieve" not in STUB_STAGES and not _stance_failed and _stance_configured():
            STUB_STAGES.discard("stance")
        return None
    model = _get_stance_model()
    if model is None:
        STUB_STAGES.add("stance")  # sempat dilepas di jalur tanpa-padanan, lalu model gagal dimuat
        return {"label": "unrelated", "confidence": 0.5, "claim_id": None}
    STUB_STAGES.discard("stance")
    from indonesian_hoax_checker.stance.predict import predict_stance, predict_stance_bidirectional

    candidates = retrieval.get("candidates") or []
    predict = predict_stance_bidirectional if _pipeline_section("stance").get("bidirectional") else predict_stance
    return predict(model, text, candidates[0] if candidates else None)
