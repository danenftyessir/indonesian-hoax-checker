# Skrip tahapan dalam pipeline

# NER: NERStage membaca checkpoint dari configs/pipeline.yaml, Bila folder checkpoint belum ada

from __future__ import annotations

import os
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
from indonesian_hoax_checker.pipeline.textproc import strip_report_frame

PIPELINE_CONFIG = "configs/pipeline.yaml"
ROOT = Path(__file__).resolve().parents[3]  # root repo (src/indonesian_hoax_checker/pipeline/ -> 3 tingkat)

STUB_STAGES: set[str] = {"ner", "classify", "retrieve", "stance"}

MAX_SEGMENTS = 40          # kalimat input yang dijadikan kueri retrieval tambahan
FULL_TEXT_MAX_WORDS = 30   # teks multi-kalimat di atas ini tidak dijadikan kueri utuh
MIN_CLAIM_SPAN_WORDS = 4   # span CLAIM lebih pendek dari ini tidak dijadikan kueri
MAX_CLAIM_SPANS = 10


def resolve_path(path: str | None) -> str | None:
    """Path relatif dicari dari folder kerja dulu, lalu dari root repo, supaya `hoax-checker` yang
    dipanggil dari folder lain tetap menemukan config, checkpoint, dan basis klaim. Path yang tidak
    ada di keduanya dikembalikan apa adanya (pemanggil yang memutuskan: stub atau galat)."""
    if not path:
        return path
    p = Path(path)
    if p.is_absolute() or p.exists():
        return path
    alt = ROOT / p
    return str(alt) if alt.exists() else path


def config_path() -> str:
    # HOAX_CHECKER_CONFIG menunjuk pipeline.yaml lain (mis. config demo atau config stub untuk tes)
    return os.environ.get("HOAX_CHECKER_CONFIG") or PIPELINE_CONFIG


def _read_pipeline_config(path: str | None = None) -> dict:
    # Selalu UTF-8: bawaan Windows (cp1252) bisa gagal pada komentar non-ASCII di YAML
    data = yaml.safe_load(Path(resolve_path(path or config_path())).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


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
    def from_config(cls, path: str | None = None) -> "NERStage":
        cfg = _read_pipeline_config(path).get("ner") or {}
        return cls(resolve_path(cfg.get("checkpoint")), cfg.get("device", "auto"),
                   resolve_path(cfg.get("claim_checkpoint")))

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
        except (OSError, ValueError, yaml.YAMLError):
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
        cfg = _pipeline_section("classification")
        checkpoint = resolve_path(cfg.get("checkpoint"))
        if not checkpoint or not Path(checkpoint).exists():
            return None
        from indonesian_hoax_checker.classification.predict import HoaxClassifier

        _classifier = HoaxClassifier(checkpoint, cfg.get("max_length", 256), cfg.get("device", "auto"))
    return _classifier


def classifier_min_words() -> int:
    # Batas kata minimum untuk classifier (classification.min_words di configs/pipeline.yaml); 0 = tanpa batas
    # model dilatih pada artikel Mafindo; pada teks pendek ia menjawab "hoax" hampir selalu
    return int(_pipeline_section("classification").get("min_words", 0) or 0)


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
        section = _read_pipeline_config().get(name)
    except (OSError, ValueError, yaml.YAMLError):
        return {}
    return section if isinstance(section, dict) else {}


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
            model_cfg = load_config(resolve_path(cfg["config"]))
        except (OSError, ValueError, yaml.YAMLError) as err:
            _warn(f"config retrieval {cfg['config']} tidak terbaca ({err}); retrieval memakai stub")
            _retriever_failed = True
            return None
        for key in ("min_score", "top_k", "entity_filter"):
            if key in cfg:
                model_cfg[key] = cfg[key]
        for key in ("claims_path", "corpus_entities_path", "cache_dir"):
            if model_cfg.get(key):
                model_cfg[key] = resolve_path(model_cfg[key])
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


def _empty_retrieval() -> RetrievalResult:
    return {"matched_claim": None, "clarification": None, "similarity_score": None,
            "candidates": [], "scorer": "bm25"}


def run_retrieve(text: str, ner: NERResult | None, segments: list[str] | None = None) -> RetrievalResult:
    """Cari klaim terbantah untuk `text`. Bila `segments` (kalimat-kalimat input) diberikan dan
    retrieval.segment_queries aktif, tiap kalimat juga dijadikan kueri dan hasil berskor tertinggi
    yang dipakai: klaim hoaks yang terselip di teks panjang tidak tenggelam oleh kalimat lain.
    Teks multi-kalimat di atas retrieval.full_text_max_words kata TIDAK dijadikan kueri utuh:
    embedding dan stance atas ratusan kata bertopik campur tidak andal (semua `supports` palsu pada
    uji artikel valid berasal dari teks penuh). `query_segment` = bagian input yang menghasilkan
    padanan itu."""
    retriever = _get_retriever()
    if retriever is None:
        return _empty_retrieval()  # tidak ada klaim terbantah yang cocok
    STUB_STAGES.discard("retrieve")
    entities = ner["entities"] if ner else None
    cfg = _pipeline_section("retrieval")
    queries = [text]
    full_text_first = True
    if segments and cfg.get("segment_queries", True):
        segs = [seg for seg in segments if seg != text][:MAX_SEGMENTS]
        max_words = cfg.get("full_text_max_words", FULL_TEXT_MAX_WORDS)
        if segs and max_words and len(text.split()) > int(max_words):
            queries, full_text_first = segs, False
        else:
            queries += segs
    if ner and cfg.get("claim_span_queries", False):
        # Span CLAIM dari NER sebagai kueri tambahan. DITOLAK sebagai default (uji 15 Nov 2026): span
        # CLAIM membuang subjek ("Nestle Indonesia sedang membuka lowongan" -> "sedang membuka lowongan
        # pekerjaan") sehingga cocok dengan klaim entitas lain; parafrasa test 88 -> 83 benar dengan 2
        # verdict ke klaim yang salah. Bingkai laporan ditangani orchestrator.strip_report_frame.
        spans = [e["text"] for e in ner["entities"]
                 if e["label"] == "CLAIM" and len(e["text"].split()) >= MIN_CLAIM_SPAN_WORDS]
        queries += [span for span in dict.fromkeys(spans) if span not in queries][:MAX_CLAIM_SPANS]
    best_i = 0
    if len(queries) > 1:
        tops = retriever.top_scores(queries, entities)
        for i in range(1, len(tops)):
            # Skor seri dengan teks penuh (lazim di BM25) -> pilih kalimatnya, karena lebih spesifik
            if tops[i] > tops[best_i] or (tops[i] == tops[best_i] and best_i == 0 and full_text_first):
                best_i = i
    best = retriever.retrieve(queries[best_i], entities)
    best["query_segment"] = queries[best_i]
    return best


def is_strong_match(retrieval: RetrievalResult | None) -> bool:
    """True bila skor padanan teratas >= retrieval.strong_match_score (skor setingkat judul klaim itu
    sendiri). Hanya bermakna untuk scorer yang skornya dibandingkan dengan ambang itu (cosine E5)."""
    threshold = _pipeline_section("retrieval").get("strong_match_score")
    if not threshold or retrieval is None or retrieval.get("similarity_score") is None:
        return False
    return float(retrieval["similarity_score"]) >= float(threshold)


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
        name = resolve_path(cfg.get("model"))
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
    name = resolve_path(_pipeline_section("stance").get("model"))
    if not name:
        return False
    local = name != "similarity" and (Path(name).exists() or name.startswith(("artifacts/", "./", "/")))
    return not local or Path(name).exists()


_WORD = re.compile(r"\w+")


def _content_tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.lower()) if len(t) >= 4}


def stance_hypotheses(text: str, segments: list[str] | None, matched: str | None) -> list[str]:
    """Bagian input yang dinilai stance terhadap klaim teratas = bagian yang menghasilkan padanan
    retrieval (kalimat paling mirip klaim), atau teks penuh bila input satu kalimat.

    Kalimat lain sengaja TIDAK ikut dinilai: pada uji teks panjang (15 Nov 2026), kalimat berita
    di sekitar klaim sering dinilai `refutes` atau `supports` hanya karena berbagi tokoh, sehingga
    14 dari 60 hoaks terselip tertahan dan 2 artikel valid mendapat `supports` palsu. Sangkalan di
    kalimat lain ditangani penanda bantahan (orchestrator.has_debunk_marker)."""
    del segments
    part = matched or text
    # Bingkai laporan ("beredar kabar bahwa X") membuat model stance menilai X sebagai tidak
    # ditegaskan; yang dinilai adalah isi klaimnya.
    return [strip_report_frame(part) or part]


def entity_mismatch(segment: str, ner: NERResult | None, candidate: dict) -> list[str]:
    """Entitas PER/ORG/LOC di `segment` yang tidak disebut sama sekali di klaim terbantah maupun
    klarifikasinya (mis. input menyebut Shell, klaimnya tentang Pertamina). Dicek per kata (>= 4
    huruf) supaya singkatan dan nama sebagian tetap cocok."""
    if not ner:
        return []
    reference = _content_tokens(f"{candidate.get('claim', '')} {candidate.get('clarification', '')}")
    low = segment.lower()
    missing: list[str] = []
    for ent in ner["entities"]:
        if ent["label"] not in ("PER", "ORG", "LOC") or ent["text"].lower() not in low:
            continue
        tokens = _content_tokens(ent["text"])
        if tokens and not (tokens & reference) and ent["text"] not in missing:
            missing.append(ent["text"])
    return missing


def run_stance(
    text: str,
    retrieval: RetrievalResult | None,
    segments: list[str] | None = None,
    ner: NERResult | None = None,
) -> StanceResult | None:
    # premise = klaim terbantah teratas, hypothesis = klaim input (atau kalimat-kalimatnya)
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
    from indonesian_hoax_checker.stance.predict import predict_stance_segments

    cfg = _pipeline_section("stance")
    candidates = retrieval.get("candidates") or []
    if not candidates:
        return {"label": "unrelated", "confidence": 1.0, "claim_id": None}
    candidate = candidates[0]
    hyps = stance_hypotheses(text, segments, retrieval.get("query_segment"))
    max_words = cfg.get("one_way_max_words")
    result = predict_stance_segments(
        model, hyps, candidate,
        bidirectional=bool(cfg.get("bidirectional")),
        one_way_min=float(cfg.get("one_way_min_confidence", 0.0) or 0.0),
        one_way_max_words=int(max_words) if max_words else None,
    )
    if result["label"] == "supports" and cfg.get("entity_guard", True):
        missing = entity_mismatch(result.get("segment", text), ner, candidate)
        if missing:
            result["note"] = "entitas tidak ada di klaim terbantah: " + ", ".join(missing)
    return result
