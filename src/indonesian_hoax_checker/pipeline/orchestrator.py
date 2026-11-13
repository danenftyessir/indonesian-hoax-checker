# Menjalankan tahap sesuai urutan: NER -> klasifikasi dan retrieval paralel -> stance -> gabung

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from indonesian_hoax_checker.common.schemas import (
    ClassificationResult,
    NERResult,
    RetrievalResult,
    StanceResult,
)
from indonesian_hoax_checker.pipeline import stages
from indonesian_hoax_checker.pipeline.confidence import Verdict, combine

TASKS = ("all", "ner", "classify", "retrieve", "stance")


@dataclass
class PipelineResult:
    text: str
    task: str
    ner: NERResult | None = None
    classification: ClassificationResult | None = None
    retrieval: RetrievalResult | None = None
    stance: StanceResult | None = None
    verdict: Verdict | None = None
    stubs: list[str] = field(default_factory=list) 


RETRIEVAL_ENTITY_LABELS = ("PER", "ORG", "LOC")


def input_claim(text: str, ner: NERResult | None) -> str:
    # Klaim input (hypothesis stance): core_claim bila ada, kalau kosong teks penuh
    core = (ner["core_claim"] if ner else "").strip()
    return core or text


def retrieval_query(text: str, ner: NERResult | None) -> str:
    # Aturan kueri dari Danendra: core_claim + entitas PER/ORG/LOC; bila core_claim kosong, teks penuh
    core = (ner["core_claim"] if ner else "").strip()
    if not core:
        return text
    extra: list[str] = []
    for e in ner["entities"]:
        if e["label"] in RETRIEVAL_ENTITY_LABELS and e["text"] not in core and e["text"] not in extra:
            extra.append(e["text"])
    return " ".join([core, *extra])


def run_pipeline(text: str, task: str = "all") -> PipelineResult:
    if task not in TASKS:
        raise ValueError(f"task tidak dikenal: {task!r}; pilihan: {TASKS}")
    if not text.strip():
        raise ValueError("teks kosong")

    res = PipelineResult(text=text, task=task)
    need_ner = task in ("all", "ner", "retrieve", "stance")
    need_classify = task in ("all", "classify")
    need_retrieve = task in ("all", "retrieve", "stance")

    if need_ner:
        res.ner = stages.run_ner(text)

    # Klasifikasi dan retrieval dijalankan paralel
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_cls = pool.submit(stages.run_classify, text, res.ner) if need_classify else None
        f_ret = pool.submit(stages.run_retrieve, retrieval_query(text, res.ner), res.ner) if need_retrieve else None
        res.classification = f_cls.result() if f_cls else None
        res.retrieval = f_ret.result() if f_ret else None

    if task in ("all", "stance"):
        res.stance = stages.run_stance(input_claim(text, res.ner), res.retrieval)

    if task == "all":
        res.verdict = combine(res.classification, res.retrieval, res.stance)

    ran = {
        "ner": res.ner is not None,
        "classify": res.classification is not None,
        "retrieve": res.retrieval is not None,
        "stance": task in ("all", "stance"),
    }
    res.stubs = [s for s, was_run in ran.items() if was_run and s in stages.STUB_STAGES]
    return res
