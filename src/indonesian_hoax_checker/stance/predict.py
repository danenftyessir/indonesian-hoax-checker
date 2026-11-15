# Prediksi stance -> StanceResult, dan evaluasi satu model pada file pasangan JSONL.
#
#   python -m indonesian_hoax_checker.stance.predict --model similarity --threshold 0.5 \
#       --pairs data/processed/stance/team_pairs.jsonl --run-name rq4_simthr_team
#   python -m indonesian_hoax_checker.stance.predict --model artifacts/checkpoints/stance_indonli \
#       --pairs data/processed/stance/indonli_test_expert.jsonl --run-name rq4_nli_test_expert

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from indonesian_hoax_checker.common.schemas import RetrievalCandidate, StanceLabel, StanceResult
from indonesian_hoax_checker.retrieval.corpus import read_jsonl, write_jsonl
from indonesian_hoax_checker.stance import metrics as M
from indonesian_hoax_checker.stance.dataset import has_negation


class StanceModel(Protocol):
    def predict_batch(self, pairs: Sequence[tuple[str, str]]) -> list[tuple[StanceLabel, float]]: ...


def load_model(name: str, threshold: float = 0.5, device: str | None = None) -> StanceModel:
    """'similarity' -> baseline ambang TF-IDF; selain itu nama/path model HF."""
    if name == "similarity":
        from indonesian_hoax_checker.stance.models.similarity_threshold import SimilarityThresholdStance

        return SimilarityThresholdStance(threshold)
    from indonesian_hoax_checker.stance.models.nli_model import NLIStanceModel

    return NLIStanceModel(name, device=device)


def predict_stance(
    model: StanceModel, input_claim: str, candidate: RetrievalCandidate | None
) -> StanceResult:
    """Stance klaim input terhadap SATU kandidat klaim terbantah (premise = candidate.claim)."""
    if candidate is None:
        return {"label": "unrelated", "confidence": 1.0, "claim_id": None}
    label, conf = model.predict_batch([(candidate["claim"], input_claim)])[0]
    return {"label": label, "confidence": conf, "claim_id": candidate["claim_id"]}


def combine_directions(
    forward: tuple[StanceLabel, float],
    backward: tuple[StanceLabel, float],
    one_way_min: float = 0.0,
    allow_one_way: bool = True,
) -> tuple[StanceLabel, float]:
    """Gabungkan dua arah NLI. `forward` = (premise klaim terbantah, hypothesis input; keputusan 0002),
    `backward` = arah sebaliknya. Input yang panjang jarang di-entail judul klaim yang pendek, padahal
    input itu meng-entail judulnya; jadi `supports`/`refutes` cukup muncul di satu arah, asalkan arah
    lain tidak berlawanan. Dua arah berlawanan (supports vs refutes) -> unrelated, confidence 0,5.

    `one_way_min`: `supports` yang hanya muncul di SATU arah harus punya confidence minimal sebesar
    ini; di bawahnya dianggap unrelated. `supports` adalah satu-satunya label yang menghasilkan
    verdict otomatis, dan `supports` satu-arah berkeyakinan sekitar 0,5 terbukti sering keliru
    (entitas ditukar, teks panjang). `allow_one_way=False`: `supports` wajib muncul di DUA arah
    (dipakai untuk hypothesis panjang). `refutes` tidak dibatasi karena selalu ditinjau manusia."""
    labels = {forward[0], backward[0]}
    for label, other in (("supports", "refutes"), ("refutes", "supports")):
        if label in labels and other not in labels:
            conf = max(c for lab, c in (forward, backward) if lab == label)
            one_way = forward[0] != backward[0]
            if label == "supports" and one_way and (not allow_one_way or conf < one_way_min):
                return "unrelated", 0.5
            return label, conf
    if labels == {"unrelated"}:
        return "unrelated", min(forward[1], backward[1])
    return "unrelated", 0.5


def predict_stance_bidirectional(
    model: StanceModel, input_claim: str, candidate: RetrievalCandidate | None
) -> StanceResult:
    """Seperti predict_stance, tetapi menilai dua arah dan menggabungkannya (combine_directions)."""
    if candidate is None:
        return {"label": "unrelated", "confidence": 1.0, "claim_id": None}
    forward, backward = model.predict_batch([(candidate["claim"], input_claim), (input_claim, candidate["claim"])])
    label, conf = combine_directions(forward, backward)
    return {"label": label, "confidence": conf, "claim_id": candidate["claim_id"]}


def predict_stance_segments(
    model: StanceModel,
    hypotheses: Sequence[str],
    candidate: RetrievalCandidate,
    bidirectional: bool = True,
    one_way_min: float = 0.0,
    one_way_max_words: int | None = None,
) -> StanceResult:
    """Stance beberapa bagian input (kalimat) terhadap SATU klaim terbantah, dalam satu batch.
    Agregasi: bila ada bagian yang `refutes`, hasilnya `refutes` (input memuat sangkalan, jadi
    tidak boleh disimpulkan otomatis); kalau tidak, `supports` berkeyakinan tertinggi; kalau tidak
    ada keduanya, `unrelated`. `segment` = bagian input yang menghasilkan label itu.
    `one_way_max_words`: bagian yang lebih panjang dari ini hanya boleh `supports` bila dua arah
    sepakat (entailment satu arah dari teks panjang ke judul pendek sering kebetulan)."""
    claim = candidate["claim"]
    pairs = [(claim, h) for h in hypotheses]
    if bidirectional:
        pairs += [(h, claim) for h in hypotheses]
    preds = model.predict_batch(pairs)
    n = len(hypotheses)
    per_segment = []
    for i, hyp in enumerate(hypotheses):
        if not bidirectional:
            per_segment.append(preds[i])
            continue
        short = one_way_max_words is None or len(hyp.split()) <= one_way_max_words
        per_segment.append(combine_directions(preds[i], preds[n + i], one_way_min, allow_one_way=short))
    for wanted in ("refutes", "supports"):
        hits = [(conf, i) for i, (lab, conf) in enumerate(per_segment) if lab == wanted]
        if hits:
            conf, i = max(hits)
            return {"label": wanted, "confidence": conf, "claim_id": candidate["claim_id"],
                    "segment": hypotheses[i]}
    conf = min(c for _, c in per_segment)
    return {"label": "unrelated", "confidence": conf, "claim_id": candidate["claim_id"],
            "segment": hypotheses[0]}


def predict_against_candidates(
    model: StanceModel, input_claim: str, candidates: Sequence[RetrievalCandidate]
) -> list[StanceResult]:
    if not candidates:
        return []
    preds = model.predict_batch([(c["claim"], input_claim) for c in candidates])
    return [{"label": lab, "confidence": conf, "claim_id": c["claim_id"]}
            for c, (lab, conf) in zip(candidates, preds)]


def evaluate_file(model: StanceModel, pairs: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    preds = model.predict_batch([(r["premise"], r["hypothesis"]) for r in pairs])
    y_true = [r["label"] for r in pairs]
    y_pred = [lab for lab, _ in preds]
    neg = [bool(r.get("has_negation", has_negation(r["hypothesis"]))) for r in pairs]
    result = M.evaluate(y_true, y_pred, neg)
    rows = [{"pair_id": r["pair_id"], "gold": r["label"], "pred": lab, "confidence": round(c, 4)}
            for r, (lab, c) in zip(pairs, preds)]
    return result, rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluasi stance RQ4.")
    p.add_argument("--model", required=True, help="'similarity' atau nama/path model HF")
    p.add_argument("--pairs", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--threshold", type=float, default=0.5, help="untuk model similarity")
    p.add_argument("--fit-threshold-on", default=None, help="file dev untuk memilih ambang baseline")
    p.add_argument("--device", default=None)
    p.add_argument("--metrics-dir", default="outputs/metrics")
    p.add_argument("--predictions-dir", default="outputs/predictions")
    args = p.parse_args(argv)

    model = load_model(args.model, args.threshold, args.device)
    if args.fit_threshold_on and args.model == "similarity":
        dev = list(read_jsonl(args.fit_threshold_on))
        model.fit_threshold([(r["premise"], r["hypothesis"]) for r in dev], [r["label"] for r in dev])  # type: ignore[attr-defined]
    pairs = list(read_jsonl(args.pairs))
    result, rows = evaluate_file(model, pairs)
    result["meta"] = {
        "run_name": args.run_name,
        "model": args.model,
        "threshold": getattr(model, "threshold", None),
        "pairs": args.pairs,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = Path(args.metrics_dir) / f"{args.run_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(rows, Path(args.predictions_dir) / f"{args.run_name}.jsonl")
    print(json.dumps({k: v for k, v in result.items() if k not in ("meta",)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
