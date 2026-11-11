# Metrik NER: F1 tingkat entitas (seqeval, mode strict IOB2), keseluruhan dan per tipe.

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2


def entity_scores(true_tags: list[list[str]], pred_tags: list[list[str]]) -> dict:
    report = classification_report(
        true_tags, pred_tags, mode="strict", scheme=IOB2, output_dict=True, zero_division=0
    )
    micro = report["micro avg"]
    return {
        "precision": micro["precision"],
        "recall": micro["recall"],
        "f1": micro["f1-score"],
        "per_type": {
            kind: {"precision": r["precision"], "recall": r["recall"], "f1": r["f1-score"], "support": int(r["support"])}
            for kind, r in report.items()
            if not kind.endswith("avg")
        },
    }
