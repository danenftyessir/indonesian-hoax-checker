# Metrik stance RQ4: macro-F1 supports/refutes/unrelated, F1 per kelas, confusion
# (khususnya supports vs refutes), dan hasil terpisah untuk hypothesis dengan/tanpa negasi.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from indonesian_hoax_checker.stance.labels import STANCE_LABELS

LABELS = list(STANCE_LABELS)


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))


def per_class(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, dict[str, float]]:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return {
        lab: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
              "f1": round(float(f[i]), 4), "support": int(s[i])}
        for i, lab in enumerate(LABELS)
    }


def confusion(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    i_s, i_r = LABELS.index("supports"), LABELS.index("refutes")
    n_sr = int(cm[i_s, i_r] + cm[i_r, i_s])
    n_true_sr = int(cm[i_s].sum() + cm[i_r].sum())
    return {
        "labels": LABELS,
        "matrix": cm.tolist(),  # baris = gold, kolom = prediksi
        "supports_as_refutes": int(cm[i_s, i_r]),
        "refutes_as_supports": int(cm[i_r, i_s]),
        # porsi pasangan supports/refutes yang arahnya tertukar: metrik risiko "salah stance" BEAM
        "direction_swap_rate": round(n_sr / n_true_sr, 4) if n_true_sr else float("nan"),
    }


def evaluate(
    y_true: Sequence[str], y_pred: Sequence[str], has_negation: Sequence[bool] | None = None
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n": len(y_true),
        "macro_f1": round(macro_f1(y_true, y_pred), 4),
        "per_class": per_class(y_true, y_pred),
        "confusion": confusion(y_true, y_pred),
    }
    if has_negation is not None:
        for flag, name in ((True, "with_negation"), (False, "without_negation")):
            idx = [i for i, h in enumerate(has_negation) if bool(h) == flag]
            if idx:
                yt = [y_true[i] for i in idx]
                yp = [y_pred[i] for i in idx]
                out[name] = {"n": len(idx), "macro_f1": round(macro_f1(yt, yp), 4),
                             "per_class": per_class(yt, yp)}
    return out
