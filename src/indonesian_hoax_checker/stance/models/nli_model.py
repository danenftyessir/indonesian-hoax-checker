# Model NLI untuk stance (model kuat RQ4). Membungkus AutoModelForSequenceClassification:
# - zero-shot dengan model NLI siap pakai (id2label entailment/neutral/contradiction), atau
# - model hasil stance/train.py (id2label supports/refutes/unrelated).
# transformers/torch diimpor malas.

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from indonesian_hoax_checker.common.schemas import StanceLabel
from indonesian_hoax_checker.stance.labels import STANCE_LABELS, to_stance


class NLIStanceModel:
    def __init__(
        self,
        model_name_or_path: str | Path,
        device: str | None = None,
        max_length: int = 256,
        batch_size: int = 32,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = str(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.max_length = max_length
        self.batch_size = batch_size
        # indeks logit -> label stance
        id2label = self.model.config.id2label
        self.index_to_stance: list[StanceLabel] = [to_stance(id2label[i]) for i in range(len(id2label))]
        missing = set(STANCE_LABELS) - set(self.index_to_stance)
        if missing:
            raise ValueError(f"id2label model {id2label} tidak mencakup {missing}")

    def probabilities(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """Matriks (n, 3) peluang dalam urutan STANCE_LABELS (supports, refutes, unrelated)."""
        import torch

        out = np.zeros((len(pairs), len(STANCE_LABELS)), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(pairs), self.batch_size):
                chunk = pairs[start : start + self.batch_size]
                enc = self.tokenizer(
                    [p for p, _ in chunk],
                    [h for _, h in chunk],
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                probs = torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy()
                for j, stance in enumerate(self.index_to_stance):
                    out[start : start + len(chunk), STANCE_LABELS.index(stance)] += probs[:, j]
        return out

    def predict_batch(self, pairs: Sequence[tuple[str, str]]) -> list[tuple[StanceLabel, float]]:
        probs = self.probabilities(pairs)
        return [(STANCE_LABELS[int(i)], float(probs[k, i])) for k, i in enumerate(probs.argmax(axis=1))]

    def predict(self, premise: str, hypothesis: str) -> tuple[StanceLabel, float]:
        return self.predict_batch([(premise, hypothesis)])[0]
