# Skrip inference hoax classifier hoaks model IndoBERT-p1 hasil fine-tune

# Label: kelas 1 = hoax, kelas 0 = valid, `confidence` = peluang kelas yang dipilih
# threshold = 0,5

from __future__ import annotations

from pathlib import Path

from indonesian_hoax_checker.common.schemas import ClassificationResult


class HoaxClassifier:
    def __init__(self, checkpoint: str, max_length: int = 256, device: str = "auto"):
        self.checkpoint = checkpoint
        self.max_length = max_length
        self.device = device
        self._tok = None
        self._model = None

    def _load(self):
        if self._model is None:
            if not Path(self.checkpoint).exists():
                raise FileNotFoundError(f"Checkpoint classifier {self.checkpoint} belum ada.")
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.device = ("cuda" if torch.cuda.is_available() else "cpu") if self.device == "auto" else self.device
            self._tok = AutoTokenizer.from_pretrained(self.checkpoint)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint).to(self.device).eval()
        return self._tok, self._model

    def predict_proba(self, texts: list[str]) -> list[float]:
        # peluang kelas hoax untuk tiap teks
        import torch

        tok, model = self._load()
        batch = tok(texts, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            probs = torch.softmax(model(**batch).logits.float(), dim=-1)[:, 1]
        return [float(p) for p in probs.cpu()]

    def __call__(self, text: str) -> ClassificationResult:
        p_hoax = self.predict_proba([text])[0]
        label = "hoax" if p_hoax >= 0.5 else "valid"
        return {"label": label, "confidence": max(p_hoax, 1.0 - p_hoax)}
