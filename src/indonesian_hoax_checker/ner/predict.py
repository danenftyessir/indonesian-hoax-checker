# Langkah 4 task NER: teks mentah -> NERResult (entitas dengan posisi karakter + core_claim).
# Dipakai task 2 dan 3 untuk mendapat entitas prediksi.
#
# Contoh:
#   python -m indonesian_hoax_checker.ner.predict --checkpoint artifacts/checkpoints/ner/ner_indobert_seed42 "teks klaim"
#
#   from indonesian_hoax_checker.ner.predict import NERPredictor
#   ner = NERPredictor("artifacts/checkpoints/ner/ner_indobert_seed42")
#   ner.predict("Pemerintah menghentikan bantuan MBG mulai bulan depan")

import argparse
import json
from pathlib import Path

from indonesian_hoax_checker.common.schemas import Entity, NERResult
from indonesian_hoax_checker.ner.labels import get_spans, to_iob2
from indonesian_hoax_checker.ner.models import bilstm_crf, indobert
from indonesian_hoax_checker.ner.train import get_device, load_tokens
from indonesian_hoax_checker.preprocessing.tokenizer import Token, split_sentences, tokenize


# Ubah label per token menjadi daftar Entity dengan posisi karakter di teks asli
def to_entities(text: str, tokens: list[Token], tags: list[str]) -> list[Entity]:
    entities = []
    for kind, first, last in get_spans(tags):
        start, end = tokens[first][1], tokens[last - 1][2]
        entities.append({"text": text[start:end], "label": kind, "start": start, "end": end})
    return entities


# core_claim = span CLAIM pertama, string kosong bila tidak ada (docs/decisions/0011)
def to_result(entities: list[Entity]) -> NERResult:
    claims = [e["text"] for e in entities if e["label"] == "CLAIM"]
    return {"entities": entities, "core_claim": claims[0] if claims else ""}


class NERPredictor:
    # Jenis model dikenali dari isi folder checkpoint: model.pt = BiLSTM-CRF, selain itu IndoBERT
    def __init__(self, checkpoint_dir: str, device: str = "auto"):
        path = Path(checkpoint_dir)
        self.device = get_device(device)
        if (path / "model.pt").exists():
            self.kind = "bilstm_crf"
            if self.device.type == "mps":
                self.device = get_device("cpu")  # BiLSTM-CRF crash di MPS (Conv1d karakter)
            self.model, self.words, self.chars, cfg = bilstm_crf.load(path, self.device)
            self.lowercase = cfg.get("lowercase", False)
            self.normalize = cfg.get("normalize", False)
        else:
            from transformers import AutoModelForTokenClassification, AutoTokenizer

            self.kind = "indobert"
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForTokenClassification.from_pretrained(path).to(self.device)
            train_cfg = path / "train_config.json"
            cfg = json.loads(train_cfg.read_text()) if train_cfg.exists() else {}
            self.max_length = cfg.get("max_length", 256)
            self.lowercase = cfg.get("lowercase", False)
            self.normalize = cfg.get("normalize", False)

    # Prediksi label untuk daftar kalimat yang sudah ditokenisasi.
    # Praproses sama dengan saat training (normalize/lowercase dari config checkpoint).
    def tag(self, token_lists: list[list[str]]) -> list[list[str]]:
        token_lists = [load_tokens(tokens, self.lowercase, self.normalize) for tokens in token_lists]
        if self.kind == "bilstm_crf":
            preds = bilstm_crf.predict(self.model, self.words, self.chars, token_lists, self.device)
        else:
            preds = indobert.predict(self.model, self.tokenizer, token_lists, self.device, self.max_length)
        return [to_iob2(p) for p in preds]

    def predict(self, text: str) -> NERResult:
        return self.predict_batch([text])[0]

    # Semua kalimat dari semua teks diprediksi sekaligus supaya cepat
    def predict_batch(self, texts: list[str]) -> list[NERResult]:
        sentences = [split_sentences(tokenize(text)) for text in texts]
        flat = [s for per_text in sentences for s in per_text]
        tags = iter(self.tag([[tok for tok, _, _ in s] for s in flat]))

        results = []
        for text, per_text in zip(texts, sentences):
            entities = []
            for sentence in per_text:
                entities += to_entities(text, sentence, next(tags))
            results.append(to_result(entities))
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Prediksi entitas dari teks.")
    parser.add_argument("--checkpoint", required=True, help="Folder checkpoint hasil ner.train")
    parser.add_argument("--device", default="auto")
    parser.add_argument("text", nargs="+", help="Satu atau lebih teks")
    args = parser.parse_args()

    predictor = NERPredictor(args.checkpoint, args.device)
    for result in predictor.predict_batch(args.text):
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
