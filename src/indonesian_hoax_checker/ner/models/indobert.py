# Model kuat NER: IndoBERT token classification.
# Label hanya diberikan ke subword pertama tiap kata; subword lain diberi -100 (diabaikan loss).

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from indonesian_hoax_checker.ner.labels import LABEL2ID, LABELS


def load(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=LABEL2ID,
    )
    return tokenizer, model


def encode_batch(tokenizer, batch_tokens, batch_tags=None, max_length=512):
    enc = tokenizer(batch_tokens, is_split_into_words=True, truncation=True, max_length=max_length,
                    padding=True, return_tensors="pt")
    if batch_tags is not None:
        labels = torch.full(enc["input_ids"].shape, -100, dtype=torch.long)
        for i, tags in enumerate(batch_tags):
            prev = None
            for j, word in enumerate(enc.word_ids(i)):
                if word is not None and word != prev:
                    labels[i, j] = LABEL2ID[tags[word]]
                prev = word
        enc["labels"] = labels
    return enc


# Ambil prediksi subword pertama tiap kata; kata yang terpotong (truncation) diberi O
def word_predictions(enc, logits, batch_tokens) -> list[list[str]]:
    pred_ids = logits.argmax(-1).tolist()
    results = []
    for i, tokens in enumerate(batch_tokens):
        tags = ["O"] * len(tokens)
        prev = None
        for j, word in enumerate(enc.word_ids(i)):
            if word is not None and word != prev:
                tags[word] = LABELS[pred_ids[i][j]]
            prev = word
        results.append(tags)
    return results
