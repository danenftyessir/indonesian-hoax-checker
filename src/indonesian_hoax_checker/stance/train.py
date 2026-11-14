# Fine-tuning model stance (RQ4) dengan HF Trainer, dua tahap:
#   tahap 1: IndoNLI train (label e/n/c -> supports/refutes/unrelated)
#   tahap 2: lanjut dari checkpoint tahap 1 pada pasangan BRIN (+ parafrasa) dan/atau tim
#
#   python -m indonesian_hoax_checker.stance.train configs/models/stance_nli.yaml \
#       --train data/processed/stance/indonli_train.jsonl \
#       --dev data/processed/stance/indonli_val.jsonl \
#       --output artifacts/checkpoints/stance_indonli --seed 42
#   python -m indonesian_hoax_checker.stance.train configs/models/stance_nli.yaml \
#       --init artifacts/checkpoints/stance_indonli \
#       --train data/processed/stance/brin_refutes.jsonl data/processed/stance/team_train.jsonl \
#       --dev data/processed/stance/team_dev.jsonl --output artifacts/checkpoints/stance_brin
#
# Butuh requirements/training.txt dan GPU (Colab/Kaggle). Tidak dijalankan di laptop demo.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from indonesian_hoax_checker.retrieval.corpus import read_jsonl
from indonesian_hoax_checker.stance.labels import ID2LABEL, LABEL2ID, STANCE_LABELS
from indonesian_hoax_checker.stance.metrics import macro_f1


def load_pairs(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in paths:
        rows.extend(read_jsonl(p))
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fine-tune model stance NLI.")
    p.add_argument("config", help="configs/models/stance_nli.yaml")
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--init", default=None, help="checkpoint awal (tahap 2); default model_name config")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--max-train", type=int, default=None, help="potong data latih (uji cepat)")
    args = p.parse_args(argv)

    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    model_name = args.init or cfg.get("model_name")
    if not model_name:
        raise SystemExit("model_name di config masih null; isi dulu (lihat docs/decisions)")
    hp = cfg.get("hyperparameters", {})
    set_seed(args.seed)

    train_rows = load_pairs(args.train)
    if args.max_train:
        train_rows = train_rows[: args.max_train]
    dev_rows = load_pairs([args.dev])

    tok = AutoTokenizer.from_pretrained(model_name)
    max_len = int(hp.get("max_length", 256))

    def encode(batch):
        enc = tok(batch["premise"], batch["hypothesis"], truncation=True, max_length=max_len)
        enc["labels"] = [LABEL2ID[lab] for lab in batch["label"]]
        return enc

    cols = ["premise", "hypothesis", "label"]
    # remove_columns: kolom `label` (string) harus dibuang. DataCollatorWithPadding mengganti nama
    # `label` -> `labels`, sehingga string itu menimpa `labels` integer dan gagal dijadikan tensor.
    ds_train = Dataset.from_list([{k: r[k] for k in cols} for r in train_rows]).map(
        encode, batched=True, remove_columns=cols)
    ds_dev = Dataset.from_list([{k: r[k] for k in cols} for r in dev_rows]).map(
        encode, batched=True, remove_columns=cols)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(STANCE_LABELS),
        id2label={i: lab for i, lab in ID2LABEL.items()},
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,  # kepala klasifikasi model NLI siap pakai diganti bila jumlah label beda
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        pred = np.argmax(logits, axis=-1)
        return {"macro_f1": macro_f1([ID2LABEL[int(i)] for i in labels], [ID2LABEL[int(i)] for i in pred])}

    # transformers 5 menghapus `warmup_ratio`; hitung warmup_steps sendiri supaya jalan di semua versi
    bs = int(hp.get("batch_size", 16))
    epochs = float(args.epochs or hp.get("epochs", 3))
    total_steps = max(1, int(-(-len(train_rows) // bs) * epochs))
    warmup_steps = int(total_steps * float(hp.get("warmup_ratio", 0.06)))

    targs = TrainingArguments(
        output_dir=args.output,
        seed=args.seed,
        learning_rate=float(hp.get("learning_rate", 2e-5)),
        per_device_train_batch_size=int(hp.get("batch_size", 16)),
        per_device_eval_batch_size=int(hp.get("batch_size", 16)) * 2,
        num_train_epochs=float(args.epochs or hp.get("epochs", 3)),
        weight_decay=float(hp.get("weight_decay", 0.01)),
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        save_total_limit=1,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_train,
        eval_dataset=ds_dev,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    dev_metrics = trainer.evaluate()
    trainer.save_model(args.output)
    tok.save_pretrained(args.output)
    meta = {
        "base_model": model_name,
        "train_files": args.train,
        "n_train": len(train_rows),
        "dev_file": args.dev,
        "seed": args.seed,
        "hyperparameters": hp,
        "dev_metrics": dev_metrics,
    }
    Path(args.output, "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
