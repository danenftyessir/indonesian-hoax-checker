# Latih dan evaluasi model NER (BiLSTM-CRF atau IndoBERT) untuk beberapa seed.
# Jalankan: python -m indonesian_hoax_checker.ner.train --config configs/models/ner_bilstm_crf.yaml
#
# Per seed menyimpan:
#   artifacts/checkpoints/ner/{name}_seed{seed}/   model terbaik (F1 dev tertinggi)
#   outputs/metrics/{name}_seed{seed}.json         skor dev dan test
#   outputs/predictions/{name}_seed{seed}.conll    token, label gold, label prediksi (test)
# Ringkasan semua seed (rata-rata dan simpangan baku): outputs/metrics/{name}_summary.json

import argparse
import json
import random
import statistics
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

from indonesian_hoax_checker.ner.dataset import read_conll
from indonesian_hoax_checker.ner.labels import LABEL2ID, LABELS, to_iob2
from indonesian_hoax_checker.ner.metrics import entity_scores
from indonesian_hoax_checker.ner.models import bilstm_crf, indobert


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def load_split(path: str, lowercase: bool):
    sentences = read_conll(path)
    if lowercase:
        sentences = [([t.lower() for t in tokens], tags) for tokens, tags in sentences]
    return sentences


def batches(items: list, size: int, shuffle: bool = False):
    order = list(range(len(items)))
    if shuffle:
        random.shuffle(order)
    for i in range(0, len(order), size):
        yield [items[j] for j in order[i:i + size]]


# ---------- BiLSTM-CRF ----------

class BiLSTMRunner:
    def __init__(self, cfg, train, device):
        self.cfg, self.device = cfg, device
        self.words, self.chars = bilstm_crf.build_vocabs([tokens for tokens, _ in train])
        self.model = bilstm_crf.BiLSTMCRF(len(self.words), len(self.chars), len(LABELS), **cfg["model_args"]).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["lr"])

    def train_epoch(self, train):
        self.model.train()
        for batch in batches(train, self.cfg["batch_size"], shuffle=True):
            word_ids, char_ids, mask = bilstm_crf.encode_batch([t for t, _ in batch], self.words, self.chars)
            label_ids = torch.zeros_like(word_ids)
            for i, (_, tags) in enumerate(batch):
                label_ids[i, :len(tags)] = torch.tensor([LABEL2ID[t] for t in tags])
            loss = self.model.loss(word_ids.to(self.device), char_ids.to(self.device), mask.to(self.device),
                                   label_ids.to(self.device))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

    @torch.no_grad()
    def predict(self, token_lists):
        self.model.eval()
        preds = []
        for batch in batches(token_lists, 64):
            word_ids, char_ids, mask = bilstm_crf.encode_batch(batch, self.words, self.chars)
            for ids in self.model.decode(word_ids.to(self.device), char_ids.to(self.device), mask.to(self.device)):
                preds.append([LABELS[i] for i in ids])
        return preds

    def save(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out_dir / "model.pt")
        (out_dir / "vocab.json").write_text(json.dumps({"words": self.words.itos, "chars": self.chars.itos}))
        (out_dir / "config.json").write_text(json.dumps(self.cfg, indent=2))

    def load_best(self, out_dir: Path):
        self.model.load_state_dict(torch.load(out_dir / "model.pt", map_location=self.device))


# ---------- IndoBERT ----------

class IndoBERTRunner:
    def __init__(self, cfg, train, device):
        from transformers import get_linear_schedule_with_warmup

        self.cfg, self.device = cfg, device
        self.tokenizer, self.model = indobert.load(cfg["model_name"])
        self.model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.01))
        total = cfg["epochs"] * ((len(train) + cfg["batch_size"] - 1) // cfg["batch_size"])
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, int(cfg.get("warmup_ratio", 0.1) * total), total)

    def train_epoch(self, train):
        self.model.train()
        for batch in batches(train, self.cfg["batch_size"], shuffle=True):
            enc = indobert.encode_batch(self.tokenizer, [t for t, _ in batch], [g for _, g in batch], self.cfg["max_length"])
            loss = self.model(**{k: v.to(self.device) for k, v in enc.items()}).loss
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def predict(self, token_lists):
        self.model.eval()
        preds = []
        for batch in batches(token_lists, 32):
            enc = indobert.encode_batch(self.tokenizer, batch, max_length=self.cfg["max_length"])
            logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits.cpu()
            preds.extend(indobert.word_predictions(enc, logits, batch))
        return preds

    def save(self, out_dir: Path):
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        (out_dir / "train_config.json").write_text(json.dumps(self.cfg, indent=2))

    def load_best(self, out_dir: Path):
        from transformers import AutoModelForTokenClassification

        self.model = AutoModelForTokenClassification.from_pretrained(out_dir).to(self.device)


RUNNERS = {"bilstm_crf": BiLSTMRunner, "indobert": IndoBERTRunner}


# ---------- Alur latih per seed ----------

def evaluate(runner, sentences):
    # to_iob2 merapikan prediksi I-X yatim supaya span tetap terbaca
    preds = [to_iob2(p) for p in runner.predict([tokens for tokens, _ in sentences])]
    return entity_scores([tags for _, tags in sentences], preds), preds


def run(cfg: dict, seed: int) -> dict:
    set_seed(seed)
    device = get_device(cfg.get("device", "auto"))
    lowercase = cfg.get("lowercase", False)
    train, dev, test = (load_split(cfg["data"][s], lowercase) for s in ("train", "dev", "test"))

    run_name = f"{cfg['name']}_seed{seed}"
    ckpt_dir = Path("artifacts/checkpoints/ner") / run_name
    runner = RUNNERS[cfg["model"]](cfg, train, device)

    best_f1, best_epoch, waited = -1.0, 0, 0
    for epoch in range(1, cfg["epochs"] + 1):
        runner.train_epoch(train)
        dev_f1 = evaluate(runner, dev)[0]["f1"]
        print(f"[{run_name}] epoch {epoch}: dev F1 {dev_f1:.4f}", flush=True)
        if dev_f1 > best_f1:
            best_f1, best_epoch, waited = dev_f1, epoch, 0
            runner.save(ckpt_dir)
        else:
            waited += 1
            if waited >= cfg["patience"]:
                break

    runner.load_best(ckpt_dir)
    dev_scores, _ = evaluate(runner, dev)
    test_scores, test_preds = evaluate(runner, test)

    result = {
        "name": cfg["name"], "model": cfg["model"], "seed": seed, "device": str(device),
        "git_commit": git_commit(), "best_epoch": best_epoch, "data": cfg["data"],
        "dev": dev_scores, "test": test_scores,
    }
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    Path(f"outputs/metrics/{run_name}.json").write_text(json.dumps(result, indent=2))

    Path("outputs/predictions").mkdir(parents=True, exist_ok=True)
    lines = []
    for (tokens, gold), pred in zip(test, test_preds):
        lines += [f"{t}\t{g}\t{p}" for t, g, p in zip(tokens, gold, pred)] + [""]
    Path(f"outputs/predictions/{run_name}.conll").write_text("\n".join(lines), encoding="utf-8")

    print(f"[{run_name}] best epoch {best_epoch}, test F1 {test_scores['f1']:.4f}", flush=True)
    return result


def summarize(name: str, results: list[dict]) -> dict:
    def mean_std(values):
        return {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0}

    types = sorted({k for r in results for k in r["test"]["per_type"]})
    summary = {
        "name": name,
        "seeds": [r["seed"] for r in results],
        "test_f1": mean_std([r["test"]["f1"] for r in results]),
        "test_f1_per_type": {t: mean_std([r["test"]["per_type"].get(t, {"f1": 0.0})["f1"] for r in results]) for t in types},
        "dev_f1": mean_std([r["dev"]["f1"] for r in results]),
    }
    Path(f"outputs/metrics/{name}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Latih model NER.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 2026])
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    results = [run(cfg, seed) for seed in args.seeds]
    print(json.dumps(summarize(cfg["name"], results), indent=2))


if __name__ == "__main__":
    main()
