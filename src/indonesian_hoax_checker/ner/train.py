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
from indonesian_hoax_checker.preprocessing.augment import augment_corpus
from indonesian_hoax_checker.preprocessing.normalize import normalize as normalize_tokens


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


# path boleh satu file atau daftar file (digabung), mis. NERGrit + NERP untuk varian lowercase.
# normalize/lowercase diterapkan ke token saja, label tidak berubah.
def load_split(path, lowercase: bool = False, normalize: bool = False):
    paths = path if isinstance(path, list) else [path]
    sentences = [s for p in paths for s in read_conll(p)]
    if normalize:
        sentences = [(normalize_tokens(tokens), tags) for tokens, tags in sentences]
    if lowercase:
        sentences = [([t.lower() for t in tokens], tags) for tokens, tags in sentences]
    return sentences


def load_tokens(tokens, lowercase=False, normalize=False):
    if normalize:
        tokens = normalize_tokens(tokens)
    return [t.lower() for t in tokens] if lowercase else tokens


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
        if cfg.get("init_from"):
            # Tahap 2: lanjutkan dari model tahap 1, kamus diperluas dengan kata baru
            self.model, self.words, self.chars, first_cfg = bilstm_crf.load(cfg["init_from"], device)
            self.cfg = {**cfg, "model_args": first_cfg["model_args"]}
            bilstm_crf.grow_vocab(self.model, self.words, self.chars, [tokens for tokens, _ in train])
        else:
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

    def predict(self, token_lists):
        return bilstm_crf.predict(self.model, self.words, self.chars, token_lists, self.device)

    def save(self, out_dir: Path):
        bilstm_crf.save(out_dir, self.model, self.words, self.chars, self.cfg)

    def load_best(self, out_dir: Path):
        self.model.load_state_dict(torch.load(out_dir / "model.pt", map_location=self.device))


# ---------- IndoBERT ----------

class IndoBERTRunner:
    def __init__(self, cfg, train, device):
        from transformers import get_linear_schedule_with_warmup

        self.cfg, self.device = cfg, device
        # Tahap 2: mulai dari checkpoint tahap 1, bukan dari model dasar
        self.tokenizer, self.model = indobert.load(cfg.get("init_from") or cfg["model_name"])
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

    def predict(self, token_lists):
        return indobert.predict(self.model, self.tokenizer, token_lists, self.device, self.cfg["max_length"])

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
    if cfg.get("init_from"):
        # init_from boleh memuat {seed}, mis. artifacts/checkpoints/ner/ner_indobert_seed{seed}
        cfg = {**cfg, "init_from": cfg["init_from"].format(seed=seed)}
    device = get_device(cfg.get("device", "auto"))
    lowercase, normalize = cfg.get("lowercase", False), cfg.get("normalize", False)
    train, dev, test = (load_split(cfg["data"][s], lowercase, normalize) for s in ("train", "dev", "test"))
    if cfg.get("augment"):
        # Tambah salinan informal dari data train (RQ1); dev/test tidak diubah
        copies = augment_corpus(train, copies=cfg["augment"].get("copies", 1), seed=seed)
        train = train + [(load_tokens(t, lowercase, normalize), g) for t, g in copies]

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
    # Evaluasi tambahan, mis. NERGrit test pada model tahap 2 untuk cek PER/ORG/LOC tidak turun
    for name, path in cfg.get("extra_eval", {}).items():
        if not Path(path).exists():
            print(f"[{run_name}] lewati evaluasi {name}: {path} belum ada", flush=True)
            continue
        result.setdefault("extra", {})[name] = evaluate(runner, load_split(path, lowercase, normalize))[0]
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    Path(f"outputs/metrics/{run_name}.json").write_text(json.dumps(result, indent=2))

    Path("outputs/predictions").mkdir(parents=True, exist_ok=True)
    lines = []
    for (tokens, gold), pred in zip(test, test_preds):
        lines += [f"{t}\t{g}\t{p}" for t, g, p in zip(tokens, gold, pred)] + [""]
    Path(f"outputs/predictions/{run_name}.conll").write_text("\n".join(lines), encoding="utf-8")

    print(f"[{run_name}] best epoch {best_epoch}, test F1 {test_scores['f1']:.4f}", flush=True)
    return result


# Config boleh punya "base: <file>" untuk mewarisi config lain; isi file ini menimpa base
def load_config(path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if "base" in cfg:
        parent = load_config(cfg.pop("base"))
        cfg = {**parent, **cfg, "data": {**parent.get("data", {}), **cfg.get("data", {})}}
    return cfg


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
        "extra_f1": {k: mean_std([r["extra"][k]["f1"] for r in results if k in r.get("extra", {})])
                     for k in results[0].get("extra", {})},
    }
    Path(f"outputs/metrics/{name}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Latih model NER.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 2026])
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = [run(cfg, seed) for seed in args.seeds]
    print(json.dumps(summarize(cfg["name"], results), indent=2))


if __name__ == "__main__":
    main()
