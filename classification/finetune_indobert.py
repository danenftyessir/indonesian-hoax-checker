# Skrip Fine-tuning IndoBERT-p1 pada dataset Mafindo-2024

# Train data: Mafindo kolom `content`, 10% dipisah sebagai validasi (split tetap, seed 13)
# Tes memakai `cross_dataset.evaluate`dengan threshold 0,5 pada Mafindo test tanpa judul yang ada di train
# Mendeley 600 (semua, tanpa penanda bantahan, hanya penanda bantahan)
# Tiga seed (default 13 42 2026, docs/experiment_protocol.md)

import argparse, json, random, re, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_dataset as cd 
from short_claims_probe import load_mafindo_frames 
from data_audit import mendeley_audit as ma  

MODEL = "indobenchmark/indobert-base-p1"
METRICS = ("macro_f1", "auroc", "hoax_precision", "hoax_recall")
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip() 


def aggregate(per_seed: list[dict]) -> dict:
    # per_seed: daftar {nama_tes: hasil evaluate}, mengembalikan mean dan sd (ddof=1) tiap metrik antar seed
    out = {}
    for name in per_seed[0]:
        out[name] = {}
        for m in METRICS:
            v = np.array([r[name][m] for r in per_seed], dtype=float)
            out[name][m] = {"mean": round(float(v.mean()), 3),
                            "sd": round(float(v.std(ddof=1)), 3) if len(v) > 1 else None,
                            "values": [round(float(x), 3) for x in v]}
    return out


def eval_sets(df_m, m_y, t_titles, t_y, train_titles, scores_m, scores_t, probe=None):
    # susun hasil evaluate untuk semua tes dari skor probabilitas kelas hoax
    keep_nd = (~df_m.debunk_marker).to_numpy()
    t_new = np.array([norm(t) not in train_titles for t in t_titles])
    res = {
        "mafindo_test": cd.evaluate(t_y, scores_t),
        "mafindo_test_titles_not_in_train": cd.evaluate(t_y[t_new], scores_t[t_new]),
        "mendeley_all": cd.evaluate(m_y, scores_m),
        "mendeley_no_debunk": cd.evaluate(m_y[keep_nd], scores_m[keep_nd]),
        "mendeley_debunk_only": cd.evaluate(m_y[~keep_nd], scores_m[~keep_nd]),
    }
    if probe is not None:
        res["short_claims_probe"] = cd.evaluate(probe[1], probe[0])
    return res


def set_seed(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def encode(tok, texts, max_len):
    return tok(list(texts), truncation=True, max_length=max_len)["input_ids"]


def collate(batch_ids, labels, pad_id, device):
    import torch
    L = max(len(x) for x in batch_ids)
    ids = torch.full((len(batch_ids), L), pad_id, dtype=torch.long)
    for i, x in enumerate(batch_ids):
        ids[i, :len(x)] = torch.tensor(x)
    mask = (ids != pad_id).long()
    out = {"input_ids": ids.to(device), "attention_mask": mask.to(device)}
    if labels is not None:
        out["labels"] = torch.tensor(labels, dtype=torch.long).to(device)
    return out


def predict(model, ids, pad_id, device, bs=128):
    # probabilitas kelas hoax
    import torch
    model.eval()
    order = np.argsort([len(x) for x in ids])
    out = np.zeros(len(ids))
    with torch.no_grad():
        for s in range(0, len(ids), bs):
            idx = order[s:s + bs]
            b = collate([ids[i] for i in idx], None, pad_id, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(**b).logits
            out[idx] = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
    return out


def train_one_seed(a, seed, data, device):
    import torch
    from sklearn.metrics import f1_score
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    set_seed(seed)
    tok = data["tok"]
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(device)
    tr_ids, tr_y, va_ids, va_y = data["tr_ids"], data["tr_y"], data["va_ids"], data["va_y"]
    steps = a.epochs * ((len(tr_ids) + a.batch_size - 1) // a.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    rng = np.random.default_rng(seed)
    best_f1, best_state, log = -1.0, None, []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); losses = []
        perm = rng.permutation(len(tr_ids))
        for s in range(0, len(perm), a.batch_size):
            idx = perm[s:s + a.batch_size]
            b = collate([tr_ids[i] for i in idx], [tr_y[i] for i in idx], tok.pad_token_id, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = model(**b).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(float(loss.detach()))
        p = predict(model, va_ids, tok.pad_token_id, device) >= 0.5
        f1 = float(f1_score(va_y, p, average="macro"))
        log.append({"epoch": ep + 1, "train_loss": round(float(np.mean(losses)), 4), "val_macro_f1": round(f1, 4),
                    "seconds": round(time.time() - t0)})
        print(f"seed {seed} epoch {ep + 1}/{a.epochs} loss {log[-1]['train_loss']} val macroF1 {f1:.4f} ({log[-1]['seconds']}s)", flush=True)
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, log, best_f1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True, help="folder Mendeley mentah")
    ap.add_argument("--out-dir", type=Path, default=Path("results/indobert"))
    ap.add_argument("--save-dir", type=Path, default=None, help="simpan model terbaik tiap seed")
    ap.add_argument("--probe", type=Path, default=None, help="short_claims.jsonl (opsional)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 2026])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--smoke", action="store_true", help="uji alur: data kecil, 1 epoch, 1 seed, panjang 128")
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.seeds, a.max_length, a.out_dir = 1, a.seeds[:1], 128, a.out_dir / "smoke"

    import torch
    from sklearn.model_selection import train_test_split
    from transformers import AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not a.smoke:
        sys.exit("Tidak ada GPU. Jalankan di Colab (Runtime > Change runtime type > GPU), atau pakai --smoke.")
    print(f"device {device} | model {MODEL} | max_length {a.max_length} | epochs {a.epochs} | seeds {a.seeds}")

    # Data
    tr, te = load_mafindo_frames()
    txt = tr.content.fillna("").astype(str).to_numpy(dtype=object); y = tr.is_hoax.astype(int).to_numpy()
    idx_tr, idx_va = train_test_split(np.arange(len(tr)), test_size=0.1, stratify=y, random_state=13)
    t_txt = te.content.fillna("").astype(str).to_numpy(dtype=object); t_y = te.is_hoax.astype(int).to_numpy()
    df_m, _ = ma.build_manifest(a.data_dir)
    m_txt, m_y = df_m.text.to_numpy(dtype=object), df_m.y.to_numpy()
    if a.smoke:
        rng = np.random.default_rng(0)
        idx_tr, idx_va = rng.choice(idx_tr, 512, replace=False), rng.choice(idx_va, 256, replace=False)
        sub = rng.choice(len(te), 400, replace=False)
        te, t_txt, t_y = te.iloc[sub], t_txt[sub], t_y[sub]
    train_titles = {norm(t) for t in tr.title.fillna("")}
    if a.probe and a.probe.exists():
        import pandas as pd
        pr = pd.read_json(a.probe, lines=True)
        probe_txt, probe_y = pr.text.to_numpy(dtype=object), (pr.label == "hoax").astype(int).to_numpy()
    else:
        probe_txt = probe_y = None

    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = lambda xs: encode(tok, xs, a.max_length) 
    data = {"tok": tok, "tr_ids": enc(txt[idx_tr]), "tr_y": y[idx_tr].tolist(), "va_ids": enc(txt[idx_va]), "va_y": y[idx_va]}
    t_ids, m_ids = enc(t_txt), enc(m_txt)
    p_ids = enc(probe_txt) if probe_txt is not None else None
    print(f"train {len(idx_tr)} | val {len(idx_va)} | mafindo test {len(t_y)} | mendeley {len(m_y)}"
          f"{' | probe ' + str(len(probe_y)) if p_ids else ''}")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    per_seed = []
    for seed in a.seeds:
        f = a.out_dir / f"seed{seed}.json"
        if f.exists():
            print(f"seed {seed}: hasil sudah ada, dilewati"); per_seed.append(json.loads(f.read_text())["results"]); continue
        model, log, best_val = train_one_seed(a, seed, data, device)
        pad = tok.pad_token_id
        s_t, s_m = predict(model, t_ids, pad, device), predict(model, m_ids, pad, device)
        s_p = predict(model, p_ids, pad, device) if p_ids else None
        res = eval_sets(df_m, m_y, te.title.fillna("").tolist(), t_y, train_titles, s_m, s_t,
                        (s_p, probe_y) if s_p is not None else None)
        f.write_text(json.dumps({"seed": seed, "best_val_macro_f1": round(best_val, 4), "epochs": log, "results": res},
                                indent=2, ensure_ascii=False))
        np.savez_compressed(a.out_dir / f"scores_seed{seed}.npz", mafindo_test=s_t, mendeley=s_m,
                            **({"probe": s_p} if s_p is not None else {}))
        if a.save_dir and not a.smoke:
            model.save_pretrained(a.save_dir / f"seed{seed}"); tok.save_pretrained(a.save_dir / f"seed{seed}")
        per_seed.append(res)
        for k in ("mafindo_test", "mendeley_all", "mendeley_no_debunk"):
            r = res[k]; print(f"  seed {seed} {k:20s} macroF1 {r['macro_f1']:.3f} AUROC {r['auroc']:.3f}", flush=True)
        del model; torch.cuda.empty_cache()

    agg = aggregate(per_seed)
    meta = {"model": MODEL, "seeds": a.seeds, "epochs": a.epochs, "batch_size": a.batch_size, "lr": a.lr,
            "max_length": a.max_length, "text_field": "content", "threshold": 0.5, "val_split": "10% stratified, seed 13",
            "smoke": a.smoke, "torch": torch.__version__}
    (a.out_dir / "summary.json").write_text(json.dumps({"meta": meta, "aggregate": agg}, indent=2, ensure_ascii=False))
    print("\nRingkasan antar seed (mean ± sd):")
    for name, ms in agg.items():
        sd = lambda m: "n/a" if ms[m]["sd"] is None else f"{ms[m]['sd']:.3f}" 
        print(f"  {name:36s} macroF1 {ms['macro_f1']['mean']:.3f} ± {sd('macro_f1')} | AUROC {ms['auroc']['mean']:.3f} ± {sd('auroc')}")
    print(f"\nhasil -> {a.out_dir}")


if __name__ == "__main__":
    main()
