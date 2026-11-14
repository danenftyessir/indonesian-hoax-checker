# Freeze hash data uji klasifikasi 
import argparse, hashlib, json, subprocess
from datetime import datetime, timezone
from pathlib import Path

LOCAL_FILES = [
    "data/raw/mendeley/600 news with valid hoax label.csv",
    "data/raw/mendeley/600 news labelling process.xlsx",
    "data/manifests/mendeley_manifest.csv",
    "results/mendeley_baseline_repeated.json",
    "outputs/cross_dataset.json",
    "outputs/indobert_results/scores_seed13.npz",
    "outputs/indobert_results/scores_seed42.npz",
    "outputs/indobert_results/scores_seed2026.npz",
    "data/interim/classifier/no_match_queries.jsonl",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_entry(p: str) -> dict:
    path = Path(p)
    if not path.exists():
        return {"file": p, "missing": True}
    return {"file": p, "bytes": path.stat().st_size, "sha256": sha256(path)}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def mafindo_hash() -> dict:
    from datasets import load_dataset
    from huggingface_hub import HfApi
    name = "umn-cyber/mafindo-2024"
    rev = HfApi().dataset_info(name).sha
    out = {"dataset": name, "hf_revision": rev}
    ds = load_dataset(name, revision=rev)
    for split in ("train", "test"):
        d = ds[split].to_pandas()
        rows = sorted(hashlib.sha1((str(t) + "\x1f" + str(c)).encode("utf-8")).hexdigest()
                      for t, c in zip(d.title.fillna(""), d.content.fillna("")))
        out[split] = {"n": int(len(d)), "hoax_rate": round(float(d.is_hoax.mean()), 4),
                      "sorted_row_sha1_digest": hashlib.sha256("\n".join(rows).encode()).hexdigest()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/splits/classification_manifest.json"))
    ap.add_argument("--mafindo-only", action="store_true")
    ap.add_argument("--merge-mafindo", type=Path, default=None)
    a = ap.parse_args()

    if a.mafindo_only:
        res = mafindo_hash()
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2)); return

    man = {"frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": git_commit(),
           "note": "hash saja, tanpa teks pihak ketiga; bekukan ulang pada 18 Nov 2026 dan catat tanggal di sini",
           "files": [file_entry(p) for p in LOCAL_FILES]}
    import csv  
    with open("data/manifests/mendeley_manifest.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    man["mendeley"] = {"n": len(rows), "hoax": sum(int(r["y"]) for r in rows),
                       "text_sha1_digest": hashlib.sha256("\n".join(r["text_sha1"] for r in rows).encode()).hexdigest()}
    if a.merge_mafindo:
        man["mafindo"] = json.loads(a.merge_mafindo.read_text())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(man, indent=2, ensure_ascii=False))
    missing = [f["file"] for f in man["files"] if f.get("missing")]
    print(f"{len(man['files']) - len(missing)} berkas di-hash; hilang: {missing or 'tidak ada'}")
    print(f"Mendeley: {man['mendeley']}\nMafindo: {'digabung' if 'mafindo' in man else 'belum (jalankan di Colab)'}\n-> {a.out}")


if __name__ == "__main__":
    main()
