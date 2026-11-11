import csv

from indonesian_hoax_checker.ner.annotation import split_seed
from indonesian_hoax_checker.ner.dataset import write_conll
from indonesian_hoax_checker.ner.error_analysis import analyze
from indonesian_hoax_checker.ner.seed_sentences import komdigi_sentences, mafindo_sentences, select, split_text
from indonesian_hoax_checker.ner.train import load_config, load_split


def test_split_text_handles_missing_space():
    assert split_text("Rabu (14/8) sore. Gus Yahya tiba?Pantauan ada.") == ["Rabu (14/8) sore.", "Gus Yahya tiba?", "Pantauan ada."]


def test_komdigi_takes_first_beredar_sentence(tmp_path):
    path = tmp_path / "k.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["id", "published_at", "body_text"])
        w.writeheader()
        w.writerow({"id": "7", "published_at": "2026-10-05T08:00:00", "body_text":
                    "Penjelasan: Beredar pesan yang mengeklaim Kominfo akan memutus internet nasional. Faktanya tidak benar."})
        w.writerow({"id": "8", "published_at": "2026-01-01", "body_text": "Tidak ada kalimat yang cocok di artikel ini."})
    rows = komdigi_sentences(path)
    assert len(rows) == 1
    assert rows[0]["text"].startswith("Beredar pesan") and rows[0]["published_at"] == "2026-10-05"


def test_mafindo_keeps_only_valid_articles():
    articles = [
        {"is_hoax": 0, "content": "Presiden Joko Widodo meresmikan bandara baru di Kediri. Kalimat kedua."},
        {"is_hoax": 1, "content": "Artikel cek fakta yang membantah sebuah hoaks beredar luas."},
    ]
    rows = mafindo_sentences(articles)
    assert [r["text"] for r in rows] == ["Presiden Joko Widodo meresmikan bandara baru di Kediri."]


def test_select_marks_calibration_by_source_ratio():
    komdigi = [{"text": f"k{i}", "source": "komdigi"} for i in range(30)]
    mafindo = [{"text": f"m{i}", "source": "mafindo"} for i in range(10)]
    rows = select(komdigi, mafindo, n_komdigi=30, n_mafindo=10, n_calibration=8)
    calibration = [r for r in rows if r["batch"] == "calibration"]
    assert len(rows) == 40
    assert sum(r["source"] == "komdigi" for r in calibration) == 6 and len(calibration) == 8


def test_split_seed_excludes_informal():
    rows = [{"id": "i1", "source": "informal"}] + [{"id": f"m{i}", "source": "mafindo"} for i in range(10)]
    train, dev, test = split_seed(rows, n_dev=2, n_test=2)
    assert "i1" not in {r["id"] for r in train + dev + test}


def test_load_config_inherits_base(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("name: a\nlr: 0.1\ndata:\n  train: t.conll\n  test: x.conll\n")
    child = tmp_path / "child.yaml"
    child.write_text(f"base: {base}\nname: b\ndata:\n  train: [t.conll, u.conll]\n")
    assert load_config(child) == {"name": "b", "lr": 0.1, "data": {"train": ["t.conll", "u.conll"], "test": "x.conll"}}


def test_load_split_merges_files_and_normalizes(tmp_path):
    write_conll(tmp_path / "a.conll", [(["Joko", "bsk"], ["B-PER", "O"])])
    write_conll(tmp_path / "b.conll", [(["Bandung"], ["B-LOC"])])
    data = load_split([str(tmp_path / "a.conll"), str(tmp_path / "b.conll")], lowercase=True, normalize=True)
    assert data == [(["joko", "besok"], ["B-PER", "O"]), (["bandung"], ["B-LOC"])]


def test_error_analysis_categories():
    tokens = ["Joko", "Widodo", "di", "Bank", "Mandiri", "dan", "Bali", "x"]
    gold = ["B-PER", "I-PER", "O", "B-ORG", "I-ORG", "O", "B-LOC", "O"]
    pred = ["B-PER", "O", "O", "B-LOC", "I-LOC", "O", "O", "B-ORG"]
    report = analyze([(tokens, gold, pred)])
    assert report["per_tipe"]["PER"]["salah_batas"] == 1
    assert report["per_tipe"]["ORG"]["salah_tipe"] == 1
    assert report["per_tipe"]["LOC"]["terlewat"] == 1
    assert report["salah_deteksi"] == {"ORG": 1}
    assert report["salah_tipe_menjadi"] == {"ORG": {"LOC": 1}}
