import csv

from indonesian_hoax_checker.ner.dataset import write_conll
from indonesian_hoax_checker.ner.error_analysis import analyze
from indonesian_hoax_checker.ner.seed_sentences import komdigi_sentences, mafindo_sentences, select, split_text
from indonesian_hoax_checker.ner.train import load_config, load_split


def test_split_text_handles_missing_space():
    assert split_text("Rabu (14/8) sore. Gus Yahya tiba?Pantauan ada.") == ["Rabu (14/8) sore.", "Gus Yahya tiba?", "Pantauan ada."]


def test_split_text_keeps_quoted_sentence_whole():
    text = "Beredar narasi “HEBOH!! Pemerintah menghentikan bantuan.” di Facebook. Faktanya salah."
    assert split_text(text) == ["Beredar narasi “HEBOH!! Pemerintah menghentikan bantuan.” di Facebook.", "Faktanya salah."]


def test_komdigi_prefix_with_category_is_removed(tmp_path):
    path = tmp_path / "k.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, ["id", "published_at", "body_text"])
        w.writeheader()
        w.writerow({"id": "9", "published_at": "2025-02-01", "body_text": "KATEGORI: HOAKS Penjelasan : Beredar unggahan yang mengeklaim bansos dihentikan."})
    assert komdigi_sentences(path)[0]["text"].startswith("Beredar unggahan")


def test_komdigi_takes_first_beredar_sentence(tmp_path):
    path = tmp_path / "k.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # pakai BOM seperti CSV Kaggle
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


def test_select_assigns_splits_and_keeps_test_period_out_of_train():
    komdigi = [{"id": f"k{i}", "text": f"k{i}", "source": "komdigi",
                "published_at": "2026-10-01" if i < 5 else "2025-01-01"} for i in range(30)]
    mafindo = [{"id": f"m{i}", "text": f"m{i}", "source": "mafindo", "published_at": None} for i in range(20)]
    composition = {"train": {"komdigi": 10, "mafindo": 8}, "dev": {"komdigi": 3, "mafindo": 2},
                   "test": {"komdigi": 4, "mafindo": 3}}
    rows = select(komdigi, mafindo, composition)
    by_split = {s: [r for r in rows if r["split"] == s] for s in ("train", "dev", "test")}
    assert [len(by_split[s]) for s in ("train", "dev", "test")] == [18, 5, 7]
    test_komdigi = [r for r in by_split["test"] if r["source"] == "komdigi"]
    assert all(r["published_at"] >= "2026-09-01" for r in test_komdigi)
    assert all(r["published_at"] < "2026-09-01" for r in by_split["train"] + by_split["dev"] if r["source"] == "komdigi")


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
