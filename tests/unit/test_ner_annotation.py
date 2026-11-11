import json

from indonesian_hoax_checker.ner.annotation import (
    adjudicate,
    kappa_report,
    make_tasks,
    read_export,
    spans_to_tags,
    split_seed,
)
from indonesian_hoax_checker.ner.time_rules import add_time_tags, time_spans
from indonesian_hoax_checker.preprocessing.tokenizer import tokenize


def words(text):
    return [t for t, _, _ in tokenize(text)]


def found(text):
    w = words(text)
    return [" ".join(w[a:b]) for a, b in time_spans(w)]


def test_time_rules():
    assert found("Pemerintah menghentikan bantuan mulai bulan depan") == ["bulan depan"]
    assert found("Internet diputus selama tiga hari") == ["selama tiga hari"]
    assert found("Pada Senin, 3 Nov 2026 dan 12/11/2026") == ["Senin , 3 Nov 2026", "12 / 11 / 2026"]
    assert found("stop bansos bsk gaes") == ["bsk"]
    assert found("Jumlah hari kerja di Bank Mandiri") == []


def test_add_time_tags_keeps_existing_entities():
    tokens = ["Jakarta", "besok"]
    assert add_time_tags(tokens, ["B-LOC", "O"]) == ["B-LOC", "B-TIME"]


def test_make_tasks_prefills_time():
    task = make_tasks([{"id": "s1", "text": "Bantuan dihentikan bulan depan"}])[0]
    result = task["predictions"][0]["result"]
    assert task["data"]["id"] == "s1"
    assert [r["value"]["text"] for r in result] == ["bulan depan"]
    assert result[0]["value"]["labels"] == ["TIME"]


def test_spans_to_tags_aligns_partial_spans():
    tokens = tokenize("Pemerintah menghentikan bantuan")
    results = [
        {"type": "labels", "value": {"start": 0, "end": 10, "labels": ["ORG"]}},
        {"type": "labels", "value": {"start": 13, "end": 31, "labels": ["CLAIM"]}},  # mulai di tengah kata
    ]
    assert spans_to_tags(tokens, results) == ["B-ORG", "B-CLAIM", "I-CLAIM"]


def _export(tmp_path):
    def ann(who, spans):
        return {"completed_by": {"email": who}, "result": [
            {"type": "labels", "value": {"start": s, "end": e, "labels": [k]}} for s, e, k in spans]}

    text = "Pemerintah menghentikan bantuan besok"
    tasks = [
        {"data": {"id": "a", "text": text, "source": "komdigi", "published_at": "2026-10-01"},
         "annotations": [ann("x", [(0, 10, "ORG"), (32, 37, "TIME")]), ann("y", [(0, 10, "ORG"), (32, 37, "TIME")])]},
        {"data": {"id": "b", "text": text, "source": "komdigi", "published_at": "2025-01-01"},
         "annotations": [ann("x", [(11, 31, "CLAIM")]), ann("y", [(0, 10, "ORG")])]},
    ]
    path = tmp_path / "export.json"
    path.write_text(json.dumps(tasks))
    return read_export(path)


def test_kappa_and_adjudication(tmp_path):
    items = _export(tmp_path)
    report = kappa_report(items)
    assert report["annotators"] == ["x", "y"]
    assert report["mean"]["TIME"] == 1.0
    assert report["mean"]["CLAIM"] < 0.6

    final, unresolved = adjudicate(items)
    assert [r["id"] for r in final] == ["a"] and unresolved == ["b"]
    final, unresolved = adjudicate(items, adjudicator="x")
    assert len(final) == 2 and final[1]["tags"] == ["O", "B-CLAIM", "I-CLAIM", "O"]


def test_split_forces_test_period_into_test():
    rows = [{"id": f"k{i}", "source": "komdigi", "published_at": "2026-10-01"} for i in range(3)]
    rows += [{"id": f"m{i}", "source": "mafindo", "published_at": None} for i in range(20)]
    train, dev, test = split_seed(rows, n_dev=5, n_test=5)
    assert {r["id"] for r in rows if r["source"] == "komdigi"} <= {r["id"] for r in test}
    assert (len(train), len(dev), len(test)) == (13, 5, 5)
