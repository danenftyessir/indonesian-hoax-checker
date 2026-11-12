import json

import numpy as np
import pytest

from indonesian_hoax_checker.evaluation import splits as S
from indonesian_hoax_checker.retrieval import corpus as C
from indonesian_hoax_checker.retrieval import metrics as M
from indonesian_hoax_checker.retrieval.bm25 import BM25, tokenize
from indonesian_hoax_checker.retrieval.candidate_filter import EntityCandidateFilter, entity_keys
from indonesian_hoax_checker.retrieval.search import Retriever

CLAIMS = [
    {"claim_id": "1", "claim": "Pemerintah menghentikan bantuan Makan Bergizi Gratis", "claim_sentence": None,
     "clarification": "Faktanya program tetap berjalan.", "url": "u1", "published_at": "2026-09-10T00:00:00+00:00"},
    {"claim_id": "2", "claim": "Tautan pendaftaran bansos BLT Rp600 ribu dari Kemensos", "claim_sentence": None,
     "clarification": "", "url": "u2", "published_at": "2026-03-01T00:00:00+00:00"},
    {"claim_id": "3", "claim": "Link pendaftaran bansos BLT Rp600 ribu Kemensos", "claim_sentence": None,
     "clarification": "", "url": "u3", "published_at": "2026-10-01T00:00:00+00:00"},
    {"claim_id": "4", "claim": "Video gempa besar mengguncang Bali hari ini", "claim_sentence": None,
     "clarification": "", "url": "u4", "published_at": ""},
]


# --- corpus -----------------------------------------------------------------

def test_strip_prefix_variants():
    assert C.strip_prefix("[HOAKS] Pemerintah stop MBG") == ("Pemerintah stop MBG", ["HOAKS"])
    assert C.strip_prefix("[HOAKS] [DISINFORMASI] Teks") == ("Teks", ["HOAKS", "DISINFORMASI"])
    assert C.strip_prefix("Tanpa prefiks") == ("Tanpa prefiks", [])
    assert C.strip_prefix("[hoaks]: judul") == ("judul", ["HOAKS"])


def test_extract_claim_and_fact_sentence():
    body = ("Penjelasan: Beredar sebuah unggahan yang mengeklaim pemerintah menghentikan MBG. "
            "Unggahan itu viral. Faktanya, program tetap berjalan sesuai jadwal. Komdigi mengimbau warga.")
    assert C.extract_claim_sentence(body).startswith("Beredar sebuah unggahan")
    assert C.extract_fact_sentence(body) == "Faktanya, program tetap berjalan sesuai jadwal."
    assert C.extract_claim_sentence("Tidak ada apa-apa di sini.") is None


def test_clean_html_and_row_to_claim():
    row = {"id": "9", "title": "[HOAKS] Judul &amp; lain", "body_text": "",
           "body_html": "<p>Beredar pesan <b>berantai</b>.</p><p>Faktanya, tidak benar.</p>",
           "url": "http://x", "published_at": "2026-09-05 10:00:00", "tags": '["Kesehatan", "Penipuan"]', "topics": "politik"}
    c = C.row_to_claim(row)
    assert c["claim"] == "Judul & lain"
    assert c["claim_sentence"] == "Beredar pesan berantai."
    assert c["fact_sentence"] == "Faktanya, tidak benar."
    assert c["tags"] == ["kesehatan", "penipuan", "politik"]
    assert c["published_at"].startswith("2026-09-05")


def test_build_claims_drops_duplicates_and_empty(tmp_path):
    import pandas as pd

    df = pd.DataFrame([
        {"id": "1", "title": "[HOAKS] A", "published_at": "2026-01-01"},
        {"id": "1", "title": "[HOAKS] A lagi", "published_at": "2026-01-01"},
        {"id": "2", "title": "[HOAKS]", "published_at": "2026-01-01"},
    ])
    assert [c["claim_id"] for c in C.build_claims(df)] == ["1"]
    df.to_csv(tmp_path / "k.csv", index=False)
    out = tmp_path / "o.jsonl"
    assert C.main([str(tmp_path / "k.csv"), str(out)]) == 0
    assert len(C.load_claims(out)) == 1


# --- near-duplicate + split ---------------------------------------------------

def test_cluster_near_duplicates_groups_similar_titles():
    texts = [c["claim"] for c in CLAIMS]
    cids, pairs, sim = S.cluster_near_duplicates(texts, threshold=0.6)
    assert cids[1] == cids[2]
    assert cids[0] != cids[1] and cids[3] not in (cids[0], cids[1])
    assert pairs and pairs[0][:2] == (1, 2)
    assert sim.shape == (4, 4) and sim[0, 0] == 0


def test_assign_time_split_uses_newest_in_cluster():
    cids = [0, 1, 1, 2]
    sp = S.assign_time_split(CLAIMS, cids, test_start="2026-09-01", dev_start="2026-07-01")
    assert sp == ["test", "test", "test", "train"]  # klaster 1: anggota Mar + Okt -> test


def test_assign_time_split_rejects_bad_dates():
    with pytest.raises(ValueError):
        S.assign_time_split(CLAIMS, [0, 1, 2, 3], test_start="2026-01-01", dev_start="2026-06-01")


def test_splits_main_writes_outputs(tmp_path):
    raw = tmp_path / "raw.jsonl"
    C.write_jsonl(CLAIMS, raw)
    out = tmp_path / "claims.jsonl"
    assert S.main([str(raw), "--output", str(out), "--review-dir", str(tmp_path / "rev"),
                   "--splits-dir", str(tmp_path / "splits"), "--threshold", "0.6"]) == 0
    claims = C.load_claims(out)
    assert all("cluster_id" in c and "split" in c for c in claims)
    manifest = json.loads((tmp_path / "splits" / "manifest.json").read_text())
    assert manifest["n_clusters"] == 3 and len(manifest["files"]) == 2
    assert (tmp_path / "rev" / "near_duplicate_review.csv").exists()
    assert (tmp_path / "splits" / "refuted_claims_split.csv").exists()
    # hash di manifest harus sama di semua OS: file yang di-hash tidak boleh berisi CRLF
    assert b"\r" not in (tmp_path / "splits" / "refuted_claims_split.csv").read_bytes()
    assert b"\r" not in (tmp_path / "splits" / "manifest.json").read_bytes()


# --- BM25 -------------------------------------------------------------------

def test_tokenize():
    assert tokenize("Rp600 ribu, BLT!") == ["rp600", "ribu", "blt"]


def test_bm25_ranks_relevant_first_and_roundtrips(tmp_path):
    docs = [c["claim"] for c in CLAIMS]
    bm = BM25().fit(docs)
    s = bm.score("bansos BLT Kemensos")
    assert set(np.argsort(-s)[:2]) == {1, 2}
    assert s[3] == 0
    bm.save(tmp_path / "bm25.json")
    s2 = BM25.load(tmp_path / "bm25.json").score("bansos BLT Kemensos")
    assert np.allclose(s, s2)


# --- candidate filter -----------------------------------------------------------

def test_entity_filter_mask_and_fallback():
    corpus_entities = [
        [{"text": "Pemerintah", "label": "ORG", "start": 0, "end": 10}],
        [{"text": "Kemensos", "label": "ORG", "start": 0, "end": 8}],
        [{"text": "Kemensos", "label": "ORG", "start": 0, "end": 8}],
        [{"text": "Bali", "label": "LOC", "start": 0, "end": 4}],
    ]
    f = EntityCandidateFilter(corpus_entities)
    q = [{"text": "Kemensos", "label": "ORG", "start": 0, "end": 8}]
    assert f.mask(q).tolist() == [False, True, True, False]
    assert f.mask([]).all()  # tanpa entitas: semua lolos
    assert f.mask([{"text": "Jokowi", "label": "PER", "start": 0, "end": 6}]).all()  # fallback
    assert f.mask([{"text": "besok", "label": "TIME", "start": 0, "end": 5}]).all()  # TIME diabaikan
    assert "kementerian" in entity_keys([{"text": "Kementerian Kesehatan RI", "label": "ORG", "start": 0, "end": 1}])


def test_retriever_returns_schema_and_applies_filter():
    r = Retriever(CLAIMS, scorer="bm25", top_k=3, min_score=0.1)
    res = r.retrieve("bansos BLT Kemensos")
    assert res["scorer"] == "bm25"
    assert res["matched_claim"] == res["candidates"][0]["claim"]
    assert res["similarity_score"] == res["candidates"][0]["score"]
    assert {c["claim_id"] for c in res["candidates"]} == {"2", "3"}
    assert set(res["candidates"][0]) == {"claim_id", "claim", "clarification", "url", "published_at", "score"}

    empty = r.retrieve("xyz tidak ada")
    assert empty["candidates"] == [] and empty["matched_claim"] is None

    f = EntityCandidateFilter([[{"text": "Bali", "label": "LOC", "start": 0, "end": 4}]] + [[]] * 3)
    r2 = Retriever(CLAIMS, scorer="bm25", entity_filter=f)
    ids = r2.ranked_ids("bansos BLT", entities=[{"text": "Bali", "label": "LOC", "start": 0, "end": 4}])
    assert ids == ["1"]  # hanya dokumen yang berbagi entitas yang tersisa


# --- metrics ----------------------------------------------------------------

def test_ranking_metrics():
    run = {"a": ["1", "2", "3"], "b": ["9", "9", "2"], "c": []}
    gold = {"a": {"1"}, "b": {"2", "3"}, "c": {"5"}}
    m = M.evaluate_rankings(run, gold, ks=(1, 3))
    assert m["success@1"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["success@3"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["mrr"] == pytest.approx((1 + 1 / 3 + 0) / 3, abs=1e-4)
    assert m["recall@3"] == pytest.approx((1 + 0.5 + 0) / 3, abs=1e-4)


def test_no_match_rejection_and_threshold_search():
    top = {"a": 5.0, "b": 0.5, "c": None, "d": 4.0}
    nm = {"a": False, "b": True, "c": True, "d": False}
    r = M.no_match_rejection(top, nm, threshold=1.0)
    assert r["rejection_rate_no_match"] == 1.0 and r["false_rejection_rate"] == 0.0
    best = M.best_rejection_threshold(top, nm)
    assert 0.5 < best["threshold"] <= 4.0


def test_recall_by_claim_age_buckets():
    run = {"a": ["1"], "b": ["x"]}
    gold = {"a": {"1"}, "b": {"2"}}
    dates = {"1": "2026-10-01T00:00:00+00:00", "2": "2025-01-01T00:00:00+00:00"}
    r = M.recall_by_claim_age(run, gold, dates, "2026-11-18", k=10)
    assert r["<3m"]["n"] == 1 and r["<3m"]["success@10"] == 1.0
    assert r[">=12m"]["n"] == 1 and r[">=12m"]["success@10"] == 0.0


def test_annotate_entities_skips_empty_text_and_batches():
    from indonesian_hoax_checker.retrieval import annotate_entities as A

    calls = []

    def fake_predict(texts):
        calls.append(list(texts))
        return [{"entities": [{"text": t.split()[0], "label": "PER", "start": 0, "end": len(t.split()[0])}],
                 "core_claim": t} for t in texts]

    rows = [{"query": "Jokowi ke Bali"}, {"query": ""}, {"query": "  "}, {"query": "Prabowo teken SK"}]
    A.annotate_rows(rows, "query", fake_predict, batch_size=1)
    assert calls == [["Jokowi ke Bali"], ["Prabowo teken SK"]]  # teks kosong tidak dikirim ke model
    assert rows[0]["entities"][0]["text"] == "Jokowi" and rows[1]["entities"] == [] and rows[2]["core_claim"] == ""
    s = A.summarize(rows)
    assert s["n"] == 4 and s["with_per_org_loc"] == 2 and s["label_counts"] == {"PER": 2}


def test_no_match_prefer_short_and_whitespace():
    from indonesian_hoax_checker.data import claim_verification_templates as T

    claims = [{"claim_id": str(i), "claim": f"klaim {i}", "cluster_id": i, "split": "test"} for i in range(4)]
    valid = [{"id": "a", "text": "panjang  sekali kalimat ini ya", "n_words": 5},
             {"id": "b", "text": "pendek saja", "n_words": 2}]
    rows = T.build_retrieval_queries(claims, valid, n_per_type=2, prefer_short=True)
    nm = [r for r in rows if r["type"] == "no_match"]
    assert [r["source_id"] for r in nm] == ["b", "a"]
    assert nm[1]["query"] == "panjang sekali kalimat ini ya"  # spasi ganda dirapikan, sumber tidak diubah
