import json

from indonesian_hoax_checker.ner.annotation import (
    adjudicate,
    auto_label,
    build,
    correction_report,
    kappa_report,
    make_tasks,
    read_export,
    spans_to_tags,
    split_rows,
)
from indonesian_hoax_checker.ner.claim_rules import add_claim_tag
from indonesian_hoax_checker.ner.labels import get_spans
from indonesian_hoax_checker.ner.time_rules import add_time_tags, time_spans
from indonesian_hoax_checker.preprocessing.tokenizer import tokenize


def words(text):
    return [t for t, _, _ in tokenize(text)]


def found(text):
    w = words(text)
    return [" ".join(w[a:b]) for a, b in time_spans(w)]


def spans(tokens, tags):
    return [(" ".join(tokens[a:b]), k) for k, a, b in get_spans(tags)]


def test_time_rules():
    assert found("Pemerintah menghentikan bantuan mulai bulan depan") == ["bulan depan"]
    assert found("Internet diputus selama tiga hari") == ["selama tiga hari"]
    assert found("Pada Senin, 3 Nov 2026 dan 12/11/2026") == ["Senin , 3 Nov 2026", "12 / 11 / 2026"]
    assert found("stop bansos bsk gaes") == ["bsk"]
    assert found("Jumlah hari kerja di Bank Mandiri") == []


def test_add_time_tags_keeps_existing_entities():
    assert add_time_tags(["Jakarta", "besok"], ["B-LOC", "O"]) == ["B-LOC", "B-TIME"]


def test_claim_after_marker_skips_subject_entity():
    t = words("Beredar unggahan yang mengeklaim bahwa Tentara Nasional Indonesia (TNI) akan memberikan pelatihan militer.")
    tags = ["O"] * len(t)
    i = t.index("Tentara")
    tags[i:i + 3] = ["B-ORG", "I-ORG", "I-ORG"]
    assert spans(t, add_claim_tag(t, tags)) == [("Tentara Nasional Indonesia", "ORG"),
                                                 ("akan memberikan pelatihan militer", "CLAIM")]


def test_claim_news_sentence_excludes_trailing_time():
    t = words("Pemerintah menghentikan seluruh bantuan Makan Bergizi Gratis mulai bulan depan")
    tags = add_time_tags(t, ["B-ORG"] + ["O"] * (len(t) - 1))
    assert spans(t, add_claim_tag(t, tags)) == [("Pemerintah", "ORG"),
                                                 ("menghentikan seluruh bantuan Makan Bergizi Gratis", "CLAIM"),
                                                 ("bulan depan", "TIME")]


def test_news_sentence_uses_main_predicate():
    # tanpa penanda dan tanpa subjek entitas: predikat utama "ditemukan" ("bernama" bukan predikat)
    t = words("Seorang nelayan bernama Yoni ditemukan tewas di perairan Rancapinang, Kecamatan Cimanggu.")
    assert spans(t, add_claim_tag(t, ["O"] * len(t))) == [("ditemukan tewas di perairan Rancapinang", "CLAIM")]
    # "Beredar ..." tanpa penanda dan tanpa judul: tidak ditebak
    t = words("Beredar video lama di media sosial.")
    assert "B-CLAIM" not in add_claim_tag(t, ["O"] * len(t))


def test_attribution_tail_is_cut():
    t = words("Pertamina menyebut harga Pertalite naik menjadi Rp13.000 per liter, kata Juru Bicara Pertamina.")
    assert spans(t, add_claim_tag(t, ["B-ORG"] + ["O"] * (len(t) - 1)))[-1] == ("harga Pertalite naik menjadi Rp13.000 per liter", "CLAIM")


def test_title_alignment_for_komdigi():
    text = ("Beredar sebuah unggahan di media sosial yang mengeklaim bahwa Tentara Nasional Indonesia (TNI) "
            "akan memberikan pelatihan militer bagi pasien yang mengalami gangguan jiwa.")
    t = words(text)
    tags = ["O"] * len(t); i = t.index("Tentara"); tags[i:i + 3] = ["B-ORG", "I-ORG", "I-ORG"]
    title = "TNI Bakal Berikan Pasien Gangguan Jiwa Pelatihan Militer"
    assert spans(t, add_claim_tag(t, tags, title))[-1] == (
        "akan memberikan pelatihan militer bagi pasien yang mengalami gangguan jiwa", "CLAIM")
    # tanpa penanda, judul tetap menemukan klaimnya
    t = words("Beredar unggahan video di media sosial Facebook Agus Suryonugroho mengumumkan pajak kendaraan digratiskan.")
    tags = ["O"] * len(t); i = t.index("Agus"); tags[i:i + 2] = ["B-PER", "I-PER"]
    assert spans(t, add_claim_tag(t, tags, "Kakorlantas Umumkan Pajak Kendaraan Digratiskan"))[-1] == (
        "mengumumkan pajak kendaraan digratiskan", "CLAIM")


def test_stems_match_affixed_forms():
    from indonesian_hoax_checker.ner.claim_rules import _same_word, stems
    assert _same_word(stems("berikan"), stems("memberikan"))
    assert _same_word(stems("Umumkan"), stems("mengumumkan"))
    assert _same_word(stems("sebut"), stems("menyebut"))
    assert not _same_word(stems("pajak"), stems("pasien"))


def test_auto_label_and_tasks_keep_auto_tags():
    rows = auto_label([{"id": "s1", "text": "Bantuan dihentikan bulan depan", "split": "dev"}])
    assert rows[0]["tags"] == ["O", "O", "B-TIME", "I-TIME"]
    task = make_tasks(rows)[0]
    assert task["data"]["auto_tags"] == rows[0]["tags"] and "tokens" not in task["data"]
    assert [r["value"]["text"] for r in task["predictions"][0]["result"]] == ["bulan depan"]


def test_spans_to_tags_aligns_partial_spans():
    tokens = tokenize("Pemerintah menghentikan bantuan")
    results = [
        {"type": "labels", "value": {"start": 0, "end": 10, "labels": ["ORG"]}},
        {"type": "labels", "value": {"start": 13, "end": 31, "labels": ["CLAIM"]}},  # mulai di tengah kata
    ]
    assert spans_to_tags(tokens, results) == ["B-ORG", "B-CLAIM", "I-CLAIM"]


def _ann(who, ranges):
    return {"completed_by": {"email": who}, "result": [
        {"type": "labels", "value": {"start": s, "end": e, "labels": [k]}} for s, e, k in ranges]}


def _write_export(tmp_path, name, tasks):
    path = tmp_path / name
    path.write_text(json.dumps(tasks))
    return path


TEXT = "Pemerintah menghentikan bantuan besok"


def test_kappa_and_adjudication(tmp_path):
    items = read_export(_write_export(tmp_path, "e.json", [
        {"data": {"id": "a", "text": TEXT}, "annotations": [_ann("x", [(0, 10, "ORG"), (32, 37, "TIME")]),
                                                         _ann("y", [(0, 10, "ORG"), (32, 37, "TIME")])]},
        {"data": {"id": "b", "text": TEXT}, "annotations": [_ann("x", [(11, 31, "CLAIM")]), _ann("y", [(0, 10, "ORG")])]},
    ]))
    report = kappa_report(items)
    assert report["mean"]["TIME"] == 1.0 and report["mean"]["CLAIM"] < 0.6
    final, unresolved = adjudicate(items)
    assert [r["id"] for r in final] == ["a"] and unresolved == ["b"]
    final, _ = adjudicate(items, adjudicator="x")
    assert final[1]["tags"] == ["O", "B-CLAIM", "I-CLAIM", "O"]


def test_read_export_merges_files_from_separate_instances(tmp_path):
    a = _write_export(tmp_path, "jonathan.json", [{"data": {"id": "a", "text": TEXT}, "annotations": [_ann("admin", [(0, 10, "ORG")])]}])
    b = _write_export(tmp_path, "dzaky.json", [{"data": {"id": "a", "text": TEXT}, "annotations": [_ann("admin", [(32, 37, "TIME")])]}])
    assert set(read_export([a, b])["a"]["by"]) == {"admin", "admin (dzaky)"}


def test_single_annotator_correction_report_and_build(tmp_path):
    auto_rows = [
        {"id": "d1", "text": TEXT, "split": "dev", "tokens": words(TEXT), "tags": ["B-ORG", "O", "O", "B-TIME"]},
        {"id": "d2", "text": TEXT, "split": "test", "tokens": words(TEXT), "tags": ["O"] * 4},
        {"id": "t1", "text": TEXT, "split": "train", "tokens": words(TEXT), "tags": ["B-ORG", "B-CLAIM", "I-CLAIM", "O"]},
        {"id": "t2", "text": TEXT, "split": "train", "tokens": words(TEXT), "tags": ["B-ORG", "O", "O", "O"]},
    ]
    # Annotator menambah CLAIM pada d1; d2 belum dikoreksi
    export = _write_export(tmp_path, "e.json", [{"data": {"id": "d1", "text": TEXT, "split": "dev", "auto_tags": auto_rows[0]["tags"]},
                                                 "annotations": [_ann("jonathan", [(0, 10, "ORG"), (11, 31, "CLAIM"), (32, 37, "TIME")])]}])
    final, _ = adjudicate(read_export(export))
    assert "auto_tags" not in final[0]

    report = correction_report(final, {r["id"]: r for r in auto_rows})
    assert report["sentences"] == 1 and report["unchanged_sentences"] == 0.0
    assert report["auto_vs_human"]["per_type"]["CLAIM"]["recall"] == 0.0

    rows, missing, dropped = build(final, auto_rows)
    assert {(r["id"], r["quality"]) for r in rows} == {("d1", "human"), ("t1", "silver")}
    assert missing == ["d2"]  # dev/test yang belum dikoreksi tidak diisi label otomatis
    assert dropped == 1       # t2: train tanpa CLAIM dibuang


def test_split_rows_uses_split_field_and_informal_source():
    rows = [{"id": "a", "split": "train"}, {"id": "b", "split": "test"}, {"id": "c", "source": "informal", "split": "informal_test"}, {"id": "d"}]
    parts = split_rows(rows)
    assert [r["id"] for r in parts["train"]] == ["a"] and [r["id"] for r in parts["test"]] == ["b"]
    assert [r["id"] for r in parts["informal_test"]] == ["c"] and parts["dev"] == []


def test_claim_rule_fixes():
    # penanda kuat "diklaim" dipakai, bukan "berisi"; kata "narasi" tidak masuk
    t = words("Beredar unggahan berisi narasi dan tautan yang diklaim membagikan hadiah umrah.")
    assert spans(t, add_claim_tag(t, ["O"] * len(t))) == [("membagikan hadiah umrah", "CLAIM")]
    # kurung tidak terpotong
    t = words("Beredar unggahan yang mengeklaim pemerintah memungut pajak dari pekerja seks komersial (PSK).")
    assert spans(t, add_claim_tag(t, ["O"] * len(t)))[0][0].endswith("( PSK )")
    # subjek majemuk "Sekjen PDIP atau Hasto": klaim tetap mulai dari predikat, bukan "atau"
    t = words("Sekjen PDIP atau Hasto dilaporkan ke polisi.")
    assert spans(t, add_claim_tag(t, ["O", "B-ORG"] + ["O"] * (len(t) - 2)))[-1] == ("dilaporkan ke polisi", "CLAIM")
    # jabatan dan nomor urut sebelum nama dilewati
    t = words("Capres nomor urut 2 Prabowo Subianto mendorong pendekatan sistemik.")
    tags = ["O"] * len(t); tags[4:6] = ["B-PER", "I-PER"]
    assert spans(t, add_claim_tag(t, tags))[-1] == ("mendorong pendekatan sistemik", "CLAIM")


def test_claim_after_marker_skips_title_and_entity():
    t = words("Beredar unggahan yang mengeklaim Presiden Prabowo Subianto meminta warga Aceh mengeroyok Bahlil.")
    tags = ["O"] * len(t); i = t.index("Prabowo"); tags[i:i + 2] = ["B-PER", "I-PER"]
    assert spans(t, add_claim_tag(t, tags))[-1] == ("meminta warga Aceh mengeroyok Bahlil", "CLAIM")


def test_time_rules_ignore_regulation_numbers():
    assert found("Peraturan Pemerintah No 28 Tahun 2024 tentang Kesehatan") == []


def test_claim_skips_long_capitalized_subject_and_weak_marker():
    t = words("Beredar unggahan yang mengeklaim bahwa Menteri Pertahanan Sjafrie Sjamsoeddin mengumumkan dana bantuan.")
    tags = ["O"] * len(t); i = t.index("Sjafrie"); tags[i:i + 2] = ["B-PER", "I-PER"]
    assert spans(t, add_claim_tag(t, tags))[-1] == ("mengumumkan dana bantuan", "CLAIM")
    t = words("Beredar unggahan dengan narasi Indonesia pimpin negara ASEAN rebut Laut Cina Selatan.")
    tags = ["O"] * len(t); tags[t.index("Indonesia")] = "B-LOC"
    assert spans(t, add_claim_tag(t, tags))[-1] == ("pimpin negara ASEAN rebut Laut Cina Selatan", "CLAIM")
    # penanda lemah tanpa subjek entitas: tidak diberi klaim
    t = words("Beredar foto menampilkan kartu bertuliskan ATM MBG.")
    assert "B-CLAIM" not in add_claim_tag(t, ["O"] * len(t))


def test_digit_is_not_a_title_unless_after_nomor_urut():
    t = words("Salam 4 jari disebut simbol perlawanan.")
    assert spans(t, add_claim_tag(t, ["O"] * len(t))) == [("disebut simbol perlawanan", "CLAIM")]


def test_title_span_extends_to_phrase_end_and_marker_wins():
    # tanpa penanda: judul menentukan awal, akhir diperpanjang sampai koma/titik
    t = words("Beredar unggahan video TNI menyelidiki pembalakan liar penyebab bencana di Sumatra.")
    tags = ["O"] * len(t); tags[t.index("TNI")] = "B-ORG"
    assert spans(t, add_claim_tag(t, tags, "TNI Menyelidiki Illegal Logging Penyebab Banjir Sumatera"))[-1] == (
        "menyelidiki pembalakan liar penyebab bencana di Sumatra", "CLAIM")
    # ada penanda kuat: aturan penanda dipakai, "terkait" dilewati
    t = words("Beredar informasi yang mengeklaim terkait penonaktifan fitur TikTok Live oleh Kementerian Komdigi.")
    assert spans(t, add_claim_tag(t, ["O"] * len(t), "Fitur TikTok Live Dinonaktifkan"))[-1] == (
        "penonaktifan fitur TikTok Live oleh Kementerian Komdigi", "CLAIM")


def test_compound_subject_is_skipped():
    t = words("Beredar unggahan yang mengeklaim bahwa DPR RI dan Polri telah meresmikan SIM berlaku seumur hidup.")
    tags = ["O"] * len(t); i = t.index("DPR"); tags[i:i + 2] = ["B-ORG", "I-ORG"]
    assert spans(t, add_claim_tag(t, tags))[-1] == ("telah meresmikan SIM berlaku seumur hidup", "CLAIM")


def test_build_accept_auto_uses_auto_labels_for_checked_dev_test():
    auto_rows = [
        {"id": "d1", "split": "dev", "tokens": ["a"], "tags": ["B-CLAIM"]},
        {"id": "t1", "split": "test", "tokens": ["b"], "tags": ["O"]},
        {"id": "r1", "split": "train", "tokens": ["c"], "tags": ["B-CLAIM"]},
    ]
    rows, missing, dropped = build([], auto_rows, accept_auto=True)
    assert {(r["id"], r["quality"]) for r in rows} == {("d1", "human"), ("t1", "human"), ("r1", "silver")}
    assert missing == [] and dropped == 0
