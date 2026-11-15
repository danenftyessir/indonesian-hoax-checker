# Ketahanan pipeline dan CLI terhadap input dan lingkungan yang tidak rapi. Semua tes di sini
# memakai model tiruan atau config stub, jadi cepat dan tidak butuh checkpoint.

import json
import os
import subprocess
import sys

import pytest

from indonesian_hoax_checker import cli
from indonesian_hoax_checker.pipeline import orchestrator, stages
from indonesian_hoax_checker.pipeline.confidence import combine
from indonesian_hoax_checker.pipeline.orchestrator import clean_text, has_debunk_marker, split_segments
from indonesian_hoax_checker.stance.predict import combine_directions, predict_stance_segments

CLAIMS = [
    {"claim_id": "1", "claim": "Air nanas panas sembuhkan kanker", "clarification": "Faktanya tidak benar",
     "url": "u1", "published_at": "2026-09-01"},
    {"claim_id": "2", "claim": "Pertamina batasi pembelian BBM di SPBU", "clarification": "Pertamina membantah",
     "url": "u2", "published_at": "2026-09-02"},
]
CANDIDATE = {**CLAIMS[1], "score": 1.0}


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """pipeline.yaml sementara: BM25 atas dua klaim, stance tiruan (lihat FakeStance), tanpa NER/classifier."""
    claims = tmp_path / "claims.jsonl"
    claims.write_text("\n".join(json.dumps(c) for c in CLAIMS) + "\n", encoding="utf-8")
    model_cfg = tmp_path / "retrieval.yaml"
    model_cfg.write_text(f"scorer: bm25\nclaims_path: {claims.as_posix()}\nfields: [claim]\n", encoding="utf-8")
    path = tmp_path / "pipeline.yaml"

    def write(stance: str = "  model: similarity\n  bidirectional: true", retrieval_extra: str = "") -> None:
        path.write_text(
            f"retrieval:\n  config: {model_cfg.as_posix()}\n  min_score: 0.5\n{retrieval_extra}\nstance:\n{stance}\n",
            encoding="utf-8")

    write()
    monkeypatch.setenv("HOAX_CHECKER_CONFIG", str(path))
    for name, value in [("_retriever", None), ("_retriever_failed", False), ("_stance_model", None),
                        ("_stance_failed", False), ("_ner_stage", None), ("_classifier", None)]:
        monkeypatch.setattr(stages, name, value)
    monkeypatch.setattr(stages, "STUB_STAGES", {"ner", "classify", "retrieve", "stance"})
    return write


class FakeStance:
    """supports bila hypothesis memuat premise persis (huruf kecil); refutes bila hypothesis memuat 'tidak'."""

    def predict_batch(self, pairs):
        out = []
        for premise, hyp in pairs:
            p, h = premise.lower(), hyp.lower()
            if "tidak" in h and p.split()[0] in h:
                out.append(("refutes", 0.9))
            elif p in h or h in p:
                out.append(("supports", 0.9))
            else:
                out.append(("unrelated", 0.8))
        return out


# ---------- pembersihan dan pemecahan input ----------

@pytest.mark.parametrize("raw,expected", [
    ("  a\tb \r\n c  ", "a b c"),
    ("a\x00b\x07c", "a b c"),
    ("\n\t  \x00", ""),
    ("satu  dua", "satu dua"),
])
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


def test_split_segments_sentences_and_lines():
    text = "Air nanas panas sembuhkan kanker. Selain itu besok hujan deras!\nPesan ini wajib disebar? ya"
    assert split_segments(text) == ["Air nanas panas sembuhkan kanker.", "Selain itu besok hujan deras!",
                                    "Pesan ini wajib disebar?"]
    assert split_segments("satu kalimat saja tanpa titik") == ["satu kalimat saja tanpa titik"]
    assert split_segments("ya. ok. !!!") == []  # potongan di bawah 3 kata dibuang


@pytest.mark.parametrize("text,expected", [
    ("Faktanya, klaim itu keliru", True), ("Kabar itu HOAKS", True), ("[SALAH] Air nanas", True),
    ("Tidak benar bahwa BBM dibatasi", True), ("Air nanas panas sembuhkan kanker", False),
    ("Itu bohong besar, jangan percaya", True), ("Khasiatnya belum terbukti", True),
    ("hoki besar di awal tahun", False),
])
def test_debunk_marker(text, expected):
    assert has_debunk_marker(text) is expected


# ---------- stance: gerbang satu arah, agregasi kalimat, pengaman entitas ----------

def test_one_way_gate_only_blocks_weak_one_direction_supports():
    assert combine_directions(("unrelated", 0.9), ("supports", 0.54), 0.6) == ("unrelated", 0.5)
    assert combine_directions(("unrelated", 0.9), ("supports", 0.75), 0.6) == ("supports", 0.75)
    assert combine_directions(("supports", 0.55), ("supports", 0.52), 0.6) == ("supports", 0.55)  # dua arah: lolos
    assert combine_directions(("unrelated", 0.9), ("refutes", 0.4), 0.6) == ("refutes", 0.4)      # refutes tak dibatasi


def test_segments_refutes_wins_over_supports():
    res = predict_stance_segments(FakeStance(), ["pertamina batasi pembelian bbm di spbu",
                                                 "pertamina tidak membatasi apa pun"], CANDIDATE)
    assert res["label"] == "refutes" and res["segment"] == "pertamina tidak membatasi apa pun"


def test_segments_supports_reports_segment():
    res = predict_stance_segments(FakeStance(), ["besok hujan deras", "katanya pertamina batasi pembelian bbm di spbu"],
                                  CANDIDATE)
    assert res["label"] == "supports" and res["segment"].startswith("katanya pertamina")
    assert res["claim_id"] == "2"


def test_segments_all_unrelated():
    res = predict_stance_segments(FakeStance(), ["besok hujan deras", "harga cabai naik"], CANDIDATE)
    assert res["label"] == "unrelated"


def test_stance_hypothesis_is_the_matched_segment():
    text = "Besok hujan deras. Pertamina batasi pembelian BBM mulai hari ini. Harga cabai naik."
    segs = split_segments(text)
    assert stages.stance_hypotheses(text, segs, segs[1]) == [segs[1]]  # kalimat lain tidak ikut dinilai
    assert stages.stance_hypotheses("satu kalimat", [], None) == ["satu kalimat"]
    # bingkai laporan dibuang walau retrieval memilih versi berbingkainya
    framed = "Beredar kabar bahwa Pertamina batasi pembelian BBM di semua SPBU"
    assert stages.stance_hypotheses(framed, [], framed) == ["Pertamina batasi pembelian BBM di semua SPBU"]


def test_long_segment_needs_two_way_supports():
    class BackwardOnly:  # hanya arah (input -> klaim) yang supports
        def predict_batch(self, pairs):
            return [("supports", 0.9) if p != CANDIDATE["claim"] else ("unrelated", 0.8) for p, h in pairs]

    short = "katanya pertamina batasi pembelian bbm di spbu"
    long_text = " ".join(["kalimat pengisi yang panjang"] * 10 + [short])
    kw = {"one_way_min": 0.6, "one_way_max_words": 30}
    assert predict_stance_segments(BackwardOnly(), [short], CANDIDATE, **kw)["label"] == "supports"
    assert predict_stance_segments(BackwardOnly(), [long_text], CANDIDATE, **kw)["label"] == "unrelated"
    assert predict_stance_segments(BackwardOnly(), [long_text], CANDIDATE, one_way_min=0.6)["label"] == "supports"
    assert combine_directions(("supports", 0.9), ("supports", 0.9), 0.6, allow_one_way=False) == ("supports", 0.9)


def test_entity_mismatch():
    ner = {"entities": [{"text": "Shell", "label": "ORG", "start": 0, "end": 5},
                        {"text": "SPBU", "label": "LOC", "start": 30, "end": 34},
                        {"text": "besok", "label": "TIME", "start": 40, "end": 45}], "core_claim": ""}
    assert stages.entity_mismatch("Shell batasi pembelian BBM di SPBU besok", ner, CANDIDATE) == ["Shell"]
    ok = {"entities": [{"text": "PT Pertamina", "label": "ORG", "start": 0, "end": 12}], "core_claim": ""}
    assert stages.entity_mismatch("PT Pertamina batasi BBM", ok, CANDIDATE) == []
    assert stages.entity_mismatch("apa saja", None, CANDIDATE) == []


def test_combine_holds_supports_with_note_or_debunk_marker():
    match = {"matched_claim": "k", "clarification": "c", "similarity_score": 0.9, "candidates": [], "scorer": "bm25"}
    cls = {"label": "valid", "confidence": 0.5}
    sup = {"label": "supports", "confidence": 0.9, "claim_id": "1"}
    assert combine(cls, match, sup)["indication"] == "kredibilitas rendah"
    noted = combine(cls, match, {**sup, "note": "entitas tidak ada di klaim terbantah: Shell"})
    assert noted["needs_review"] and "Shell" in noted["explanation"]
    assert combine(cls, match, sup, None, True)["needs_review"]


# ---------- pipeline dengan config sementara ----------

def test_pipeline_finds_claim_buried_in_multi_sentence_text(cfg, monkeypatch):
    monkeypatch.setattr(stages, "_stance_model", FakeStance())
    text = ("Pemerintah daerah mengumumkan jadwal pembangunan jalan baru.\n"
            "Selain itu beredar kabar air nanas panas sembuhkan kanker.\nWarga diminta tetap tenang.")
    res = orchestrator.run_pipeline(text)
    assert res.retrieval["candidates"][0]["claim_id"] == "1"
    assert res.retrieval["query_segment"] == "Selain itu beredar kabar air nanas panas sembuhkan kanker."
    assert res.stance["label"] == "supports" and res.verdict["indication"] == "kredibilitas rendah"
    assert "\n" not in res.text


def test_pipeline_segment_queries_can_be_disabled(cfg, monkeypatch):
    cfg(retrieval_extra="  segment_queries: false")
    monkeypatch.setattr(stages, "_stance_model", FakeStance())
    res = orchestrator.run_pipeline("Besok hujan deras sekali. Air nanas panas sembuhkan kanker.")
    assert res.retrieval["query_segment"] == res.text


def test_pipeline_debunk_text_is_not_low_credibility(cfg, monkeypatch):
    monkeypatch.setattr(stages, "_stance_model", FakeStance())
    res = orchestrator.run_pipeline("Beredar kabar air nanas panas sembuhkan kanker. Faktanya kabar itu keliru.")
    assert res.stance["label"] == "supports" and res.verdict["needs_review"]
    assert "penanda bantahan" in res.verdict["explanation"]


def test_pipeline_entity_swap_goes_to_review(cfg, monkeypatch):
    monkeypatch.setattr(stages, "_stance_model", FakeStance())
    monkeypatch.setattr(stages, "run_ner", lambda text: {
        "entities": [{"text": "Shell", "label": "ORG", "start": 0, "end": 5}], "core_claim": ""})
    monkeypatch.setattr(FakeStance, "predict_batch", lambda self, pairs: [("supports", 0.9)] * len(pairs))
    res = orchestrator.run_pipeline("Shell batasi pembelian BBM di SPBU")
    assert res.stance["label"] == "supports" and "Shell" in res.stance["note"]
    assert res.verdict["indication"] == "perlu tinjauan manusia"


@pytest.mark.parametrize("text", ["?", "...", "12345", "\U0001F602\U0001F602", "https://bit.ly/x", "a" * 5000,
                                  "<b>x</b> y z", "'; DROP TABLE claims; --", "السلام عليكم", "kata " * 3000])
def test_pipeline_survives_odd_input(cfg, text):
    res = orchestrator.run_pipeline(text)
    assert res.verdict["indication"] in {"perlu tinjauan manusia", "kredibilitas rendah"}
    assert cli.format_result(res, True)


@pytest.mark.parametrize("text", ["", "   ", "\n\t", "\x00\x07"])
def test_pipeline_rejects_empty_input(cfg, text):
    with pytest.raises(ValueError, match="teks kosong"):
        orchestrator.run_pipeline(text)


# ---------- config dan path ----------

def test_config_found_from_another_working_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("HOAX_CHECKER_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)  # tidak ada configs/ di sini; harus jatuh ke root repo
    assert "min_words" in stages._pipeline_section("classification")
    resolved = stages.resolve_path("configs/models/retrieval_bm25.yaml")
    assert os.path.isabs(resolved) and os.path.exists(resolved)
    assert stages.resolve_path("tidak/ada.yaml") == "tidak/ada.yaml"
    assert stages.resolve_path(None) is None


def test_config_with_non_ascii_comment_is_read_as_utf8(tmp_path, monkeypatch):
    path = tmp_path / "pipeline.yaml"
    # byte 0x9D ("”" dalam UTF-8) tidak terdefinisi di cp1252: gagal bila dibaca dengan encoding bawaan Windows
    path.write_text("# ambang “ketat” ≥ 0,8\nclassification:\n  min_words: 7\n", encoding="utf-8")
    monkeypatch.setenv("HOAX_CHECKER_CONFIG", str(path))
    assert stages.classifier_min_words() == 7
    assert stages.NERStage.from_config().is_stub


@pytest.mark.parametrize("content", ["", "- bukan\n- mapping\n", "classification: [1, 2]\n", "a: [tidak ditutup\n"])
def test_broken_config_falls_back_to_stub(tmp_path, monkeypatch, content):
    path = tmp_path / "pipeline.yaml"
    path.write_text(content, encoding="utf-8")
    monkeypatch.setenv("HOAX_CHECKER_CONFIG", str(path))
    for name, value in [("_retriever", None), ("_retriever_failed", False), ("_stance_model", None),
                        ("_stance_failed", False), ("_ner_stage", None), ("_classifier", None)]:
        monkeypatch.setattr(stages, name, value)
    monkeypatch.setattr(stages, "STUB_STAGES", {"ner", "classify", "retrieve", "stance"})
    res = orchestrator.run_pipeline("Pemerintah menghentikan bantuan mulai besok")
    assert set(res.stubs) == {"ner", "classify", "retrieve", "stance"}


def test_missing_config_file_falls_back_to_stub(tmp_path, monkeypatch):
    monkeypatch.setenv("HOAX_CHECKER_CONFIG", str(tmp_path / "tidak_ada.yaml"))
    assert stages._pipeline_section("stance") == {} and stages.classifier_min_words() == 0


# ---------- CLI ----------

def test_cli_text_starting_with_dash(cfg, capsys):
    # argparse menerima teks berawalan "-" bila memuat spasi; satu kata berawalan "-" butuh pemisah "--"
    assert cli.main(["-5 derajat suhu di Bandung besok pagi"]) == 0
    assert cli.main(["--", "-hoaks"]) == 0
    assert capsys.readouterr().out.count("Indikasi") == 2
    with pytest.raises(SystemExit):
        cli.main(["-hoaks"])


def test_cli_empty_text_exit_code(cfg, capsys):
    assert cli.main(["   "]) == 2
    assert "teks kosong" in capsys.readouterr().err


def test_cli_unexpected_error_does_not_crash(cfg, monkeypatch, capsys):
    def boom(text, task):
        raise RuntimeError("model rusak")

    monkeypatch.setattr(cli, "run_pipeline", boom)
    assert cli.main(["apa saja"]) == 1
    assert "RuntimeError: model rusak" in capsys.readouterr().err


def test_cli_interactive_continues_after_error_and_stops_on_blank(cfg, monkeypatch, capsys):
    calls = []

    def fake(text, task):
        calls.append(text)
        if text == "rusak":
            raise RuntimeError("x")
        return orchestrator.run_pipeline(text, task)

    lines = iter(["rusak", "Air nanas panas sembuhkan kanker", "", "tidak terbaca"])
    monkeypatch.setattr(cli, "run_pipeline", fake)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    assert cli.main([]) == 0
    assert calls == ["rusak", "Air nanas panas sembuhkan kanker"]
    assert "Indikasi" in capsys.readouterr().out


@pytest.mark.parametrize("exc", [EOFError, KeyboardInterrupt])
def test_cli_interactive_exits_cleanly(cfg, monkeypatch, exc):
    def raiser(prompt=""):
        raise exc

    monkeypatch.setattr("builtins.input", raiser)
    assert cli.main([]) == 0


def test_cli_output_survives_console_that_cannot_encode(tmp_path):
    """Keluaran di-pipe dengan encoding ASCII ketat (pengganti cp1252 Windows): emoji di input ikut tercetak
    di baris NER; tanpa harden_streams ini berakhir dengan UnicodeEncodeError."""
    stub = tmp_path / "pipeline.yaml"
    stub.write_text("ner:\n  checkpoint: null\n", encoding="utf-8")
    env = {**os.environ, "HOAX_CHECKER_CONFIG": str(stub), "PYTHONIOENCODING": "ascii", "PYTHONUTF8": "0"}
    src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
    env["PYTHONPATH"] = os.pathsep.join([os.path.abspath(src), env.get("PYTHONPATH", "")])
    proc = subprocess.run(
        [sys.executable, "-m", "indonesian_hoax_checker", "Vaksin bikin mandul \U0001F631 “sebarkan”", "-v"],
        capture_output=True, env=env, timeout=120)
    assert proc.returncode == 0, proc.stderr.decode("ascii", "replace")
    assert b"Indikasi" in proc.stdout and b"UnicodeEncodeError" not in proc.stderr


def test_long_multi_sentence_text_is_not_queried_as_a_whole(cfg, monkeypatch):
    monkeypatch.setattr(stages, "_stance_model", FakeStance())
    # tanpa kata yang ada di dua klaim tiruan ("di", "BBM", ...): BM25 atas dua dokumen cocok pada satu kata saja
    filler = "Pemerintah daerah mengumumkan jadwal pembangunan jalan baru wilayah utara kota. " * 6
    text = filler + "Kabarnya air nanas panas sembuhkan kanker."
    assert len(text.split()) > 30
    res = orchestrator.run_pipeline(text)
    # yang dicocokkan adalah kalimatnya (dengan atau tanpa bingkai "Kabarnya"), bukan teks penuh
    assert res.retrieval["query_segment"].endswith("air nanas panas sembuhkan kanker.")
    assert res.retrieval["query_segment"] != res.text
    assert res.stance["segment"] == "air nanas panas sembuhkan kanker."  # stance: tanpa bingkai
    no_match = orchestrator.run_pipeline(filler + "Harga cabai rawit naik pekan ini.")
    assert no_match.retrieval["matched_claim"] is None  # teks penuh tidak dipakai sebagai kueri cadangan


def test_claim_span_queries_are_opt_in(cfg, monkeypatch):
    monkeypatch.setattr(stages, "_stance_model", FakeStance())
    text = "Menurut tetangga saya air nanas panas sembuhkan kanker lho"
    span = "air nanas panas sembuhkan kanker"
    monkeypatch.setattr(stages, "run_ner", lambda t: {"core_claim": span, "entities": [
        {"text": span, "label": "CLAIM", "start": 22, "end": 54},
        {"text": "lho", "label": "CLAIM", "start": 55, "end": 58},  # terlalu pendek: tidak jadi kueri
    ]})
    assert orchestrator.run_pipeline(text).retrieval["query_segment"] == text  # default: mati
    cfg(retrieval_extra="  claim_span_queries: true")
    res = orchestrator.run_pipeline(text)
    assert res.retrieval["query_segment"] == span and res.stance["segment"] == span


@pytest.mark.parametrize("sentence,expected", [
    ("Selain itu beredar kabar bahwa air nanas panas sembuhkan kanker.", "air nanas panas sembuhkan kanker."),
    ("Katanya Pertamina batasi pembelian BBM di SPBU", "Pertamina batasi pembelian BBM di SPBU"),
    ("Ada info di WhatsApp kalau besok semua SPBU tutup", "besok semua SPBU tutup"),
    ("Beredar video yang mengeklaim Gunung Lawu meletus kemarin", "Gunung Lawu meletus kemarin"),
    ("Viral unggahan di media sosial yang menyebut harga BBM naik lagi", "harga BBM naik lagi"),
    ("Zulhas menyebut tugas rakyat hanya bayar pajak", None),    # atribusi tokoh: bagian dari klaim
    ("Pertamina batasi pembelian BBM di SPBU", None),             # tanpa bingkai
    ("Katanya begitu", None),                                     # sisanya terlalu pendek
    ("Ada kabar baik untuk warga Bandung hari ini", None),        # bukan bingkai laporan (sisa terlalu pendek/"baik ...")
])
def test_strip_report_frame(sentence, expected):
    from indonesian_hoax_checker.pipeline.orchestrator import strip_report_frame

    assert strip_report_frame(sentence) == expected


def test_reported_rumour_is_matched_without_its_frame(cfg, monkeypatch):
    monkeypatch.setattr(stages, "_stance_model", FakeStance())
    res = orchestrator.run_pipeline("Selain itu beredar kabar bahwa air nanas panas sembuhkan kanker.")
    assert res.retrieval["query_segment"] == "air nanas panas sembuhkan kanker."
    assert res.stance["label"] == "supports"
