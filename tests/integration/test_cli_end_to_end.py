# Uji end-to-end CLI dengan model sungguhan (NER, classifier, E5, mDeBERTa) dan basis klaim Komdigi.
#
# Lambat (memuat lima model, beberapa menit di CPU), jadi hanya jalan bila diminta:
#
#   HOAX_CHECKER_E2E=1 python -m pytest tests/integration/test_cli_end_to_end.py -q
#   (PowerShell: $env:HOAX_CHECKER_E2E=1; python -m pytest tests/integration/test_cli_end_to_end.py -q)
#
# Yang diuji adalah PERILAKU AMAN, bukan akurasi model: verdict otomatis "kredibilitas rendah" hanya
# boleh keluar untuk input yang memang mengulang klaim terbantah, dan tidak boleh keluar untuk
# sangkalan, klaim yang entitas/angkanya diubah, berita tanpa padanan, atau input aneh. Kasus yang
# hasilnya wajar jatuh ke dua arah (parafrasa jauh, typo berat) hanya dicek tidak salah cap.
# Klaim rujukan: 65177 (air nanas), 65458 (Pertamina/SPBU), 65140 (Hotel Shangri-La), 65704 (Kemenag).

import os
import subprocess
import sys
from pathlib import Path

import pytest

from indonesian_hoax_checker.pipeline import stages

ROOT = Path(__file__).resolve().parents[2]
NEEDED = [
    "artifacts/checkpoints/ner/ner_indobert_entity_seed42",
    "artifacts/checkpoints/ner/ner_indobert_claim_seed42",
    "artifacts/checkpoints/classifier/indobert_seed13",
    "data/processed/refuted_claims/claims.jsonl",
]
MISSING = [p for p in NEEDED if not (ROOT / p).exists()]
pytestmark = pytest.mark.skipif(
    os.environ.get("HOAX_CHECKER_E2E") != "1" or bool(MISSING),
    reason="set HOAX_CHECKER_E2E=1 untuk uji end-to-end" + (f"; belum ada: {MISSING}" if MISSING else ""),
)

LOW = "kredibilitas rendah"
REVIEW = "perlu tinjauan manusia"
NANAS = "Air nanas panas dapat sembuhkan kanker dan penyakit berat"
FILLER = ("Dinas pekerjaan umum menyiapkan jalur alternatif dan petugas yang mengatur arus kendaraan di persimpangan utama. "
          "Laporan kemajuan pekerjaan akan diumumkan setiap bulan di papan pengumuman kantor desa dan di laman resmi pemerintah daerah. ")


def words(n: int) -> str:
    w = (FILLER * (n // len(FILLER.split()) + 2)).split()
    return " ".join(w[:n])


@pytest.fixture(scope="module")
def run():
    os.environ.pop("HOAX_CHECKER_CONFIG", None)
    from indonesian_hoax_checker.pipeline.orchestrator import run_pipeline

    cache: dict[str, object] = {}

    def _run(text: str):
        if text not in cache:
            cache[text] = run_pipeline(text, "all")
        return cache[text]

    return _run


def top(res):
    c = res.retrieval["candidates"]
    return c[0]["claim_id"] if c else None


# ---------- 1. mengulang klaim terbantah: harus "kredibilitas rendah" dengan klaim yang benar ----------

REPEATS = [
    ("judul_persis", NANAS, "65177"),
    ("huruf_besar", NANAS.upper(), "65177"),
    ("huruf_kecil", NANAS.lower(), "65177"),
    ("spasi_dan_baris_baru", f"   \n  {NANAS}  \t ", "65177"),
    ("tanda_tanya_beruntun", NANAS + "???", "65177"),
    ("bentuk_pertanyaan", "Benarkah air nanas panas dapat menyembuhkan kanker dan penyakit berat?", "65177"),
    ("karakter_kontrol", "Air nanas panas\x00 dapat\x07 sembuhkan\tkanker\r\ndan penyakit berat", "65177"),
    ("tag_html", "<b>Air nanas panas</b> dapat <i>sembuhkan kanker</i> &amp; penyakit berat", "65177"),
    ("emoji", "Vaksin HPV bikin anak laki-laki impoten \U0001F631\U0001F631 sebarkan!!!", "65287"),
    ("pertamina", "Aturan baru Pertamina batasi pembelian BBM di semua SPBU", "65458"),
    ("angka_sama", "Hotel Shangri-La Surabaya dijual Rp2,3 triliun", "65140"),
    ("parafrasa", "Mulai sekarang setiap SPBU hanya boleh menjual BBM dalam jumlah terbatas karena aturan baru dari Pertamina.",
     "65458"),
    ("dua_kalimat_satu_hoaks", "Air nanas panas dapat sembuhkan kanker. Selain itu, besok Bandung diperkirakan hujan ringan.",
     "65177"),
    ("bingkai_katanya", "Katanya air nanas panas dapat sembuhkan kanker dan penyakit berat", "65177"),
    ("bingkai_beredar_kabar", "Beredar kabar bahwa Pertamina batasi pembelian BBM di semua SPBU", "65458"),
    ("hoaks_di_tengah_pesan", "Selamat pagi semuanya.\nAir nanas panas dapat sembuhkan kanker dan penyakit berat.\n"
                              "Tolong teruskan ke grup keluarga.", "65177"),
]


@pytest.mark.parametrize("name,text,claim", REPEATS, ids=[r[0] for r in REPEATS])
def test_repeated_claim_is_low_credibility(run, name, text, claim):
    res = run(text)
    assert res.stubs == []
    assert top(res) == claim
    assert res.stance["label"] == "supports"
    assert res.verdict["indication"] == LOW


# ---------- 2. bukan pengulangan: tidak boleh "kredibilitas rendah" ----------

NOT_REPEATS = [
    # sangkalan
    ("negasi_tidak", "Air nanas panas tidak dapat menyembuhkan kanker dan penyakit berat"),
    ("negasi_bukan", "Air nanas panas bukan obat kanker"),
    ("negasi_ganda", "Tidak benar bahwa Pertamina tidak membatasi pembelian BBM di semua SPBU"),
    ("disebut_hoaks", "Kabar Pertamina batasi pembelian BBM di semua SPBU itu hoaks"),
    ("awalan_hoaks", "[HOAKS] Air Nanas Panas Dapat Sembuhkan Kanker dan Penyakit Berat"),
    ("laporan_bantahan", "Komdigi menyatakan kabar air nanas panas dapat menyembuhkan kanker adalah hoaks."),
    ("klaim_lalu_bantahan", "Beredar kabar air nanas panas dapat sembuhkan kanker dan penyakit berat. Faktanya kabar itu keliru."),
    ("klaim_lalu_jangan_percaya", "Air nanas panas dapat sembuhkan kanker dan penyakit berat. Itu bohong, jangan percaya."),
    # isi klaim diubah
    ("entitas_ditukar_org", "Shell batasi pembelian BBM di semua SPBU"),
    ("entitas_ditukar_buah", "Air jeruk panas dapat sembuhkan kanker dan penyakit berat"),
    ("entitas_ditukar_orang", "Megawati resmi pensiun dari partai, pilih Puan jadi Ketua PDIP yang baru"),
    ("angka_diubah", "Hotel Shangri-La Surabaya dijual Rp23 triliun"),
    ("tanggal_diubah", "Luhut dicopot dan digantikan Zulkifli Hasan pada Oktober 2027"),
    ("peristiwa_berlawanan", "Gaji guru turun 60% mulai tahun 2027"),
    # tanpa padanan
    ("berita_benar", "Bank Indonesia mempertahankan suku bunga acuan pada rapat dewan gubernur bulan ini."),
    ("fiksi", "Seekor naga terlihat terbang di atas Monas tadi malam dan menyemburkan api ke arah Istana."),
    ("opini", "Menurut saya nasi goreng lebih enak daripada mi goreng."),
    ("salam", "Selamat pagi semuanya, semoga hari ini menyenangkan."),
    ("demo_taman_baca", "Pemerintah Kota Bandung akan membangun taman baca di setiap kelurahan pada tahun 2027."),
    ("demo_pln_80_persen", "PLN mengumumkan diskon listrik 80 persen untuk semua pelanggan selama libur Natal."),
    ("satir", "BREAKING: kucing di Jakarta resmi dilantik jadi ketua RT karena warga capek rapat (satir)."),
    # bahasa lain dan input aneh
    ("inggris", "Hot pineapple water can cure cancer and other serious diseases"),
    ("campur_bahasa", "Guys, the government says bansos will be stopped next month, beneran ga sih?"),
    ("arab", "السلام عليكم هذا خبر كاذب"),
    ("satu_kata", "Tautan"), ("satu_kata_nama", "Prabowo"), ("dua_kata", "vaksin HPV"),
    ("tanda_baca", "?"), ("titik_titik", "..."), ("angka", "12345"), ("nomor_telepon", "081234567890"),
    ("emoji_saja", "\U0001F602\U0001F602\U0001F602"), ("url_saja", "https://bit.ly/bansos2026"),
    ("sql", "'; DROP TABLE claims; -- Pertamina batasi pembelian BBM"),
    ("leet", "A1r n4nas p4nas d4pat s3mbuhkan k4nker"),
    ("satu_token_panjang", "a" * 6000),
]


@pytest.mark.parametrize("name,text", NOT_REPEATS, ids=[r[0] for r in NOT_REPEATS])
def test_non_repeat_is_never_low_credibility(run, name, text):
    res = run(text)
    assert res.stubs == []
    assert res.verdict["indication"] != LOW, (res.retrieval["matched_claim"], res.stance)
    assert res.verdict["indication"] == REVIEW  # semua kasus ini di bawah 200 kata: tidak ada verdict classifier


@pytest.mark.parametrize("text", ["?", "...", "12345", "\U0001F602\U0001F602\U0001F602", "Tautan", "a" * 6000])
def test_contentless_input_matches_nothing(run, text):
    res = run(text)
    assert res.retrieval["matched_claim"] is None and res.stance is None


@pytest.mark.parametrize("name,text,claim", [
    ("negasi", "Air nanas panas tidak dapat menyembuhkan kanker dan penyakit berat", "65177"),
    ("disebut_hoaks", "Kabar Pertamina batasi pembelian BBM di semua SPBU itu hoaks", "65458"),
])
def test_denial_still_shows_the_refuted_claim(run, name, text, claim):
    res = run(text)
    assert top(res) == claim and res.stance["label"] == "refutes" and res.verdict["needs_review"]


def test_entity_swap_is_explained(run):
    res = run("Shell batasi pembelian BBM di semua SPBU")
    assert res.verdict["indication"] == REVIEW
    if res.stance["label"] == "supports":  # tertahan pengaman entitas, bukan oleh model
        assert "Shell" in res.stance["note"] and "Shell" in res.verdict["explanation"]


# ---------- 3. teks panjang: gerbang 200 kata, hoaks terselip, artikel cek fakta ----------

def test_classifier_word_gate(run):
    short, long_ = run(words(199)), run(words(200))
    assert short.classifier_skipped and short.verdict["indication"] == REVIEW
    assert not long_.classifier_skipped and long_.classification["label"] in ("valid", "hoax")
    assert long_.verdict["indication"] in ("tidak terindikasi hoaks", "terindikasi hoaks", REVIEW)


def test_hoax_buried_in_long_text_is_found(run):
    # 256 kata berita biasa (kalimat utuh) + satu kalimat hoaks: classifier menilai "valid", tetapi
    # kalimat hoaksnya harus tetap ditemukan lewat kueri per kalimat
    text = FILLER * 8 + "Selain itu beredar kabar bahwa air nanas panas dapat menyembuhkan kanker dan penyakit berat."
    res = run(text)
    assert len(text.split()) >= 200 and not res.classifier_skipped
    assert top(res) == "65177"
    # kalimat hoaksnya dicocokkan tanpa bingkai laporan "Selain itu beredar kabar bahwa"
    assert res.retrieval["query_segment"] == "air nanas panas dapat menyembuhkan kanker dan penyakit berat."
    assert res.stance["label"] == "supports"
    assert res.verdict["indication"] == LOW  # tidak boleh lolos sebagai "tidak terindikasi hoaks"


def test_run_on_sentence_is_a_known_limitation(run):
    # Keterbatasan yang didokumentasikan: hoaks yang menyatu dengan kalimat panjang tanpa tanda baca
    # tidak bisa dipecah, jadi bisa tidak ditemukan. Yang dijaga: tidak pernah salah cap.
    text = words(190) + " Selain itu beredar kabar bahwa air nanas panas dapat menyembuhkan kanker dan penyakit berat."
    res = run(text)
    assert res.stubs == [] and res.verdict["indication"] != LOW


def test_fact_check_article_is_not_accused(run):
    text = ("Beredar unggahan di media sosial yang mengeklaim air nanas panas dapat menyembuhkan kanker dan penyakit berat. "
            "Faktanya, klaim tersebut tidak benar. Belum ada bukti ilmiah yang mendukung klaim itu. " + words(185))
    res = run(text)
    # artikel cek fakta memuat klaimnya, tetapi tidak boleh dicap mengulang hoaks
    assert top(res) == "65177" and res.verdict["indication"] in (REVIEW, "tidak terindikasi hoaks")


def test_very_long_input_finishes(run):
    res = run(words(5000))
    assert res.stubs == [] and res.verdict["indication"] != LOW
    repeated = run(" ".join([NANAS + "."] * 50))
    assert top(repeated) == "65177"


# ---------- 4. CLI sebagai proses terpisah ----------

def cli(args, cwd, stdin=None, env_extra=None):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONIOENCODING", "PYTHONUTF8", "HOAX_CHECKER_CONFIG")}
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), env.get("PYTHONPATH", "")])
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-m", "indonesian_hoax_checker", *args], cwd=cwd, env=env,
                          input=stdin, capture_output=True, timeout=900)


def test_cli_single_shot_from_another_directory_without_utf8_console(tmp_path):
    # folder kerja lain (config dan checkpoint ditemukan lewat root repo), encoding konsol bawaan sistem
    proc = cli(["Aturan baru Pertamina batasi pembelian BBM di semua SPBU \U0001F631", "-v"], tmp_path)
    out = proc.stdout.decode("utf-8", "replace")
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")[-500:]
    assert "STUB" not in out and "Indikasi:" in out
    assert "Retrieval: 'Aturan Baru Pertamina" in out and "bantahan resmi: https://www.komdigi.go.id" in out
    assert "Stance:" in out and "Klasifikasi: dilewati" in out


def test_cli_interactive_with_piped_input():
    lines = "Air nanas panas dapat sembuhkan kanker dan penyakit berat\n   \nbaris ini tidak dibaca\n"
    proc = cli([], ROOT, stdin=lines.encode("utf-8"), env_extra={"PYTHONIOENCODING": "utf-8"})
    out = proc.stdout.decode("utf-8", "replace")
    assert proc.returncode == 0 and out.count("Indikasi:") == 1 and "klaim>" not in out


@pytest.mark.parametrize("task,expected", [
    ("ner", "NER: entitas="), ("classify", "Klasifikasi: dilewati"), ("retrieve", "Retrieval: 'Air Nanas"),
    ("stance", "Stance: supports"),
])
def test_cli_single_task(task, expected):
    proc = cli([NANAS, "--task", task], ROOT, env_extra={"PYTHONIOENCODING": "utf-8"})
    out = proc.stdout.decode("utf-8", "replace")
    assert proc.returncode == 0 and expected in out and "STUB" not in out and "Indikasi" not in out


def test_cli_empty_input_exit_code():
    proc = cli(["   "], ROOT)
    assert proc.returncode == 2 and b"teks kosong" in proc.stderr


def test_all_stages_use_real_models():
    assert stages.resolve_path("configs/pipeline.yaml")
    for section in ("ner", "classification", "retrieval", "stance"):
        assert stages._pipeline_section(section), section
