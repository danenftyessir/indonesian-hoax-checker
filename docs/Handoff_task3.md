**Handoff Task 3 (retrieval + stance) dan penyambungan CLI, dari Danendra**

Per 15 Nov 2026. Balasan untuk dua handoff di `docs/Handoff.md`. Rincian angka ada di `docs/experiment_protocol.md` (tiga bagian terakhir); keputusan pipeline diusulkan di `docs/decisions/0013-penyambungan-retrieval-dan-stance-di-pipeline.md`.

**1. Status**

- Semua tahap CLI memakai model sungguhan: NER (entitas + CLAIM baru), classifier demo `indobert_seed13`, retrieval E5, stance mDeBERTa. Tidak ada lagi `[STUB]`.
- `pytest`: 179 tes unit lulus. Uji end-to-end dengan model sungguhan: 74 kasus lulus (`HOAX_CHECKER_E2E=1 python -m pytest tests/integration/test_cli_end_to_end.py -q`, sekitar 5 menit di CPU).
- Clone baru bisa menghasilkan basis klaim yang hash-nya sama dengan `data/splits/manifest.json`.

**2. Yang harus ada di mesin yang menjalankan CLI** (tidak dilacak Git)

| Apa | Dari mana | Ke mana |
|---|---|---|
| NER entitas + CLAIM | Drive `hasil_training/ner_20261112-1801/ner_checkpoints_seed42.zip`, lalu zip CLAIM `ner_20261114-1153` (ekstrak berurutan) | `artifacts/checkpoints/ner/` |
| Classifier demo | Drive `hasil_training/indobert_classifier_seed13.zip` (ekstrak di root repo) | `artifacts/checkpoints/classifier/indobert_seed13/` |
| Basis klaim | CSV Komdigi v396 (Drive tim `data/archive/komdigi_hoaks.csv`), lalu `retrieval.corpus` dan `evaluation.splits --splits-dir <folder sementara>` | `data/processed/refuted_claims/claims.jsonl` |
| E5 dan mDeBERTa | diunduh otomatis dari HuggingFace saat pertama dipakai (masing-masing sekitar 1,1 GB) | cache HF |

Tanpa basis klaim, retrieval dan stance diam-diam menjadi stub. `notebooks/03_classification/ner_cli_test_colab.ipynb` sekarang mengerjakan keempat baris di atas.

**3. Jonathan (NER)**

- `annotate_entities` sudah dijalankan ulang dengan CLAIM baru. Entitas PER/ORG/LOC/TIME tidak berubah; `core_claim` berubah di 1.122 dari 3.048 judul.
- `core_claim` pada judul Komdigi sering terpotong: terisi 2.987 → 2.825; satu kata 156 → 478; isinya hanya "Tautan" 19 → 279 (dari 437 judul berawalan "Tautan", 286 menjadi satu kata). Contoh: "Tautan Pembukaan Rekening Bank Otomatis dan Gratis via Telegram" → lama "Rekening Bank Otomatis dan Gratis via Telegram", baru "Tautan".
- Sebagai hypothesis stance, `core_claim` jauh lebih buruk dari teks penuh: `supports` benar 77 vs 94 dari 100 judul asli; `supports` keliru di `no_match` 10 vs 3. Pipeline memakai teks input untuk retrieval dan stance; `core_claim` hanya ditampilkan.
- Span CLAIM sebagai kueri tambahan juga dicoba dan ditolak: span membuang subjek ("Nestle Indonesia sedang membuka lowongan" → "sedang membuka lowongan pekerjaan") lalu cocok dengan klaim perusahaan lain (2 verdict salah dari 100 parafrasa test). Ini sejalan dengan catatan aturan `subject` di bagian NER.
- Yang dipakai dari NER di pipeline sekarang: entitas PER/ORG/LOC sebagai **pengaman** (verdict ditahan bila input menyebut entitas yang tidak ada di klaim terbantah). Filter entitas di retrieval tidak menaikkan success@k (tabel RQ3).
- Mohon dicek: CLAIM untuk judul berbentuk frasa benda ("Tautan ...", "Lowongan ...", "Video ...").

**4. Dzaky (klasifikasi, CLI, orchestrator)**

- `pipeline/` adalah wilayahmu (0006); perubahan di sana kuajukan sebagai usulan 0013 dan menunggu persetujuanmu. Ringkasnya: `stages.py` (retrieval/stance sungguhan, kueri per kalimat, pengaman), `textproc.py` (baru: pembersihan input, pemecahan kalimat, penanda bantahan, bingkai laporan), `confidence.py` (urutan verdict), `cli.py` (tahan galat dan encoding).
- Aturan verdict sekarang: "kredibilitas rendah" hanya bila ada padanan + `supports`, dan tidak ditahan pengaman entitas atau penanda bantahan. `refutes` → tinjauan manusia. Padanan sangat mirip (skor ≥ 0,87) tanpa keputusan stance → tinjauan manusia, **verdict classifier tidak dikeluarkan**. Selain itu jatuh ke classifier seperti semula.
- Classifier: checkpoint demo terpasang dari zip-mu; gerbang 200 kata tetap. Dua teks panjang dari notebookmu dinilai `valid` (1,00) dan `hoax` (0,99).
- Notebook `ner_cli_test_colab.ipynb` kuubah: unduh dua zip NER + zip classifier + CSV Komdigi lewat Drive API, bangun basis klaim dan cocokkan hash-nya, uji CLI tanpa stub. Sel evaluasi panjang kalimat tetap ada tetapi dilewati bila `indobert_results/` tidak ada di Drive yang dipakai.
- **Pembekuan hash 18 Nov** (`classification/freeze_hashes.py --merge-mafindo ...`) tidak kujalankan: skripnya membaca file Mendeley dan skor IndoBERT yang hanya ada di mesinmu, jadi dari mesinku manifestnya akan berisi entri "missing". Sisi task 3 sudah kucek: `claims.jsonl` dan `refuted_claims_split.csv` cocok dengan `data/splits/manifest.json` (beku 12 Nov), juga dari clone baru.

**5. Angka yang bisa dipakai, dengan statusnya**

Semua parafrasa dan pasangan stance masih **draf LLM yang belum diverifikasi manusia**; set teks panjang adalah buatan dan dipakai memilih aturan. Sebut status itu bila mengutip.

| Hal | Angka |
|---|---|
| Parafrasa test → "kredibilitas rendah" dengan klaim benar | 88 dari 100 (dev 42 dari 44) |
| Kueri tanpa padanan yang salah dicap | 0 dari 144 |
| Hoaks terselip di antara 4 kalimat berita | 56 dari 60, tanpa klaim salah |
| Artikel berita valid buatan yang salah dicap | 0 dari 60 |
| Artikel cek fakta Komdigi yang salah dicap | 0 dari 40 |
| RQ3 success@1 parafrasa: BM25 / +filter entitas / E5 / +filter entitas | 0,98 / 0,97 / 0,95 / 0,95 |
| RQ3 kueri berpadanan yang ikut tertolak pada ambang dev: BM25 / E5 | 0,38 / 0,10 |
| RQ4 macro-F1 pada 300 pasangan tim (draf): similarity / mDeBERTa / IndoBERT-IndoNLI | 0,370 / 0,638 / 0,800 |

**Perhatian untuk laporan:** pada pasangan tim, IndoBERT-IndoNLI mengungguli mDeBERTa secara tiga kelas, kebalikan dari IndoNLI test_expert (0,569 vs 0,776). CLI tetap memakai mDeBERTa karena unggul pada `supports` (F1 0,95 vs 0,88), satu-satunya label yang menghasilkan verdict otomatis. Jangan menulis "mDeBERTa model stance terbaik" tanpa catatan ini.

**6. Yang masih menunggu manusia**

- **Verifikasi anotasi.** 200 hypothesis stance, 100 pasangan `unrelated`, dan 144 parafrasa kueri berstatus `annotated` dengan `annotators: ["llm:claude"]`. Belum ada yang `verified`. Set uji final stance butuh dua annotator manusia per pasangan. Cara kerja dan pengamannya ada di `data/annotations/claim_verification/README.md`.
- **Pengesahan 0013** oleh Dzaky dan Jonathan.
- **Set teks panjang yang independen** untuk laporan: set buatan yang sekarang dipakai memilih aturan.
- **Stance tahap 2 (BRIN)** belum dilatih; butuh akses dataset gated `nlp-brin-id/fakenews-mafindo` dan GPU.

**7. Keterbatasan yang diketahui**

- Hoaks yang menyatu dengan kalimat panjang tanpa tanda baca tidak dipecah; di dalam teks 200 kata lebih ia bisa lolos dengan verdict classifier saja.
- Kutipan langsung, pesan berantai bergaya bebas, dua klaim dalam satu kalimat, dan teks informal berat sering dinilai `unrelated`: masuk tinjauan manusia, tidak salah cap.
- Sistem sengaja condong ke "perlu tinjauan manusia"; beban tinjauan perlu dilaporkan di RQ5.
