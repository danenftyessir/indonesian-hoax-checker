**Handoff Task 2 (klasifikasi hoaks/valid) utk Danendra**

Bagian classification sudah selesai dan sudah merge ke `main`. Yang tersisa adalah menyambungkan retrieval dan stance ke pipeline.

**1. Yang perlu disambungkan (`src/indonesian_hoax_checker/pipeline/stages.py`)**

- run_retrieve(text, ner) dan run_stance(text, retrieval) di main masih stub.
- Modulnya sudah cocok bentuknya, jadi kerjanya tinggal membungkus:
  - `retrieve`: build_retriever(config) lalu `Retriever.retrieve(query, ner["entities"])`. Hasilnya sudah RetrievalResult lengkap dengan `scorer`.
  - `stance`: load_model(...) lalu `predict_stance(model, text, retrieval["candidates"][0])`, atau None bila tidak ada kandidat.
- Polanya sama dengan _get_classifier() dan `_get_ner_stage()`: baca bagian config di `configs/pipeline.yaml`, muat model secara malas, lalu STUB_STAGES.discard("retrieve") atau discard("stance") saat model benar-benar dipakai. Dengan begitu label [STUB] di CLI hilang sendiri.
- Orchestrator menjalankan classification dan retrieval paralel (`docs/decisions/0010`), lalu stance. confidence.combine menerima stance bernilai None dan mendahulukan bukti retrieval/stance atas skor classifier.
- Query retrieval saat ini adalah teks penuh (`retrieval_query`). Ubah di orchestrator.py bila mau memakai klaim inti dari NER.
- Menyiapkan korpus klaim terbantah dan checkpoint stance di mesin yang menjalankan CLI. periksa apakah build_retriever jalan tanpa file tambahan.

**2. Classifier (sudah jadi)**

- Checkpoint: `artifacts/checkpoints/classifier/indobert_seed13`, dibaca dari bagian classification: di `configs/pipeline.yaml`.
- Gerbang `classification.min_words: 200`. Teks lebih pendek dilewati dan hasilnya "perlu tinjauan manusia"; CLI menampilkan "dilewati". Jadi untuk klaim pendek, keputusan akhir **bergantung penuh pada retrieval dan stance**.
- Skor classifier ditandai "belum dikalibrasi". Classifier hanya sinyal lemah.

**3. Angka yang bisa dikutip**
Semua dihitung dari kode di repo dan dicatat di `docs/dataset_documentation.md`.

- Dalam domain (Mafindo test): macro-F1 0,987. Lintas-sumber (Mendeley 600): macro-F1 sekitar 0,52–0,57, AUROC sekitar 0,59. Jadi classifier belum bisa dipakai sebagai pemutus.
- 100 query no_match yang diisi: kemiripan char n-gram dengan judul Komdigi tertinggi 0,431, tidak ada yang ≥ 0,5. Itu dihitung dengan `classification/no_match_check.py`. Skrip itu hanya mengukur kemiripan teks, bukan kualitas retrieval.

**4. Yang belum terverifikasi**

- Ambang 200 kata belum diuji pada data berlabel di luar Mafindo.
- Perbandingan TF-IDF vs IndoBERT di Mendeley belum diuji signifikansinya.
- Belum ada pengujian end-to-end CLI dengan retrieval dan stance sungguhan. Itu tergantung penyambunganmu.

**5. Tanggal**

- Pembekuan ulang hash test set: **18 Nov 2026**. menjalankan `python classification/freeze_hashes.py --merge-mafindo data/splits/mafindo_test_hash.json`.

Halo Dzaky, Danendra, update NER (task 1) dari aku:

1. Model CLAIM baru (wajib diunduh ulang)
   Model CLAIM sekarang dilatih dengan augmentasi teks informal (hasil RQ1), plus ada pasca-proses yang memotong ekor CLAIM (waktu, “?”, “!”). F1 CLAIM di test naik dari 0,618 ke 0,647. Model entitas tidak berubah.

Unduh dua zip, lalu ekstrak berurutan (zip kedua menimpa model CLAIM lama):

Zip lama (model entitas + CLAIM lama), hasil_training/ner_20261112-1801: https://drive.google.com/file/d/1c_ouOXwCIdMgqD2_rJzF_oXL-5qS0oEN/view
Zip baru (hanya model CLAIM), hasil_training/ner_20261114-1153: https://drive.google.com/file/d/1s5mZu7px3fsNGz1_BtCq66cl7nl6k0Uv/view
unzip -o ner_checkpoints_seed42.zip -d artifacts/checkpoints/ner/        # zip lama dulu
unzip -o "ner_checkpoints_seed42 (1).zip" -d artifacts/checkpoints/ner/  # lalu zip baru

Kalau nama file hasil unduhan beda, sesuaikan nama di perintah. configs/pipeline.yaml tidak perlu diubah, karena nama foldernya sama.

2. Dzaky

ner_cli_test_colab.ipynb masih cuma mengunduh zip lama (NER_ZIP_FILE_ID = 1c_ouOX...). Tambahkan unduhan zip baru (1s5mZu7px3fsNGz1_BtCq66cl7nl6k0Uv) dan ekstrak sesudah zip lama.
data/demo_cases.txt sudah ada: 25 kasus untuk 8 kategori demo dari protokol, sebagian merujuk ID klaim Komdigi asli. Bisa langsung dipakai untuk demo CLI. Hasil NER-nya ada di reports/tables/ner_demo_cases.md.

3. Danendra

Tolong jalankan ulang retrieval.annotate_entities dengan --checkpoint ner_indobert_entity_seed42 dan --claim-checkpoint ner_indobert_claim_seed42, karena core_claim di claims_entities.jsonl dan retrieval_queries.jsonl sekarang berubah.
Kalau sempat, cek stance pakai core_claim vs pakai teks penuh sebagai hypothesis. Kalau core_claim jauh lebih jelek, kabari aku supaya CLAIM-nya kucek lagi.

4. Branch
   Semua perubahan ada di feat/ner dan akan aku merge ke main lewat PR. Hasil lengkap dan alasannya ada di docs/experiment_protocol.md. Draf bagian NER untuk laporan ada di reports/ner_section.md, tabel dan grafik di reports/tables dan reports/figures.

Thanks!


Halo Dzaky, Danendra, update NER (task 1) dari aku:

1. Model CLAIM baru (wajib diunduh ulang)
   Model CLAIM sekarang dilatih dengan augmentasi teks informal (hasil RQ1), plus ada pasca-proses yang memotong ekor CLAIM (waktu, “?”, “!”). F1 CLAIM di test naik dari 0,618 ke 0,647. Model entitas tidak berubah.

Unduh dua zip, lalu ekstrak berurutan (zip kedua menimpa model CLAIM lama):

Zip lama (model entitas + CLAIM lama), hasil_training/ner_20261112-1801: https://drive.google.com/file/d/1c_ouOXwCIdMgqD2_rJzF_oXL-5qS0oEN/view
Zip baru (hanya model CLAIM), hasil_training/ner_20261114-1153: https://drive.google.com/file/d/1s5mZu7px3fsNGz1_BtCq66cl7nl6k0Uv/view
unzip -o ner_checkpoints_seed42.zip -d artifacts/checkpoints/ner/        # zip lama dulu
unzip -o "ner_checkpoints_seed42 (1).zip" -d artifacts/checkpoints/ner/  # lalu zip baru

Kalau nama file hasil unduhan beda, sesuaikan nama di perintah. configs/pipeline.yaml tidak perlu diubah, karena nama foldernya sama.

2. Dzaky

ner_cli_test_colab.ipynb masih cuma mengunduh zip lama (NER_ZIP_FILE_ID = 1c_ouOX...). Tambahkan unduhan zip baru (1s5mZu7px3fsNGz1_BtCq66cl7nl6k0Uv) dan ekstrak sesudah zip lama.
data/demo_cases.txt sudah ada: 25 kasus untuk 8 kategori demo dari protokol, sebagian merujuk ID klaim Komdigi asli. Bisa langsung dipakai untuk demo CLI. Hasil NER-nya ada di reports/tables/ner_demo_cases.md.

3. Danendra

Tolong jalankan ulang retrieval.annotate_entities dengan --checkpoint ner_indobert_entity_seed42 dan --claim-checkpoint ner_indobert_claim_seed42, karena core_claim di claims_entities.jsonl dan retrieval_queries.jsonl sekarang berubah.
Kalau sempat, cek stance pakai core_claim vs pakai teks penuh sebagai hypothesis. Kalau core_claim jauh lebih jelek, kabari aku supaya CLAIM-nya kucek lagi.

4. Branch
   Semua perubahan ada di feat/ner dan akan aku merge ke main lewat PR. Hasil lengkap dan alasannya ada di docs/experiment_protocol.md. Draf bagian NER untuk laporan ada di reports/ner_section.md, tabel dan grafik di reports/tables dan reports/figures.

Thanks!

Halo Dzaky, Danendra, update NER (task 1) dari aku:

1. Model CLAIM baru (wajib diunduh ulang)
   Model CLAIM sekarang dilatih dengan augmentasi teks informal (hasil RQ1), plus ada pasca-proses yang memotong ekor CLAIM (waktu, “?”, “!”). F1 CLAIM di test naik dari 0,618 ke 0,647. Model entitas tidak berubah.

Unduh dua zip, lalu ekstrak berurutan (zip kedua menimpa model CLAIM lama):

Zip lama (model entitas + CLAIM lama), hasil_training/ner_20261112-1801: https://drive.google.com/file/d/1c_ouOXwCIdMgqD2_rJzF_oXL-5qS0oEN/view
Zip baru (hanya model CLAIM), hasil_training/ner_20261114-1153: https://drive.google.com/file/d/1s5mZu7px3fsNGz1_BtCq66cl7nl6k0Uv/view
unzip -o ner_checkpoints_seed42.zip -d artifacts/checkpoints/ner/        # zip lama dulu
unzip -o "ner_checkpoints_seed42 (1).zip" -d artifacts/checkpoints/ner/  # lalu zip baru

Kalau nama file hasil unduhan beda, sesuaikan nama di perintah. configs/pipeline.yaml tidak perlu diubah, karena nama foldernya sama.

2. Dzaky

ner_cli_test_colab.ipynb masih cuma mengunduh zip lama (NER_ZIP_FILE_ID = 1c_ouOX...). Tambahkan unduhan zip baru (1s5mZu7px3fsNGz1_BtCq66cl7nl6k0Uv) dan ekstrak sesudah zip lama.
data/demo_cases.txt sudah ada: 25 kasus untuk 8 kategori demo dari protokol, sebagian merujuk ID klaim Komdigi asli. Bisa langsung dipakai untuk demo CLI. Hasil NER-nya ada di reports/tables/ner_demo_cases.md.

3. Danendra

Tolong jalankan ulang retrieval.annotate_entities dengan --checkpoint ner_indobert_entity_seed42 dan --claim-checkpoint ner_indobert_claim_seed42, karena core_claim di claims_entities.jsonl dan retrieval_queries.jsonl sekarang berubah.
Kalau sempat, cek stance pakai core_claim vs pakai teks penuh sebagai hypothesis. Kalau core_claim jauh lebih jelek, kabari aku supaya CLAIM-nya kucek lagi.

4. Branch
   Semua perubahan ada di feat/ner dan akan aku merge ke main lewat PR. Hasil lengkap dan alasannya ada di docs/experiment_protocol.md. Draf bagian NER untuk laporan ada di reports/ner_section.md, tabel dan grafik di reports/tables dan reports/figures.

Thanks!

Halo Dzaky, Danendra, update NER (task 1) dari aku:

1. Model CLAIM baru (wajib diunduh ulang)
   Model CLAIM sekarang dilatih dengan augmentasi teks informal (hasil RQ1), plus ada pasca-proses yang memotong ekor CLAIM (waktu, “?”, “!”). F1 CLAIM di test naik dari 0,618 ke 0,647. Model entitas tidak berubah.

Unduh dua zip, lalu ekstrak berurutan (zip kedua menimpa model CLAIM lama):

Zip lama (model entitas + CLAIM lama), hasil_training/ner_20261112-1801: https://drive.google.com/file/d/1c_ouOXwCIdMgqD2_rJzF_oXL-5qS0oEN/view
Zip baru (hanya model CLAIM), hasil_training/ner_20261114-1153: https://drive.google.com/file/d/1s5mZu7px3fsNGz1_BtCq66cl7nl6k0Uv/view
unzip -o ner_checkpoints_seed42.zip -d artifacts/checkpoints/ner/        # zip lama dulu
unzip -o "ner_checkpoints_seed42 (1).zip" -d artifacts/checkpoints/ner/  # lalu zip baru

Kalau nama file hasil unduhan beda, sesuaikan nama di perintah. configs/pipeline.yaml tidak perlu diubah, karena nama foldernya sama.

2. Dzaky

ner_cli_test_colab.ipynb masih cuma mengunduh zip lama (NER_ZIP_FILE_ID = 1c_ouOX...). Tambahkan unduhan zip baru (1s5mZu7px3fsNGz1_BtCq66cl7nl6k0Uv) dan ekstrak sesudah zip lama.
data/demo_cases.txt sudah ada: 25 kasus untuk 8 kategori demo dari protokol, sebagian merujuk ID klaim Komdigi asli. Bisa langsung dipakai untuk demo CLI. Hasil NER-nya ada di reports/tables/ner_demo_cases.md.

3. Danendra

Tolong jalankan ulang retrieval.annotate_entities dengan --checkpoint ner_indobert_entity_seed42 dan --claim-checkpoint ner_indobert_claim_seed42, karena core_claim di claims_entities.jsonl dan retrieval_queries.jsonl sekarang berubah.
Kalau sempat, cek stance pakai core_claim vs pakai teks penuh sebagai hypothesis. Kalau core_claim jauh lebih jelek, kabari aku supaya CLAIM-nya kucek lagi.

4. Branch
   Semua perubahan ada di feat/ner dan akan aku merge ke main lewat PR. Hasil lengkap dan alasannya ada di docs/experiment_protocol.md. Draf bagian NER untuk laporan ada di reports/ner_section.md, tabel dan grafik di reports/tables dan reports/figures.

Thanks!
