# 0011 — Keputusan teknis tambahan

**Status:** Final

Hal-hal yang sebelumnya masih terbuka di rencana pengerjaan, ditetapkan di sini supaya tidak perlu dibahas lagi.

## 1. Kontrak `schemas.py`
- `Entity` tetap memakai field `label` (bukan `type`), nilainya PER, ORG, LOC, TIME, atau CLAIM.
- Tahap retrieval mengembalikan `list[RetrievalResult]`: top-k klaim terbantah, urut skor kemiripan dari tertinggi. List kosong berarti tidak ada klaim yang mirip.
- `NERResult.core_claim` = teks span CLAIM pertama di input; string kosong bila tidak ada CLAIM. Untuk artikel multi-kalimat, CLAIM pertama biasanya ada di kalimat pembuka.

## 2. Parafrasa untuk pasangan stance `supports`
Ditulis **manual** oleh annotator, tanpa model generatif, lalu diverifikasi orang kedua (sesuai [0007](0007-seed-set-dan-annotator.md)).

## 3. Base model stance
`indobenchmark/indobert-base-p1`, sama dengan NER dan klasifikasi. Satu keluarga model untuk ketiga task memudahkan perbandingan dan reproduksi.

## 4. Compute
- **Semua training model dijalankan lewat notebook di Google Colab (GPU)**, satu notebook per task di `notebooks/` (NER: `notebooks/02_ner/train_ner_colab.ipynb` untuk baseline dan tahap 2, `rq1_ner_colab.ipynb` untuk RQ1). Notebook hanya memanggil kode di `src/`; logika tidak ditulis di notebook.
- Alasan: MPS di Mac 16 GB kehabisan memori pada IndoBERT dan crash pada BiLSTM-CRF, dan semua angka laporan harus dari environment yang sama.
- Laptop dipakai untuk menulis kode dan menjalankan test unit.
- Hasil (ringkasan metrik dan zip checkpoint/prediksi) di-upload ke folder Drive tim `hasil_training` (https://drive.google.com/drive/folders/1Eq9C3SIW42xYpRn5UVAwoiLrRkOY52Pn), satu subfolder per run.

## 5. Akses data
Tidak ada permintaan izin ke penulis mana pun. Data yang dipakai hanya yang bisa diunduh langsung atau lewat akun sendiri (login Kaggle, persetujuan gated HF untuk dataset BRIN). Dataset yang lisensinya tidak jelas (Cahyo dkk., 2025) tidak dipakai.
