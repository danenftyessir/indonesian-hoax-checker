# 0013 — Penyambungan retrieval dan stance di pipeline

**Status:** Usulan (Danendra, 2026-11-14), belum disahkan bertiga. Menyentuh [0002](0002-arah-pasangan-stance.md) (arah stance) dan `pipeline/` milik Dzaky ([0006](0006-pemilik-cli-dan-orchestrator.md)).

## Konteks
`run_retrieve` dan `run_stance` di `pipeline/stages.py` masih stub. Classifier dilewati untuk teks di bawah 200 kata, jadi untuk klaim pendek hasil akhir bergantung penuh pada retrieval dan stance. Angka pendukung ada di `docs/experiment_protocol.md` ("Hasil task 3: penyambungan pipeline").

## Keputusan
1. **Retrieval di CLI = E5** (`configs/models/retrieval_embeddings.yaml`), `top_k: 5`, ambang tolak cosine **0,837** (persentil ke-90 skor top-1 kueri `no_match` set dev). Kueri = teks input penuh.
2. **Stance = mDeBERTa zero-shot** (`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`). `stance_final` (tahap 2 BRIN) belum dilatih.
3. **Hypothesis stance = teks input penuh**, bukan `core_claim`.
4. **Stance dinilai dua arah** (`stance.bidirectional: true`): arah 0002 (premise = klaim terbantah, hypothesis = input) ditambah arah sebaliknya; `supports`/`refutes` cukup muncul di satu arah asalkan arah lain tidak berlawanan. Definisi label 0002 tidak berubah.
5. **`refutes` tidak lagi menghasilkan "kemungkinan benar"**; hasilnya "perlu tinjauan manusia" dengan bantahan resmi ditampilkan.
6. Set dev kueri retrieval dibuat (`data/annotations/claim_verification/dev/`), 44 per tipe, tanpa tumpang tindih dengan set test.

## Alasan
- Hypothesis teks penuh: `supports` benar 94 vs 77 dari 100 kueri judul asli; `core_claim` model CLAIM baru sering terpotong.
- Dua arah: judul klaim pendek jarang meng-entail input yang lebih panjang, padahal input itu mengulang klaimnya. Parafrasa test 81 → 88 `supports`, tanpa tambahan `supports` palsu di `no_match`.
- `refutes`: 5 dari 8 `refutes` pada kasus demo keliru (klaim lain bertopik sama, satir, pesan yang justru mengulang hoaks, teks Inggris).
- Ambang 0,837, bukan ambang "terbaik" dev 0,848: parafrasa mendapat skor lebih rendah dari judul asli; pada 0,848 hanya 36 dari 44 parafrasa dev yang lolos, pada 0,837 lolos 42.

## Konsekuensi
- Mesin demo perlu mengunduh dua model HF (masing-masing sekitar 1,1 GB) dan menghitung embedding basis klaim sekali; pemanggilan CLI pertama lambat.
- Verdict otomatis hanya "kredibilitas rendah" (retrieval cocok + `supports`); selain itu jatuh ke classifier atau tinjauan manusia. Beban tinjauan naik dan harus dilaporkan di RQ5.
- Parafrasa dev dan test yang dipakai menguji butir 1 dan 4 adalah **draf LLM yang belum diverifikasi manusia** (`draft_source`). Butir itu ditinjau ulang setelah anotasi selesai.
- Aturan dua arah dan aturan `refutes` dipilih sambil melihat `data/demo_cases.txt` dan kueri test; kasus demo itu tidak boleh dipakai lagi sebagai bukti hasil.
