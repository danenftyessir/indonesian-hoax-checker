# 0013 — Penyambungan retrieval dan stance di pipeline

**Status:** Usulan. Disetujui Danendra (pemilik task 3) pada 2026-11-15; menunggu konfirmasi Dzaky (pemilik `pipeline/`, [0006](0006-pemilik-cli-dan-orchestrator.md)) dan Jonathan. Menyentuh [0002](0002-arah-pasangan-stance.md) (arah stance).

## Konteks
`run_retrieve` dan `run_stance` di `pipeline/stages.py` masih stub. Classifier dilewati untuk teks di bawah 200 kata, jadi untuk klaim pendek hasil akhir bergantung penuh pada retrieval dan stance. Angka pendukung ada di `docs/experiment_protocol.md` ("Hasil task 3: penyambungan pipeline" dan "Uji kasus tricky dan teks panjang").

## Keputusan

### Model dan kueri
1. **Retrieval di CLI = E5** (`configs/models/retrieval_embeddings.yaml`), `top_k: 5`, ambang tolak cosine **0,837** (persentil ke-90 skor top-1 kueri `no_match` set dev).
2. **Stance = mDeBERTa zero-shot** (`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`). `stance_final` (tahap 2 BRIN) belum dilatih. Pilihan ini bertumpu pada label `supports` saja dan wajib ditinjau ulang, lihat Konsekuensi.
3. **Kueri retrieval dan hypothesis stance = teks input, bukan `core_claim`.**
4. **Kueri per kalimat** (`retrieval.segment_queries`): untuk input multi-kalimat, tiap kalimat juga dijadikan kueri dan padanan berskor tertinggi yang dipakai. Input multi-kalimat di atas **30 kata** tidak dijadikan kueri utuh (`full_text_max_words`), hanya kalimat-kalimatnya.
5. **Bingkai laporan dibuang untuk kueri tambahan** (`orchestrator.strip_report_frame`): "beredar kabar bahwa X", "katanya X", "ada info di WhatsApp kalau X" → X, dengan subjek tetap utuh. Stance juga dinilai pada versi tanpa bingkai (bagi model NLI, "beredar kabar bahwa X" bukan penegasan X). Atribusi tokoh ("Zulhas menyebut ...") tidak dibuang karena bagian dari klaim.
6. **Span CLAIM dari NER tidak dipakai sebagai kueri** (`claim_span_queries: false`). Sudah dicoba dan ditolak, lihat Alasan.

### Stance
7. **Dinilai pada bagian input yang menghasilkan padanan** (kalimat paling mirip klaim), bukan pada seluruh teks dan bukan pada kalimat lain.
8. **Dua arah** (`stance.bidirectional`): arah 0002 (premise = klaim terbantah, hypothesis = input) ditambah arah sebaliknya; `supports`/`refutes` cukup muncul di satu arah asalkan arah lain tidak berlawanan. Definisi label 0002 tidak berubah.
9. **Gerbang `supports` satu-arah**: keyakinan minimal **0,6** (`one_way_min_confidence`), dan hanya untuk bagian input sepanjang **30 kata atau kurang** (`one_way_max_words`); bagian yang lebih panjang harus `supports` di dua arah.

### Verdict
10. Verdict otomatis dari bukti hanya satu: **"kredibilitas rendah"** = ada padanan + `supports` + tidak tertahan butir 12-13.
11. **`refutes` → "perlu tinjauan manusia"** (bukan "kemungkinan benar"), bantahan resmi ditampilkan.
12. **Pengaman entitas** (`stance.entity_guard`): `supports` ditahan bila bagian input menyebut PER/ORG/LOC (prediksi NER) yang tidak ada di klaim terbantah maupun klarifikasinya.
13. **Penanda bantahan** (`orchestrator.has_debunk_marker`: "faktanya", "hoaks", "tidak benar", "jangan percaya", "belum terbukti", ...): `supports` ditahan bila input memuatnya, karena teks itu kemungkinan membahas atau menyangkal klaim.
14. **Padanan kuat tanpa keputusan stance** (`retrieval.strong_match_score: 0.87`, skor terendah judul asli di set dev): bila padanan sekuat itu tetapi stance `unrelated`, hasilnya "perlu tinjauan manusia"; verdict classifier ("tidak terindikasi hoaks") tidak dikeluarkan.

### Lain-lain
15. Set dev kueri retrieval dibuat (`data/annotations/claim_verification/dev/`), 44 per tipe, tanpa tumpang tindih dengan set test.
16. Input dibersihkan sekali di orchestrator (karakter kontrol dibuang, spasi dan baris baru dirapatkan). Path di config dicari dari folder kerja lalu dari root repo; `HOAX_CHECKER_CONFIG` menunjuk `pipeline.yaml` lain.

## Alasan
- Teks penuh sebagai hypothesis: `supports` benar 94 vs 77 dari 100 kueri judul asli; `core_claim` model CLAIM baru sering terpotong.
- Dua arah: judul klaim pendek jarang meng-entail input yang lebih panjang, padahal input itu mengulang klaimnya. Parafrasa test 81 → 88 `supports`.
- Kueri per kalimat: tanpa itu klaim yang terselip di teks multi-kalimat tenggelam. Hoaks terselip di antara empat kalimat berita: 56 dari 60 terdeteksi dengan klaim yang benar.
- Teks penuh tidak dipakai untuk input panjang: pada 60 artikel berita valid buatan, keempat `supports` palsu berasal dari teks penuh 100+ kata (satu di antaranya `supports` di dua arah). Setelah aturan ini: 0 `supports` palsu.
- Gerbang satu-arah: 15 `supports` satu-arah yang benar pada parafrasa dev+test semuanya ≥ 0,75; dua verdict salah yang ditemukan uji kasus tricky (entitas ditukar 0,54; teks panjang 0,48) di bawahnya.
- `refutes`: 5 dari 8 `refutes` pada kasus demo keliru. Model ini juga sering menyebut "kontradiksi" untuk kalimat yang hanya berbagi tokoh (34-50 dari 100 kueri `no_match`).
- Penanda bantahan: dari 40 artikel cek fakta Komdigi, 26 dinilai `supports` oleh model (artikelnya memang memuat kalimat klaim); semuanya tertahan aturan ini.
- Pengaman entitas memblokir 0 dari 100 parafrasa test dan 1 dari 44 parafrasa dev ("Bank Dunia" vs "World Bank").
- Span CLAIM sebagai kueri tambahan ditolak: span CLAIM membuang subjek ("Nestle Indonesia sedang membuka lowongan" → "sedang membuka lowongan pekerjaan") sehingga cocok dengan klaim entitas lain. Parafrasa test 88 → 83 benar dengan 2 verdict ke klaim yang salah; parafrasa dev 42 → 39 dengan 1 salah. Pengupas bingkai berbasis aturan tidak mengubah satu pun hasil di set regresi dan set teks panjang.
- Padanan kuat: pada 204 input tanpa padanan (144 kueri `no_match` + 60 artikel valid buatan) skor tertinggi 0,847, jadi ambang 0,87 tidak pernah terpicu di sana; padanan yang benar bermedian 0,87.
- Ambang 0,837, bukan ambang "terbaik" dev 0,848: pada 0,848 hanya 36 dari 44 parafrasa dev yang lolos, pada 0,837 lolos 42.

## Konsekuensi
- **Pilihan model stance belum mantap.** Pada 300 pasangan tim (draf LLM), IndoBERT-IndoNLI mengungguli mDeBERTa secara tiga kelas (macro-F1 0,800 vs 0,638), kebalikan dari IndoNLI test_expert (0,569 vs 0,776). mDeBERTa unggul pada `supports` (F1 0,95 vs 0,88), satu-satunya label yang menghasilkan verdict otomatis, tetapi sering menukar `refutes` dan `unrelated`. Setelah pasangan diverifikasi: uji IndoBERT-IndoNLI di dalam pipeline; butir 8 dan 9 dipilih untuk mDeBERTa dan harus dipilih ulang bila modelnya diganti.
- Mesin demo perlu mengunduh dua model HF (masing-masing sekitar 1,1 GB) dan menghitung embedding basis klaim sekali; pemanggilan CLI pertama lambat.
- Sistem sengaja condong ke "perlu tinjauan manusia". Beban tinjauan naik dan harus dilaporkan di RQ5.
- **Keterbatasan yang diketahui:** (a) hoaks yang menyatu dengan kalimat panjang tanpa tanda baca tidak dipecah, jadi bisa tidak ditemukan; (b) parafrasa jauh, kutipan tidak langsung dengan bingkai di luar daftar aturan, dan pesan berantai bergaya bebas sering dinilai `unrelated` (masuk tinjauan, tidak salah cap); (c) teks 200 kata lebih tanpa padanan mendapat verdict dari classifier saja, termasuk bila hoaksnya tidak ada di basis Komdigi; (d) teks informal berat: retrieval menemukan klaimnya, stance sering tidak; (e) satu kalimat yang memuat dua klaim sekaligus dinilai `unrelated`.
- Parafrasa dev dan test yang dipakai menguji butir 1, 8, dan 9 adalah **draf LLM yang belum diverifikasi manusia** (`draft_source`). Butir itu ditinjau ulang setelah anotasi diverifikasi.
- Butir 4, 5, 7, 9, 12, 13, 14 dipilih sambil melihat `data/demo_cases.txt`, kasus tricky, dan set teks panjang buatan (konteksnya kalimat berita Mafindo dari set `no_match`). Angka pada set-set itu adalah pemeriksaan kewajaran, bukan hasil evaluasi; untuk laporan perlu set teks panjang baru yang tidak dipakai memilih aturan.
