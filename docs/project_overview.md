# Project Overview

Tugas besar IF5153 Pemrosesan Bahasa Alami, ITB 2026.

## Tujuan

Indonesian Hoax Checker menerima klaim atau artikel berbahasa Indonesia (satu kalimat sampai beberapa paragraf, formal atau informal) dan mengembalikan **indikasi kredibilitas** beserta **bantahan resmi Komdigi** bila ada.

Sistem ini alat bantu penyaringan awal, **bukan pemutus akhir**:

- keputusan berskor rendah diserahkan ke manusia;
- klaim tanpa padanan di basis klaim terbantah dilaporkan dengan lebih hati-hati;
- cakupan hanya bahasa Indonesia dan klaim yang beririsan dengan basis rujukan.

## Tim dan task

| Task | Pemilik | Input | Output |
|---|---|---|---|
| 1. Named Entity Recognition | Jonathan Kenan Budianto (13523139) | Teks mentah | Entitas PER, ORG, LOC, TIME dan span CLAIM |
| 2. Klasifikasi hoaks | Muhammad Dzaky Atha Fadhilah (18223124) | Teks + entitas | Label `hoax`/`valid` + skor 0–1 |
| 3. Similarity dan stance | Danendra Shafi Athallah (13523136) | Teks, entitas | Klaim terbantah terdekat, skor kemiripan, stance |

Pembagian tambahan ([0006](decisions/0006-pemilik-cli-dan-orchestrator.md)): Dzaky memegang CLI dan orchestrator, Jonathan memimpin anotasi, Danendra memegang retrieval dan stance.

## Contoh

Input: *"Pemerintah menghentikan seluruh bantuan Makan Bergizi Gratis mulai bulan depan"*

1. NER: `Pemerintah` = ORG, `menghentikan seluruh bantuan Makan Bergizi Gratis` = CLAIM, `bulan depan` = TIME.
2. Klasifikasi: dugaan hoaks dengan skor tertentu.
3. Retrieval dan stance: klaim terbantah terdekat di basis Komdigi, stance `supports` → **kredibilitas rendah, bantahan resmi disertakan**.

## Research questions

| RQ | Task | Pertanyaan |
|---|---|---|
| RQ1 | NER | Apakah normalisasi dan augmentasi teks informal memperbaiki ekstraksi entitas dan klaim inti? |
| RQ2 | Klasifikasi | Apakah fitur entitas memperbaiki klasifikasi hoaks dibanding teks saja? |
| RQ3 | Similarity | Apakah mempersempit kandidat dengan entitas bersama membantu menemukan klaim terbantah yang tepat? |
| RQ4 | Stance | Apakah stance detection memisahkan hoaks yang diulang dari koreksinya, yang tidak bisa dilakukan similarity saja? |
| RQ5 | End-to-end | Apakah pipeline lengkap dengan ambang keyakinan memberi indikasi lebih aman daripada classifier saja? |

## Deliverable

- Pipeline yang bisa dijalankan lewat command prompt (demo dengan beberapa input, termasuk kasus sulit).
- Eksperimen lengkap per task: baseline, model kuat, metrik, ablasi.
- PPT: penjelasan algoritma, hasil eksperimen, error analysis. Presentasi minggu 14–15.

## Dokumen terkait

- [architecture.md](architecture.md) — alur sistem dan modul
- [dataset_documentation.md](dataset_documentation.md) — dataset dan audit
- [experiment_protocol.md](experiment_protocol.md) — split, metrik, ablasi
- [label_mapping.md](label_mapping.md) — label tiap task
- [annotation_guidelines.md](annotation_guidelines.md) — pedoman anotasi CLAIM, TIME, stance
- [decisions/](decisions/README.md) — keputusan desain
- [implementation_plan.md](implementation_plan.md) — rencana pengerjaan
