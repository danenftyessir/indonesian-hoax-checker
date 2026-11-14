NER pada `data/demo_cases.txt` (entitas: ner_indobert_entity_seed42, CLAIM: ner_indobert_claim_seed42, dengan pasca-proses CLAIM).

| Kategori | Teks | Entitas | core_claim |
|---|---|---|---|
| 1 | Video sebut Kemenag buka bantuan dana hibah Arab Saudi | Arab Saudi/LOC | buka bantuan dana hibah Arab Saudi |
| 1 | Aturan baru Pertamina batasi pembelian BBM di semua SPBU | Pertamina/ORG | batasi pembelian BBM di semua SPBU |
| 1 | Air nanas panas dapat sembuhkan kanker dan penyakit berat | - | dapat sembuhkan kanker dan penyakit berat |
| 2 | Kementerian Agama katanya membagikan dana hibah dari pemerintah Arab Saudi, silakan daftar lewat video ini. | Kementerian/ORG, Arab Saudi/LOC | membagikan dana hibah dari pemerintah Arab Saudi |
| 2 | Mulai sekarang setiap SPBU hanya boleh menjual BBM dalam jumlah terbatas karena aturan baru dari Pertamina. | Pertamina/ORG | hanya boleh menjual BBM dalam jumlah terbatas karena aturan baru dari Pertamina |
| 2 | Minum air rebusan nanas yang masih panas bisa menyembuhkan kanker, sebarkan ke keluarga. | - | bisa menyembuhkan kanker |
| 3 | Faktanya, Kementerian Agama tidak pernah membuka bantuan dana hibah dari Arab Saudi. | Kementerian/ORG, Arab Saudi/LOC | tidak pernah membuka bantuan dana hibah dari Arab Saudi |
| 3 | Tidak benar bahwa air nanas panas dapat menyembuhkan kanker; belum ada bukti ilmiahnya. | - | air nanas panas dapat menyembuhkan kanker |
| 4 | Pemerintah Kota Bandung akan membangun taman baca di setiap kelurahan pada tahun 2027. | Kota Bandung/LOC, tahun 2027/TIME | akan membangun taman baca di setiap kelurahan |
| 4 | Beredar pesan bahwa Bank Indonesia akan mencetak uang pecahan Rp250.000 mulai Januari 2027. | Indonesia/ORG, Januari 2027/TIME | akan mencetak uang pecahan Rp250.000 |
| 4 | PLN mengumumkan diskon listrik 80 persen untuk semua pelanggan selama libur Natal. | - | mengumumkan diskon listrik 80 persen untuk semua pelanggan selama libur Natal |
| 5 | katanya kemenag bagi2 dana hibah arab saudi gaes, buruan daftar sblm ditutup!!! | arab saudi/LOC | bagi2 dana hibah arab saudi gaes |
| 5 | INFO PENTING!!! pertamina batasin beli bbm di smua spbu mulai bsk | pertamina/ORG | batasin beli bbm di smua spbu |
| 5 | pemerintah stop bansos mulai bsk gaes!!! sebarin ke grup kluarga | - | mulai bsk gaes |
| 5 | Kementrian Kesehtan bilang vaksin booster bikin mandul?? hati2 ya bun | Kementrian Kesehtan/ORG | vaksin booster bikin mandul |
| 6 | Pertamina tidak membatasi pembelian BBM di SPBU. | Pertamina/ORG | tidak membatasi pembelian BBM di SPBU |
| 6 | Harga Pertalite naik menjadi Rp15 ribu per liter mulai 1 Desember 2026. | 1 Desember 2026/TIME | naik menjadi Rp15 ribu per liter |
| 6 | Harga Pertalite naik menjadi Rp13 ribu per liter mulai 1 Desember 2026. | 1 Desember 2026/TIME | naik menjadi Rp13 ribu per liter |
| 7 | Presiden Prabowo Subianto meresmikan Bandara Dhoho di Kediri, Jawa Timur. | Prabowo Subianto/PER, Bandara Dhoho/LOC, Kediri/LOC, Jawa Timur/LOC | meresmikan Bandara Dhoho di Kediri |
| 7 | Menteri Keuangan mengatakan pemerintah akan menjaga harga BBM subsidi tetap stabil hingga akhir tahun. | - | pemerintah akan menjaga harga BBM subsidi tetap stabil hingga akhir tahun |
| 7 | BREAKING: kucing di Jakarta resmi dilantik jadi ketua RT karena warga capek rapat (satir). | Jakarta/LOC | resmi dilantik jadi ketua RT karena warga capek rapat (satir) |
| 8 | hoaks? | - | - |
| 8 | (kosong) | - | - |
| 8 | Beredar pesan berantai di WhatsApp. / Pesan itu mengeklaim Kominfo akan memutus internet nasional selama tiga hari mulai Senin. / Pengirim meminta pesan diteruskan ke sepuluh grup agar nomor tidak diblokir. | selama tiga hari/TIME, Senin/TIME | akan memutus internet nasional |
| 8 | Guys, the government says bansos will be stopped next month, beneran ga sih? | - | will be stopped next month |
