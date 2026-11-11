# Perintah ringkas untuk audit dan baseline klasifikasi. Data mentah tidak ada di Git: lihat data/README.md.
PY ?= python
MENDELEY_DIR ?= data/raw/mendeley

.PHONY: audit-mendeley audit-mafindo baseline-mendeley

# Bangun manifest (tanpa teks artikel) dan baseline sekali jalan -> data/manifests/, results/mendeley_baseline.json
audit-mendeley:
	$(PY) classification/data_audit/mendeley_audit.py --data-dir $(MENDELEY_DIR)

# Mengunduh umn-cyber/mafindo-2024 dari Hugging Face saat pertama kali jalan (butuh paket `datasets`).
audit-mafindo:
	$(PY) classification/data_audit/mafindo_audit.py

# Baseline TF-IDF + LR, protokol diulang (rata-rata dan sebaran) -> results/mendeley_baseline_repeated.json
baseline-mendeley:
	$(PY) classification/baseline.py --data-dir $(MENDELEY_DIR)
