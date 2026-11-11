# Tahap-tahap pipeline. Tiap tahap = objek yang bisa dipanggil seperti fungsi.
# Model baru dimuat saat pertama dipakai, supaya import dan --help tetap cepat.
#
# Contoh:
#   from indonesian_hoax_checker.pipeline.stages import NERStage
#   ner = NERStage.from_config()          # baca configs/pipeline.yaml
#   ner("Pemerintah menghentikan bantuan MBG mulai bulan depan")   # -> NERResult

from pathlib import Path

import yaml

from indonesian_hoax_checker.common.schemas import NERResult

PIPELINE_CONFIG = "configs/pipeline.yaml"


class NERStage:
    name = "ner"

    # checkpoint=None -> mode stub (selalu entitas kosong), sesuai docs/decisions/0005
    def __init__(self, checkpoint: str | None, device: str = "auto"):
        self.checkpoint = checkpoint
        self.device = device
        self._predictor = None

    @classmethod
    def from_config(cls, path: str = PIPELINE_CONFIG) -> "NERStage":
        cfg = yaml.safe_load(Path(path).read_text()).get("ner", {})
        return cls(cfg.get("checkpoint"), cfg.get("device", "auto"))

    @property
    def is_stub(self) -> bool:
        return self.checkpoint is None

    def _load(self):
        if self._predictor is None:
            if not Path(self.checkpoint).exists():
                raise FileNotFoundError(
                    f"Checkpoint NER {self.checkpoint} belum ada. Unduh ner_results.zip dari Drive hasil_training "
                    "dan ekstrak di root repo, atau set ner.checkpoint: null di configs/pipeline.yaml untuk mode stub."
                )
            from indonesian_hoax_checker.ner.predict import NERPredictor

            self._predictor = NERPredictor(self.checkpoint, self.device)
        return self._predictor

    def __call__(self, text: str) -> NERResult:
        return self.run_batch([text])[0]

    def run_batch(self, texts: list[str]) -> list[NERResult]:
        if self.is_stub:
            return [{"entities": [], "core_claim": ""} for _ in texts]
        return self._load().predict_batch(texts)
