# Label NER proyek dan pemetaan dari NERGrit/NERP. Harus sama dengan configs/labels.yaml.

TYPES = ["PER", "ORG", "LOC", "TIME", "CLAIM"]
LABELS = ["O"] + [f"{p}-{t}" for t in TYPES for p in ("B", "I")]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}

# Tipe yang tidak ada di pemetaan menjadi O
NERGRIT_MAPPING = {"PERSON": "PER", "ORGANISATION": "ORG", "PLACE": "LOC"}
NERP_MAPPING = {"PPL": "PER", "PLC": "LOC"}


def map_tag(tag: str, mapping: dict[str, str]) -> str:
    if tag == "O":
        return "O"
    prefix, _, kind = tag.partition("-")
    new_kind = mapping.get(kind)
    return f"{prefix}-{new_kind}" if new_kind else "O"


# Ubah ke IOB2: I-X yang tidak didahului B-X/I-X diganti menjadi B-X
def to_iob2(tags: list[str]) -> list[str]:
    fixed = []
    prev = "O"
    for tag in tags:
        if tag.startswith("I-") and prev[2:] != tag[2:]:
            tag = "B-" + tag[2:]
        fixed.append(tag)
        prev = tag
    return fixed


# Ambil span entitas sebagai (tipe, token_awal, token_akhir_eksklusif)
def get_spans(tags: list[str]) -> list[tuple[str, int, int]]:
    spans = []
    start, kind = None, None
    for i, tag in enumerate(tags + ["O"]):
        if start is not None and not (tag.startswith("I-") and tag[2:] == kind):
            spans.append((kind, start, i))
            start, kind = None, None
        if tag.startswith("B-"):
            start, kind = i, tag[2:]
    return spans
