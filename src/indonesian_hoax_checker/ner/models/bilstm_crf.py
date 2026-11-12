# Baseline NER: BiLSTM-CRF dengan embedding kata + fitur karakter (CNN).
# Kata di-lowercase untuk embedding; karakter tetap cased supaya huruf kapital tetap terbaca.

import json
from pathlib import Path

import torch
from torch import nn
from torchcrf import CRF

from indonesian_hoax_checker.ner.labels import LABELS

PAD, UNK = "<pad>", "<unk>"


class Vocab:
    def __init__(self, items: list[str]):
        self.itos = [PAD, UNK] + sorted(set(items))
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    # Bangun ulang dari daftar yang disimpan di checkpoint
    @classmethod
    def from_itos(cls, itos: list[str]) -> "Vocab":
        vocab = cls([])
        vocab.itos = itos
        vocab.stoi = {s: i for i, s in enumerate(itos)}
        return vocab

    def __len__(self) -> int:
        return len(self.itos)

    def get(self, item: str) -> int:
        return self.stoi.get(item, 1)


def build_vocabs(sentences: list[list[str]]) -> tuple[Vocab, Vocab]:
    words = Vocab([tok.lower() for tokens in sentences for tok in tokens])
    chars = Vocab([ch for tokens in sentences for tok in tokens for ch in tok])
    return words, chars


# Ubah satu batch kalimat menjadi tensor (word_ids, char_ids, mask)
def encode_batch(batch: list[list[str]], words: Vocab, chars: Vocab, max_word_len: int = 30):
    max_len = max(len(tokens) for tokens in batch)
    max_chars = min(max_word_len, max(len(tok) for tokens in batch for tok in tokens))
    word_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    char_ids = torch.zeros(len(batch), max_len, max_chars, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, tokens in enumerate(batch):
        for j, tok in enumerate(tokens):
            word_ids[i, j] = words.get(tok.lower())
            for k, ch in enumerate(tok[:max_chars]):
                char_ids[i, j, k] = chars.get(ch)
            mask[i, j] = True
    return word_ids, char_ids, mask


class BiLSTMCRF(nn.Module):
    def __init__(self, n_words, n_chars, n_labels, word_dim=100, char_dim=30, char_filters=50, hidden=200, dropout=0.5):
        super().__init__()
        self.word_emb = nn.Embedding(n_words, word_dim, padding_idx=0)
        self.char_emb = nn.Embedding(n_chars, char_dim, padding_idx=0)
        self.char_cnn = nn.Conv1d(char_dim, char_filters, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(word_dim + char_filters, hidden // 2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden, n_labels)
        self.crf = CRF(n_labels, batch_first=True)

    def emissions(self, word_ids, char_ids, mask):
        b, t, l = char_ids.shape
        chars = self.char_emb(char_ids.view(b * t, l)).transpose(1, 2)  # (b*t, dim, l)
        char_feat = torch.relu(self.char_cnn(chars)).max(dim=2).values.view(b, t, -1)
        x = self.dropout(torch.cat([self.word_emb(word_ids), char_feat], dim=-1))
        lengths = mask.sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=t)
        return self.out(self.dropout(out))

    def loss(self, word_ids, char_ids, mask, label_ids):
        return -self.crf(self.emissions(word_ids, char_ids, mask), label_ids, mask=mask, reduction="mean")

    def decode(self, word_ids, char_ids, mask) -> list[list[int]]:
        return self.crf.decode(self.emissions(word_ids, char_ids, mask), mask=mask)


@torch.no_grad()
def predict(model, words, chars, token_lists, device, batch_size=64) -> list[list[str]]:
    model.eval()
    preds = []
    for i in range(0, len(token_lists), batch_size):
        word_ids, char_ids, mask = encode_batch(token_lists[i:i + batch_size], words, chars)
        for ids in model.decode(word_ids.to(device), char_ids.to(device), mask.to(device)):
            preds.append([LABELS[j] for j in ids])
    return preds


# Checkpoint = model.pt (bobot) + vocab.json + config.json (config training)
def save(out_dir, model, words, chars, cfg) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "vocab.json").write_text(json.dumps({"words": words.itos, "chars": chars.itos}))
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))


def load(out_dir, device):
    out_dir = Path(out_dir)
    cfg = json.loads((out_dir / "config.json").read_text())
    vocab = json.loads((out_dir / "vocab.json").read_text())
    words, chars = Vocab.from_itos(vocab["words"]), Vocab.from_itos(vocab["chars"])
    model = BiLSTMCRF(len(words), len(chars), len(LABELS), **cfg["model_args"])
    model.load_state_dict(torch.load(out_dir / "model.pt", map_location=device))
    return model.to(device), words, chars, cfg


# Untuk training tahap 2: tambahkan kata/karakter baru dari data baru ke kamus,
# bobot embedding lama disalin, baris baru diinisialisasi acak
def grow_vocab(model, words, chars, sentences) -> None:
    for vocab, emb_name, items in (
        (words, "word_emb", {tok.lower() for tokens in sentences for tok in tokens}),
        (chars, "char_emb", {ch for tokens in sentences for tok in tokens for ch in tok}),
    ):
        new_items = sorted(items - set(vocab.stoi))
        if not new_items:
            continue
        for item in new_items:
            vocab.stoi[item] = len(vocab.itos)
            vocab.itos.append(item)
        old = getattr(model, emb_name)
        grown = nn.Embedding(len(vocab), old.embedding_dim, padding_idx=0).to(old.weight.device)
        grown.weight.data[: old.num_embeddings] = old.weight.data
        setattr(model, emb_name, grown)
