# Baseline NER: BiLSTM-CRF dengan embedding kata + fitur karakter (CNN).
# Kata di-lowercase untuk embedding; karakter tetap cased supaya huruf kapital tetap terbaca.

import torch
from torch import nn
from torchcrf import CRF

PAD, UNK = "<pad>", "<unk>"


class Vocab:
    def __init__(self, items: list[str]):
        self.itos = [PAD, UNK] + sorted(set(items))
        self.stoi = {s: i for i, s in enumerate(self.itos)}

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
