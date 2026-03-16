"""Byte-level tokenizer with common n-gram merges (via NLTK)."""

import json
from collections import Counter
from pathlib import Path

from nltk import ngrams


class ByteNgramTokenizer:
    """Byte-level tokenizer augmented with frequent byte n-grams.

    Tokens 0-255 represent individual bytes.
    Tokens 256+ represent frequent byte n-grams learned from a corpus.
    """

    def __init__(self):
        self._ngram_to_id: dict[tuple[int, ...], int] = {}
        self._id_to_ngram: dict[int, tuple[int, ...]] = {}
        self._max_ngram_len: int = 1

    @property
    def n_vocab(self) -> int:
        return 256 + len(self._ngram_to_id)

    def fit(
        self, texts: list[str], max_vocab_size: int = 10_000, max_ngram_order: int = 6
    ) -> "ByteNgramTokenizer":
        """Count byte n-grams in *texts* and keep the most frequent ones."""
        max_merges = max(0, max_vocab_size - 256)
        if max_merges == 0:
            return self

        counts: Counter = Counter()
        for text in texts:
            byte_seq = list(text.encode("utf-8"))
            for n in range(2, max_ngram_order + 1):
                counts.update(ngrams(byte_seq, n))

        for rank, (gram, _count) in enumerate(counts.most_common(max_merges)):
            token_id = 256 + rank
            self._ngram_to_id[gram] = token_id
            self._id_to_ngram[token_id] = gram

        if self._ngram_to_id:
            self._max_ngram_len = max(len(g) for g in self._ngram_to_id)
        return self

    def encode(self, text: str) -> list[int]:
        """Greedy longest-match encoding over bytes."""
        data = list(text.encode("utf-8"))
        tokens: list[int] = []
        i = 0
        while i < len(data):
            matched = False
            for length in range(
                min(self._max_ngram_len, len(data) - i), 1, -1
            ):
                gram = tuple(data[i : i + length])
                tid = self._ngram_to_id.get(gram)
                if tid is not None:
                    tokens.append(tid)
                    i += length
                    matched = True
                    break
            if not matched:
                tokens.append(data[i])
                i += 1
        return tokens

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs back to a UTF-8 string."""
        raw: list[int] = []
        for t in tokens:
            if t < 256:
                raw.append(t)
            else:
                ngram = self._id_to_ngram.get(t)
                if ngram is not None:
                    raw.extend(ngram)
        return bytes(raw).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        """Persist learned n-gram vocabulary to JSON."""
        merges = [None] * len(self._ngram_to_id)
        for gram, token_id in self._ngram_to_id.items():
            merges[token_id - 256] = list(gram)
        Path(path).write_text(json.dumps({"merges": merges}), encoding="utf-8")

    def load(self, path: str | Path) -> "ByteNgramTokenizer":
        """Load a previously saved n-gram vocabulary."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self._ngram_to_id.clear()
        self._id_to_ngram.clear()
        for rank, gram_list in enumerate(data["merges"]):
            gram = tuple(gram_list)
            token_id = 256 + rank
            self._ngram_to_id[gram] = token_id
            self._id_to_ngram[token_id] = gram
        if self._ngram_to_id:
            self._max_ngram_len = max(len(g) for g in self._ngram_to_id)
        else:
            self._max_ngram_len = 1
        return self


def build_tokenizer(
    texts: list[str], max_vocab_size: int = 10_000
) -> ByteNgramTokenizer:
    """Build and return a byte n-gram tokenizer fitted on *texts*."""
    return ByteNgramTokenizer().fit(texts, max_vocab_size)


def get_tokenizer(vocab_path: str | Path | None = None) -> ByteNgramTokenizer:
    """Return a tokenizer -- loading saved vocab if *vocab_path* is given."""
    tok = ByteNgramTokenizer()
    if vocab_path is not None and Path(vocab_path).exists():
        tok.load(vocab_path)
    return tok


if __name__ == "__main__":
    enc = get_tokenizer()
    print(f"Vocab size: {enc.n_vocab}")
