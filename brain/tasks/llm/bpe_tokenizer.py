"""BPE tokenizer wrapper using HuggingFace `tokenizers` lib.

Drop-in replacement for WordTokenizer with the same API:
  fit(texts) → trains BPE merges
  encode(text) → list of token ids
  decode(ids) → text
  get_vocab_size() → V

Key advantages over WordTokenizer at scale:
  - No <UNK> tokens — every input is composable from BPE units
  - Subword granularity preserves morphology ("running" → "run" + "ning")
  - Smaller vocab needed for same coverage
  - Standard for transformer LMs; matches TinyStories paper anchor

Phase 2+ migration: replace `WordTokenizer` with `BPETokenizer` and
substrate-LLM should handle vocab beyond ~5K naturally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


# Lazy-imported to allow optional dependency
def _lazy_imports():
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace
        return Tokenizer, BPE, BpeTrainer, Whitespace
    except ImportError as e:
        raise ImportError(
            "HuggingFace `tokenizers` not available. "
            "Install via: ~/.venv-substrate/bin/pip install tokenizers"
        ) from e


@dataclass
class BPETokenizer:
    """BPE tokenizer with the same interface as WordTokenizer."""
    max_vocab: int = 8000
    min_freq: int = 2
    lowercase: bool = True

    PAD: str = '<pad>'
    UNK: str = '<unk>'
    BOS: str = '<bos>'
    EOS: str = '<eos>'

    _tokenizer: Optional[object] = None
    token_to_id: Dict[str, int] = None
    id_to_token: Dict[int, str] = None

    def __post_init__(self):
        if self.token_to_id is None:
            self.token_to_id = {}
        if self.id_to_token is None:
            self.id_to_token = {}

    def fit(self, texts: Iterable[str]) -> None:
        """Train BPE on the given corpus."""
        Tokenizer, BPE, BpeTrainer, Whitespace = _lazy_imports()

        tok = Tokenizer(BPE(unk_token=self.UNK))
        tok.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            vocab_size=self.max_vocab,
            min_frequency=self.min_freq,
            special_tokens=[self.PAD, self.UNK, self.BOS, self.EOS],
        )

        # train_from_iterator wants strings (and may consume the iterator)
        texts_list = [t.lower() if self.lowercase else t for t in texts]
        tok.train_from_iterator(texts_list, trainer=trainer)
        self._tokenizer = tok

        # Sync token_to_id / id_to_token for compatibility
        vocab = tok.get_vocab()
        self.token_to_id = dict(vocab)
        self.id_to_token = {v: k for k, v in vocab.items()}

    def encode(self, text: str, *, add_bos: bool = True,
                add_eos: bool = True) -> List[int]:
        if self._tokenizer is None:
            raise RuntimeError('Tokenizer not fit yet. Call fit(texts) first.')
        if self.lowercase:
            text = text.lower()
        ids: List[int] = []
        if add_bos:
            ids.append(self.token_to_id[self.BOS])
        ids.extend(self._tokenizer.encode(text).ids)
        if add_eos:
            ids.append(self.token_to_id[self.EOS])
        return ids

    def decode(self, ids: List[int], *, skip_special: bool = True) -> str:
        if self._tokenizer is None:
            raise RuntimeError('Tokenizer not fit yet.')
        specials = {self.token_to_id[s] for s in
                    (self.PAD, self.UNK, self.BOS, self.EOS)
                    if s in self.token_to_id}
        if skip_special:
            ids = [i for i in ids if i not in specials]
        return self._tokenizer.decode(ids)

    def get_vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size() if self._tokenizer else 0
