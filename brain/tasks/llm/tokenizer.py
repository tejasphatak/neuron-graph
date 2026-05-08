"""Simple word-level tokenizer for substrate-LLM Phase 1.

Phase 1 uses word-level tokens (split on whitespace + punctuation
isolation) for simplicity and speed of iteration. Phase 2+ swaps in
BPE via the `tokenizers` library.

API mirrors HuggingFace tokenizers minimally — fit, encode, decode,
get_vocab_size — so the swap is one-line later.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Regex: words OR punctuation as separate tokens
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


@dataclass
class WordTokenizer:
    """Whitespace-and-punctuation tokenizer with frequency-pruned vocab."""
    max_vocab: int = 5000
    min_freq: int = 2
    lowercase: bool = True

    # Special tokens (must be present)
    PAD: str = '<pad>'
    UNK: str = '<unk>'
    BOS: str = '<bos>'
    EOS: str = '<eos>'

    token_to_id: Dict[str, int] = field(default_factory=dict)
    id_to_token: Dict[int, str] = field(default_factory=dict)

    def __post_init__(self):
        # Always reserve special-token ids
        if not self.token_to_id:
            for special in (self.PAD, self.UNK, self.BOS, self.EOS):
                tid = len(self.token_to_id)
                self.token_to_id[special] = tid
                self.id_to_token[tid] = special

    def _split(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        return _TOKEN_RE.findall(text)

    def fit(self, texts) -> None:
        """Build vocab from an iterable of texts."""
        counts: Counter = Counter()
        for t in texts:
            counts.update(self._split(t))
        # Most-common tokens that meet min_freq, capped at max_vocab
        budget = self.max_vocab - len(self.token_to_id)  # already has 4 specials
        for tok, freq in counts.most_common():
            if freq < self.min_freq:
                break
            if budget <= 0:
                break
            if tok in self.token_to_id:
                continue
            tid = len(self.token_to_id)
            self.token_to_id[tok] = tid
            self.id_to_token[tid] = tok
            budget -= 1

    def encode(self, text: str, *, add_bos: bool = True,
                add_eos: bool = True) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.token_to_id[self.BOS])
        unk_id = self.token_to_id[self.UNK]
        for tok in self._split(text):
            ids.append(self.token_to_id.get(tok, unk_id))
        if add_eos:
            ids.append(self.token_to_id[self.EOS])
        return ids

    def decode(self, ids: List[int], *, skip_special: bool = True) -> str:
        specials = {self.PAD, self.UNK, self.BOS, self.EOS}
        out = []
        for tid in ids:
            tok = self.id_to_token.get(tid, self.UNK)
            if skip_special and tok in specials:
                continue
            out.append(tok)
        # Simple detokenization: join with spaces, fix spacing around punct
        text = ' '.join(out)
        text = re.sub(r'\s+([.,!?;:\'"])', r'\1', text)
        return text

    def get_vocab_size(self) -> int:
        return len(self.token_to_id)
