"""Character IDs and contiguous train/validation splits."""
import hashlib
import json
from pathlib import Path

import torch


class CharacterTokenizer:
    def __init__(self, characters):
        self.characters = list(characters)
        if not self.characters or len(set(self.characters)) != len(self.characters):
            raise ValueError("Vocabulary must contain unique characters")
        self.to_id = {char: index for index, char in enumerate(self.characters)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    def encode(self, text):
        unknown = set(text) - self.to_id.keys()
        if unknown:
            raise ValueError(f"Characters absent from vocabulary: {sorted(unknown)!r}")
        return [self.to_id[char] for char in text]

    def decode(self, ids):
        return "".join(self.characters[index] for index in ids)

    def save(self, path):
        Path(path).write_text(json.dumps(self.characters, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


def load_data(path, context_length, tokenizer=None):
    text = Path(path).read_text(encoding="utf-8")
    tokenizer = tokenizer or CharacterTokenizer.from_text(text)
    # Vocabulary sees character identities only; no validation sequences enter training.
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    boundary = int(len(tokens) * 0.9)
    splits = {"train": tokens[:boundary], "val": tokens[boundary:]}
    for name, split in splits.items():
        if len(split) < context_length + 1:
            raise ValueError(f"{name} needs at least {context_length + 1} characters; got {len(split)}")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return tokenizer, splits, digest


def batch(tokens, batch_size, context_length, device, generator=None):
    starts = torch.randint(len(tokens) - context_length, (batch_size,), generator=generator)
    x = torch.stack([tokens[i:i + context_length] for i in starts])
    # Each target is the character immediately AFTER its corresponding input.
    y = torch.stack([tokens[i + 1:i + context_length + 1] for i in starts])
    return x.to(device), y.to(device)
