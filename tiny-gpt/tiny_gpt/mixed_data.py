"""Verified per-book splits; training windows never cross book/split boundaries."""
import hashlib
import json
from pathlib import Path

import torch

from .data import batch


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_mixture(manifest_path, tokenizer, context_length):
    path = Path(manifest_path)
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest['tokenizer'] != tokenizer.characters:
        raise ValueError('Mixture vocabulary differs from checkpoint character IDs')
    books = {}
    for name, book in manifest['books'].items():
        books[name] = {}
        for split in ('train', 'val'):
            record = book[split]
            text = (path.parent / record['file']).read_text(encoding='utf-8')
            if sha256_bytes(text.encode('utf-8')) != record['sha256']:
                raise ValueError(f'{name}/{split} text differs from prepared manifest')
            ids = tokenizer.encode(text)
            if len(ids) < context_length + 1:
                raise ValueError(f'{name}/{split} too short for context {context_length}')
            books[name][split] = torch.tensor(ids, dtype=torch.long)
    if set(books) != {'alice', 'time_machine'}:
        raise ValueError('Expected Alice and Time Machine splits')
    return books, manifest, sha256_bytes(raw)


def mixed_batch(books, batch_size, context_length, device, generator=None):
    if batch_size < 2 or batch_size % 2:
        raise ValueError('Use an even batch size >= 2 for equal per-book mixing')
    pairs = [batch(books[name]['train'], batch_size // 2, context_length, device, generator)
             for name in ('alice', 'time_machine')]
    return torch.cat([p[0] for p in pairs]), torch.cat([p[1] for p in pairs])


@torch.no_grad()
def validation_losses(model, books, batch_size=16, eval_batches=20, seed=1338):
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    result = {}
    try:
        for name, splits in books.items():
            rng = torch.Generator().manual_seed(seed)
            losses = []
            for _ in range(eval_batches):
                x, y = batch(splits['val'], batch_size, model.config.context_length, device, rng)
                losses.append(model(x, y)[1].item())
            result[name] = sum(losses) / len(losses)
    finally:
        model.train(was_training)
    return result


@torch.no_grad()
def full_validation_losses(model, books):
    """Score every holdout target exactly once in contiguous context-sized windows."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    result = {}
    try:
        for name, splits in books.items():
            tokens = splits['val']
            total = 0.0
            count = 0
            for start in range(0, len(tokens) - 1, model.config.context_length):
                length = min(model.config.context_length, len(tokens) - 1 - start)
                x = tokens[start:start + length].unsqueeze(0).to(device)
                y = tokens[start + 1:start + length + 1].unsqueeze(0).to(device)
                loss = model(x, y)[1].item()
                total += loss * length
                count += length
            result[name] = {'loss': total / count, 'targets': count}
    finally:
        model.train(was_training)
    return result
