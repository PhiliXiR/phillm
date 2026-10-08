"""Prepare separate book splits using the original checkpoint's exact vocabulary."""
import argparse
from collections import Counter
import json
from pathlib import Path
from urllib.request import urlopen

import torch

from .data import CharacterTokenizer
from .mixed_data import sha256_bytes

URL = 'https://www.gutenberg.org/ebooks/35.txt.utf-8'
NORMALIZATION = {'æ': 'ae', 'ü': 'u', '…': '...', 'œ': 'oe', 'ç': 'c'}


def strip_wrapper(raw):
    text = raw.decode('utf-8-sig').replace('\r\n', '\n').replace('\r', '\n')
    start = text.find('*** START OF THE PROJECT GUTENBERG EBOOK')
    end = text.find('*** END OF THE PROJECT GUTENBERG EBOOK')
    if start < 0 or end <= start:
        raise ValueError('Missing or invalid Gutenberg boundary markers')
    return text[text.index('\n', start) + 1:end].strip() + '\n'


def normalize(text, tokenizer):
    unsupported = Counter(c for c in text if c not in tokenizer.to_id)
    unknown = unsupported.keys() - NORMALIZATION.keys()
    if unknown:
        raise ValueError(f'Unplanned unsupported characters: {sorted(unknown)!r}; inspect before extending normalization')
    normalized = ''.join(NORMALIZATION.get(c, c) if c in unsupported else c for c in text)
    tokenizer.encode(normalized)
    return normalized, {c: {'count': n, 'replacement': NORMALIZATION[c], 'codepoint': f'U+{ord(c):04X}'}
                        for c, n in unsupported.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='runs/alice-1000/latest.pt')
    parser.add_argument('--alice', default='data/alice.txt')
    parser.add_argument('--out', default='data/alice-time-machine')
    parser.add_argument('--source-file', help='Use a previously downloaded Gutenberg source instead of downloading')
    args = parser.parse_args()
    out = Path(args.out)
    if (out / 'manifest.json').exists():
        raise ValueError('Prepared dataset already exists; use another --out to preserve it')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    tokenizer = CharacterTokenizer(checkpoint['tokenizer'])
    alice = Path(args.alice).read_text(encoding='utf-8')
    alice_digest = sha256_bytes(alice.encode())
    if alice_digest != checkpoint['data_sha256']:
        raise ValueError('Alice text must exactly match the original checkpoint corpus')
    if args.source_file:
        raw = Path(args.source_file).read_bytes()
    else:
        with urlopen(URL, timeout=60) as response:
            raw = response.read()
    body = strip_wrapper(raw)
    alice_boundary = int(len(alice) * 0.9)
    time_boundary = int(len(body) * 0.9)
    splits = {'alice': {'train': alice[:alice_boundary], 'val': alice[alice_boundary:]},
              'time_machine': {'train': body[:time_boundary], 'val': body[time_boundary:]}}
    counts = {}
    # Split BEFORE normalization so expansions cannot move held-out text into training.
    for split in ('train', 'val'):
        splits['time_machine'][split], counts[split] = normalize(splits['time_machine'][split], tokenizer)
    context = checkpoint['model_config']['context_length']
    for book in splits.values():
        for text in book.values():
            tokenizer.encode(text)
            if len(text) < context + 1:
                raise ValueError('Every book split must exceed the context length')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'time-machine-source.txt').write_bytes(raw)
    (out / 'time-machine-body.txt').write_text(body, encoding='utf-8')
    records = {}
    for name, book in splits.items():
        records[name] = {}
        for split, text in book.items():
            filename = f'{name}-{split}.txt'
            (out / filename).write_text(text, encoding='utf-8')
            records[name][split] = {'file': filename, 'characters': len(text), 'sha256': sha256_bytes(text.encode())}
    manifest = {'version': 1, 'tokenizer': tokenizer.characters, 'books': records,
                'origin_checkpoint_sha256': sha256_bytes(Path(args.checkpoint).read_bytes()),
                'origin_checkpoint_step': checkpoint['step'], 'alice_data_sha256': alice_digest,
                'alice_boundary': alice_boundary, 'time_machine_boundary_before_normalization': time_boundary,
                'time_machine_body_characters': len(body), 'normalization': counts,
                'source': {'title': 'The Time Machine', 'author': 'H. G. Wells', 'url': URL,
                           'catalog': 'https://www.gutenberg.org/ebooks/35',
                           'license': 'Public domain in the USA, per Project Gutenberg catalog',
                           'raw_sha256': sha256_bytes(raw),
                           'processing': 'UTF-8 BOM removal, newline normalization, Gutenberg wrapper removal; explicit unsupported-character mappings after 90/10 split'},
                'mixing': 'Equal numbers of per-book training windows in every even-sized batch; no concatenation'}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
