"""Mixture-specific behavior: explicit normalization and no validation leakage."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from tiny_gpt.data import CharacterTokenizer
from tiny_gpt.mixed_data import load_mixture, mixed_batch, sha256_bytes
from tiny_gpt.prepare_mix import normalize, strip_wrapper


class MixtureTests(unittest.TestCase):
    def test_equal_mix_excludes_validation_and_preserves_targets(self):
        books = {'alice': {'train': torch.ones(30, dtype=torch.long), 'val': torch.full((30,), 9)},
                 'time_machine': {'train': torch.full((30,), 2), 'val': torch.full((30,), 8)}}
        x, y = mixed_batch(books, 16, 8, 'cpu', torch.Generator().manual_seed(42))
        self.assertEqual(x.shape, (16, 8))
        self.assertTrue((x[:8] == 1).all())
        self.assertTrue((x[8:] == 2).all())
        torch.testing.assert_close(y, x)
        with self.assertRaises(ValueError):
            mixed_batch(books, 3, 8, 'cpu')

    def test_explicit_normalization_and_wrapper(self):
        tokenizer = CharacterTokenizer.from_text('aeu.oc\n')
        text, counts = normalize('æü…œç', tokenizer)
        self.assertEqual(text, 'aeu...oec')
        self.assertEqual(counts['…']['replacement'], '...')
        with self.assertRaises(ValueError):
            normalize('é', tokenizer)
        raw = b'header\r\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\r\n\r\nBody\r\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\r\nfooter'
        self.assertEqual(strip_wrapper(raw), 'Body\n')
        with self.assertRaises(ValueError):
            strip_wrapper(b'No markers')

    def test_manifest_rejects_reordered_vocabulary_and_tampering(self):
        tokenizer = CharacterTokenizer(['a', 'b'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            books = {}
            for name, char in [('alice', 'a'), ('time_machine', 'b')]:
                books[name] = {}
                for split in ('train', 'val'):
                    text = char * 20
                    filename = f'{name}-{split}.txt'
                    (root / filename).write_text(text)
                    books[name][split] = {'file': filename, 'sha256': sha256_bytes(text.encode())}
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'tokenizer': tokenizer.characters, 'books': books}))
            loaded, _, _ = load_mixture(manifest, tokenizer, 8)
            self.assertEqual(len(loaded['alice']['val']), 20)
            with self.assertRaises(ValueError):
                load_mixture(manifest, CharacterTokenizer(['b', 'a']), 8)
            (root / 'alice-val.txt').write_text('b' * 20)
            with self.assertRaises(ValueError):
                load_mixture(manifest, tokenizer, 8)


if __name__ == '__main__':
    unittest.main()
