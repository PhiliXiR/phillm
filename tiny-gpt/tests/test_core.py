"""Behavior checks for tokenization, future leakage, gradients, and reload."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch

from tiny_gpt.data import CharacterTokenizer, batch, load_data
from tiny_gpt.generate import load_checkpoint, sample
from tiny_gpt.model import ModelConfig, TinyGPT


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(123)
        self.tokenizer = CharacterTokenizer.from_text("abc\n")
        self.model = TinyGPT(ModelConfig(4, context_length=8, embedding_width=16, heads=4, layers=2))

    def test_tokenizer_round_trip_and_save(self):
        text = "cab\nabc"
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vocab.json"
            self.tokenizer.save(path)
            self.assertEqual(CharacterTokenizer.load(path).encode(text), self.tokenizer.encode(text))
        with self.assertRaises(ValueError):
            self.tokenizer.encode("z")

    def test_shapes_and_no_future_leakage(self):
        tokens = torch.randint(4, (2, 8))
        logits, loss = self.model(tokens, tokens)
        self.assertEqual(logits.shape, (2, 8, 4))
        self.assertEqual(loss.shape, torch.Size([]))
        changed = tokens.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % 4
        changed_logits, _ = self.model(changed)
        torch.testing.assert_close(logits[:, :4], changed_logits[:, :4], rtol=0, atol=0)
        self.assertFalse(torch.allclose(logits[:, 4:], changed_logits[:, 4:]))
        mask = self.model.blocks[0].attention.mask
        self.assertFalse(mask.triu(diagonal=1).any())
        self.assertTrue(mask.diag().all())

    def test_finite_gradients_and_checkpoint_generation(self):
        tokens = torch.randint(4, (2, 8))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            _, loss = self.model(tokens, tokens.roll(-1, dims=1))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for parameter in self.model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            optimizer.step()
        expected = sample(self.model, self.tokenizer, tokens=16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.pt"
            torch.save({"model": self.model.state_dict(), "model_config": asdict(self.model.config),
                        "tokenizer": self.tokenizer.characters, "optimizer": optimizer.state_dict(), "step": 3}, path)
            restored, tokenizer, state = load_checkpoint(path, "cpu")
            torch.testing.assert_close(self.model(tokens)[0], restored(tokens)[0], rtol=0, atol=0)
            self.assertEqual(sample(restored, tokenizer, tokens=16), expected)
            self.assertEqual(state["step"], 3)
            other_optimizer = torch.optim.AdamW(restored.parameters())
            other_optimizer.load_state_dict(state["optimizer"])
            self.assertEqual(len(other_optimizer.state), len(optimizer.state))

    def test_contiguous_splits_shift_and_short_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "text.txt"
            text = "a" * 90 + "b" * 10
            path.write_text(text)
            tokenizer, splits, _ = load_data(path, 8)
            self.assertTrue((splits["train"] == tokenizer.to_id["a"]).all())
            self.assertTrue((splits["val"] == tokenizer.to_id["b"]).all())
            sequence = torch.arange(20)
            x, y = batch(sequence, 4, 8, "cpu")
            torch.testing.assert_close(y, x + 1)
            path.write_text("abc")
            with self.assertRaises(ValueError):
                load_data(path, 8)


if __name__ == "__main__":
    unittest.main()
