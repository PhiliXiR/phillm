"""Sample one character at a time from a saved checkpoint."""
import argparse

import torch

from .data import CharacterTokenizer
from .model import ModelConfig, TinyGPT


def choose_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; try --device cpu and check nvidia-smi outside the sandbox")
    return requested


@torch.no_grad()
def sample(model, tokenizer, prompt="\n", tokens=300, temperature=0.8, top_k=20, seed=2026):
    if temperature <= 0 or tokens < 0 or (top_k is not None and top_k <= 0):
        raise ValueError("Temperature/top-k must be positive and token count nonnegative")
    if not prompt:
        raise ValueError("Prompt cannot be empty")
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    try:
        for _ in range(tokens):
            logits, _ = model(ids[:, -model.config.context_length:])
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, indices = torch.topk(logits, min(top_k, logits.shape[-1]))
                filtered = torch.full_like(logits, float("-inf"))
                logits = filtered.scatter(-1, indices, values)
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=rng)
            ids = torch.cat((ids, next_id), dim=1)
        return tokenizer.decode(ids[0].tolist())
    finally:
        model.train(was_training)


def load_checkpoint(path, device):
    # weights_only rejects arbitrary pickle objects; load only your own checkpoints.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = TinyGPT(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, CharacterTokenizer(checkpoint["tokenizer"]), checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20, help="0 disables top-k")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, tokenizer, _ = load_checkpoint(args.checkpoint, choose_device(args.device))
    print(sample(model, tokenizer, args.prompt, args.tokens, args.temperature, args.top_k or None, args.seed))


if __name__ == "__main__":
    main()
