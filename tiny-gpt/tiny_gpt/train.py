"""Train from random weights, or resume with optimizer and RNG state."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from .data import batch, load_data
from .generate import choose_device, load_checkpoint, sample
from .model import ModelConfig, TinyGPT


@torch.no_grad()
def evaluate(model, splits, settings, device):
    was_training = model.training
    model.eval()
    # Fixed evaluation windows make comparisons fair and leave training RNG alone.
    rng = torch.Generator().manual_seed(settings["seed"] + 1)
    result = {}
    try:
        for name, tokens in splits.items():
            losses = []
            for _ in range(settings["eval_batches"]):
                x, y = batch(tokens, settings["batch_size"], model.config.context_length, device, rng)
                _, loss = model(x, y)
                losses.append(loss.item())
            result[name] = sum(losses) / len(losses)
    finally:
        model.train(was_training)
    return result


def save_checkpoint(path, model, optimizer, tokenizer, settings, step, digest, history):
    state = {"model": model.state_dict(), "model_config": asdict(model.config),
             "optimizer": optimizer.state_dict(), "tokenizer": tokenizer.characters,
             "step": step, "training_config": settings, "data_sha256": digest,
             "history": history, "cpu_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else []}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--data", default="data/alice.txt")
    parser.add_argument("--out", default="runs/default")
    parser.add_argument("--resume")
    for key in ("steps", "batch-size", "eval-interval", "eval-batches", "threads"):
        parser.add_argument("--" + key, type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text())
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        settings = checkpoint["training_config"].copy()
    for key in ("steps", "batch_size", "eval_interval", "eval_batches", "threads", "learning_rate", "device"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    for key in ("steps", "batch_size", "eval_interval", "eval_batches", "threads"):
        if settings[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if settings["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")
    torch.set_num_threads(settings["threads"])
    torch.manual_seed(settings["seed"])
    device = choose_device(settings["device"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "latest.pt").exists() and not args.resume:
        raise ValueError("Output already contains a checkpoint; choose a new --out or use --resume")
    if checkpoint:
        model, tokenizer, checkpoint = load_checkpoint(args.resume, device)
        tokenizer, splits, digest = load_data(args.data, model.config.context_length, tokenizer)
        if digest != checkpoint["data_sha256"]:
            raise ValueError("Resume corpus differs from checkpoint; start a fresh run for new data")
    else:
        tokenizer, splits, digest = load_data(args.data, settings["context_length"])
        config = ModelConfig(vocab_size=len(tokenizer.characters), **{
            key: settings[key] for key in ("context_length", "embedding_width", "heads", "layers")})
        model = TinyGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"])
    start_step = 0
    history = []
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = settings["learning_rate"]
        start_step = checkpoint["step"]
        history = checkpoint["history"]
        torch.set_rng_state(checkpoint["cpu_rng"])
        if device == "cuda" and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    if settings["steps"] <= start_step:
        raise ValueError("--steps is the total target step count and must exceed the saved step")
    tokenizer.save(out / "tokenizer.json")
    (out / "config.json").write_text(json.dumps(settings, indent=2))
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"device={device} parameters={parameter_count:,} vocabulary={len(tokenizer.characters)} "
          f"train_chars={len(splits['train'])} val_chars={len(splits['val'])}", flush=True)
    if device == "cuda":
        print(f"GPU={torch.cuda.get_device_name(0)}", flush=True)
        torch.cuda.reset_peak_memory_stats()
    # Local sampling generator ensures before/after samples do not alter training.
    (out / ("resume_before.txt" if checkpoint else "before.txt")).write_text(sample(model, tokenizer), encoding="utf-8")
    initial = evaluate(model, splits, settings, device)
    history.append({"step": start_step, **initial})
    print(f"step={start_step} train={initial['train']:.4f} val={initial['val']:.4f}", flush=True)
    started = time.perf_counter()
    model.train()
    for step in range(start_step + 1, settings["steps"] + 1):
        x, y = batch(splits["train"], settings["batch_size"], model.config.context_length, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % settings["eval_interval"] == 0 or step == settings["steps"]:
            metrics = evaluate(model, splits, settings, device)
            history.append({"step": step, **metrics})
            elapsed = time.perf_counter() - started
            print(f"step={step} train={metrics['train']:.4f} val={metrics['val']:.4f} "
                  f"batch_loss={loss.item():.4f} grad_norm={norm.item():.4f} elapsed={elapsed:.1f}s", flush=True)
            (out / "losses.json").write_text(json.dumps(history, indent=2))
            save_checkpoint(out / "latest.pt", model, optimizer, tokenizer, settings, step, digest, history)
    (out / "after.txt").write_text(sample(model, tokenizer), encoding="utf-8")
    elapsed = time.perf_counter() - started
    summary = {"parameters": parameter_count, "device": device, "steps_this_run": settings["steps"] - start_step,
               "seconds_including_validation_and_saving": elapsed,
               "training_tokens_per_second_including_overhead":
                   (settings["steps"] - start_step) * settings["batch_size"] * model.config.context_length / elapsed,
               "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
               "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if device == "cuda" else None,
               "sampling": {"prompt": "\n", "tokens": 300, "temperature": 0.8, "top_k": 20, "seed": 2026},
               "torch": str(torch.__version__), "initial": initial, "final": metrics}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
