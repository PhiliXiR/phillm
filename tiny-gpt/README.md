# Tiny GPT: learn by training from scratch

A character-level decoder-only transformer, initialized with random weights. This is an educational next-character predictor. It has no pretrained weights, retrieval, agents, web UI, or fine-tuning.

## Setup in WSL

Commands below run in Ubuntu WSL, from this project directory. The existing environment is ready; the creation/install commands also document how to recreate it. Python 3.12 is supported. The installed wheel is PyTorch 2.11.0+cu128, selected from the official CUDA 12.8 index; the detected Windows driver supports CUDA 13.4, which is newer than this wheel's runtime. You do not need a separately installed CUDA toolkit to run this wheel.

```bash
cd /home/phil/dev/phillm/tiny-gpt
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")'
python -m tiny_gpt.prepare_data
python -m unittest discover -s tests -v
```

PyTorch may print a missing NumPy warning at import. This project does not use NumPy; no extra dependency is required.

In a restricted agent sandbox, `nvidia-smi` and CUDA can fail even though normal WSL has GPU access. This session verified GPU access outside the sandbox. If your normal WSL terminal also fails, check `nvidia-smi` and the Windows NVIDIA driver first. Updating WSL with `wsl.exe --update` and restarting it with `wsl.exe --shutdown` are possible troubleshooting steps from Windows PowerShell; shutdown stops other WSL work too. **Do not install Linux NVIDIA display drivers in WSL.** System-wide changes are not performed by this project. Use CPU mode while investigating.

## Run, resume, generate

Use a fresh `--out` for each new experiment. Training refuses to silently overwrite an existing checkpoint. `--steps` means the total target step count, including steps already completed when resuming.

```bash
# Small CPU smoke run (works without GPU access)
python -m tiny_gpt.train --device cpu --steps 3 --eval-interval 3 --eval-batches 2 --out runs/my-cpu-smoke

# GPU smoke run before longer training
python -m tiny_gpt.train --device cuda --steps 10 --eval-interval 5 --eval-batches 4 --out runs/my-gpu-smoke

# Longer run with the conservative defaults
python -m tiny_gpt.train --device cuda --steps 1000 --out runs/my-alice

# Resume 1,000 completed steps and train another 1,000
python -m tiny_gpt.train --device cuda --resume runs/my-alice/latest.pt --steps 2000 --out runs/my-alice

# Generate from the measured checkpoint supplied in this workspace
python -m tiny_gpt.generate runs/alice-1000/latest.pt --device cuda --prompt 'Alice ' --tokens 300 --temperature 0.8 --top-k 20 --seed 2026

# CPU generation; 0 disables top-k sampling
python -m tiny_gpt.generate runs/alice-1000/latest.pt --device cpu --tokens 300 --temperature 1.0 --top-k 0

# Train a new model on your own UTF-8 text
python -m tiny_gpt.train --data /path/to/my-text.txt --steps 1000 --out runs/my-text
```

`--batch-size`, `--learning-rate`, `--eval-interval`, `--eval-batches`, `--threads`, and `--device` override the JSON defaults. Edit `config.json` (or pass `--config another.json`) for architecture changes. Resume uses the checkpoint's architecture and saved training settings, then applies explicit CLI overrides. The corpus hash must match on resume; start a new run when changing the text. Unknown prompt characters give a clear error because this tokenizer has no unknown-character token.

Defaults: context 128, width 128, 4 heads, 4 layers, batch 16, AdamW learning rate 0.0003, seed 1337. Float32 throughout, no compilation or mixed precision. If CUDA runs out of memory, lower batch size first: add `--batch-size 8`, then 4 if needed. Keep the architecture unchanged when resuming. A fixed seed makes same-device experiments repeatable, but bitwise equality across CPU/GPU or different library/hardware versions is not promised.

## What the pieces do

- `tiny_gpt/data.py`: saves a sorted character vocabulary and converts text to/from integer IDs. It splits the text at 90% into contiguous training and validation portions **before** sampling windows. Vocabulary includes the entire corpus's character identities, but training never receives validation sequences. Each portion needs at least 129 characters at the default context size (roughly 1,290 total).
- `tiny_gpt/model.py`: token embeddings turn IDs into learned vectors; position embeddings identify each slot in the window. Four transformer blocks mix past information using four attention heads, then transform it with a feed-forward network. Residual additions carry information and gradients around each sublayer. Layer normalization keeps vector scales manageable. The output head produces 75 scores per position for this corpus. There are **829,003 trainable parameters**.
- `tiny_gpt/train.py`: samples input windows `x` and targets `y` shifted forward one character. For `Alice`, an input starting `Alic` has targets `lice`. Cross-entropy penalizes assigning low probability to the actual next character. `loss.backward()` computes how each weight contributed to that error; AdamW updates weights in a direction that reduces it. Gradient clipping limits unusually large updates. Nonfinite loss or gradient norms stop training.
- `tiny_gpt/generate.py`: loads a checkpoint and predicts one next character, samples it, appends it, then repeats. Only the last 128 characters are visible. Temperature below 1 sharpens the distribution; above 1 makes it flatter. Top-k restricts candidates to the k highest scores. Generation uses evaluation mode and no gradients.
- `tiny_gpt/prepare_data.py`: downloads and records the public-domain corpus. No private documents are accessed.

Within attention, queries ask what information a position needs, keys describe what other positions offer, and values supply that information. Query/key dot products become attention weights. A lower-triangular causal mask allows each position to see itself and earlier inputs. Future scores become negative infinity before softmax, so future weights are zero. Without this mask, training could cheat by seeing the answer it is supposed to predict. The test changes future characters and verifies earlier logits are exactly unchanged.

## Outputs and checks

Each run writes `latest.pt`, `tokenizer.json`, `config.json`, `losses.json`, `summary.json`, `before.txt`, and `after.txt`. Checkpoints contain model weights, architecture, tokenizer, optimizer state, completed step, training settings, corpus hash, history, and CPU/CUDA random generator states. They are saved at validation intervals and the final step using an atomic replacement. A stopped run can resume from its last saved validation step; unsaved steps are lost. Load checkpoints you created yourself.

Validation runs in evaluation mode with no gradients, using fixed sampled validation windows. It measures next-character loss on text excluded from training. Samples use the same newline prompt, 300 characters, temperature 0.8, top-k 20, and seed 2026 before and after training. Sampling and validation do not consume the training sampler's RNG. The model has no dropout, but the evaluation mode switches are explicit for learning and future changes.

Tests cover tokenizer round-trip/save/reload, shifted batches and contiguous splits, rejection of insufficient data, tensor shapes, causal masking, finite loss and gradients, model reload, optimizer reload, and repeatable generation. CPU and GPU smoke runs, modest training, and actual checkpoint resume/generation are recorded in `reports/observations.md`.

Plausible-looking text is **not proof of factual knowledge or reasoning**. This small model learns local spelling and text patterns, can memorize passages, and will invent words and sentences. Loss improvement alone does not establish useful conversation ability.

## Corpus and teaching attribution

Corpus: Lewis Carroll, *Alice's Adventures in Wonderland* (1865), [Project Gutenberg ebook 11](https://www.gutenberg.org/ebooks/11), downloaded from https://www.gutenberg.org/ebooks/11.txt.utf-8. The catalog states public domain in the USA; the underlying text is also public domain in Canada. Check your jurisdiction when redistributing. The downloader excludes Gutenberg's header and footer and normalizes newlines; it saves provenance in `data/source.json`. The first 90% trains and the final 10% validates, so this is a single-book holdout, not a broad language benchmark.

Teaching reference: Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html), especially the language modeling/GPT lessons. The implementation here is written independently; no reference code or pretrained weights were copied. Start by inspecting the shifted targets, then the mask, then the loss/backward/update sequence.

Installation references: [PyTorch Start Locally](https://pytorch.org/get-started/locally/) and [NVIDIA CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html). The PyTorch page's rendered selector showed older release text during inspection; the official cu128 package index supplied the actual installed 2.11.0 build. Only `torch` is a direct dependency; CUDA libraries installed by its wheel are managed inside the virtual environment.
