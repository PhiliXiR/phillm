# Adding The Time Machine to PhilLM

The original Tiny GPT source and observations were pushed first to https://github.com/PhiliXiR/phillm on `main`, commit `b3b6536`. This addition preserves the original downloaded Alice text, its validation split, and all existing checkpoint files. New data and training artifacts are separate.

## Preparation and vocabulary compatibility

Source: H. G. Wells, *The Time Machine* (1895), [Project Gutenberg ebook 35](https://www.gutenberg.org/ebooks/35), plain text https://www.gutenberg.org/ebooks/35.txt.utf-8. Its catalog identifies this edition as public domain in the USA. The downloaded source is retained in `data/alice-time-machine/time-machine-source.txt`; the ebook wrapper outside the Gutenberg boundary markers is excluded from the training body. UTF-8 BOM and CRLF/newlines are normalized. The raw source SHA-256 is `2892e919000e17c83e1dac51b30f4675db50536b644d7579fe8a89bb399a9bdc`.

The original checkpoint has exactly 75 character IDs. We load its ordered character list rather than creating a new tokenizer. The model architecture and 829,003 parameter count stay unchanged. Five unsupported characters in the new book required explicit text replacements:

| Original | Code point | Replacement | Training occurrences | Validation occurrences |
| --- | --- | --- | --- | --- |
| æ | U+00E6 | ae | 3 | 2 |
| ü | U+00FC | u | 1 | 0 |
| … | U+2026 | ... | 4 | 1 |
| œ | U+0153 | oe | 3 | 0 |
| ç | U+00E7 | c | 1 | 0 |

These substitutions lose some typography/diacritics but retain readable approximations without changing token IDs. No other character substitution is applied. A future unrecognized character stops preparation rather than being dropped or silently creating a vocabulary. The entire body has 179,699 characters before these substitutions; normalization adds 18 characters.

Each book is split contiguously at 90% **before** normalization or training-window sampling. Alice is read from its existing file, checked against the original checkpoint's corpus digest, then split at exactly its original character boundary. The new book is split at raw-body character 161,729; expanding unsupported characters afterward cannot change which text belongs to validation.

| Book | Training characters | Validation characters |
| --- | --- | --- |
| Alice | 130,140 | 14,460 |
| The Time Machine | 161,743 | 17,974 |

Training samples eight independent 128-character windows from each book per batch of 16. Every window and its shifted targets stay inside that book's training portion. The books are not concatenated. Alice's exact training and validation token sequences were compared to the original loader and matched. The manifest saves the ordered vocabulary, split lengths/hashes, normalization counts, source/license, and originating checkpoint digest. Runtime loading verifies both the mapping and every split hash.

## Continuation and optimizer behavior

The old `tiny_gpt.train --resume` is deliberately strict: it requires the same corpus hash. It restores AdamW's optimizer moments and step counters, CPU/CUDA RNG states, and saved settings; `--steps` is the total target step, not additional updates. It remains suitable for the original single-book experiment.

The new `tiny_gpt.continue_train --from-checkpoint` is an explicit change of dataset. It requires a manifest prepared against that exact Alice checkpoint, restores its weights **and AdamW state**, preserves step 1,000 as the starting count, and starts a separate mixture loss history. `--steps 3000` means 2,000 further updates. The new checkpoint embeds the mixture provenance and original loss history. To resume the new dataset, use this module's `--resume`; it checks that the manifest is unchanged. Both modes reject overwriting an unrelated output checkpoint, and initial continuation cannot write into the original checkpoint directory.

Actual long-run configuration: context 128, embedding width 128, four heads, four layers, batch 16 (8+8), learning rate 0.0003, AdamW betas (0.9, 0.999), epsilon 1e-8, weight decay 0.01, gradient clipping at norm 1.0, seed 1337, four CPU threads, float32, CUDA. Target total step 3,000 from 1,000; sampled per-book validation every 200 steps with 20 batches. Existing RNG states are restored; the saved seed governs fixed validation windows. Local sample generators do not alter the training sampler's RNG.

## Exact commands

Run these in WSL from the project directory. The existing virtual environment and original checkpoint are used; no new dependencies or system changes are needed.

```bash
cd /home/phil/dev/phillm/tiny-gpt
source .venv/bin/activate

# Download and prepare a NEW dataset directory (refuses an existing manifest)
python -m tiny_gpt.prepare_mix \
  --checkpoint runs/alice-1000/latest.pt \
  --alice data/alice.txt \
  --out data/alice-time-machine

# Offline preparation from the source already downloaded in this session
# Use another --out if the prepared manifest already exists.
python -m tiny_gpt.prepare_mix \
  --source-file data/alice-time-machine/time-machine-source.txt \
  --out data/alice-time-machine-reprepared

python -m unittest discover -s tests -v

# GPU smoke test from the preserved original checkpoint
python -m tiny_gpt.continue_train \
  --from-checkpoint runs/alice-1000/latest.pt \
  --manifest data/alice-time-machine/manifest.json \
  --out runs/alice-time-machine-smoke \
  --device cuda --steps 1010 --eval-interval 5 --eval-batches 4

# Actual longer run: 2,000 additional updates
python -m tiny_gpt.continue_train \
  --from-checkpoint runs/alice-1000/latest.pt \
  --manifest data/alice-time-machine/manifest.json \
  --out runs/alice-time-machine-3000 \
  --device cuda --steps 3000 --eval-interval 200 --eval-batches 20

# Resume the mixed dataset; this example would add another 1,000 updates
python -m tiny_gpt.continue_train \
  --resume runs/alice-time-machine-3000/latest.pt \
  --manifest data/alice-time-machine/manifest.json \
  --out runs/alice-time-machine-3000 --device cuda --steps 4000

# Full per-book validation and nine identical prompt/seed pairs
python -m tiny_gpt.compare \
  --original runs/alice-1000/latest.pt \
  --updated runs/alice-time-machine-3000/latest.pt \
  --manifest data/alice-time-machine/manifest.json \
  --out reports/alice-time-machine-comparison \
  --device cuda --tokens 200 --temperature 0.8 --top-k 20

# Individual generation from either checkpoint with identical settings
python -m tiny_gpt.generate runs/alice-1000/latest.pt \
  --device cuda --prompt 'The Time Traveller ' --tokens 200 --temperature 0.8 --top-k 20 --seed 42
python -m tiny_gpt.generate runs/alice-time-machine-3000/latest.pt \
  --device cuda --prompt 'The Time Traveller ' --tokens 200 --temperature 0.8 --top-k 20 --seed 42
```

Training and comparison commands preserve existing checkpoints/reports: use fresh output directories when repeating an experiment. The listed actual run directories now contain their completed results. Downloaded editions can change upstream; retain the recorded raw source for exact preparation. The `--steps 4000` resume example is documentation, not a claim that it was run.

## What changes we are studying

The character vocabulary cannot acquire new IDs in this experiment; what changes are the weights' learned probabilities for existing characters and their combinations. Additional training may teach spelling, recurring phrases, and book-specific patterns. Balanced book sampling limits the chance that Alice is overwhelmed by the larger book, but cannot guarantee that earlier phrasing or behavior survives. The sample seeds are 42, 1337, and 2026, each with prompts newline, `Alice `, and `The Time Traveller `. Both models use the same CUDA device, temperature 0.8, top-k 20, and 200 new characters.

The final comparison scores every validation target (except each holdout's first character) once, using contiguous context windows up to 128 characters and token-weighted loss. The context resets at window boundaries for both models. This is separate from periodically sampled training-time validation and should not be directly equated with the first project's sampled metric. Nine matched output pairs are reported without selecting only favorable samples. This approximately 829,000-parameter model is an educational text generator; coherent storytelling, conversational answers, factual knowledge, or reasoning are not guaranteed.

## Actual measured results and verification

The seven unittest checks passed. New checks cover equal per-book sampling without validation tokens, explicit character normalization and Gutenberg wrapper stripping, rejection of reordered vocabulary, and rejection of changed split files. Existing round-trip, causality, gradients and reload checks still pass. Python compilation and `pip check` succeeded.

The original data was downloaded separately and inspected first; actual preparation used:

```bash
python -m tiny_gpt.prepare_mix --source-file data/alice-time-machine/time-machine-source.txt
```

The ten-step GPU smoke run restored all optimizer state. Time Machine's sampled validation loss fell 2.0957 → 2.0253, while Alice rose 1.8452 → 1.8712. These small early changes illustrate that mixing does not guarantee immediate preservation. Its four evaluation batches differ from the longer run's twenty, so their absolute numbers are not directly equivalent.

The longer run completed the documented 1,000 → 3,000 step command, adding 2,000 updates and 4,096,000 sampled input tokens (2,048,000 per book, with overlapping windows). All batch losses and gradient norms remained finite.

| Total step | Sampled Alice validation loss | Sampled Time Machine validation loss |
| --- | --- | --- |
| 1000 | 1.8248 | 2.1157 |
| 1200 | 1.7586 | 1.9291 |
| 1400 | 1.6938 | 1.8361 |
| 1600 | 1.6514 | 1.7812 |
| 1800 | 1.5943 | 1.7380 |
| 2000 | 1.5624 | 1.7003 |
| 2200 | 1.5285 | 1.6677 |
| 2400 | 1.5014 | 1.6455 |
| 2600 | 1.4925 | 1.6320 |
| 2800 | 1.4801 | 1.6139 |
| 3000 | 1.4617 | 1.6056 |

The measured longer-run interval was **11.84 seconds**, including periodic validation, checkpoint writes, and the final sample; excluding process startup, initial validation and initial sample. PyTorch peak allocated memory was **116.77 MiB**, reserved memory 132 MiB. These counters exclude Windows desktop and driver/context overhead. No out-of-memory event occurred and batch size stayed 16. This is an observation, not a promise about other runs.

The completed full-holdout comparison used the documented `tiny_gpt.compare` command. It scored 14,459 Alice targets and 17,973 Time Machine targets for each model:

| Book | Original loss at step 1,000 | Updated loss at step 3,000 |
| --- | --- | --- |
| alice | 1.8485 | 1.4871 |
| time_machine | 2.1184 | 1.6093 |

Both books improved by this metric. This experiment has no Alice-only continuation control, so it does not isolate the effect of mixing from simply doing more training, and it does not establish preservation on other texts or prompts. [The complete matched samples and evaluation details](alice-time-machine-comparison/comparison.md) include every seed/prompt pair. The original newline sample at seed 42 is mostly whitespace; it is retained in the comparison. Updated samples contain more recognizable phrasing, alongside invented words and broken grammar.

A separate resume test actually ran:

```bash
python -m tiny_gpt.continue_train \
  --resume runs/alice-time-machine-3000/latest.pt \
  --manifest data/alice-time-machine/manifest.json \
  --out runs/alice-time-machine-resume-check \
  --device cuda --steps 3020 --eval-interval 20 --eval-batches 20
```

It completed 20 further updates, with sampled validation losses Alice 1.4617 → 1.4656 and Time Machine 1.6056 → 1.5978. Every AdamW parameter state's step counter reached 3,020; CPU/CUDA RNG states are saved. The 3,000-step comparison checkpoint remains separate and unchanged.

Both documented individual generation commands were run, and their full outputs exactly matched the corresponding comparison examples. The original and updated ordered vocabulary lists and model configurations were checked for equality. All **31** original data/run artifacts still match their initial byte hashes, recorded in `alice-preservation-hashes.json` and checked in `alice-preservation-verification.json`.

New checkpoints/data remain git-ignored under `runs/alice-time-machine-*` and `data/alice-time-machine/`; the source, tests, commands, full comparison JSON/Markdown, and this report are suitable for GitHub. No dependency, display driver, pretrained weight, or system-wide setting was added.
