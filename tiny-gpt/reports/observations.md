# Actual observations: 2026-10-07

Ubuntu 24.04.3 LTS, WSL2 kernel 6.18.33.2-microsoft-standard-WSL2, Python 3.12.3.
About 934 GiB available disk space before setup. No applicable AGENTS.md was found in the project or its parent directories.
The preexisting `tiny-gpt` directory contained only an environment and empty data folder; it was reused with the user's authorization.

The sandbox blocked initial GPU/NVML access. Outside the sandbox, `nvidia-smi` reported an NVIDIA GeForce RTX 4070 SUPER with 12,282 MiB VRAM, Windows KMD 617.14, and CUDA UMD 13.4.
PyTorch 2.11.0+cu128 (CUDA runtime 12.8) returned `torch.cuda.is_available() == True`, named the GPU correctly, and executed a CUDA tensor operation. No display drivers or system packages were installed.

## Verification

Four unittest checks passed: character round-trip and vocabulary persistence; contiguous splits, shifted targets and too-short data; tensor shapes and future-input perturbation; finite gradients, weights/optimizer reload and identical generation on CPU.
A three-step CPU smoke run had finite loss/gradients and validation loss 4.3212 → 3.8124.
A ten-step GPU smoke run had finite loss/gradient norms and validation loss 4.3239 → 3.5271.
`pip check` found no broken requirements. Python modules compiled successfully.

## Fresh GPU training

Corpus: Lewis Carroll's public-domain *Alice's Adventures in Wonderland*, Project Gutenberg ebook 11.
144,600 characters after newline normalization and wrapper removal; training 130,140, validation 14,460; 75 character tokens.
Configuration: context 128, width 128, four heads, four layers, batch 16, learning rate 0.0003, seed 1337, float32.
Parameter count: **829,003**. Trained from random weights for 1,000 steps, or 2,048,000 sampled input tokens. Windows overlap, so this is not that many unique characters.

| Step | Training loss | Validation loss |
| --- | --- | --- |
| 0 | 4.3190 | 4.3213 |
| 100 | 2.5632 | 2.5478 |
| 200 | 2.4310 | 2.4247 |
| 300 | 2.3664 | 2.3677 |
| 400 | 2.2936 | 2.2937 |
| 500 | 2.2111 | 2.2157 |
| 600 | 2.1196 | 2.1354 |
| 700 | 2.0266 | 2.0497 |
| 800 | 1.9460 | 1.9710 |
| 900 | 1.8680 | 1.8963 |
| 1000 | 1.7991 | 1.8349 |

Loss is next-character cross-entropy in natural-log units, averaged over 20 batches for each split. Evaluation uses the same sampled windows at each interval.
All monitored loss/gradient values remained finite. Training and validation losses both fell; this small run does not establish generalization beyond this book.

Measured interval: **8.28 seconds**, including periodic validation, checkpoint writes and final generation, excluding process/import/startup, initial generation and initial evaluation. Throughput including that overhead: about 247,236 sampled tokens/second.
PyTorch peak allocated GPU memory: **116.77 MiB**; peak reserved: 132 MiB. These allocator counters exclude Windows desktop use and CUDA driver/context overhead. No out-of-memory error occurred; batch size remained 16. Measurements apply to this run rather than predicting future runtimes.

## Same-settings before/after samples

Both samples use newline prompt, 300 new characters, temperature 0.8, top-k 20, sampling seed 2026, evaluation mode, and no gradients.

Before training:

```text

(Z.rr:IO jL?ldb—DP’ùSYe )gV—0E:BY*imrJ‘s“sBIsp“_H-a*c(h!o(ZmASfL“sar:t]LOgaglbr*0]bdad:?Pz3k““US’ISkdntogHprgZ(“ELxi-“‘nt’iOoBp]w*spW*EhrhIgZ
EX—Lv(keA00Ilmùm—mtùOMXmrF?0IhwiOnlpEkuml,AXpIXc0—WmdtmOPteE“0B]gcoBZ“WONpePBE“ok(eWwOeùPOAAi(lnw’eù—]puBviG!gB)aZ;e Ml*e!wpGoEwO(pEpIZnd?Wl h-vwePtAmEXh!G!pt
```

After 1,000 steps:

```text

     she couldn ton the roouse a little thats sionace all imome that
thers begrentinglinf, int formed to look cantere, sthis same she
downd and on ofe they, and the she swintent shume, she dit moom nott
the conterpples of bring a donginttere. Alice had the hould a fort she
paters of the the had sall
```

The initial sample has scattered characters and punctuation. The trained sample has word spacing, common short words, and `Alice`, alongside invented words such as `begrentinglinf` and broken grammar. It has learned some spelling and book patterns. Plausible text is not proof of factual knowledge or reasoning.

Artifacts remain in `runs/alice-1000/`: full samples, JSON loss history, summary, vocabulary, configuration and `latest.pt`. These generated artifacts and the downloaded data/environment are git-ignored. This report preserves the observed results outside the ignored folders.

## Actual checkpoint resume and generation

Loaded `runs/alice-1000/latest.pt` on CUDA and regenerated the full after-training sample using identical settings. It matched `after.txt` exactly (the CLI adds one trailing newline). CPU checkpoint generation with prompt `Alice ` also succeeded.

Resumed the 1,000-step checkpoint into `runs/resume-check/` and completed 20 additional steps. Validation loss moved from 1.8349 to 1.8192. The resulting checkpoint reports step 1,020, and every AdamW parameter state also reports step 1,020. CPU/CUDA RNG states are present and the corpus hash matches. The original 1,000-step checkpoint remains available for reproducing the report's before/after comparison.
