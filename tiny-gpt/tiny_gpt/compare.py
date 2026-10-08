"""Compare two saved models on identical per-book holdouts and seeded prompts."""
import argparse
import json
from pathlib import Path

import torch

from .generate import choose_device, load_checkpoint, sample
from .mixed_data import full_validation_losses, load_mixture, sha256_bytes

PROMPTS = ['\n', 'Alice ', 'The Time Traveller ']
SEEDS = [42, 1337, 2026]


def display_sample(text):
    return '\n'.join(line.rstrip() for line in text.split('\n'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', default='runs/alice-1000/latest.pt')
    parser.add_argument('--updated', required=True)
    parser.add_argument('--manifest', default='data/alice-time-machine/manifest.json')
    parser.add_argument('--out', required=True, help='New directory for JSON results and Markdown comparison')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=20, help='0 disables top-k')
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise ValueError('Use a new comparison output directory to preserve previous reports')
    torch.set_num_threads(4)
    device = choose_device(args.device)
    original, tokenizer, old = load_checkpoint(args.original, device)
    updated, new_tokenizer, new = load_checkpoint(args.updated, device)
    if tokenizer.characters != new_tokenizer.characters or old['model_config'] != new['model_config']:
        raise ValueError('Vocabulary IDs and architecture must match for this experiment')
    books, manifest, digest = load_mixture(args.manifest, tokenizer, original.config.context_length)
    if sha256_bytes(Path(args.original).read_bytes()) != manifest['origin_checkpoint_sha256']:
        raise ValueError('Original checkpoint differs from dataset provenance')
    if new.get('mixture', {}).get('manifest_sha256') != digest:
        raise ValueError('Updated checkpoint was trained with a different manifest')
    losses = {'original': full_validation_losses(original, books), 'updated': full_validation_losses(updated, books)}
    settings = {'temperature': args.temperature, 'top_k': args.top_k or None, 'tokens': args.tokens,
                'prompts': PROMPTS, 'seeds': SEEDS, 'device': device}
    examples = []
    for prompt in PROMPTS:
        for seed in SEEDS:
            record = {'prompt': prompt, 'seed': seed}
            for label, model in [('original', original), ('updated', updated)]:
                record[label] = sample(model, tokenizer, prompt, args.tokens, args.temperature, args.top_k or None, seed)
            examples.append(record)
    result = {'original': {'path': args.original, 'step': old['step'], 'sha256': sha256_bytes(Path(args.original).read_bytes())},
              'updated': {'path': args.updated, 'step': new['step'], 'sha256': sha256_bytes(Path(args.updated).read_bytes())},
              'manifest_sha256': digest, 'validation': losses, 'sampling': settings, 'examples': examples}
    out.mkdir(parents=True)
    (out / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = []
    for name in books:
        before = losses['original'][name]['loss']
        after = losses['updated'][name]['loss']
        rows.append(f'| {name} | {losses["original"][name]["targets"]:,} | {before:.4f} | {after:.4f} | {after-before:+.4f} |')
    report = ['# Alice + The Time Machine: original versus updated\n',
              f'Original step {old["step"]}; updated step {new["step"]}. Exact vocabulary and architecture match.\n',
              'Every holdout target except the first character is scored exactly once. Windows are contiguous, non-overlapping, at most 128 inputs; the context resets at each window. Loss is token-weighted cross-entropy in natural-log units. These full-holdout losses differ from the earlier randomly sampled validation metric. Both models use identical windows.\n',
              '| Book | Validation targets | Original loss | Updated loss | Change |',
              '| --- | --- | --- | --- | --- |', *rows,
              '\n## Identical sampling settings\n',
              f'Both models: device {device}, temperature {args.temperature}, top-k {args.top_k or "disabled"}, {args.tokens} new characters, seeds {SEEDS}. Prompt text is included in each output. Evaluation mode, no gradients.\n',
              'Markdown removes trailing spaces from sample lines; comparison.json preserves exact output. The nine matched prompt/seed pairs show variations, rather than selecting a single flattering output. Better holdout loss is evidence of better next-character prediction on these books, not coherent storytelling, factual knowledge, or reasoning. Mixing reduces the risk of forgetting; it does not guarantee preservation of prior outputs.\n']
    for example in examples:
        report += [f'## Prompt {json.dumps(example["prompt"])}; seed {example["seed"]}\n',
                   'Original:\n', '```text', display_sample(example['original']), '```\n',
                   'Updated:\n', '```text', display_sample(example['updated']), '```\n']
    (out / 'comparison.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps({'validation': losses, 'sampling': settings, 'output': str(out)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
