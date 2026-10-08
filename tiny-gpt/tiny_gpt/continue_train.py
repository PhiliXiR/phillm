"""Explicitly continue an Alice checkpoint on a verified two-book mixture."""
import argparse
import json
from pathlib import Path
import time

import torch

from .generate import choose_device, load_checkpoint, sample
from .mixed_data import load_mixture, mixed_batch, sha256_bytes, validation_losses
from .train import save_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--from-checkpoint', help='Start new mixed-corpus continuation from original Alice checkpoint')
    source.add_argument('--resume', help='Resume an existing mixed-corpus continuation checkpoint')
    parser.add_argument('--manifest', default='data/alice-time-machine/manifest.json')
    parser.add_argument('--out', required=True)
    parser.add_argument('--steps', type=int, required=True, help='Total target step, not additional steps')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--eval-interval', type=int)
    parser.add_argument('--eval-batches', type=int)
    parser.add_argument('--threads', type=int)
    args = parser.parse_args()
    source_path = Path(args.resume or args.from_checkpoint)
    out = Path(args.out)
    if not args.resume and out.resolve() == source_path.parent.resolve():
        raise ValueError('Continuation must use a separate output directory from the source checkpoint')
    if (out / 'latest.pt').exists() and (not args.resume or (out / 'latest.pt').resolve() != source_path.resolve()):
        raise ValueError('Output checkpoint exists; use a new --out or resume its own latest.pt')
    source_state = torch.load(source_path, map_location='cpu', weights_only=True)
    settings = source_state['training_config'].copy()
    for key in ('steps', 'device', 'batch_size', 'learning_rate', 'eval_interval', 'eval_batches', 'threads'):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    for key in ('steps', 'batch_size', 'eval_interval', 'eval_batches', 'threads'):
        if settings[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if settings['batch_size'] < 2 or settings['batch_size'] % 2 or settings['learning_rate'] <= 0:
        raise ValueError('Use positive learning rate and an even batch size >= 2')
    torch.set_num_threads(settings['threads'])
    torch.manual_seed(settings['seed'])
    device = choose_device(settings['device'])
    model, tokenizer, checkpoint = load_checkpoint(source_path, device)
    books, manifest, digest = load_mixture(args.manifest, tokenizer, model.config.context_length)
    start_step = checkpoint['step']
    if settings['steps'] <= start_step:
        raise ValueError('--steps must exceed the saved total step count')
    if args.resume:
        if checkpoint.get('mixture', {}).get('manifest_sha256') != digest or checkpoint['data_sha256'] != digest:
            raise ValueError('Resume requires the exact saved mixture manifest')
        metadata = checkpoint['mixture']
        history = checkpoint['history'].copy()
    else:
        if 'mixture' in checkpoint:
            raise ValueError('Use --resume for an already mixed-corpus checkpoint')
        if sha256_bytes(source_path.read_bytes()) != manifest['origin_checkpoint_sha256']:
            raise ValueError('Prepared mixture belongs to a different source checkpoint')
        if checkpoint['data_sha256'] != manifest['alice_data_sha256']:
            raise ValueError('Original Alice dataset digest differs')
        metadata = {'manifest_sha256': digest, 'origin_checkpoint_sha256': manifest['origin_checkpoint_sha256'],
                    'origin_step': start_step, 'sampling': '50% Alice / 50% Time Machine windows',
                    'manifest': manifest, 'original_history': checkpoint['history']}
        history = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for group in optimizer.param_groups:
        group['lr'] = settings['learning_rate']
    torch.set_rng_state(checkpoint['cpu_rng'])
    if device == 'cuda' and checkpoint['cuda_rng']:
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        torch.cuda.reset_peak_memory_stats()
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out / 'tokenizer.json')
    (out / 'config.json').write_text(json.dumps(settings, indent=2))
    (out / 'dataset-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'device={device} parameters={sum(p.numel() for p in model.parameters()):,} '
          f'start_step={start_step} target_step={settings["steps"]} optimizer_restored=True', flush=True)
    if device == 'cuda':
        print(f'GPU={torch.cuda.get_device_name(0)}', flush=True)
    def evaluate():
        return validation_losses(model, books, settings['batch_size'], settings['eval_batches'], settings['seed'] + 1)
    initial = evaluate()
    history.append({'step': start_step, 'validation': initial})
    print(f'step={start_step} validation={initial}', flush=True)
    (out / ('resume_before.txt' if args.resume else 'before.txt')).write_text(sample(model, tokenizer), encoding='utf-8')
    started = time.perf_counter()
    model.train()
    for step in range(start_step + 1, settings['steps'] + 1):
        x, y = mixed_batch(books, settings['batch_size'], model.config.context_length, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % settings['eval_interval'] == 0 or step == settings['steps']:
            metrics = evaluate()
            history.append({'step': step, 'validation': metrics, 'training_batch_loss': loss.item()})
            print(f'step={step} validation={metrics} batch_loss={loss.item():.4f} '
                  f'grad_norm={norm.item():.4f} elapsed={time.perf_counter()-started:.1f}s', flush=True)
            (out / 'losses.json').write_text(json.dumps(history, indent=2))
            save_checkpoint(out / 'latest.pt', model, optimizer, tokenizer, settings, step, digest, history, mixture=metadata)
    (out / 'after.txt').write_text(sample(model, tokenizer), encoding='utf-8')
    summary = {'source_checkpoint': str(source_path), 'source_step': start_step, 'final_step': step,
               'additional_steps': step - start_step, 'optimizer_restored': True, 'config': settings,
               'parameters': sum(p.numel() for p in model.parameters()), 'device': device,
               'seconds_including_periodic_validation_saves_final_sample': time.perf_counter() - started,
               'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated() / 2**20 if device == 'cuda' else None,
               'peak_cuda_reserved_mib': torch.cuda.max_memory_reserved() / 2**20 if device == 'cuda' else None,
               'initial_validation': initial, 'final_validation': metrics, 'torch': str(torch.__version__)}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
