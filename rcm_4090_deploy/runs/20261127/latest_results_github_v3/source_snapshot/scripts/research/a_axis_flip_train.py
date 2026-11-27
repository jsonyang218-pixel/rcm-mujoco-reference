"""Four fixed 1000-batch epochs, exact 8% replay, CUDA AMP and resumable state."""
from __future__ import annotations

import gc
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from lerobot_training import dit_state_attention as attention
from lerobot_training.dit_experience_replay import (
    IndexSubset, _load_policy, _optimizer, _train_loader, _validation_loader,
)
from lerobot_training.train_diffusion_policy import RCMDataset, image_transform_for_dataset
from lerobot_training.train_dit_rectified_flow import _atomic_torch_save, _loss, _move_batch
from lerobot_training.navigation_overlay import file_sha256
from scripts.research.a_axis_flip_common import (
    DATA, OLD, VAL, apply_abc_overlay, gpu_check, read, record, signature, write_json,
)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(s):
    random.setstate(s['python'])
    np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch'])
    torch.cuda.set_rng_state_all(s['cuda'])


def finite_parameters(model):
    for name, value in model.state_dict().items():
        if torch.is_floating_point(value) and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f'nonfinite parameter/buffer: {name}')


def checked_validation(model, loader, device, args):
    model.eval()
    total, n = 0., 0
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(args.validation_seed)
        with torch.no_grad():
            for n, batch in enumerate(loader, 1):
                batch = _move_batch(batch, device)
                with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float16):
                    loss = _loss(model, batch)
                value = float(loss.detach())
                if not math.isfinite(value):
                    raise FloatingPointError(f'nonfinite validation loss, batch {n}')
                total += value
                if n % 100 == 0:
                    print(f'validation {n}/{len(loader)}, loss={total/n:.8f}', flush=True)
    if n == 0:
        raise ValueError('empty frozen validation loader')
    return total/n


def train(replay, destination, identity):
    gpu_check()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(mode='ratio', epochs=4, steps_per_epoch=1000,
        batch_size=16, val_batch_size=64, num_workers=4, seed=20261118,
        backbone_lr=1e-6, attention_lr=1e-5, weight_decay=1e-4,
        validation_seed=20261119, validation_max_batches=None, device='cuda', amp=True)
    contract = {'branch': replay['branch'], 'replay': replay, 'args': vars(args),
                'initial_checkpoint': str(OLD), 'initial_checkpoint_sha256': file_sha256(OLD),
                'source_dataset_sha256': file_sha256(DATA / 'data/dataset.npz'),
                'validation_dataset_sha256': file_sha256(VAL / 'data/dataset.npz'),
                'correction_ratio': .08, 'dagger_repeat': 1, 'identity': identity,
                'normalization': 'Frozen initial task2 step4000 statistics',
                'optimizer': 'New AdamW, original replay learning rates, FP16 AMP default GradScaler',
                'budgets': {'stage_timeout_s': None, 'training_timeout_s': None},
                'resume': 'Atomic state every100 batches; optimizer/scaler/RNG/quota restored, completed epochs not repeated.'}
    contract_path = destination / 'training_contract.json'
    if contract_path.exists():
        if read(contract_path) != contract:
            raise ValueError('training resume provenance differs')
    else:
        if any(destination.iterdir()):
            raise FileExistsError('unrecognized occupied training directory')
        write_json(contract_path, contract)
    receipt_path = destination / 'training_receipt.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt['contract_sha256'] != signature(contract) or any(
                file_sha256(p) != sha for p, sha in receipt['checkpoint_hashes'].items()):
            raise ValueError('completed training evidence changed')
        return receipt
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    transform = image_transform_for_dataset(DATA)
    kwargs = dict(transform=transform, n_obs_steps=2, n_action_steps=16,
                  include_velocity=True, include_previous_action=True)
    original_data = RCMDataset(DATA, **kwargs)
    validation = RCMDataset(VAL, **kwargs)
    original = IndexSubset(original_data, np.arange(len(original_data)), source=0)
    if replay['branch'] == 'A':
        correction_data = RCMDataset(DATA, **kwargs)
        correction_data.actions = apply_abc_overlay(correction_data.actions, replay['overlay'],
                                                     contract['source_dataset_sha256'])
        with np.load(replay['correction_window_path'], allow_pickle=False) as f:
            wanted_rows = f['rows'].copy()
        valid = np.asarray(correction_data.valid_indices)
        correction_indices = np.flatnonzero(np.isin(valid, wanted_rows))
        if len(correction_indices) != len(wanted_rows):
            raise ValueError('ABC correction valid_window_indices no longer match')
    else:
        correction_data = RCMDataset(Path(replay['data_dir']), **kwargs)
        if file_sha256(Path(replay['data_dir']) / 'data/dataset.npz') != replay['dataset_sha256']:
            raise ValueError('same-observation replay dataset changed')
        correction_indices = np.arange(len(correction_data))
    correction = IndexSubset(correction_data, correction_indices, source=1)
    if min(len(original), len(correction), len(validation)) < 1:
        raise ValueError('empty training/replay/validation windows')
    training = ConcatDataset([original, correction])
    device = torch.device('cuda')
    model, config, _ = _load_policy(OLD, device)
    optimizer = _optimizer(model, args)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    config.update({'training_module': 'a_axis_flip_replay_v1', 'data_dir': str(DATA),
        'correction_data_dir': replay.get('data_dir'), 'correction_ratio': .08,
        'init_from': str(OLD), 'dagger_repeat': 1, 'amp': True,
        'n_obs_steps': 2, 'n_action_steps': 16, 'include_velocity': True,
        'include_previous_action': True, 'flow_integration_steps': 4})
    write_json(destination / 'training_config.json', config)
    val_loader = _validation_loader(args, validation)
    manifest = {'original_windows': len(original), 'correction_windows': len(correction),
                'validation_windows': len(validation), 'epoch_correction_samples': 1280,
                'epoch_original_samples': 14720, 'total_batches': 4000,
                'original_window_index_sha256': signature(np.asarray(original_data.valid_indices).tolist()),
                'correction_window_index_sha256': signature(correction_indices.tolist()),
                'validation_window_index_sha256': signature(np.asarray(validation.valid_indices).tolist()),
                'validation_max_batches': None, 'baseline': str(OLD)}
    write_json(destination / 'training_manifest.json', manifest)
    latest = destination / 'resume_state.pth'
    state = {'epoch': 0, 'batch_offset': 0, 'global_step': 0, 'optimizer_updates': 0,
             'epoch_loss_sum': 0., 'history': [], 'counts': [0, 0], 'epoch_counts': [0, 0]}
    restored_rng = None
    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        if saved['contract_sha256'] != signature(contract):
            raise ValueError('resume checkpoint contract differs')
        model.load_state_dict(saved['model_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        scaler.load_state_dict(saved['scaler_state_dict'])
        state, restored_rng = saved['progress'], saved['rng_state']
        print(f"Resume exactly after batch {state['global_step']}/4000", flush=True)
        del saved

    def save_resume():
        finite_parameters(model)
        _atomic_torch_save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict(),
            'rng_state': rng_state(), 'progress': state, 'contract_sha256': signature(contract)}, latest)
        write_json(destination / 'partial_summary.json', state)

    metrics = (destination / 'training_metrics.jsonl').open('a', buffering=1)
    try:
        for epoch in range(state['epoch'], 4):
            loader = _train_loader(args, training, 'ratio', len(original), len(correction), .08, epoch)
            offset = state['batch_offset']
            model.train()
            for batch_index, batch in enumerate(loader):
                if batch_index < offset:
                    continue
                if restored_rng is not None:
                    restore_rng(restored_rng)
                    restored_rng = None
                batch = _move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float16):
                    loss = _loss(model, batch)
                value = float(loss.detach())
                if not math.isfinite(value):
                    raise FloatingPointError(f"nonfinite training loss at batch {state['global_step']+1}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                before_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                updated = scaler.get_scale() >= before_scale
                if not math.isfinite(float(norm)) and updated:
                    raise FloatingPointError('nonfinite unscaled gradient on an optimizer update')
                # AMP overflow skips are the original GradScaler behavior, explicitly logged.
                counts = [int((batch['sample.source'] == j).sum()) for j in (0, 1)]
                for j in (0, 1):
                    state['counts'][j] += counts[j]
                    state['epoch_counts'][j] += counts[j]
                state['global_step'] += 1
                state['optimizer_updates'] += int(updated)
                state['epoch_loss_sum'] += value
                state['batch_offset'] = batch_index+1
                row = {'step': state['global_step'], 'epoch': epoch, 'loss': value,
                       'optimizer_updated': updated, 'amp_scale': scaler.get_scale(),
                       'gradient_norm': float(norm) if math.isfinite(float(norm)) else 'AMP_overflow_skipped',
                       'sampling_counts_original_correction': state['counts']}
                metrics.write(json.dumps(row, allow_nan=False) + '\n')
                if state['global_step'] % 100 == 0:
                    print(json.dumps(row), flush=True)
                    save_resume()
            if state['epoch_counts'] != [14720, 1280] or state['batch_offset'] != 1000:
                raise ValueError(f"8% quota/epoch batch mismatch: {state}")
            if restored_rng is not None:
                restore_rng(restored_rng)
                restored_rng = None
            validation_loss = checked_validation(model, val_loader, device, args)
            finite_parameters(model)
            epoch_row = {'epoch': epoch, 'batches_seen': state['global_step'],
                         'train_loss': state['epoch_loss_sum']/1000, 'val_loss': validation_loss,
                         'optimizer_updates': state['optimizer_updates'],
                         'sampling_counts': state['counts'].copy()}
            payload = {'schema_version': 1, 'policy_type': model.policy_type,
                       'architecture': attention.ARCHITECTURE,
                       'model_state_dict': model.state_dict(), 'config': config,
                       'epoch': epoch, 'train_loss': epoch_row['train_loss'], 'val_loss': validation_loss}
            _atomic_torch_save(payload, destination / f"step_{state['global_step']:06d}.pth")
            state['history'].append(epoch_row)
            write_json(destination / 'training_history.json', state['history'])
            record('training_epoch', 'completed', destination,
                   f'branch={replay["branch"]}; dataset={DATA}; init={OLD}; {epoch_row}; AMP=True; correction8%')
            state.update(epoch=epoch+1, batch_offset=0, epoch_loss_sum=0., epoch_counts=[0, 0])
            save_resume()
            print('Training epoch result:', epoch_row, flush=True)
        if state['global_step'] != 4000 or state['counts'] != [58880, 5120]:
            raise ValueError('full four-epoch 8% replay budget not completed')
        checkpoint_hashes = {str(destination / f'step_{step:06d}.pth'):
                             file_sha256(destination / f'step_{step:06d}.pth') for step in (2000, 4000)}
        receipt = {'status': 'completed', 'contract_sha256': signature(contract),
                   'checkpoint_hashes': checkpoint_hashes, 'progress': state, 'manifest': manifest}
        write_json(receipt_path, receipt)
        record('training', 'completed', destination, receipt)
        return receipt
    except BaseException as error:
        # Preserve the last finite atomic state; never save corrupt weights over it.
        write_json(destination / 'failure.json', {'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                   'reason': repr(error), 'progress': state, 'last_finite_resume': str(latest)})
        record('training', 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', destination,
               f'init={OLD}; data={DATA}; branch={replay["branch"]}; step={state["global_step"]}; {error!r}')
        raise
    finally:
        metrics.close()
        del model, optimizer, scaler, val_loader, training
        gc.collect()
        torch.cuda.empty_cache()
