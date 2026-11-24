"""Compare CPU and MPS inference on BEELINE with identical trainer settings.

Example:
    python -m test_scripts.benchmark_mps --data-dir data --seeds 42 142 242

Runs devices sequentially to avoid timing concurrent CPU/GPU workloads.
Results are saved after each run and include the full training loss history.
"""
import argparse
import functools
import gc
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

import regdiffusion as rd
import regdiffusion.trainer as trainer_module


DATASETS = ['hESC', 'hHep', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']


class ProgressLogger(rd.LightLogger):
    def __init__(self, result_dir, label):
        super().__init__(result_dir=str(result_dir))
        self.label = label
        self.started = time.perf_counter()

    def log(self, log_dict, step=None):
        super().log(log_dict, step=step)
        completed = self.mem[self.current_log]['current_step'] + 1
        if completed % 100 == 0:
            elapsed = time.perf_counter() - self.started
            print(f'{self.label}: step {completed}, loss={log_dict["train_loss"]:.5f}, '
                  f'elapsed={elapsed:.1f}s', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data', type=Path)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--devices', nargs='+', choices=['cpu', 'mps'], default=['mps', 'cpu'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--memory-efficient', action='store_true')
    parser.add_argument('--mps-amp', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('results/mps_beeline/results.json'))
    args = parser.parse_args()
    if 'mps' in args.devices and not torch.backends.mps.is_available():
        parser.error('MPS is unavailable; this benchmark must run on the actual GPU.')
    torch.set_num_threads(args.threads)
    trainer_module.tqdm = functools.partial(trainer_module.tqdm, disable=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'macos': platform.mac_ver()[0], 'machine': platform.machine()},
        'settings': {'benchmark_setting': '1000_STRING', 'n_steps': args.steps,
                     'batch_size': 128, 'threads': args.threads, 'compile': False,
                     'memory_efficient': args.memory_efficient, 'mps_amp': args.mps_amp,
                     'seeds': args.seeds, 'devices': args.devices,
                     'data_dir': str(args.data_dir.resolve())},
        'datasets': {}, 'runs': [],
    }
    print(json.dumps(report['environment']), flush=True)
    for dataset in args.datasets:
        data, truth = rd.data.load_beeline(
            data_dir=str(args.data_dir), benchmark_data=dataset,
            benchmark_setting='1000_STRING')
        evaluator = rd.GRNEvaluator(truth, data.var_names)
        dataset_path = args.data_dir / 'BEELINE' / f'1000_STRING_{dataset}'
        report['datasets'][dataset] = {
            'cells': data.n_obs, 'genes': data.n_vars,
            'sha256': {name: hashlib.sha256((dataset_path / name).read_bytes()).hexdigest()
                       for name in ('data.csv', 'label.csv')},
        }
        for seed in args.seeds:
            for device in args.devices:
                label = f'{dataset}-{device}-seed{seed}'
                print(f'Starting {label}: {data.shape}', flush=True)
                logger = ProgressLogger(args.output.parent / 'logs', label)
                trainer = rd.RegDiffusionTrainer(
                    data.X, device=device, n_steps=args.steps, batch_size=128,
                    memory_efficient=args.memory_efficient,
                    use_amp=args.mps_amp and device == 'mps',
                    compile=False, seed=seed, logger=logger)
                if device == 'mps':
                    torch.mps.synchronize()
                started = time.perf_counter()
                trainer.train()
                if device == 'mps':
                    torch.mps.synchronize()
                seconds = time.perf_counter() - started
                adjacency = trainer.get_adj()
                history = trainer.logger.to_df().train_loss.to_numpy(dtype=float)
                if not np.isfinite(adjacency).all() or not np.isfinite(history).all():
                    raise RuntimeError(f'{label} produced nonfinite outputs')
                metrics = {key: float(value) for key, value in evaluator.evaluate(adjacency).items()}
                adjacency_path = args.output.parent / f'{label}-adj.npy'
                np.save(adjacency_path, adjacency)
                result = {'dataset': dataset, 'device': device, 'seed': seed,
                          'seconds': seconds, 'metrics': metrics,
                          'train_loss': history.tolist(), 'adjacency_path': str(adjacency_path)}
                report['runs'].append(result)
                pending = args.output.with_suffix('.tmp')
                pending.write_text(json.dumps(report, indent=2))
                pending.replace(args.output)
                print(f'Finished {label}: {seconds:.1f}s, {json.dumps(metrics)}', flush=True)
                del trainer
                gc.collect()
                if device == 'mps':
                    torch.mps.empty_cache()
    print(f'Results: {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
