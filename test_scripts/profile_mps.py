"""Profile MPS training and test equivalent contractions without changing the model.

Run from the repository root:
    python -m test_scripts.profile_mps --data-dir data

Stage timings fence the GPU, include dispatch/wait costs, and perturb overlap.
End-to-end timings use the ordinary backward pass with no extra stage fences.
The reassociated contraction keeps the final bias AFTER gene aggregation.
"""
import argparse
import copy
import gc
import hashlib
import json
import platform
import statistics
import subprocess
import tempfile
import time
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import regdiffusion as rd


def features(model, x, t, ct):
    time_embedding = model.time_mlp(t)
    cell_embedding = model.celltype_emb(ct)
    original = x.unsqueeze(-1)
    h = model.gene_emb(x)
    for i, block in enumerate(model.blocks):
        if i:
            h = torch.cat((h, original), dim=-1)
        h = block(h, time_embedding, cell_embedding)
    return h


def flat_mix(h, a):
    batch, genes, dims = h.shape
    return (h.transpose(1, 2).reshape(batch * dims, genes) @ a).reshape(
        batch, dims, genes).transpose(1, 2)


def experimental_forward(self, x, t, ct):
    h = features(self, x, t, ct)
    a = self.I_minus_A()
    if self._profile_variant == 'baseline':
        return self.final(torch.einsum('ogd,gh->ohd', h, a)).squeeze(-1)
    if self._profile_variant == 'flat_mm':
        return self.final(flat_mix(h, a)).squeeze(-1)
    # Linear reassociation: sum_g sum_d h[b,g,d] * a[g,j] * w[d] + bias.
    projected = F.linear(h, self.final.weight, bias=None).squeeze(-1)
    z = projected @ a
    return z if self.final.bias is None else z + self.final.bias.to(dtype=z.dtype)


@torch.no_grad()
def sampled_edges_only(self):
    rows, cols = self.sampled_adj_row_nonparam, self.sampled_adj_col_nonparam
    values = self.soft_thresholding(self.adj_A[rows, cols], self.gene_reg_norm / 2)
    return values * (rows != cols).to(values.dtype)


def configure(model, variant):
    # Keep the pre-0.2.5 baseline explicit after the production forward changes.
    model._profile_variant = (
        'baseline' if variant in ('baseline', 'sample_only', 'monitor_every10')
        else variant)
    model.forward = types.MethodType(experimental_forward, model)
    if variant in ('sample_only', 'project_first_sample_only'):
        model.get_sampled_adj_ = types.MethodType(sampled_edges_only, model)


def stats(values):
    return dict(mean_ms=statistics.mean(values), median_ms=statistics.median(values),
                min_ms=min(values), max_ms=max(values), samples=len(values))


class StageTimer:
    def __init__(self):
        self.samples = defaultdict(list)
        self.enabled = False

    def call(self, name, fn):
        if not self.enabled:
            return fn()
        torch.mps.synchronize()
        start = time.perf_counter()
        result = fn()
        torch.mps.synchronize()
        self.samples[name].append((time.perf_counter() - start) * 1000)
        return result


def inputs(trainer, step):
    torch.manual_seed(42 + step)
    x, ct = next(iter(trainer.train_dataloader))
    return x, ct


def noise_inputs(trainer, batch):
    x, ct = (v.to('mps') for v in batch)
    t = torch.randint(0, trainer.hp['T'], (x.shape[0],), device='mps')
    noisy, noise = trainer.forward_pass(x, t)
    return noisy, noise, t, ct


def ordinary_step(trainer, step, previous, monitor=True):
    batch = inputs(trainer, step)
    x, noise, t, ct = noise_inputs(trainer, batch)
    trainer.opt.zero_grad()
    prediction = trainer.model(x, t, ct)
    loss = F.mse_loss(prediction, noise) + .25 * trainer.model.get_sparse_loss()
    loss.backward()
    trainer.opt.step()
    if monitor:
        loss_value = loss.item()
        current = trainer.model.get_sampled_adj_()
        change = (current - previous).mean().item() * (trainer.n_gene - 1)
        trainer.logger.log(dict(train_loss=loss_value, adj_change=change))
        return current
    return previous


def split_step(trainer, step, previous, timer, variant):
    model = trainer.model
    batch = timer.call('seed_and_dataloader', lambda: inputs(trainer, step))
    x, noise, t, ct = timer.call('device_transfer_and_diffusion', lambda: noise_inputs(trainer, batch))
    timer.call('zero_grad', trainer.opt.zero_grad)
    h = timer.call('feature_network_forward', lambda: features(model, x, t, ct))
    a = timer.call('adjacency_transform_forward', model.I_minus_A)
    if variant == 'baseline':
        mixed = timer.call('gene_mix_forward', lambda: torch.einsum('ogd,gh->ohd', h, a))
        mse = timer.call('output_projection_and_mse', lambda: F.mse_loss(model.final(mixed).squeeze(-1), noise))
    else:
        projected = timer.call('feature_projection_forward', lambda: F.linear(h, model.final.weight, None).squeeze(-1))
        mixed = timer.call('gene_mix_forward', lambda: projected @ a)
        mse = timer.call('output_bias_and_mse', lambda: F.mse_loss(mixed + model.final.bias, noise))
    sparse = timer.call('sparse_loss_forward', lambda: .25 * model.get_sparse_loss())
    if variant == 'baseline':
        grad_mix, grad_w, grad_b = timer.call('output_projection_and_mse_backward', lambda: torch.autograd.grad(
            mse, (mixed, model.final.weight, model.final.bias)))
        model.final.weight.grad, model.final.bias.grad = grad_w, grad_b
        grad_h, grad_a = timer.call('gene_mix_backward', lambda: torch.autograd.grad(mixed, (h, a), grad_mix))
        timer.call('features_adjacency_sparse_backward', lambda: torch.autograd.backward(
            (h, a, sparse), (grad_h, grad_a, torch.ones_like(sparse))))
    else:
        grad_mix, grad_b = timer.call('output_bias_and_mse_backward', lambda: torch.autograd.grad(
            mse, (mixed, model.final.bias)))
        model.final.bias.grad = grad_b
        grad_p, grad_a = timer.call('gene_mix_backward', lambda: torch.autograd.grad(mixed, (projected, a), grad_mix))
        timer.call('features_projection_adjacency_sparse_backward', lambda: torch.autograd.backward(
            (projected, a, sparse), (grad_p, grad_a, torch.ones_like(sparse))))
    timer.call('adam_update', trainer.opt.step)

    def monitor():
        loss = (mse + sparse).item()
        current = model.get_sampled_adj_()
        change = (current - previous).mean().item() * (trainer.n_gene - 1)
        trainer.logger.log(dict(train_loss=loss, adj_change=change))
        return current
    return timer.call('adjacency_monitor_and_scalar_logging', monitor)


def verify_variants(make_trainer):
    report = {}
    original = make_trainer('baseline')
    x, noise, t, ct = noise_inputs(original, inputs(original, 30))
    original.model.eval()
    # Include inhibitory edges and entries around the threshold, plus a nonzero bias.
    with torch.no_grad():
        original.model.adj_A.copy_(torch.randn_like(original.model.adj_A) * original.model.gene_reg_norm)
        original.model.final.bias.fill_(.37)
    reference = original.model(x, t, ct)
    (F.mse_loss(reference, noise) + .25 * original.model.get_sparse_loss()).backward()
    for variant in ('flat_mm', 'project_first'):
        model = copy.deepcopy(original.model)
        model.zero_grad(set_to_none=True)
        configure(model, variant)
        actual = model(x, t, ct)
        torch.testing.assert_close(actual, reference, rtol=2e-4, atol=2e-5)
        (F.mse_loss(actual, noise) + .25 * model.get_sparse_loss()).backward()
        errors = []
        for (name, ref), (_, parameter) in zip(original.model.named_parameters(), model.named_parameters()):
            torch.testing.assert_close(parameter.grad, ref.grad, rtol=2e-3, atol=3e-5,
                                       msg=lambda msg: name + ': ' + msg)
            errors.append(float((parameter.grad - ref.grad).abs().max().item()))
        report[variant] = dict(output_max_abs=float((actual-reference).abs().max().item()),
                               gradient_max_abs=max(errors))
        del model, actual
    torch.testing.assert_close(sampled_edges_only(original.model), original.model.get_sampled_adj_(), rtol=0, atol=0)
    report['sample_only'] = 'Exact match, including diagonal masking'
    del original, reference
    gc.collect()
    torch.mps.empty_cache()

    # Check that splitting the backward pass for attribution preserves updates.
    for variant in ('baseline', 'project_first'):
        first, second = make_trainer(variant), make_trainer(variant)
        second.model.load_state_dict(first.model.state_dict())
        prev_first, prev_second = first.model.get_sampled_adj_(), second.model.get_sampled_adj_()
        ordinary_step(first, 30, prev_first)
        split_step(second, 30, prev_second, StageTimer(), variant)
        max_abs = 0.
        for (name, p), (_, q) in zip(first.model.named_parameters(), second.model.named_parameters()):
            torch.testing.assert_close(p, q, rtol=2e-4, atol=3e-5, msg=lambda msg: name + ': ' + msg)
            max_abs = max(max_abs, float((p-q).abs().max().item()))
        report[variant + '_split_backward_update_max_abs'] = max_abs
        del first, second
        gc.collect()
        torch.mps.empty_cache()
    return report


def isolated_mix(genes, iterations, warmup):
    batch, dims = 128, 15
    h = torch.randn(batch, genes, dims, device='mps', requires_grad=True)
    a = (torch.randn(genes, genes, device='mps') / genes).requires_grad_()
    w = torch.randn(1, dims, device='mps', requires_grad=True)
    upstream = torch.randn(batch, genes, device='mps')
    result = {}
    functions = {
        'einsum_plus_projection': lambda: F.linear(torch.einsum('bgd,gj->bjd', h, a), w).squeeze(-1),
        'flat_mm_plus_projection': lambda: F.linear(flat_mix(h, a), w).squeeze(-1),
        'projection_first': lambda: F.linear(h, w).squeeze(-1) @ a,
    }
    for name, fn in functions.items():
        timer = StageTimer()
        for step in range(warmup + iterations):
            timer.enabled = step >= warmup
            y = timer.call('forward', fn)
            timer.call('backward', lambda: torch.autograd.grad(y, (h, a, w), upstream))
        result[name] = {key: stats(values) for key, values in timer.samples.items()}
    del h, a, w, upstream
    gc.collect()
    torch.mps.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--dataset', default='hESC')
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--warmup', type=int, default=15)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('results/mps_profile/profile.json'))
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error('A real MPS GPU is required.')
    torch.set_num_threads(4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data, truth = rd.data.load_beeline(str(args.data_dir), args.dataset, '1000_STRING')
    report = dict(environment={'torch': torch.__version__, 'python': platform.python_version(),
                              'macos': platform.mac_ver()[0],
                              'chip': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()},
                  settings={'dataset': args.dataset, 'shape': list(data.shape), 'batch_size':128,
                            'dtype':'float32', 'warmup_steps':args.warmup, 'measured_steps':args.steps,
                            'repeats':args.repeats, 'cpu_threads':4, 'compile':False,
                            'data_sha256':hashlib.sha256((args.data_dir/'BEELINE'/f'1000_STRING_{args.dataset}'/'data.csv').read_bytes()).hexdigest()},
                  methodology={'stages':'GPU-fenced wall time; includes launch and synchronization costs; splitting backward perturbs scheduling',
                               'end_to_end':'Steady-state training with original backward, active sparsity penalty, original per-step scalar logging unless variant says otherwise; synchronize only outside timed blocks',
                               'isolated':'GPU-fenced forward/backward with leaf inputs; includes final feature projection for a fair comparison'})

    def save():
        args.output.write_text(json.dumps(report, indent=2))

    with tempfile.TemporaryDirectory(prefix='regdiffusion-profile-') as logs:
        def make_trainer(variant):
            trainer = rd.RegDiffusionTrainer(data.X, device='mps', seed=42,
                                            logger=rd.LightLogger(logs))
            configure(trainer.model, variant)
            return trainer

        report['correctness'] = verify_variants(make_trainer)
        print('Correctness:', json.dumps(report['correctness']), flush=True)
        save()
        report['stages'] = {}
        for variant in ('baseline', 'project_first'):
            trainer = make_trainer(variant)
            previous = trainer.model.get_sampled_adj_()
            timer = StageTimer()
            for step in range(args.warmup + args.steps):
                timer.enabled = step >= args.warmup
                previous = split_step(trainer, step+30, previous, timer, variant)
            report['stages'][variant] = {name: stats(values) for name, values in timer.samples.items()}
            print('Stages', variant, json.dumps({k: round(v['mean_ms'],3) for k,v in report['stages'][variant].items()}), flush=True)
            del trainer, previous
            gc.collect()
            torch.mps.empty_cache()
            save()

        variants = ['baseline', 'flat_mm', 'project_first', 'sample_only', 'monitor_every10', 'project_first_sample_only']
        samples = defaultdict(list)
        # Reverse order on alternating rounds to reduce consistent ordering bias.
        for repeat in range(args.repeats):
            for variant in variants if repeat%2==0 else reversed(variants):
                trainer = make_trainer(variant)
                previous = trainer.model.get_sampled_adj_()
                for step in range(args.warmup):
                    previous = ordinary_step(trainer, step+30, previous,
                                             monitor=variant!='monitor_every10' or step%10==0)
                torch.mps.synchronize()
                started = time.perf_counter()
                for step in range(args.steps):
                    previous = ordinary_step(trainer, args.warmup+step+30, previous,
                                             monitor=variant!='monitor_every10' or step%10==0)
                torch.mps.synchronize()
                elapsed = (time.perf_counter()-started)*1000/args.steps
                samples[variant].append(elapsed)
                print(f'E2E round={repeat+1} variant={variant} ms/step={elapsed:.3f}', flush=True)
                del trainer, previous
                gc.collect()
                torch.mps.empty_cache()
                report['end_to_end'] = {name: stats(values) for name,values in samples.items()}
                save()

        report['isolated_contraction'] = {}
        for genes in (692, data.n_vars, 1620):
            report['isolated_contraction'][str(genes)] = isolated_mix(genes, args.steps, args.warmup)
            print('Isolated contraction finished:', genes, flush=True)
            save()

        # Host-only operator trace identifies einsum lowering; NOT GPU timings.
        trainer = make_trainer('baseline')
        x, noise, t, ct = noise_inputs(trainer, inputs(trainer, 30))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
            with torch.profiler.record_function('baseline_forward_backward'):
                loss = F.mse_loss(trainer.model(x,t,ct), noise) + .25*trainer.model.get_sparse_loss()
                loss.backward()
                torch.mps.synchronize()
        prof.export_chrome_trace(str(args.output.parent/'host_operator_trace.json'))
        (args.output.parent/'host_operators.txt').write_text(prof.key_averages(group_by_input_shape=True).table(
            sort_by='self_cpu_time_total', row_limit=40))
        save()
    print('Saved:', args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
