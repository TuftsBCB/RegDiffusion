"""GPU memory benchmark for RegDiffusion at different gene counts.

Tests default, AMP, and memory-efficient (ME) configurations.
"""
import numpy as np
import torch
import gc
import json
import time
import regdiffusion as rd

GENE_SIZES = [5000, 10000, 15000, 20000, 25000, 30000]
N_CELLS = 200
N_STEPS = 20
BATCH_SIZE = 128

CONFIGS = [
    {"label": "Default",  "use_amp": False, "memory_efficient": False},
    {"label": "AMP",      "use_amp": True,  "memory_efficient": False},
    {"label": "ME",       "use_amp": False, "memory_efficient": True},
    {"label": "ME+AMP",   "use_amp": True,  "memory_efficient": True},
]


def run_single(n_gene, use_amp, memory_efficient):
    np.random.seed(42)
    exp_array = np.random.rand(N_CELLS, n_gene).astype(np.float32) * 5

    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    start = time.time()
    trainer = rd.RegDiffusionTrainer(
        exp_array,
        device='cuda',
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        use_amp=use_amp,
        memory_efficient=memory_efficient,
    )
    trainer.train()
    elapsed = time.time() - start

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    time_per_step = elapsed / N_STEPS

    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return peak_mem, time_per_step


def main():
    results = {}

    for n_gene in GENE_SIZES:
        results[n_gene] = {}
        for cfg in CONFIGS:
            label = cfg["label"]
            print(f"\n{'='*60}")
            print(f"Testing {label} with {n_gene:,} genes...")
            print(f"{'='*60}")
            try:
                mem, tps = run_single(
                    n_gene, cfg["use_amp"], cfg["memory_efficient"]
                )
                results[n_gene][label] = {
                    "memory_gb": round(mem, 2),
                    "time_per_step": round(tps, 3),
                }
                print(f"  Peak memory: {mem:.2f} GB, Time/step: {tps:.3f}s")
            except torch.cuda.OutOfMemoryError:
                results[n_gene][label] = {"memory_gb": "OOM", "time_per_step": "OOM"}
                print(f"  OUT OF MEMORY")
                gc.collect()
                torch.cuda.empty_cache()

    # Print comparison table
    print(f"\n{'='*100}")
    print("RESULTS SUMMARY")
    print(f"{'='*100}")

    header = f"{'Genes':>8}"
    for cfg in CONFIGS:
        label = cfg["label"]
        header += f" | {label+' (GB)':>12} {label+' (s/step)':>14}"
    print(header)
    print("-" * len(header))

    for n_gene in GENE_SIZES:
        row = f"{n_gene:>8,}"
        for cfg in CONFIGS:
            label = cfg["label"]
            r = results[n_gene].get(label, {})
            mem = r.get("memory_gb", "N/A")
            tps = r.get("time_per_step", "N/A")
            mem_str = f"{mem}" if isinstance(mem, str) else f"{mem:.2f}"
            tps_str = f"{tps}" if isinstance(tps, str) else f"{tps:.3f}"
            row += f" | {mem_str:>12} {tps_str:>14}"
        print(row)

    # Savings comparison
    print(f"\n{'='*60}")
    print("MEMORY SAVINGS vs Default")
    print(f"{'='*60}")
    for n_gene in GENE_SIZES:
        default_mem = results[n_gene].get("Default", {}).get("memory_gb")
        if default_mem is None or default_mem == "OOM":
            continue
        row = f"{n_gene:>8,}"
        for cfg in CONFIGS[1:]:
            label = cfg["label"]
            mem = results[n_gene].get(label, {}).get("memory_gb")
            if mem is None or mem == "OOM" or default_mem == 0:
                row += f" | {label}: N/A"
            else:
                saving = (1 - mem / default_mem) * 100
                row += f" | {label}: {saving:.1f}%"
        print(row)

    with open("memory_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to memory_benchmark_results.json")


if __name__ == "__main__":
    main()
