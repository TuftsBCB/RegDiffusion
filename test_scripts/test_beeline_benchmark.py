"""Full BEELINE benchmark comparing RegDiffusion vs RegDiffusionME.

Tests all 7 BEELINE datasets with 1000_STRING, 10 repeats each.
Distributes runs across 8 GPUs in parallel.
"""
import numpy as np
import json
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

DATASETS = ['hESC', 'hHep', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']
N_REPEATS = 10
N_GPUS = 8
METRICS = ['AUROC', 'AUPRR', 'EPR']

CONFIGS = [
    {"label": "RegDiffusion", "memory_efficient": False},
    {"label": "RegDiffusionME", "memory_efficient": True},
]


def run_worker(dataset, memory_efficient, seed, gpu_id, output_path):
    """Launch a worker subprocess on a specific GPU."""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    result = subprocess.run(
        ['python', 'test_beeline_worker.py',
         dataset, str(memory_efficient), str(seed), output_path],
        capture_output=True, text=True, env=env,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    if result.returncode != 0:
        return None, result.stderr[-500:] if result.stderr else "Unknown error"
    with open(output_path) as f:
        return json.load(f), None


def main():
    # Build list of all jobs
    jobs = []
    for dataset in DATASETS:
        for cfg in CONFIGS:
            for rep in range(N_REPEATS):
                jobs.append({
                    'dataset': dataset,
                    'label': cfg['label'],
                    'memory_efficient': cfg['memory_efficient'],
                    'seed': rep * 100 + 42,
                    'rep': rep,
                })

    total = len(jobs)
    print(f"Total runs: {total} ({len(DATASETS)} datasets x {len(CONFIGS)} configs x {N_REPEATS} repeats)")
    print(f"Running on {N_GPUS} GPUs in parallel\n")

    # Run jobs with process pool (N_GPUS concurrent workers)
    all_results = {d: {c['label']: [] for c in CONFIGS} for d in DATASETS}
    completed = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        with ProcessPoolExecutor(max_workers=N_GPUS) as executor:
            futures = {}
            for i, job in enumerate(jobs):
                gpu_id = i % N_GPUS
                output_path = os.path.join(tmpdir, f"result_{i}.json")
                future = executor.submit(
                    run_worker,
                    job['dataset'], job['memory_efficient'],
                    job['seed'], gpu_id, output_path
                )
                futures[future] = job

            for future in as_completed(futures):
                job = futures[future]
                completed += 1
                res, err = future.result()
                if res is not None:
                    all_results[job['dataset']][job['label']].append(res)
                    print(f"[{completed}/{total}] {job['dataset']} | {job['label']} | rep {job['rep']+1}"
                          f"  AUROC={res['AUROC']:.4f}  AUPRR={res['AUPRR']:.4f}  EPR={res['EPR']:.2f}")
                else:
                    print(f"[{completed}/{total}] {job['dataset']} | {job['label']} | rep {job['rep']+1}"
                          f"  ERROR: {err}")

    # Compute summaries
    summary = {}
    for dataset in DATASETS:
        summary[dataset] = {}
        for cfg in CONFIGS:
            label = cfg['label']
            runs = all_results[dataset][label]
            if runs:
                s = {}
                for key in runs[0]:
                    vals = [r[key] for r in runs]
                    s[f"{key}_mean"] = round(np.mean(vals), 4)
                    s[f"{key}_std"] = round(np.std(vals), 4)
                summary[dataset][label] = {"runs": runs, "summary": s}
            else:
                summary[dataset][label] = {"runs": [], "summary": {}}

    # Print markdown tables
    for metric in METRICS:
        print(f"\n\n## {metric} Comparison (mean +/- std over {N_REPEATS} runs)\n")
        header = "| Dataset |"
        separator = "|---------|"
        for cfg in CONFIGS:
            header += f" {cfg['label']} |"
            separator += "----------------|"
        print(header)
        print(separator)

        for dataset in DATASETS:
            row = f"| {dataset} |"
            for cfg in CONFIGS:
                label = cfg['label']
                s = summary[dataset].get(label, {}).get("summary", {})
                mean = s.get(f"{metric}_mean")
                std = s.get(f"{metric}_std")
                if mean is not None:
                    row += f" {mean:.4f} +/- {std:.4f} |"
                else:
                    row += " N/A |"
            print(row)

    # Save full results
    with open("beeline_benchmark_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull results saved to beeline_benchmark_results.json")


if __name__ == "__main__":
    main()
