"""Worker script: runs a single BEELINE experiment. Called by test_beeline_benchmark.py."""
import numpy as np
import regdiffusion as rd
import torch
import json
import sys
import os


def main():
    dataset = sys.argv[1]
    memory_efficient = sys.argv[2] == 'True'
    seed = int(sys.argv[3])
    output_path = sys.argv[4]

    bl_dt, bl_gt = rd.data.load_beeline(
        benchmark_data=dataset, benchmark_setting='1000_STRING'
    )
    evaluator = rd.evaluator.GRNEvaluator(bl_gt, bl_dt.var_names)

    torch.manual_seed(seed)
    np.random.seed(seed)

    trainer = rd.RegDiffusionTrainer(
        bl_dt.X, device='cuda', n_steps=1000,
        memory_efficient=memory_efficient,
    )
    trainer.train()

    inferred_adj = trainer.get_adj()
    results = evaluator.evaluate(inferred_adj)

    with open(output_path, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
