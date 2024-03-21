import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from arboreto.algo import grnboost2, genie3
import sys
import regdiffusion as rd

# import distributed

if __name__ == '__main__':    

    method = sys.argv[1]
    dt = sys.argv[2]
    bm = sys.argv[3]

    # client = distributed.Client(processes=False)

    bl_data, bl_gt = rd.data.load_beeline(benchmark_data=dt, benchmark_setting=bm)
    evaluator = rd.evaluator.GRNEvaluator(bl_gt, bl_data.var_names)
    if method == 'grnboost2':
        start = datetime.now()
        net = grnboost2(
            bl_data.X, gene_names=bl_data.var_names, 
            tf_names=None, verbose=True)
        time_cost = datetime.now() - start

        adj = np.zeros([bl_data.n_vars, bl_data.n_vars])
        gene_idx = {n: i for i, n in enumerate(bl_data.var_names)}
        for i, r in tqdm(net.iterrows()):
            adj[gene_idx[r['TF']], gene_idx[r['target']]] = r['importance']

    elif method == 'genie3':
        start = datetime.now()
        net = genie3(
            bl_data.X, gene_names=bl_data.var_names, 
            tf_names=None, verbose=True)
        time_cost = datetime.now() - start

        adj = np.zeros([bl_data.n_vars, bl_data.n_vars])
        gene_idx = {n: i for i, n in enumerate(bl_data.var_names)}
        for i, r in tqdm(net.iterrows()):
            adj[gene_idx[r['TF']], gene_idx[r['target']]] = r['importance']

    metrics = evaluator.evaluate(adj)
    with open(f'results/{bm}/{method}.csv', 'a') as f:
        f.writelines(f"{dt},{time_cost.total_seconds()},{metrics['AUPR']},{metrics['AUPRR']},{metrics['EP']},{metrics['EPR']}\n")
