import numpy as np
import pandas as pd
from dazzle import load_beeline, extract_edges, get_metrics, DAZZLE, LightLogger, runDAZZLE, runDAZZLE_ensemble, DEFAULT_DAZZLE_CONFIGS, DEFAULT_DEEPSEM_CONFIGS
from regdiffusion import runRegDiffusion, DEFAULT_REGDIFFUSION_CONFIGS
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from arboreto.algo import grnboost2, genie3
import sys
import distributed

if __name__ == '__main__':    

    method = sys.argv[1]
    bm = sys.argv[2]

    # client = distributed.Client(processes=False)
    for dt in ['hESC', 'hHep', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']:
        bl_data, bl_gt = load_beeline(benchmark_data=dt, benchmark_setting=bm)
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

        elif method == 'dazzle':
            start = datetime.now()
            model, adjs = runDAZZLE(bl_data.X, DEFAULT_DAZZLE_CONFIGS)
            adj = model.get_adj()
            time_cost = datetime.now() - start

        elif method == 'deepsem':
            start = datetime.now()
            model, adjs = runDAZZLE(bl_data.X, DEFAULT_DEEPSEM_CONFIGS)
            adj = model.get_adj()
            time_cost = datetime.now() - start
            
        elif method == 'deepseme':
            start = datetime.now()
            models, adj = runDAZZLE_ensemble(bl_data.X, DEFAULT_DEEPSEM_CONFIGS)
            time_cost = datetime.now() - start
            
        elif method == 'regdiffusion':
            start = datetime.now()
            model = runRegDiffusion(bl_data.X, DEFAULT_REGDIFFUSION_CONFIGS)
            adj = model.get_adj()
            time_cost = datetime.now() - start
        elif method == 'regdiffusionda':
            start = datetime.now()
            regdiffusion = runRegDiffusion(bl_data.X, DEFAULT_REGDIFFUSION_CONFIGS)
            dazzle, adjs = runDAZZLE(bl_data.X, DEFAULT_DAZZLE_CONFIGS)
            regdiffusion_adj = regdiffusion.get_adj()
            dazzle_adj = dazzle.get_adj()
            adj = regdiffusion_adj + dazzle_adj
            time_cost = datetime.now() - start

        metrics = get_metrics(adj, bl_gt)
        with open(f'results/{bm}/{method}.csv', 'a') as f:
            f.writelines(f"{dt},{time_cost.total_seconds()},{metrics['AUPR']},{metrics['AUPRR']},{metrics['EP']},{metrics['EPR']}\n")
