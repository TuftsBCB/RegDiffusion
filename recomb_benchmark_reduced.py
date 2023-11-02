import numpy as np
import pandas as pd
from dazzle import load_beeline, extract_edges, get_metrics, DAZZLE, LightLogger, runDAZZLE, DEFAULT_DAZZLE_CONFIGS, DEFAULT_DEEPSEM_CONFIGS
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
    train_split = float(sys.argv[2])
    dt = 'mESC'
    bm = '1000_STRING'

    # client = distributed.Client(processes=False)

    bl_data, bl_gt = load_beeline(benchmark_data=dt, benchmark_setting=bm)
    
    random_state = np.random.RandomState(123)
    train_test_split = random_state.rand(bl_data.X.shape[0])
    train_index = train_test_split <= train_split
    test_index = train_test_split > train_split
    
    if method == 'grnboost2':
        start = datetime.now()
        net = grnboost2(
            bl_data.X[train_index, ], gene_names=bl_data.var_names, 
            tf_names=None, verbose=True)
        time_cost = datetime.now() - start

        adj = np.zeros([bl_data.n_vars, bl_data.n_vars])
        gene_idx = {n: i for i, n in enumerate(bl_data.var_names)}
        for i, r in tqdm(net.iterrows()):
            adj[gene_idx[r['TF']], gene_idx[r['target']]] = r['importance']

    elif method == 'genie3':
        start = datetime.now()
        net = genie3(
            bl_data.X[train_index, ], gene_names=bl_data.var_names, 
            tf_names=None, verbose=True)
        time_cost = datetime.now() - start

        adj = np.zeros([bl_data.n_vars, bl_data.n_vars])
        gene_idx = {n: i for i, n in enumerate(bl_data.var_names)}
        for i, r in tqdm(net.iterrows()):
            adj[gene_idx[r['TF']], gene_idx[r['target']]] = r['importance']

    elif method == 'dazzle':
        start = datetime.now()
        model, adjs = runDAZZLE(bl_data.X[train_index, ], DEFAULT_DAZZLE_CONFIGS)
        adj = model.get_adj()
        time_cost = datetime.now() - start

    elif method == 'deepsem':
        start = datetime.now()
        model, adjs = runDAZZLE(bl_data.X[train_index, ], DEFAULT_DEEPSEM_CONFIGS)
        adj = model.get_adj()
        time_cost = datetime.now() - start

    metrics = get_metrics(adj, bl_gt)
    with open(f'results/reduced_{method}.csv', 'a') as f:
        f.writelines(f"{dt},{train_split},{time_cost.total_seconds()},{metrics['AUPR']},{metrics['AUPRR']},{metrics['EP']},{metrics['EPR']}\n")
