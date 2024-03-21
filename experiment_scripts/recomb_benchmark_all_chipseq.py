import numpy as np
import pandas as pd
from other_models.dazzle import runDAZZLE, runDAZZLE_ensemble, DEFAULT_DAZZLE_CONFIGS, DEFAULT_DEEPSEM_CONFIGS
import regdiffusion as rd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
# from arboreto.algo import grnboost2, genie3
import sys
# import distributed

deepsem_config = DEFAULT_DEEPSEM_CONFIGS
deepsem_config['alpha'] = 1
deepsem_config['beta'] = 0.01
dazzle_config = DEFAULT_DAZZLE_CONFIGS
dazzle_config['alpha'] = 1
dazzle_config['beta'] = 0.01

if __name__ == '__main__':    

    method = sys.argv[1]
    bm = sys.argv[2]

    # client = distributed.Client(processes=False)
    for dt in ['hESC', 'hHep', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']:
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

        elif method == 'dazzle':
            start = datetime.now()
            model, adjs = runDAZZLE(bl_data.X, dazzle_config)
            adj = model.get_adj()
            time_cost = datetime.now() - start

        elif method == 'deepsem':
            start = datetime.now()
            model, adjs = runDAZZLE(bl_data.X, deepsem_config)
            adj = model.get_adj()
            time_cost = datetime.now() - start
            
        elif method == 'deepseme':
            start = datetime.now()
            models, adj = runDAZZLE_ensemble(bl_data.X, deepsem_config)
            time_cost = datetime.now() - start
            
        elif method == 'regdiffusion':
            start = datetime.now()
            rd_trainer = rd.RegDiffusionTrainer(bl_data.X, weight_decay_adj=0.0001, sparse_loss_coef=0.0025, init_coef=10)
            rd_trainer.train()
            adj = rd_trainer.get_adj()
            time_cost = datetime.now() - start

        metrics = evaluator.evaluate(adj)
        with open(f'results/202403/{bm}_{method}.csv', 'a') as f:
            f.writelines(f"{dt},{time_cost.total_seconds()},{metrics['AUPR']},{metrics['AUPRR']},{metrics['EP']},{metrics['EPR']}\n")
