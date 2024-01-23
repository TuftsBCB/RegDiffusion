import numpy as np
import pandas as pd
from dazzle import load_beeline, extract_edges, get_metrics, DAZZLE, LightLogger, runDAZZLE, DEFAULT_DAZZLE_CONFIGS, DEFAULT_DEEPSEM_CONFIGS
from regdiffusion import runRegDiffusion, DEFAULT_REGDIFFUSION_CONFIGS
import scanpy as sc
from datetime import datetime
import networkx as nx
from pyvis import network as net
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys

if __name__ == '__main__':

    file = sys.argv[1]

    hammond_ann = sc.read_csv(f'../grnvae/data/other_data/Hammond_processed/final/{file}_data.csv')
    hammond_ann = hammond_ann.transpose()
    hammond_ann = hammond_ann[:, ~hammond_ann.var_names.str.startswith('Gm')]
    hammond_ann = hammond_ann[:, ~hammond_ann.var_names.str.startswith('mt')]
    hammond_ann = hammond_ann[:, ~hammond_ann.var_names.str.startswith('Rpl')]
    hammond_ann = hammond_ann[:, ~hammond_ann.var_names.str.startswith('Rps')]
    
    sc.pp.filter_genes(hammond_ann, min_counts=10)
    print(hammond_ann.shape)
    
    configs = DEFAULT_DAZZLE_CONFIGS
    configs['n_epochs'] = 40
    configs['lr_adj'] = 1e-5
    configs['eval_on_n_steps'] = 1
    
    logger=LightLogger()
    hammond_net = runDAZZLE(hammond_ann.X, configs, logger=logger)
    
    adj = hammond_net[0].get_adj()
    
    with open(f'dazzle_hammond_{f}.pkl', 'wb') as f:
        pickle.dump((adj, logger), f)
