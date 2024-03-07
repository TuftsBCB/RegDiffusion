import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from collections import deque, Counter
from scipy.sparse import csr_matrix, save_npz
import pickle

def extract_edges(A, gene_names=None, TFmask=None, threshold=0.0):
    '''Extract predicted edges
    
    Extract edges from the predicted adjacency matrix
    
    Parameters
    ----------
    A: numpy.array 
        Predicted adjacency matrix. Expected size is |g| x |g|.
    gene_names: None, list or numpy.array
        (Optional) List of Gene Names. Usually accessible in the var_names 
        field of scanpy data. 
    TFmask: numpy.array
        A masking matrix indicating the position of TFs. Expected 
        size is |g| x |g|.
        
    Returns
    -------
    pandas.DataFrame
        A DataFrame including all the predicted links with predicted
        link strength.
    '''
    num_nodes = A.shape[0]
    mat_indicator_all = np.zeros([num_nodes, num_nodes])
    if TFmask is not None:
        A_masked = A * TFmask
    else:
        A_masked = A
    mat_indicator_all[abs(A_masked) > threshold] = 1
    idx_source, idx_target = np.where(mat_indicator_all)
    if gene_names is None:
        source_lbl = idx_source
        target_lbl = idx_target
    else:
        source_lbl = gene_names[idx_source]
        target_lbl = gene_names[idx_target]
    edges_df = pd.DataFrame(
        {'Source': source_lbl, 'Target': target_lbl, 
         'EdgeWeight': (A[idx_source, idx_target]),
         'AbsEdgeWeight': (np.abs(A[idx_source, idx_target]))
        })
    edges_df = edges_df.sort_values('AbsEdgeWeight', ascending=False)

    return edges_df.reset_index(drop=True)