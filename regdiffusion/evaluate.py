import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# Modified from 
# https://github.com/HantaoShu/DeepSEM/blob/master/src/utils.py

def get_metrics(A, ground_truth):
    ''' Calculate Metrics including AUPR, AUPRR, EP, and EPR
    
    Calculate EPR given predicted adjacency matrix and BEELINE 
    ground truth
    
    Parameters
    ----------
    A: numpy.array 
        Predicted adjacency matrix. Expected size is |g| x |g|.
    ground_truth: dict
        BEELINE ground truth object exported by 
        data.load_beeline_ground_truth. The first element of this
        tuple is eval_flat_mask, the boolean mask on the flatten
        adjacency matrix to identify TFs and target genes. The
        second element is the lable values y_true after flatten. 
        
    Returns
    -------
    tuple
        A tuple with AUPR, AUPR ratio, EP (in counts), and EPR
    '''
    gt = ground_truth
    if A.shape[0] == A.shape[1]:
        A = A[gt['tf_mask'], :]
    A = A[:, gt['gene_mask']]
    
    y_pred = np.abs(A.flatten())
    
    AUPR = average_precision_score(gt['y_true'], y_pred)
    AUPRR = AUPR / np.mean(gt['y_true'])
    
    num_truth_edge = int(gt['y_true'].sum())
    cutoff = np.partition(y_pred, -num_truth_edge)[-num_truth_edge]
    y_above_cutoff = y_pred > cutoff
    EP = int(np.sum(gt['y_true'][y_above_cutoff]))
    EPR = 1. * EP / ((num_truth_edge ** 2) / len(y_pred))
    
    return {'AUPR': AUPR, 'AUPRR': AUPRR, 
            'EP': EP, 'EPR': EPR}

