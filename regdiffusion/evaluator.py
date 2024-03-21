import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

class GRNEvaluator:
    """
    A generalized evaluator for GRN inference.

    Args:
        ground_truth (np.ndarray or list): Either a 2D numpy array or list of 
            list holding the ground truth. Each row is an edge and includes 
            names for the source and target nodes. For example, [['A', 'B'], 
            ['B', 'C']].
        gene_names (np.ndarray or list): Either a 1D numpy array or list of 
            gene names. Make sure the order of the gene names is the same as 
            the order of gene names in the adjacency matrix. 
        metrics (list): A list of supported evaluation metrics. Currently 
            support 'AUROC', 'AUPR', 'AUPRR', 'EP', 'EPR'.
    """
    def __init__(self, ground_truth, gene_names, 
                 metrics=['AUROC', 'AUPR', 'AUPRR', 'EP', 'EPR']):
        n_gene = len(gene_names)
        gene1 = [x[0] for x in ground_truth]
        gene2 = [x[1] for x in ground_truth]
        
        # TF is the set of genes appearing as regulators
        TF = set(gene1)
        # All_gene is the combined set of regulator and target genes
        All_gene = set(gene1) | set(gene2)
    
        tf_mask = (np.zeros(n_gene) == 1)
        # Not all provided genes are in the all_gene set
        gene_mask = (np.zeros(n_gene) == 1)
        tf_map = {}
        gene_map = {}
        for i, item in enumerate(gene_names):
            if item in TF:
                tf_mask[i] = True
                tf_map[item] = len(tf_map)
            if item in All_gene:
                gene_mask[i] = True
                gene_map[item] = len(gene_map)
        
        y_true = np.zeros([len(TF), len(All_gene)])
        for link in ground_truth:
            y_true[tf_map[link[0]], gene_map[link[1]]] = 1.0
        y_true = y_true.flatten()

        self.tf_mask = tf_mask
        self.gene_mask = gene_mask
        self.y_true = y_true
        self.y_true_mean = np.mean(y_true)
        self.num_true_edges = int(y_true.sum())
        self.ground_truth = ground_truth
        self.report_auroc = ('AUROC' in metrics)
        self.report_aupr = ('AUPR' in metrics)
        self.report_auprr = ('AUPRR' in metrics)
        self.report_ep = ('EP' in metrics)
        self.report_epr = ('EPR' in metrics)

    def evaluate(self, A):
        if A.shape[0] == A.shape[1]:
            A = A[self.tf_mask, :]
        A = A[:, self.gene_mask]
        y_pred = np.abs(A.flatten())

        eval_results = {}
        if self.report_auroc:
            eval_results['AUROC'] = roc_auc_score(self.y_true, y_pred)
        if self.report_aupr or self.report_auprr:
            eval_results['AUPR'] = average_precision_score(
                self.y_true, y_pred
            )
            eval_results['AUPRR'] = eval_results['AUPR'] / self.y_true_mean

        if self.report_ep or self.report_epr:
            cutoff = np.partition(
                y_pred, -self.num_true_edges
            )[-self.num_true_edges]
            y_above_cutoff = y_pred > cutoff
        
            eval_results['EP'] = int(np.sum(self.y_true[y_above_cutoff]))
            eval_results['EPR'] = 1. * eval_results['EP']
            eval_results['EPR'] /= ((self.num_true_edges ** 2) / len(y_pred))
        return eval_results

