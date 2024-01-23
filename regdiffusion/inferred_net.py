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

class Inferred_GRN:
    def __init__(self, inferred_adj=None, gene_names=None, tf_names=None, 
                 top_gene_percentile=None, n_cells=None, logger=None, 
                 training_losses=None, time_cost=None,training_method=None):
        if inferred_adj is None:
            raise Exception(
                "You need to at least provide  the inferred adjacency matrix. "
            )

        self.n_tfs = inferred_adj.shape[0]
        self.n_genes = inferred_adj.shape[1]
        if gene_names is None:
            gene_names = np.arange(self.n_genes).astype(str)
        if tf_names is None:
            tf_names = gene_names
        self.gene_names = gene_names
        self.tf_names = tf_names

        if top_gene_percentile is None:
            self.cutoff_threshold = 0
        else:
            # Here we are estimating the cutoff point for top 5% predicted edges
            # To speed up the process, we calculate the 95% percentile within 
            # 10,000 sampled edges instead of all edges. 
            random_row_idx = np.random.randint(0, self.n_tfs, 10000)
            random_col_idx = np.random.randint(0, self.n_genes, 10000)
            sampled_values = inferred_adj[random_row_idx, random_col_idx]
            self.cutoff_threshold = np.percentile(
                np.abs(sampled_values), 100-top_gene_percentile
            )
            inferred_adj = inferred_adj.copy()
            inferred_adj[np.abs(inferred_adj) < self.cutoff_threshold] = 0

        self.inferred_adj = inferred_adj

        self.gene_indices = {g: i for i, g in enumerate(gene_names)}
        self.tf_indices = {g: i for i, g in enumerate(tf_names)}
        
        self.n_cells=n_cells
        self.logger=logger
        self.training_losses=training_losses
        self.time_cost=time_cost
        self.training_method=training_method

    def generate_adj_list(self, k=None):
        if k is None:
            k = self.n_genes - 1
        all_edges = []
        for r in tqdm(range(self.n_tfs)):
            node_neighbors = self.inferred_adj[r, :]
            for i in range(len(node_neighbors.indices)):
                all_edges.append({
                    'Source': self.tf_names[r],
                    'Target': self.gene_names[node_neighbors.indices[i]],
                    'Weight': node_neighbors.data[i]
                })
        return pd.DataFrame(all_edges)
                

    def extract_node_sources(self, gene, k=20):
        gene_idx = self.gene_indices[gene]
        node_neighbors = self.inferred_adj[:, gene_idx]
        node_neighbors_abs = np.abs(node_neighbors)
        top_indices = np.argpartition(node_neighbors_abs, -k)[-k:]
        top_gene_names = [self.gene_names[i] for i in top_indices]
        top_edge_weights = node_neighbors[top_indices]
        output = pd.DataFrame({
            'Source': top_gene_names, 
            'Target': gene, 
            'Weight': top_edge_weights
        })    
        return output

    def extract_node_targets(self, gene, k=20):
        gene_idx = self.tf_indices[gene]
        node_neighbors = self.inferred_adj[gene_idx, :]
        node_neighbors_abs = np.abs(node_neighbors)
        top_indices = np.argpartition(node_neighbors_abs, -k)[-k:]
        top_gene_names = [self.gene_names[i] for i in top_indices]
        top_edge_weights = node_neighbors[top_indices]
        output = pd.DataFrame({
            'Source': gene,
            'Target': top_gene_names, 
            'Weight': top_edge_weights
        })    
        return output

    def extract_node_neighbors(self, gene, k=20):
        top_sources = self.extract_node_sources(gene, k)
        top_targets = self.extract_node_targets(gene, k)

        output = pd.concat([top_sources, top_targets])
        output['abs_weight'] = output.Weight.abs()
        output = output.sort_values(
            'abs_weight', ascending=False
        ).head(k).reset_index(drop=True)
        del output['abs_weight']
        return output

    def extract_node_neighborhood(self, gene, k=20, hop=2):
        # BFS
        visited = set()
        travelled_pathes = set()
        output = []
        queue = deque([(gene, 0)])
        while queue:
            node, h = queue.popleft()
            if h >= hop:
                break
                
            node_neighbors = self.extract_node_neighbors(node, k)
            for i, r in node_neighbors.iterrows():
                edge_name = f"{r['Source']}--{r['Target']}"
                if edge_name not in travelled_pathes:
                    output.append({
                        'source': r['Source'],
                        'target': r['Target'],
                        'weight': hop+1-h,
                        'score': r['Weight']
                    })
                    travelled_pathes.add(edge_name)
                    if r['Source'] not in visited:
                        visited.add(r['Source'])
                        queue.append((r['Source'], h+1))
                    if r['Target'] not in visited:
                        visited.add(r['Target'])
                        queue.append((r['Target'], h+1))
                    
        return pd.DataFrame(output)

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
            
    def __repr__(self):
        result_description_header = "Inferred GRN"
        size_description = f"{'{:,}'.format(self.n_tfs)} TFs x "
        size_description += f"{'{:,}'.format(self.n_genes)} Target Genes"
        
        message = f"{result_description_header}: {size_description}"
        return message
        
    def __str__(self):
        return self.__repr__()
