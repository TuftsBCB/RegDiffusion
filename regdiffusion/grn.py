import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from collections import deque, Counter
from scipy.sparse import csr_matrix
import h5py
import pyvis

class GRN:
    def __init__(self, adj_matrix=None, gene_names=None, tf_names=None, 
                 top_gene_percentile=None):
        if adj_matrix is None:
            raise Exception(
                "You need to at least provide the inferred adjacency matrix. "
            )

        self.n_tfs = adj_matrix.shape[0]
        self.n_genes = adj_matrix.shape[1]
        if gene_names is None:
            gene_names = np.arange(self.n_genes).astype(str)
        if tf_names is None:
            tf_names = gene_names
        self.gene_names = gene_names
        self.tf_names = tf_names

        if top_gene_percentile is None:
            self.cutoff_threshold = 0
        else:
            # Here we are estimating the cutoff point for top a% predicted edges
            # To speed up the process, we calculate the 1-a% percentile within 
            # 10,000 sampled edges instead of all edges. 
            random_row_idx = np.random.randint(0, self.n_tfs, 10000)
            random_col_idx = np.random.randint(0, self.n_genes, 10000)
            sampled_values = adj_matrix[random_row_idx, random_col_idx]
            self.cutoff_threshold = np.percentile(
                np.abs(sampled_values), 100-top_gene_percentile
            )
            adj_matrix = adj_matrix.copy()
            adj_matrix[np.abs(adj_matrix) < self.cutoff_threshold] = 0

        self.adj_matrix = adj_matrix

        self.gene_indices = {g: i for i, g in enumerate(gene_names)}
        self.tf_indices = {g: i for i, g in enumerate(tf_names)}

    def generate_adj_list(self, k=None):
        if k is None:
            k = self.n_genes - 1
        all_edges = []
        for r in tqdm(range(self.n_tfs)):
            node_neighbors = self.inferred_adj[r, :]
            for i in range(len(node_neighbors.indices)):
                all_edges.append({
                    'source': self.tf_names[r],
                    'target': self.gene_names[node_neighbors.indices[i]],
                    'weight': node_neighbors.data[i]
                })
        return pd.DataFrame(all_edges)
                

    def extract_node_sources(self, gene, k=20, return_raw_indices=False):
        gene_idx = self.gene_indices[gene]
        node_neighbors = self.adj_matrix[:, gene_idx]
        node_neighbors_abs = np.abs(node_neighbors)
        if k!=-1:
            top_indices = np.argpartition(node_neighbors_abs, -k)[-k:]
        else:
            top_indices = np.where(node_neighbors_abs!=0)[0]
        top_edge_weights = node_neighbors[top_indices]
        if return_raw_indices:
            output = {
                'tf_indices': top_indices, 
                'weights': top_edge_weights
            }
            return output
        top_gene_names = [self.tf_names[i] for i in top_indices]
        output = pd.DataFrame({
            'source': top_gene_names, 
            'target': gene, 
            'weight': top_edge_weights
        })    
        return output

    def extract_node_targets(self, gene, k=20, return_raw_indices=False):
        gene_idx = self.tf_indices[gene]
        node_neighbors = self.adj_matrix[gene_idx, :]
        node_neighbors_abs = np.abs(node_neighbors)
        if k!=-1:
            top_indices = np.argpartition(node_neighbors_abs, -k)[-k:]
        else:
            top_indices = np.where(node_neighbors_abs!=0)[0]
        top_edge_weights = node_neighbors[top_indices]
        if return_raw_indices:
            output = {
                'gene_indices': top_indices, 
                'weights': top_edge_weights
            }
            return output
        top_gene_names = [self.gene_names[i] for i in top_indices]
        output = pd.DataFrame({
            'source': gene,
            'target': top_gene_names, 
            'weight': top_edge_weights
        })    
        return output

    def extract_node_neighbors(self, gene, k=20, return_raw_indices=False):
        if return_raw_indices:
            sources = self.extract_node_sources(
                gene, k, return_raw_indices=True)
            targets = self.extract_node_targets(
                gene, k, return_raw_indices=True)
            all_weights = np.abs(np.concatenate([
                sources['weights'], targets['weights']
            ]))
            cutoff = np.partition(all_weights, -k)[-k]
            output = {
                'sources': {
                    'tf_indices': sources['tf_indices'][sources['weights'] >= cutoff],
                    'weights': sources['weights'][sources['weights'] >= cutoff]
                },
                'targets': {
                    'gene_indices': targets['gene_indices'][targets['weights'] >= cutoff],
                    'weights': targets['weights'][targets['weights'] >= cutoff]
                }
            }
            return output
        else:
            top_sources = self.extract_node_sources(gene, k)
            top_targets = self.extract_node_targets(gene, k)
    
            output = pd.concat([top_sources, top_targets])
            output['abs_weight'] = output.weight.abs()
            output = output.sort_values(
                'abs_weight', ascending=False
            ).head(k).reset_index(drop=True)
            del output['abs_weight']
            return output

    def extract_node_2hop_neighborhood(self, gene, k=20):
        hop1 = self.extract_node_neighbors(gene, k=k)
        hop1['weight'] = 0
        hop1_genes = set()
        for g in hop1.source:
            if g != gene:
                hop1_genes.add(g)
        for g in hop1.target:
            if g != gene:
                hop1_genes.add(g)
        hop2s = pd.concat([self.extract_node_neighbors(g, k=k) for g in hop1_genes])
        hop2s['weight'] = 1
        hop2_genes = set()
        for g in hop2s.source:
            if g != gene and g not in hop1_genes:
                hop2_genes.add(g)
        for g in hop2s.target:
            if g != gene and g not in hop1_genes:
                hop2_genes.add(g)
        hop3s = []
        for g in hop2_genes:
            hop3 = self.extract_node_neighbors(g, k=k)
            hop3 = hop3[hop3.source.isin(hop2_genes) & hop3.target.isin(hop2_genes) ]
            hop3s.append(hop3)
        hop3s = pd.concat(hop3s)
        hop3s['weight'] = 2
        adj_table = pd.concat([hop1, hop2s, hop3s]).reset_index(drop=True)
        return adj_table

    def visualize_local_neighborhood(
        self, gene, k=20, node_size=8, edge_widths=[2, 1, 0.5], 
        node_group_dict=None, cdn_resources='remote'):
        local_adj_table = self.extract_node_2hop_neighborhood(gene, k)
        local_adj_table.weight = local_adj_table.weight.map(
            lambda x: edge_widths[x])
        
        g = pyvis.network.Network(cdn_resources=cdn_resources)
        
        for node in set(local_adj_table['source']) | set(local_adj_table['target']):
            node_shape = 'star' if node == gene else 'dot'
            node_group = None if node_group_dict is None else node_group_dict[node]
            g.add_node(node, label=node, size=node_size, 
                       shape=node_shape, group=node_group)
        
        for _, row in local_adj_table.iterrows():
            g.add_edge(row['source'], row['target'], width=row['weight'])
        
        g.repulsion()
        return g

    def to_hdf5(self, file_path, as_sparse=False):
        if as_sparse:
            sp_adj = csr_matrix(self.adj_matrix)
        if not file_path.endswith('.hdf5'):
            file_path += '.hdf5'
        with h5py.File(file_path, 'w') as f:
            adj_group = f.create_group('adj_matrix')
            if as_sparse:
                adj_group.attrs['sparse'] = True
                adj_group.attrs['shape'] = sp_adj.shape
                adj_group.create_dataset(
                    'data', data=sp_adj.data.astype(np.float16), chunks=True, 
                    compression="gzip", compression_opts=9
                )
                adj_group.create_dataset(
                    'indices', data=sp_adj.indices, chunks=True, 
                    compression="gzip", compression_opts=9
                )
                adj_group.create_dataset(
                    'indptr', data=sp_adj.indptr, chunks=True, 
                    compression="gzip", compression_opts=9
                )
            else:
                adj_group.attrs['sparse'] = False
                adj_group.create_dataset(
                    'data', data=self.adj_matrix, chunks=True, 
                    compression="gzip", compression_opts=9
                )
            f.create_dataset(
                'gene_names', data=list(self.gene_names), chunks=True, 
                compression="gzip", compression_opts=9
            )
            f.create_dataset(
                'tf_names', data=list(self.tf_names), chunks=True, 
                compression="gzip", compression_opts=9
            )
            
    def __repr__(self):
        result_description_header = "Inferred GRN"
        size_description = f"{'{:,}'.format(self.n_tfs)} TFs x "
        size_description += f"{'{:,}'.format(self.n_genes)} Target Genes"
        
        message = f"{result_description_header}: {size_description}"
        return message
        
    def __str__(self):
        return self.__repr__()

# def load
