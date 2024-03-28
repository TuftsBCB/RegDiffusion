import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from collections import deque, Counter
from scipy.sparse import csr_matrix
import h5py
import pyvis
from typing import List, Dict, Union

class GRN:
    """ 
    A Object to save and analyze gene regulatory network

    A GRN object includes the adjacency matrix between transcriptional factors 
    and target genes. In many cases, when TFs are not specified, we have a 
    square-shaped adjacency matrix. We expected the adjacency 
    matrix to hold predicted weights/probabilities for the edges (float). 

    To create a GRN object, you need at least two things: the adjacency matrix
    and the corresponding gene names. You can further specify the TF names if
    your adjacency matrix is not a square matrix. 

    You can save a GRN object to the HDF5 format using the `.to_hdf5` method in
    the GRN class. You can load a saved GRN object using the `read_hdf5` 
    function in this package. 

    If your adjacency matrix is very large and space is a concern, you may
    consider provide a value for `top_gene_percentile`. This value will 
    calculate the a cutoff point for the values in the adjacency matrix. 
    Every value whose absolute value is below this cutoff point will be set to 
    zero. Later on, we can save the data as a sparse matrix to reduce the 
    space requirement. 
    
    The GRN object comes with many useful methods to analyze and visualize the
    network. Top top-level interfaces includes `.extract_node_2hop_neighborhood`
    and `.visualize_local_neighborhood`. 

    Args:
        adj_matrix (np.ndarray): A 2D adjacency matrix to save.
        gene_names (np.ndarray): A 1D numpy array with all the target gene 
        names.
        tf_names (np.ndarray, optional): A 1D numpy array with all the TF gene 
        names.
        top_gene_percentile (int): If this value is set, only the top k absolute
        values in the adjacency matrix will be kept. All the other values
        will be set to zero. 
    """
    def __init__(self, adj_matrix: np.ndarray, 
                 gene_names: np.ndarray, 
                 tf_names: np.ndarray = None, 
                 top_gene_percentile: int = None):
        
        self.n_tfs = adj_matrix.shape[0]
        self.n_genes = adj_matrix.shape[1]
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

    def generate_adj_list(self) -> pd.DataFrame:
        """
        Simply generate a dataframe to hold the adjacency list.

        The dataframe will have 3 columns: `source`, `target`, `weight`. 
        """
        all_edges = []
        for r in tqdm(range(self.n_tfs)):
            for c in range(self.n_genes):
                if self.adj_matrix[r, c] != 0:
                    all_edges.append({
                        'source': self.tf_names[r],
                        'target': self.gene_names[c],
                        'weight': self.adj_matrix[r, c]
                    })
        return pd.DataFrame(all_edges)
                

    def extract_node_sources_as_indices(self, gene: str, k: int = 20) -> Dict:
        """
        Generate a dictionary for the top direct edge pointing to the 
        selected gene. It is slightly faster than the dataframe version. 

        Args:
            genes (str, List(str)): A single gene or a list of genes to inspect.
            k (int): Top-k edges to inspect on each node.
            node_size (int): The size of nodes in the visualization.
        """
        gene_idx = self.gene_indices[gene]
        node_neighbors = self.adj_matrix[:, gene_idx]
        node_neighbors_abs = np.abs(node_neighbors)
        if k!=-1:
            top_indices = np.argpartition(node_neighbors_abs, -k)[-k:]
        else:
            top_indices = np.where(node_neighbors_abs!=0)[0]
        top_edge_weights = node_neighbors[top_indices]
        output = {
            'tf_indices': top_indices, 
            'weights': top_edge_weights
        }
        return output

    def extract_node_sources(self, gene: str, k: int = 20) -> pd.DataFrame:
        """
        Generate a pandas dataframe for the top direct edge pointing to the 
        selected gene. 

        The dataframe will have 3 columns: `source`, `target`, `weight`. 

        Args:
            genes (str, List(str)): A single gene or a list of genes to inspect.
            k (int): Top-k edges to inspect on each node.
            node_size (int): The size of nodes in the visualization.
        """
        source_indices = self.extract_node_sources_as_indices(gene, k)
        top_gene_names = [
            self.tf_names[i] for i in source_indices['tf_indices']
        ]
        output = pd.DataFrame({
            'source': top_gene_names, 
            'target': gene, 
            'weight': source_indices['weights']
        })    
        return output

    def extract_node_targets_as_indices(self, gene: str, k: int = 20) -> Dict:
        """
        Generate a dictionary for the top direct edge pointing from the 
        selected gene. It is slightly faster than the dataframe version. 

        Args:
            genes (str, List(str)): A single gene or a list of genes to inspect.
            k (int): Top-k edges to inspect on each node.
            node_size (int): The size of nodes in the visualization.
        """
        gene_idx = self.tf_indices[gene]
        node_neighbors = self.adj_matrix[gene_idx, :]
        node_neighbors_abs = np.abs(node_neighbors)
        if k!=-1:
            top_indices = np.argpartition(node_neighbors_abs, -k)[-k:]
        else:
            top_indices = np.where(node_neighbors_abs!=0)[0]
        top_edge_weights = node_neighbors[top_indices]
        output = {
            'gene_indices': top_indices, 
            'weights': top_edge_weights
        }
        return output

    def extract_node_targets(self, gene: str, k: int = 20) -> pd.DataFrame:
        """
        Generate a pandas dataframe for the top direct edge pointing from the 
        selected gene. 

        The dataframe will have 3 columns: `source`, `target`, `weight`. 

        Args:
            genes (str, List(str)): A single gene or a list of genes to inspect.
            k (int): Top-k edges to inspect on each node.
            node_size (int): The size of nodes in the visualization.
        """
        target_indices = self.extract_node_targets_as_indices(gene, k)
        top_gene_names = [
            self.gene_names[i] for i in target_indices['gene_indices']
        ]
        output = pd.DataFrame({
            'source': gene,
            'target': top_gene_names, 
            'weight': target_indices['weights']
        })    
        return output

    def extract_node_neighbors_as_indices(self, gene: str, k: int = 20) -> Dict:
        """
        Generate a dictionary for the top direct neighbors of selected gene. 
        It is slightly faster than the dataframe version. 

        Args:
            genes (str, List(str)): A single gene or a list of genes to inspect.
            k (int): Top-k edges to inspect on each node.
            node_size (int): The size of nodes in the visualization.
        """
        sources = self.extract_node_sources_as_indices(gene, k)
        targets = self.extract_node_targets_as_indices(gene, k)
        all_weights = np.abs(np.concatenate([
            sources['weights'], targets['weights']
        ]))
        cutoff = np.partition(all_weights, -k)[-k]
        source_crit = (sources['weights'] >= cutoff)
        target_crit = (targets['weights'] >= cutoff)
        output = {
            'sources': {
                'tf_indices': sources['tf_indices'][source_crit],
                'weights': sources['weights'][source_crit]
            },
            'targets': {
                'gene_indices': targets['gene_indices'][target_crit],
                'weights': targets['weights'][target_crit]
            }
        }
        return output

    def extract_node_neighbors(self, gene: str, k: int = 20) -> pd.DataFrame:
        """
        Generate a pandas dataframe for the top direct neighbors of selected 
        gene. The dataframe will be sorted by the absolute weight of edges. 

        The dataframe will have 3 columns: `source`, `target`, `weight`. 

        Args:
            genes (str, List(str)): A single gene or a list of genes to inspect.
            k (int): Top-k edges to inspect on each node.
            node_size (int): The size of nodes in the visualization.
        """
        top_sources = self.extract_node_sources(gene, k)
        top_targets = self.extract_node_targets(gene, k)

        output = pd.concat([top_sources, top_targets])
        output['abs_weight'] = output.weight.abs()
        output = output.sort_values(
            'abs_weight', ascending=False
        ).head(k).reset_index(drop=True)
        del output['abs_weight']
        return output

    def extract_node_2hop_neighborhood(
        self, genes: Union[str, List[str]], k: int = 20
    ) -> pd.DataFrame:
        """
        Generate a pandas dataframe for the local neighborhood (2-hop) around 
        selected gene(s). 

        Args:
            genes (str, List(str)): A single gene or a list of genes to inspect.
            k (int): Top-k edges to inspect on each node.
            node_size (int): The size of nodes in the visualization.
        """
        if isinstance(genes, str):
            genes = [genes]
        hop0_genes = set(genes)
        hop1 = pd.concat([
            self.extract_node_neighbors(g, k=k) for g in hop0_genes
        ])
        hop1['weight'] = 0
        hop1_genes = set()
        for g in hop1.source:
            if g not in hop0_genes:
                hop1_genes.add(g)
        for g in hop1.target:
            if g not in hop0_genes:
                hop1_genes.add(g)
        hop2s = pd.concat([
            self.extract_node_neighbors(g, k=k) for g in hop1_genes
        ])
        hop2s['weight'] = 1
        hop2_genes = set()
        for g in hop2s.source:
            if g not in hop0_genes and g not in hop1_genes:
                hop2_genes.add(g)
        for g in hop2s.target:
            if g not in hop0_genes and g not in hop1_genes:
                hop2_genes.add(g)
        hop3s = []
        for g in hop2_genes:
            hop3 = self.extract_node_neighbors(g, k=k)
            hop3 = hop3[
                hop3.source.isin(hop2_genes) & hop3.target.isin(hop2_genes)
            ]
            hop3s.append(hop3)
        hop3s = pd.concat(hop3s)
        hop3s['weight'] = 2
        adj_table = pd.concat([hop1, hop2s, hop3s]).reset_index(drop=True)
        return adj_table

    def visualize_local_neighborhood(
        self, genes: Union[str, List[str]], k: int = 20, 
        node_size: int = 8, edge_widths: List[int] = [2, 1, 0.5], 
        font_size: int = 30,
        node_group_dict: Dict = None, 
        cdn_resources: str = 'remote', notebook: bool = True):
        """
        Generate a vis.js network visualization of the local neighborhood 
        (2-hop) around selected gene(s). 

        Args:
            genes (str, List(str)): A single gene or a list of genes to inspect.
            k (int): Top-k edges to inspect on each node.
            node_size (int): The size of nodes in the visualization.
            edge_widths (List): The widths for edges (1st, 2nd, between 2nd 
                hops).
            font_size (int): The font size for nodes labels.
            node_group_dict (dict): A dictionary with keys being the names of 
                genes and values being the groups. Genes from the same group 
                will be colored using the same color.
            cdn_resources (str): Where to load vis.js resources. Default is
                'remote'.
            notebook (bool): Boolean value indicating whether the visualization
                happens in a jupyter notebook. 
        """
        local_adj_table = self.extract_node_2hop_neighborhood(genes, k)
        local_adj_table.weight = local_adj_table.weight.map(
            lambda x: edge_widths[x])
        
        g = pyvis.network.Network(
            cdn_resources=cdn_resources, 
            notebook=notebook
        )
        
        for node in set(local_adj_table['source']) | set(local_adj_table['target']):
            node_shape = 'star' if node in genes else 'dot'
            node_group = None if node_group_dict is None else node_group_dict[node]
            g.add_node(node, label=node, size=node_size, 
                       shape=node_shape, group=node_group, 
                       font={"size": font_size})
        
        for _, row in local_adj_table.iterrows():
            g.add_edge(row['source'], row['target'], width=row['weight'])
        
        g.repulsion()
        return g

    def to_hdf5(self, file_path: str, as_sparse: bool = False):
        """
        Save GRN into a HDF5 file. You have the option to save as a sparse 
        matrix. This option is preferred when most of the values in the 
        adjacency matrix are zeros.
    
        Args:
            file_path (str): File path to save. 
            as_sparse (bool): Whether to save as sparse matrix
        """
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

def read_hdf5(file_path: str):
    """
    Read HDF5 file as a GRN object. See the documentation for GRN for details.

    Args:
        file_path (str): File path to read. 
    """
    with h5py.File(file_path, 'r') as f:
        if f['adj_matrix'].attrs['sparse']:
            sparse_dt = csr_matrix((
                f['adj_matrix']['data'][:].astype(np.float32), 
                f['adj_matrix']['indices'][:], 
                f['adj_matrix']['indptr'][:]
            ), shape=f['adj_matrix'].attrs['shape'])
            adj_matrix = sparse_dt.toarray().astype(np.float16)
        else:
            adj_matrix = f['adj_matrix']['data'][:]
        gene_names = f['gene_names'][:]
        tf_names = f['tf_names'][:]
        return GRN(adj_matrix, gene_names, tf_names)    