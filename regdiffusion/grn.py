import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from collections import deque, Counter
from scipy.sparse import csr_matrix
import h5py
from typing import List, Dict, Union
import concurrent.futures
from .plot import plot_pyvis

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
        # TODO: make it tf safe. 
        if self.n_tfs == self.n_genes:
            self.adj_matrix_2way = np.concatenate([
                adj_matrix, adj_matrix.transpose()
            ], axis=0)
        else:
            self.adj_matrix_2way = None

        self.gene_indices = {g: i for i, g in enumerate(gene_names)}
        self.tf_indices = {g: i for i, g in enumerate(tf_names)}

        self.calculated_neighbors = {}

    def get_edgelist(self, k: int = 20, workers: int = 2) -> pd.DataFrame:
        """
        Simply generate a dataframe to hold the edge list.

        The dataframe will have 3 columns: `source`, `target`, `weight`. 

        Args:
            k (int): Top-k edges to inspect on each node. If k=-1, export all.
            workers (int): Number of concurrent workers. Default is 2. 
        """
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers) as executor:
            futures = [
                executor.submit(
                    self.extract_node_neighbors, g, k
                ) for g in list(self.gene_names)]
        
            all_edges = [
                future.result() for future in 
                concurrent.futures.as_completed(futures)]
        return pd.concat(all_edges).reset_index(drop=True)
                

    def extract_node_sources_as_indices(self, gene: str, k: int = 20) -> Dict:
        """
        Generate a dictionary for the top direct edge pointing to the 
        selected gene. It is slightly faster than the dataframe version. 

        Args:
            genes (str, List(str)): A single gene or a list of genes to inspect.
            k (int): Top-k edges to inspect on each node. If k=-1, export all
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
            k (int): Top-k edges to inspect on each node. If k=-1, export all
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
            k (int): Top-k edges to inspect on each node. If k=-1, export all
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
            k (int): Top-k edges to inspect on each node. If k=-1, export all
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
            k (int): Top-k edges to inspect on each node. If k=-1, export all
        """
        gene_idx = self.gene_indices[gene]
        if (gene_idx, k) in self.calculated_neighbors:
            return self.calculated_neighbors[(gene_idx, k)]
        node_neighbors = self.adj_matrix_2way[:, gene_idx]
        node_neighbors_abs = np.abs(node_neighbors)
        if k!=-1:
            top_indices = np.argpartition(node_neighbors_abs, -k)[-k:]
        else:
            top_indices = np.where(node_neighbors_abs!=0)[0]
        top_edge_weights = node_neighbors[top_indices]
        top_source_indices = top_indices[top_indices < self.n_tfs]
        top_source_weights = top_edge_weights[top_indices < self.n_tfs]
        top_target_indices = top_indices[top_indices >= self.n_tfs] - self.n_tfs
        top_target_weights = top_edge_weights[top_indices >= self.n_tfs]
        output = {
            'source_indices': top_source_indices, 
            'source_weights': top_source_weights,
            'target_indices': top_target_indices, 
            'target_weights': top_target_weights
        }
        self.calculated_neighbors[(gene_idx, k)] = output
        return output

    def extract_node_neighbors(self, gene: str, k: int = 20) -> pd.DataFrame:
        """
        Generate a pandas dataframe for the top direct neighbors of selected 
        gene. The dataframe will be sorted by the absolute weight of edges. 

        The dataframe will have 3 columns: `source`, `target`, `weight`. 

        Args:
            genes (str, List(str)): A single gene or a list of genes to inspect.
            k (int): Top-k edges to inspect on each node. If k=-1, export all
        """
        neighbor_indices = self.extract_node_neighbors_as_indices(gene, k)
        source_gene_names = [
            self.tf_names[i] for i in neighbor_indices['source_indices']
        ]
        source_tbl = pd.DataFrame({
            'source': source_gene_names, 
            'target': gene, 
            'weight': neighbor_indices['source_weights']
        })
        target_gene_names = [
            self.gene_names[i] for i in neighbor_indices['target_indices']
        ]
        target_tbl = pd.DataFrame({
            'source': gene,
            'target': target_gene_names, 
            'weight': neighbor_indices['target_weights']
        })    
        return pd.concat([source_tbl, target_tbl])

    def extract_local_neighborhood(
        self, genes: Union[str, List[str]], k: int = 20, hops: str = "2.5"
    ) -> pd.DataFrame:
        """
        Generate a pandas dataframe for the 2.5 or 1.5 hop local neighborhood 
        around selected gene(s). "2.5 hop local neighborhood" includes all the 
        nodes and edges reachable by a 2-hop search from the selected genes and 
        the edges connecting all the 2-hop nodes. "1.5 hop local neighborhood"
        is defined in a similar way but smaller. 

        Args:
            genes (str, List(str)): A single gene or a list of genes to inspect.
            k (int): Top-k edges to inspect on each node. If k=-1, export all
            hops (str): Number of hops to explore. We can either do a "2.5" or 
            "1.5" hop travesal around selected genes. Default is "2.5". 
        """
        if isinstance(genes, str):
            genes = [genes]
        hop0_genes = set(genes)

        # Hop 1
        hop1 = pd.concat([
            self.extract_node_neighbors(g, k=k) for g in hop0_genes
        ])
        hop1['hop'] = 0
        hop1_genes = set()
        for g in hop1.source:
            if g not in hop0_genes:
                hop1_genes.add(g)
        for g in hop1.target:
            if g not in hop0_genes:
                hop1_genes.add(g)

        # Hop 2
        if hops == "1.5":
            hop2s = []
            for g in hop1_genes:
                hop2 = self.extract_node_neighbors(g, k=k)
                hop2 = hop2[
                    hop2.source.isin(hop1_genes) & hop2.target.isin(hop1_genes)
                ]
                hop2s.append(hop2)
            hop2s = pd.concat(hop2s)
            hop2s['hop'] = 1
            adj_table = pd.concat([hop1, hop2s]).reset_index(drop=True)
            return adj_table
        elif hops == "2.5":   
            hop2s = pd.concat([
                self.extract_node_neighbors(g, k=k) for g in hop1_genes
            ])
            hop2s['hop'] = 1
            hop2_genes = set()
            for g in hop2s.source:
                if g not in hop0_genes and g not in hop1_genes:
                    hop2_genes.add(g)
            for g in hop2s.target:
                if g not in hop0_genes and g not in hop1_genes:
                    hop2_genes.add(g)
    
            # Hop 2.5
            hop3s = []
            for g in hop2_genes:
                hop3 = self.extract_node_neighbors(g, k=k)
                hop3 = hop3[
                    hop3.source.isin(hop2_genes) & hop3.target.isin(hop2_genes)
                ]
                hop3s.append(hop3)
            hop3s = pd.concat(hop3s)
            hop3s['hop'] = 2
            adj_table = pd.concat([hop1, hop2s, hop3s]).reset_index(drop=True)
            return adj_table

    def visualize_local_neighborhood(
        self, genes: Union[str, List[str]], k: int = 20, hops: str = "2.5",
        edge_widths: List[int] = [2, 1, 0.5], 
        plot_engine: str = 'pyvis', *args, **kwargs):
        """
        Generate a vis.js network visualization of the local neighborhood 
        (2-hop) around selected gene(s). 

        Args:
            genes (str, List(str)): A single gene or a list of genes to inspect.
            k (int): Top-k edges to inspect on each node. If k=-1, export all.
            hops (str): Number of hops of the neighborhood to explore. Default
            is "2.5". 
            edge_widths (List): The widths for edges for different edge width
            levels.
            plot_engine (str): Choose which network plot engine to use. Default
            is "pyvis". 
            **kwargs: Keyword arguments to be passed to ``plot_pyvis``.
        """
        if isinstance(genes, str):
            genes = [genes]
        local_adj_table = self.extract_local_neighborhood(genes, k, hops)
        local_adj_table['edge_width'] = local_adj_table.hop.map(
            lambda x: edge_widths[x])

        if plot_engine == 'pyvis':
            g = plot_pyvis(
                pandas_edgelist = local_adj_table, 
                star_genes = genes, *args, **kwargs)
        else: 
            raise Exception("Not implemented yet")
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