import pyvis
import pandas as pd
from typing import List, Dict, Union

def plot_pyvis(
    pandas_edgelist: pd.DataFrame, star_genes: List[str] = [],
    node_size: int = 8, 
    font_size: int = 30,
    node_color_dict: Dict = None, 
    cdn_resources: str = 'remote', notebook: bool = True):
    """
    Generate a vis.js network visualization given an edge list 
    representation of a graph in data frame. 

    Args:
        pandas_edgelist (pd.DataFrame): an edge list representation of 
        a graph in a pandas data frame. It should at least have columns 
        with the name `source` and `target`. You can also (optionally) 
        provide an `edge_width` column for the width of the edges
        star_genes (List): A list of genes to be starred. 
        font_size (int): The font size for nodes labels.
        node_color_dict (dict): A dictionary with keys being the names of 
            genes and values being the color. 
        cdn_resources (str): Where to load vis.js resources. Default is
            'remote'.
        notebook (bool): Boolean value indicating whether the visualization
            happens in a jupyter notebook. 
    """    
    g = pyvis.network.Network(
        cdn_resources=cdn_resources, 
        notebook=notebook
    )

    star_genes = set(star_genes)
    
    for node in set(pandas_edgelist['source']) | set(pandas_edgelist['target']):
        if len(star_genes) == 0:
            node_shape = 'dot'
            this_node_size = node_size
        else:
            node_shape = 'star' if node in star_genes else 'dot'
            this_node_size = 2 * node_size if node in star_genes else node_size
        node_color = None if node_color_dict is None else node_color_dict[node]
        g.add_node(node, label=node, size=this_node_size, 
                   shape=node_shape, color=node_color, 
                   font={"size": font_size})

    if 'edge_width' in pandas_edgelist.columns:
        for _, row in pandas_edgelist.iterrows():
            g.add_edge(row['source'], row['target'], width=row['edge_width'])
    else:
        for _, row in pandas_edgelist.iterrows():
            g.add_edge(row['source'], row['target'])
    
    g.repulsion()
    return g