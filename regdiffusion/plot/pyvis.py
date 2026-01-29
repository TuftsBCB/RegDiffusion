import pyvis
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional

def plot_pyvis(
    pandas_edgelist: pd.DataFrame, 
    star_genes: List[str] = [],
    node_size: int = 8, 
    font_size: int = 30,
    node_color_dict: Optional[Dict[str, str]] = None, 
    node_group_dict: Optional[Dict[str, str]] = None,
    edge_color_by_weight: bool = False,
    layout: str = 'force',
    physics: bool = True,
    height: str = "600px",
    width: str = "100%",
    cdn_resources: str = 'remote', 
    notebook: bool = True
) -> pyvis.network.Network:
    """
    Generate a vis.js network visualization given an edge list 
    representation of a graph in data frame. 

    Args:
        pandas_edgelist (pd.DataFrame): an edge list representation of 
            a graph in a pandas data frame. It should at least have columns 
            with the name `source` and `target`. You can also (optionally) 
            provide an `edge_width` column for the width of the edges.
        star_genes (List): A list of genes to be starred. 
        node_size (int): Base size for nodes. Default is 8.
        font_size (int): The font size for nodes labels. Default is 30.
        node_color_dict (dict): A dictionary with keys being the names of 
            genes and values being the color. 
        node_group_dict (dict): A dictionary with keys being the names of 
            genes and values being the group. 
        edge_color_by_weight (bool): If True, color edges based on their 
            weight values using a diverging colormap. Default is False.
        layout (str): Layout algorithm. Currently supports 'force'. 
            Default is 'force'.
        physics (bool): Whether to enable physics simulation. Set to False 
            for static layout after initial positioning. Default is True.
        height (str): Height of the visualization canvas. Default is "600px".
        width (str): Width of the visualization canvas. Default is "100%".
        cdn_resources (str): Where to load vis.js resources. Default is
            'remote'.
        notebook (bool): Boolean value indicating whether the visualization
            happens in a jupyter notebook. Default is True.
            
    Returns:
        pyvis.network.Network: A PyVis network object that can be displayed
            or saved to HTML.
    """    
    g = pyvis.network.Network(
        cdn_resources=cdn_resources, 
        notebook=notebook,
        height=height,
        width=width
    )

    star_genes = set(star_genes)
    
    for node in set(pandas_edgelist['source']) | set(pandas_edgelist['target']):
        if len(star_genes) == 0:
            node_shape = 'dot'
            this_node_size = node_size
        else:
            node_shape = 'star' if node in star_genes else 'dot'
            this_node_size = 2 * node_size if node in star_genes else node_size
        node_color = None if node_color_dict is None else node_color_dict.get(node)
        node_group = None if node_group_dict is None else node_group_dict.get(node)
        g.add_node(node, label=node, size=this_node_size, 
                   shape=node_shape, color=node_color, group=node_group,
                   font={"size": font_size})

    # Handle edge coloring by weight
    if edge_color_by_weight and 'weight' in pandas_edgelist.columns:
        weights = pandas_edgelist['weight'].values
        abs_weights = np.abs(weights)
        max_weight = abs_weights.max() if abs_weights.max() > 0 else 1
        
        for _, row in pandas_edgelist.iterrows():
            weight = row['weight']
            # Normalize to [0, 1] based on sign and magnitude
            norm_weight = weight / max_weight
            # Red for positive, blue for negative
            if norm_weight >= 0:
                r = int(255 * norm_weight)
                color = f'rgb({r}, 100, 100)'
            else:
                b = int(255 * abs(norm_weight))
                color = f'rgb(100, 100, {b})'
            
            edge_width = row.get('edge_width', 1) if 'edge_width' in pandas_edgelist.columns else 1
            g.add_edge(row['source'], row['target'], width=edge_width, color=color)
    elif 'edge_width' in pandas_edgelist.columns:
        for _, row in pandas_edgelist.iterrows():
            g.add_edge(row['source'], row['target'], width=row['edge_width'])
    else:
        for _, row in pandas_edgelist.iterrows():
            g.add_edge(row['source'], row['target'])
    
    if physics:
        g.repulsion()
    else:
        g.toggle_physics(False)
        
    return g