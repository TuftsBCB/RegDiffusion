import pandas as pd
from typing import List, Dict, Union, Optional

from lightgraph import net_vis, NetworkVisualization


def plot_lightgraph(
    pandas_edgelist: pd.DataFrame,
    star_genes: List[str] = [],
    node_size: float = 7,
    font_size: float = 5,
    node_color_dict: Optional[Dict[str, str]] = None,
    node_group_dict: Optional[Dict[str, str]] = None,
    node_metric: Optional[Dict[str, float]] = None,
    edge_weight_to_width: bool = True,
    edge_weight_to_opacity: bool = False,
    layout: str = 'force',
    show_arrows: bool = False,
    height: str = "600px",
    **kwargs
) -> NetworkVisualization:
    """
    Generate an interactive network visualization given an edge list
    representation of a graph in a data frame.

    Rendering is done by `lightgraph <https://haozhu233.github.io/lightgraph/>`_,
    which draws on an HTML canvas with viewport culling and therefore stays
    interactive on graphs with thousands of nodes.

    Args:
        pandas_edgelist (pd.DataFrame): an edge list representation of
            a graph in a pandas data frame. It should at least have columns
            with the name `source` and `target`. An optional `weight` column
            is used to scale edge width and/or opacity.
        star_genes (List): A list of genes to highlight. Highlighted genes are
            drawn at twice the base node size. Note that when
            ``node_group_dict`` is provided, group colors take precedence over
            per-node colors, so highlighting shows up as size only.
        node_size (float): Base size for nodes. Default is 7.
        font_size (float): The font size for node labels. Default is 5.
        node_color_dict (dict): A dictionary with keys being the names of
            genes and values being hex color strings.
        node_group_dict (dict): A dictionary with keys being the names of
            genes and values being the group. Pass the string ``'auto'`` to let
            lightgraph detect communities automatically.
        node_metric (dict): A dictionary with keys being the names of genes and
            values being a numeric score (for example degree or PageRank). The
            values are min-max normalized and mapped onto node size.
        edge_weight_to_width (bool): If True, scale edge width by the `weight`
            column. Default is True.
        edge_weight_to_opacity (bool): If True, scale edge opacity by the
            `weight` column. Default is False.
        layout (str): Layout algorithm, either 'force' or 'circular'.
            Default is 'force'.
        show_arrows (bool): Whether to draw directional arrows on edges.
            Default is False.
        height (str): Height of the visualization canvas. Default is "600px".
        **kwargs: Additional keyword arguments passed through to
            ``lightgraph.net_vis``.

    Returns:
        lightgraph.NetworkVisualization: A visualization object that renders
            itself in a Jupyter notebook. Call ``.save(path)`` to write a
            standalone HTML file or read ``.html`` for the raw markup.
    """
    if layout not in ('force', 'circular'):
        raise ValueError("layout must be either 'force' or 'circular'.")

    keep_cols = [
        c for c in ('source', 'target', 'weight') if c in pandas_edgelist
    ]
    edges = pandas_edgelist[keep_cols].copy()

    has_weight = 'weight' in keep_cols
    if has_weight:
        # RegDiffusion edge weights are signed and ranked by magnitude, so the
        # visual channels are driven by the absolute regulatory strength.
        edges['weight'] = edges['weight'].abs()
        edges = edges.sort_values('weight', ascending=False)

    # A multi-hop neighborhood walk reaches the same edge from either end, so
    # the same pair can show up more than once. Keep the strongest copy.
    edges = edges.drop_duplicates(subset=['source', 'target'])

    node_sizes = {g: 2 * node_size for g in set(star_genes)}

    params = dict(
        edges=edges,
        node_groups=node_group_dict,
        node_colors=node_color_dict,
        node_sizes=node_sizes or None,
        node_metric=node_metric,
        node_size=node_size,
        label_font_size=font_size,
        edge_weight_to_width=edge_weight_to_width and has_weight,
        edge_weight_to_opacity=edge_weight_to_opacity and has_weight,
        layout=layout,
        show_arrows=show_arrows,
        height=height,
    )
    # Anything passed through by name goes straight to lightgraph and wins, so
    # that its native argument names keep working alongside the aliases above.
    params.update(kwargs)
    return net_vis(**params)
