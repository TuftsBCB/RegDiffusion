"""
Static visualization functions using matplotlib and networkx.
These are useful for publication-quality figures.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..grn import GRN

def plot_network(
    edgelist: pd.DataFrame,
    star_genes: List[str] = [],
    figsize: Tuple[int, int] = (10, 10),
    node_size: int = 300,
    layout: str = 'spring',
    edge_width_scale: float = 1.0,
    node_color: str = 'lightblue',
    star_color: str = 'red',
    edge_color: str = 'gray',
    edge_alpha: float = 0.6,
    with_labels: bool = True,
    font_size: int = 8,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generate a static network visualization using matplotlib and networkx.
    Useful for publication-quality figures.
    
    Args:
        edgelist (pd.DataFrame): Edge list with 'source', 'target', and 
            optionally 'weight' columns.
        star_genes (List[str]): List of genes to highlight.
        figsize (tuple): Figure size. Default is (10, 10).
        node_size (int): Base node size. Default is 300.
        layout (str): Layout algorithm. Options: 'spring', 'circular', 
            'kamada_kawai', 'shell', 'spectral'. Default is 'spring'.
        edge_width_scale (float): Scale factor for edge widths. Default is 1.0.
        node_color (str): Color for regular nodes. Default is 'lightblue'.
        star_color (str): Color for starred nodes. Default is 'red'.
        edge_color (str): Color for edges. Default is 'gray'.
        edge_alpha (float): Transparency for edges. Default is 0.6.
        with_labels (bool): Whether to show node labels. Default is True.
        font_size (int): Font size for labels. Default is 8.
        title (str): Optional title for the plot.
        save_path (str): Optional path to save the figure.
        ax (plt.Axes): Optional axes to plot on.
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes objects.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required for static plotting. "
                          "Install with: pip install networkx")
    
    # Build graph
    G = nx.from_pandas_edgelist(
        edgelist, 'source', 'target', 
        edge_attr='weight' if 'weight' in edgelist.columns else None,
        create_using=nx.DiGraph()
    )
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Choose layout
    layout_funcs = {
        'spring': lambda g: nx.spring_layout(g, k=2, iterations=50, seed=42),
        'circular': nx.circular_layout,
        'kamada_kawai': nx.kamada_kawai_layout,
        'shell': nx.shell_layout,
        'spectral': nx.spectral_layout
    }
    
    if layout not in layout_funcs:
        raise ValueError(f"Unknown layout: {layout}. "
                         f"Choose from: {list(layout_funcs.keys())}")
    
    pos = layout_funcs[layout](G)
    
    # Node styling
    star_genes_set = set(star_genes)
    node_colors = [star_color if n in star_genes_set else node_color 
                   for n in G.nodes()]
    node_sizes = [node_size * 2 if n in star_genes_set else node_size 
                  for n in G.nodes()]
    
    # Edge widths
    if 'weight' in edgelist.columns:
        edge_weights = [abs(G[u][v].get('weight', 1)) for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [w / max_weight * 3 * edge_width_scale for w in edge_weights]
    else:
        edge_widths = [1 * edge_width_scale] * G.number_of_edges()
    
    # Draw
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                           node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color,
                           width=edge_widths, alpha=edge_alpha,
                           arrows=True, arrowsize=15,
                           connectionstyle="arc3,rad=0.1")
    
    if with_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size)
    
    if title:
        ax.set_title(title, fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_heatmap(
    grn: 'GRN',
    genes: Optional[List[str]] = None,
    top_k: int = 50,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'RdBu_r',
    center: float = 0,
    title: str = 'Gene Regulatory Network Heatmap',
    save_path: Optional[str] = None,
    show_values: bool = False,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot adjacency matrix as a heatmap.
    
    Args:
        grn (GRN): GRN object containing the adjacency matrix.
        genes (List[str]): Optional list of genes to include. If None, 
            selects top-k genes by regulatory strength.
        top_k (int): Number of top genes to show if genes is None. Default is 50.
        figsize (tuple): Figure size. Default is (12, 10).
        cmap (str): Colormap name. Default is 'RdBu_r'.
        center (float): Value to center the colormap at. Default is 0.
        title (str): Title for the plot.
        save_path (str): Optional path to save the figure.
        show_values (bool): Whether to annotate cells with values. Default is False.
        ax (plt.Axes): Optional axes to plot on.
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes objects.
    """
    try:
        import seaborn as sns
    except ImportError:
        raise ImportError("seaborn is required for heatmap plotting. "
                          "Install with: pip install seaborn")
    
    adj_matrix = grn.adj_matrix
    
    if genes is None:
        # Select top-k genes by total regulatory strength
        gene_strength = np.abs(adj_matrix).sum(axis=0) + np.abs(adj_matrix).sum(axis=1)
        top_indices = np.argsort(gene_strength)[-top_k:]
        genes = [grn.gene_names[i] for i in top_indices]
    
    # Get indices for selected genes
    gene_indices = [grn.gene_indices[g] for g in genes if g in grn.gene_indices]
    sub_matrix = adj_matrix[np.ix_(gene_indices, gene_indices)]
    selected_genes = [genes[i] for i in range(len(genes)) if genes[i] in grn.gene_indices]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    sns.heatmap(sub_matrix, 
                xticklabels=selected_genes, 
                yticklabels=selected_genes,
                cmap=cmap, 
                center=center, 
                ax=ax,
                annot=show_values,
                fmt='.2f' if show_values else None,
                square=True)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Target Genes', fontsize=12)
    ax.set_ylabel('Regulatory TFs', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_degree_distribution(
    grn: 'GRN',
    direction: str = 'both',
    bins: int = 50,
    figsize: Tuple[int, int] = (12, 4),
    log_scale: bool = False,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, Any]:
    """
    Plot the degree distribution of the GRN.
    
    Args:
        grn (GRN): GRN object containing the adjacency matrix.
        direction (str): 'in' (most regulated), 'out' (most regulating), 
            or 'both'. Default is 'both'.
        bins (int): Number of histogram bins. Default is 50.
        figsize (tuple): Figure size. Default is (12, 4).
        log_scale (bool): Whether to use log scale for y-axis. Default is False.
        title (str): Optional overall title.
        save_path (str): Optional path to save the figure.
        
    Returns:
        Tuple[plt.Figure, Any]: The figure and axes objects.
    """
    adj_matrix = np.abs(grn.adj_matrix)
    
    n_plots = 2 if direction == 'both' else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if direction != 'both':
        axes = [axes]
    
    plot_idx = 0
    
    if direction in ['in', 'both']:
        in_degrees = adj_matrix.sum(axis=0)
        ax = axes[plot_idx]
        ax.hist(in_degrees, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel('In-Degree (Regulatory Input)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('In-Degree Distribution', fontsize=12)
        if log_scale:
            ax.set_yscale('log')
        
        # Add statistics
        ax.axvline(np.mean(in_degrees), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(in_degrees):.2f}')
        ax.axvline(np.median(in_degrees), color='orange', linestyle='--',
                   label=f'Median: {np.median(in_degrees):.2f}')
        ax.legend()
        plot_idx += 1
    
    if direction in ['out', 'both']:
        out_degrees = adj_matrix.sum(axis=1)
        ax = axes[plot_idx]
        ax.hist(out_degrees, bins=bins, edgecolor='black', alpha=0.7, color='coral')
        ax.set_xlabel('Out-Degree (Regulatory Output)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Out-Degree Distribution', fontsize=12)
        if log_scale:
            ax.set_yscale('log')
        
        # Add statistics
        ax.axvline(np.mean(out_degrees), color='red', linestyle='--',
                   label=f'Mean: {np.mean(out_degrees):.2f}')
        ax.axvline(np.median(out_degrees), color='orange', linestyle='--',
                   label=f'Median: {np.median(out_degrees):.2f}')
        ax.legend()
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_training_curves(
    log_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, Any]:
    """
    Plot training curves from logger DataFrame.
    
    Args:
        log_df (pd.DataFrame): DataFrame from logger.to_df().
        metrics (List[str]): List of metrics to plot. If None, plots 
            'train_loss' and 'adj_change'.
        figsize (tuple): Figure size.
        save_path (str): Optional path to save the figure.
        
    Returns:
        Tuple[plt.Figure, Any]: The figure and axes objects.
    """
    if metrics is None:
        metrics = ['train_loss', 'adj_change']
    
    # Filter to available metrics
    available_metrics = [m for m in metrics if m in log_df.columns]
    
    if not available_metrics:
        raise ValueError(f"None of the requested metrics {metrics} are available. "
                         f"Available columns: {list(log_df.columns)}")
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, available_metrics):
        data = log_df[metric].dropna()
        ax.plot(data.index, data.values, linewidth=1.5)
        ax.set_xlabel('Steps', fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes
