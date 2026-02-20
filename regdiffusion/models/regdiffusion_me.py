import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
from typing import List

from .regdiffusion import SinusoidalPositionEmbeddings, Block, GeneEmbeddings


class SoftThreshold(torch.autograd.Function):
    """Memory-efficient soft thresholding.

    Standard soft_thresholding = sign(x) * relu(|x| - tau) saves two float32
    (n,n) tensors for backward (sign result + relu output). This custom
    autograd function saves only a boolean mask (1 byte per entry = 8x smaller).

    The gradient is 1 where |x| > tau, 0 elsewhere — identical to the
    standard implementation.
    """
    @staticmethod
    def forward(ctx, x, tau):
        above = x.abs() > tau
        ctx.save_for_backward(above)
        return torch.where(above, x - x.sign() * tau, torch.zeros_like(x))

    @staticmethod
    def backward(ctx, grad_output):
        above, = ctx.saved_tensors
        return grad_output * above, None


class RegDiffusionME(nn.Module):
    """
    Memory-efficient variant of RegDiffusion.

    Two optimizations reduce GPU memory for large gene networks:

    1. Custom SoftThreshold autograd: saves a boolean mask (1 byte/entry)
       instead of float32 tensors (4 bytes/entry) during backward pass.
    2. Sampled sparse loss: computes L1 regularization from random samples
       of adj_A instead of materializing the full (n_gene, n_gene) matrix.

    The model architecture and inference behavior are identical to RegDiffusion.

    Args:
        n_genes (int): Number of Genes
        time_dim (int): Dimension of time step embedding
        n_celltype (int): Number of expected cell types. If it is not provided,
        there would be no celltype embedding. Default is None.
        celltype_dim (int): Dimension of cell types
        hidden_dims (list[int]): List of integer for the dimensions of the
        hidden layers. The first hidden dimension will be used as the size
        for gene embedding.
        adj_dropout (float): A single number between 0 and 1 specifying the
        percentage of values in the adjacency matrix that are dropped
        during training.
        init_coef (int): Coefficient to multiply with gene regulation norm
        (1/(n_gene - 1)) to initialize the adjacency matrix.
    """
    def __init__(
        self, n_gene, time_dim,
        n_celltype=None, celltype_dim=4,
        hidden_dims=[16, 16, 16], adj_dropout=0.3, init_coef = 5
    ):
        super(RegDiffusionME, self).__init__()

        self.n_gene = n_gene
        self.gene_dim = hidden_dims[0]
        self.adj_dropout=adj_dropout
        self.gene_reg_norm = 1/(n_gene-1)
        self.celltype_dim = celltype_dim
        self.n_celltype = n_celltype

        adj_A = torch.ones(n_gene, n_gene) * self.gene_reg_norm * init_coef
        self.adj_A = nn.Parameter(adj_A, requires_grad =True, )
        self.sampled_adj_row_nonparam = nn.Parameter(
            torch.randint(0, n_gene, size=(10000,)),
            requires_grad=False)
        self.sampled_adj_col_nonparam = nn.Parameter(
            torch.randint(0, n_gene, size=(10000,)),
            requires_grad=False)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.Tanh()
        )

        self.gene_emb = nn.Sequential(
            GeneEmbeddings(n_gene, self.gene_dim),
            nn.Tanh()
        )

        if n_celltype is not None:
            self.celltype_emb = nn.Embedding(n_celltype, celltype_dim)

        self.blocks = nn.ModuleList([
            Block(
                hidden_dims[i], hidden_dims[i+1]-1, time_dim, celltype_dim
            ) for i in range(len(hidden_dims) - 1)
        ])

        self.final = nn.Linear(hidden_dims[-1]-1, 1)

    def soft_thresholding(self, x, tau):
        return SoftThreshold.apply(x, tau)

    def I_minus_A(self):
        eye = torch.eye(self.n_gene, device=self.adj_A.device)
        if self.training:
            A_dropout = (torch.rand_like(self.adj_A)>self.adj_dropout).float()
            A_dropout /= (1-self.adj_dropout)
            A_dropout.fill_diagonal_(0)
            mask = A_dropout
        else:
            mask = 1 - eye
        clean_A = self.soft_thresholding(self.adj_A, self.gene_reg_norm/2) * mask
        return eye - clean_A

    def get_adj_(self):
        mask = 1 - torch.eye(self.n_gene, device=self.adj_A.device)
        return self.soft_thresholding(self.adj_A, self.gene_reg_norm/2) * mask

    def get_adj(self):
        adj = self.get_adj_().detach().cpu().numpy() / self.gene_reg_norm
        return adj.astype(np.float16)

    def get_sampled_sparse_loss(self):
        """Compute mean absolute value of sampled adj entries for sparse loss.

        Instead of materializing the full (n_gene, n_gene) get_adj_() matrix,
        this samples 10,000 random entries, applies soft thresholding, and
        returns the mean. This avoids creating 3 additional (n,n) tensors
        that the full get_adj_() would require.
        """
        sampled_vals = self.adj_A[
            self.sampled_adj_row_nonparam, self.sampled_adj_col_nonparam
        ]
        sampled_clean = self.soft_thresholding(sampled_vals, self.gene_reg_norm/2)
        return sampled_clean.abs().mean()

    @torch.no_grad()
    def get_sampled_adj_(self):
        return self.get_adj_()[
            self.sampled_adj_row_nonparam, self.sampled_adj_col_nonparam
        ].detach()

    def get_gene_emb(self):
        return self.gene_emb[0].gene_emb.data.cpu().detach().numpy()

    def forward(self, x, t, ct):
        h_time = self.time_mlp(t)
        if self.n_celltype is not None:
            h_celltype = self.celltype_emb(ct)
        else:
            h_celltype = torch.zeros(
                x.shape[0], self.celltype_dim, device=x.device, dtype=x.dtype
            )
        original = x.unsqueeze(-1)
        h_x = self.gene_emb(x)
        for i, block in enumerate(self.blocks):
            if i != 0:
                h_x = torch.concat([h_x, original], dim=-1)
            h_x = block(h_x, h_time, h_celltype)

        I_minus_A = self.I_minus_A()
        hz = torch.einsum('ogd,gh->ohd', h_x, I_minus_A)
        z = self.final(hz)

        return z.squeeze(-1)
