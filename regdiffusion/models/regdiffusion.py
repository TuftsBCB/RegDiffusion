import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
from typing import List


torch.set_float32_matmul_precision('high')

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(self.theta) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(
            half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_dim, out_dim, time_dim, celltype_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_dim, out_dim)
        self.celltype_mlp = nn.Linear(celltype_dim, out_dim)
        self.l1 = nn.Linear(in_dim, out_dim)
        self.l2 = nn.Linear(out_dim, out_dim)
        self.do1 = nn.Dropout(0.1)
        self.act = nn.Tanh()
        
    def forward(self, x, t, ct):
        h = self.do1(self.act(self.l1(x)))
        time_emb = self.act(self.time_mlp(t)).unsqueeze(1)
        celltype_emb = self.act(self.celltype_mlp(ct)).unsqueeze(1)
        h = h + time_emb + celltype_emb
        h = self.act(self.l2(h))
        return h
    
class GeneEmbeddings(nn.Module):
    def __init__(self, n_gene, gene_dim):
        super().__init__()
        gene_emb = torch.randn(n_gene, gene_dim-1)
        self.gene_emb = nn.Parameter(gene_emb, requires_grad=True)

    def forward(self, x):
        n_cell = x.shape[0]
        batch_gene_emb = self.gene_emb.unsqueeze(0).repeat(n_cell, 1, 1)
        batch_gene_emb = torch.concat([x.unsqueeze(-1), batch_gene_emb], dim=-1)
        return batch_gene_emb
    
class RegDiffusion(nn.Module):
    ''' 
    
    A RegDiffusion model. For architecture details, please refer to our paper.

    > From noise to knowledge: probabilistic diffusion-based neural inference
    
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
    '''
    def __init__(
        self, n_gene, time_dim, 
        n_celltype=None, celltype_dim=4, 
        hidden_dims=[16, 16, 16], adj_dropout=0.3, init_coef = 5
    ):
        super(RegDiffusion, self).__init__()
        
        self.n_gene = n_gene
        self.gene_dim = hidden_dims[0]
        self.adj_dropout=adj_dropout
        self.gene_reg_norm = 1/(n_gene-1)
        
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
        
        self.zeros_nonparam = nn.Parameter(
            torch.zeros(n_gene, n_gene), requires_grad=False)
        self.eye_nonparam = nn.Parameter(
            torch.eye(n_gene), requires_grad=False)
        self.mask_nonparam = nn.Parameter(
            1 - torch.eye(n_gene), requires_grad=False)
        
    def soft_thresholding(self, x, tau):
        return torch.sign(x) * torch.max(
            self.zeros_nonparam, torch.abs(x) - tau)
        
    def I_minus_A(self):
        mask = self.mask_nonparam
        if self.train:
            A_dropout = (torch.rand_like(self.adj_A)>self.adj_dropout).float()
            A_dropout /= (1-self.adj_dropout)
            mask = mask * A_dropout
        clean_A = self.soft_thresholding(self.adj_A, self.gene_reg_norm/2)*mask

        return self.eye_nonparam - clean_A
        
    def get_adj_(self):
        return self.soft_thresholding(
            self.adj_A, self.gene_reg_norm/2) * self.mask_nonparam
    
    def get_adj(self):
        adj = self.get_adj_().detach().cpu().numpy() / self.gene_reg_norm
        return adj.astype(np.float16)

    @torch.no_grad()
    def get_sampled_adj_(self):
        return self.get_adj_()[
            self.sampled_adj_row_nonparam, self.sampled_adj_col_nonparam
        ].detach()
    
    def get_gene_emb(self):
        return self.gene_emb[0].gene_emb.data.cpu().detach().numpy()
    
    def forward(self, x, t, ct):
        h_time = self.time_mlp(t)
        h_celltype = self.celltype_emb(ct)
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
