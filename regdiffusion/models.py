import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(self.theta) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
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
        self.do = nn.Dropout(0.1)
        self.act = nn.Tanh()
        
    def forward(self, x, t, ct):
        h = self.do(self.act(self.l1(x)))
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
    
def soft_thresholding(x, tau):
    return torch.sign(x) * torch.max(
        torch.zeros_like(x), torch.abs(x) - tau)
    
class RegDiffusion(nn.Module):
    ''' A RegDiffusion model
    
    Parameters
    ----------
    n_genes: int
        Number of Genes
    time_dim: int
        Dimension of time step embedding
    n_celltype: int
        Number of cell types (Optional). Default None
    celltype_dim: int
        Dimension of cell types
    hidden_dims: list[int]
        List of integer for the dimensions of the hidden layers
    adj_dropout: float
        A single number between 0 and 1 specifying the percentage of 
        values in the adjacency matrix that are dropped during training. 
        
    Methods
    -------
    get_adj_
        Obtain current adjacency matrix 
    get_adj
        Obtain current adjacency matrix as a detached numpy array
    I_minus_A
        Calculate I - A
    get_gene_emb
        Obtain the first layer gene embedding
    forward(x, t, ct)
        Forward pass. Input is expression table x, time step t, and 
        cell type ct. Output is the predicted z. 
    '''
    def __init__(
        self, n_gene, time_dim, 
        n_celltype=None, celltype_dim=None, 
        hidden_dims=[1, 32, 64, 128], adj_dropout=0.1
    ):
        super(RegDiffusion, self).__init__()
        
        self.n_gene = n_gene
        self.gene_dim = hidden_dims[0]
        self.adj_dropout=adj_dropout
        
        adj_A = torch.ones(n_gene, n_gene) / (n_gene-1) * 5
        self.adj_A = nn.Parameter(adj_A, requires_grad =True)
        

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
            # self.celltype_emb = nn.Embedding(n_celltype, celltype_dim)
            self.celltype_emb = SinusoidalPositionEmbeddings(celltype_dim, theta=n_celltype)
        
        self.blocks = nn.ModuleList([
            Block(
                hidden_dims[i], hidden_dims[i+1]-1, time_dim, celltype_dim
            ) for i in range(len(hidden_dims) - 1)
        ])
        
        self.final = nn.Linear(hidden_dims[-1]-1, 1)
        
        self.eye = nn.Parameter(torch.eye(n_gene), requires_grad=False)
        self.mask = nn.Parameter(
            torch.ones(n_gene, n_gene) - torch.eye(n_gene), 
            requires_grad=False)
        
    def I_minus_A(self):
        if self.train:
            A_dropout = (torch.rand_like(self.adj_A)>self.adj_dropout)/(1-self.adj_dropout)
            mask = self.mask * A_dropout
        clean_A = soft_thresholding(self.adj_A, 0.001) * mask 
        return self.eye - clean_A
        
    def get_adj_(self):
        return soft_thresholding(self.adj_A, 0.001) * self.mask
    
    def get_adj(self):
        return self.get_adj_().cpu().detach().numpy()
    
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
