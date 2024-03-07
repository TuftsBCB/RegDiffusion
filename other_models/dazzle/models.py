import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
import math

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, activation):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, out_dim)
        self.act = activation()

    def forward(self, x):
        out1 = self.act(self.l1(x))
        out2 = self.act(self.l2(out1)) 
        return self.l3(out2)

class DAZZLE(nn.Module):
    ''' A DAZZLE model
    
    Parameters
    ----------
    n_genes: int
        Number of Genes
    hidden_dim: int
        Size of dimension in the MLP layers
    z_dim: int
        Size of dimension of Z
    A_dim: int
        Number of Adjacency matrix to be modeled at the same time
    activation: function
        A pytorch activation layer
    train_on_non_zero: bool
        Whether to train on non-zero data only
    dropout_augmentation_p: double
        Probability of augmented dropout. For example, 0.1 means that
        10% of data will be temporarily assign to zero in each forward 
        pass
    dropout_augmentation_type: str
        Choose among 'all' (default), 'belowmean', 'belowhalfmean'. This 
        option specifies where dropout augmentation would happen. If
        'belowmean' is selected, the augmentation would only happen on
        values below global mean. 
    pretrained_A: torch.tensor
        A customized initialization of A instead of random initialization. 
        
    Methods
    -------
    get_adj_
        Obtain current adjacency matrix 
    get_adj
        Obtain current adjacency matrix as a detached numpy array
    I_minus_A
        Calculate I - A
    reparameterization(z_mu, z_sigma)
        Reparameterization trick used in VAE
    dropout_augmentation(x, global_mean)
        Randomly add dropout noise to the original expression data
    forward(x, global_mean, global_std, use_dropout_augmentation)
        Forward pass
    '''
    def __init__(
        self, n_gene, hidden_dim=128, z_dim=1, A_dim=1, 
        activation=nn.Tanh, train_on_non_zero=False, 
        dropout_augmentation_p=0.1, dropout_augmentation_type='all',
        pretrained_A=None, 
    ):
        super(DAZZLE, self).__init__()
        self.n_gene = n_gene
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.A_dim = A_dim
        self.train_on_non_zero = train_on_non_zero
        
        if pretrained_A is None:
            adj_A = torch.ones(A_dim, n_gene, n_gene) / (n_gene - 1) 
            adj_A += torch.rand_like(adj_A) * 0.0002
        else:
            adj_A = pretrained_A
        self.adj_A = nn.Parameter(adj_A, requires_grad=True)
        
        self.inference_zposterior = MLP(1, hidden_dim, z_dim * 2, activation)
        self.generative_pxz = MLP(z_dim, hidden_dim, 1, activation)
        self.da_classifier = MLP(z_dim, z_dim, 1, activation)
        self.da_p = dropout_augmentation_p
        self.da_type = dropout_augmentation_type
        classifier_pos_weight = torch.FloatTensor([1.0])
        if self.da_p != 0:
            classifier_pos_weight *= (1-self.da_p) / self.da_p
        self.classifier_pos_weight = nn.Parameter(
            classifier_pos_weight, requires_grad=False
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def get_adj_(self):
        eye_tensor = torch.eye(
            self.n_gene, device = self.adj_A.device
        ).repeat(self.A_dim, 1, 1)
        mask = torch.ones_like(self.adj_A) - eye_tensor
        return (self.adj_A * mask).mean(0)
    
    def get_adj(self):
        return self.get_adj_().cpu().detach().numpy()
    
    def I_minus_A(self):
        eye_tensor = torch.eye(
            self.n_gene, device = self.adj_A.device
        ).repeat(self.A_dim, 1, 1)
        # clean up A along diagnal line
        mask = torch.ones_like(self.adj_A) - eye_tensor
        clean_A = self.adj_A * mask
        return eye_tensor - clean_A
    
    def reparameterization(self, z_mu, z_sigma):
        return z_mu + z_sigma * torch.randn_like(z_sigma)
    
    @torch.no_grad()
    def dropout_augmentation(self, x, global_mean):
        da_mask = (torch.rand_like(x) < self.da_p)
        if self.da_type == 'belowmean':
            da_mask = da_mask * (x < global_mean)
        elif self.da_type == 'belowhalfmean':
            da_mask = da_mask  * (x < (global_mean / 2))
        elif self.da_type == 'all':
            da_mask = da_mask
        noise = -1 * x * da_mask
        x = x + noise
        return x, noise, da_mask
        
    def forward(self, x, global_mean, global_std, use_dropout_augmentation=True):                
        if self.train_on_non_zero:
            eval_mask = (x != 0)
        else:
            eval_mask = torch.ones_like(x)
        
        x_init = x
        if use_dropout_augmentation:
            x, noise, da_mask = self.dropout_augmentation(x_init, global_mean)
        else:
            noise = torch.zeros_like(x)
            da_mask = (noise == 1)
        
        x = (x - global_mean) / (global_std)
        noise = (noise - global_mean) / (global_std)
        
        # Encoder --------------------------------------------------------------
        I_minus_A = self.I_minus_A()
                
        z_posterior = self.inference_zposterior(x.unsqueeze(-1))
        z_posterior = torch.einsum('ogd,agh->ohd', z_posterior, I_minus_A)
        z_mu = z_posterior[:, :, :self.z_dim]
        z_logvar = z_posterior[:, :, self.z_dim:]
        z = self.reparameterization(z_mu, torch.exp(z_logvar * 0.5))
        # z = torch.einsum('ogd,og->ogd', z, (~da_mask))
        
        # Decoder --------------------------------------------------------------
        z_inv = torch.einsum('ogd,agh->ohd', z, torch.inverse(I_minus_A))
        x_rec = self.generative_pxz(z_inv).squeeze(2)
        
        # DA classifier
        da_pred = self.da_classifier(z).squeeze(-1)
        loss_da = F.binary_cross_entropy_with_logits(
            da_pred, da_mask.float(), pos_weight=self.classifier_pos_weight)
        
        # Losses ---------------------------------------------------------------
        loss_rec_all = (x - x_rec).pow(2)
        loss_rec = torch.sum(loss_rec_all * eval_mask)
        loss_rec = loss_rec / torch.sum(eval_mask)
        
        loss_kl = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - torch.exp(z_logvar))
                        
        out = {
            'loss_rec': loss_rec, 'loss_kl': loss_kl, 
            'z_posterior': z_posterior, 'z': z, 'x_rec': x_rec,
            'da_pred': da_pred, 'da_mask': da_mask, 'loss_da': loss_da, 
        }
        return out