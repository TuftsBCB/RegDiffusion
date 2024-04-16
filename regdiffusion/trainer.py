import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from .models import RegDiffusion
from tqdm import tqdm
from .logger import LightLogger
from datetime import datetime
from .grn import GRN
from .evaluator import GRNEvaluator
from .logger import LightLogger
import matplotlib.pyplot as plt
import warnings

def linear_beta_schedule(timesteps, start_noise, end_noise):
    scale = 1000 / timesteps
    beta_start = scale * start_noise
    beta_end = scale * end_noise
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float)

def power_beta_schedule(timesteps, start_noise, end_noise, power=2):
    linspace = torch.linspace(0, 1, timesteps, dtype = torch.float)
    poweredspace = linspace ** power
    scale = 1000 / timesteps
    beta_start = scale * start_noise
    beta_end = scale * end_noise
    return beta_start + (beta_end - beta_start) * poweredspace

class RegDiffusionTrainer:
    """
    Initialize and Train a RegDiffusion model.

    For architecture and training details, please refer to our paper.

    > From noise to knowledge: probabilistic diffusion-based neural inference

    You can access the model through `RegDiffusionTrainer.model`. 
    
    Args:
        exp_array (np.ndarray): 2D numpy array. If used on single-cell RNAseq, 
            the rows are cells and the columns are genes. Data should be log 
            transformed. You may also want to remove all non expressed genes.
        cell_types (np.ndarray): (Optional) 1D integer array for cell type. If 
            you have labels in your cell type, you need to convert them to 
            interge. Default is None. 
        T (int): Total number of diffusion steps. Default: 5,000
        start_noise (float): Minimal noise level (beta) to be added. Default: 
            0.0001
        end_noise (float): Maximal noise level (beta) to be added. Default: 
            0.02
        time_dim (int): Dimension size for the time embedding. Default: 64.
        celltype_dim (int): Dimension size for the cell type embedding. 
            Default: 4.
        hidden_dim (list): Dimension sizes for the feature learning layers. We 
            use the size of the first layer as the dimension for gene embeddings
            as well. Default: [16, 16, 16].
        init_coef (int): A coefficent to control the value to initialize the 
            adjacency matrix. Here we define regulatory norm as 1 over (number 
            of genes - 1). The value which we use to initialize the model is 
            `init_coef` times of the regulatory norm. Default: 5. 
        lr_nn (float): Learning rate for the rest of the neural networks except
            the adjacency matrix. Default: 0.001
        lr_adj (float): Learning rate for the adjacency matrix. By default, it 
            equals to 0.02 * gene regulatory norm, which equals 1/(n_gene-1). 
        weight_decay_nn (float): L2 regularization coef on the rest of the 
            neural networks. Default: 0.1.
        weight_decay_adj (float): L2 regularization coef on the adj matrix.
            Default: 0.01.
        sparse_loss_coef (float): L1 regularization coef on the adj matrix. 
            Default: 0.25.
        adj_dropout (float): Probability of an edge to be zeroed. Default: 0.3.
        batch_size (int): Batch size for training. Default: 128.
        n_steps (int): Total number of training iterations. Default: 1000.
        train_split (float): Train partition. Default: 1.0.
        train_split_seed (int): Random seed for train/val partition. 
            Default: 123
        device (str or torch.device): Device where the model is running. For 
            example, "cpu", "cuda", "cuda:1", and etc. You are not recommended 
            to run this model on Apple's MPS chips. Default is "cuda" but if 
            you only has CPU, it will switch back to CPU.
        compile (boolean): Whether to compile the model before training. 
            Compile the model is a good idea on large dataset and ofter improves
            inference speed when it works. For smaller dataset, eager execution 
            if often good enough. 
        evaluator (GRNEvaluator): (Optional) A defined GRNEvaluator if ground 
            truth data is available. Evaluation will be done every 100 steps by 
            default but you can change this setting through the eval_on_n_steps 
            option. Default is None
        eval_on_n_steps (int): If an evaluator is provided, the trainer will 
            run evaluation every `eval_on_n_steps` steps. Default: 100.
        logger (LightLogger): (Optional) A LightLogger to log training process. 
            The only situation when you need to provide this is when you want 
            to save logs from different trainers into the same logger. Default 
            is None. 
    """
    def __init__(
        self, exp_array, cell_types=None, 
        T=5000, start_noise=0.0001, end_noise=0.02,
        time_dim=64, celltype_dim=4, hidden_dims=[16, 16, 16],
        init_coef = 5, 
        lr_nn=1e-3, lr_adj=None, 
        weight_decay_nn=0.1, weight_decay_adj = 0.01,
        sparse_loss_coef=0.25, adj_dropout=0.30,
        batch_size=128, n_steps=1000, 
        train_split=1.0, train_split_seed=123, 
        device='cuda', compile=False, 
        evaluator=None, eval_on_n_steps=100, logger=None
    ):
        hp = locals()
        del hp['exp_array']
        del hp['cell_types']
        del hp['logger']
        self.hp = hp
        
        if device == 'mps':
            raise Exception("We noticed unreliable training behavior on", 
                            "Apple's silicon. Consider using other devices.")
        elif device.startswith('cuda'):
            if not torch.cuda.is_available():
                print(
                    "You specified cuda as your computing device but apprently", 
                    "it's not available. Setting device to cpu for now. ")
                device = 'cpu'
        self.device = device
        self.hp['device'] = device
        
        # Logger ---------------------------------------------------------------
        if logger is None:
            self.logger = LightLogger()
        self.note_id = self.logger.start()
        
        # Define diffusion schedule
        self.betas = linear_beta_schedule(T, start_noise, end_noise)
        self.alphas = 1. - self.betas
        alpha_bars = torch.cumprod(self.alphas, axis=0)
        self.mean_schedule = torch.sqrt(alpha_bars).to(device)
        self.std_schedule = torch.sqrt(1. - alpha_bars).to(device)
    
        # Prepare Data ---------------------------------------------------------
        if (exp_array.sum(0) == 0).sum() > 0:
            warnings.warn(
                "Some columns in the exp_array contains all zero values, ",
                "which often causes trouble in inference. Please consider ",
                "removing these columns before continuing. "
            )
        if cell_types is None:
            cell_types = np.zeros(exp_array.shape[0], dtype=int)
        self.n_celltype = len(np.unique(cell_types))
        n_cell, n_gene = exp_array.shape
        self.n_cell = n_cell
        self.n_gene = n_gene

        self.evaluator = evaluator
        
        ## Normalize data
        cell_min = exp_array.min(axis=1, keepdims=True)
        cell_max = exp_array.max(axis=1, keepdims=True)
        normalized_X = (exp_array - cell_min) / (cell_max - cell_min)
        normalized_X = (normalized_X - normalized_X.mean(0))/normalized_X.std(0)
    
        ## Train/validation split
        random_state = np.random.RandomState(train_split_seed)
        train_val_split = random_state.rand(normalized_X.shape[0])
        train_index = train_val_split <= train_split
        val_index = train_val_split > train_split
    
        x_tensor_train = torch.tensor(
            normalized_X[train_index, ], dtype=torch.float32)
        celltype_tensor_train = torch.tensor(
            cell_types[train_index], dtype=int)
        x_tensor_val = torch.tensor(
            normalized_X[val_index, ], dtype=torch.float32)
        celltype_tensor_val = torch.tensor(cell_types[val_index],dtype=int)
    
        ## Setup dataset and dataloader
        self.train_dataset = torch.utils.data.TensorDataset(
            x_tensor_train, celltype_tensor_train
        )
        # Implement bootstrap for train sampler
        train_sampler = torch.utils.data.RandomSampler(
            self.train_dataset, replacement=True, num_samples=batch_size)
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, 
            sampler=train_sampler,
            batch_size = batch_size, 
            drop_last=True)
        
        self.val_dataset = torch.utils.data.TensorDataset(
            x_tensor_val, celltype_tensor_val
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, 
            shuffle=False,
            batch_size = batch_size, 
            drop_last=False)
    
        # Setup Model ----------------------------------------------------------
        gene_reg_norm = 1/(n_gene-1)
        self.model = RegDiffusion(
            n_gene=n_gene, 
            time_dim=time_dim, 
            n_celltype=self.n_celltype,
            celltype_dim = celltype_dim,
            hidden_dims=hidden_dims,
            adj_dropout=adj_dropout,
            init_coef=init_coef
        )
    
        # Setup optimizer ------------------------------------------------------
        if lr_adj is None:
            lr_adj = gene_reg_norm/50
            self.hp['lr_adj'] = lr_adj
        adj_params = []
        non_adj_params = []
        for name, param in self.model.named_parameters():
            if name.endswith('adj_A'):
                adj_params.append(param)
            else:
                if not name.endswith('_nonparam'):
                    non_adj_params.append(param)
        self.opt = torch.optim.Adam(
            [{'params': non_adj_params}, {'params': adj_params}], 
            lr=lr_nn, 
            weight_decay=weight_decay_nn, betas=[0.9, 0.99]
        )
        self.opt.param_groups[0]['lr'] = lr_nn
        self.opt.param_groups[1]['lr'] = lr_adj
        self.opt.param_groups[1]['weight_decay'] = weight_decay_adj
    
        self.model.to(device)
        if self.device.startswith('cuda') and compile:
            self.original_model = self.model
            self.model = torch.compile(self.model)
        self.total_time_cost=0
        self.losses_on_gene=None
        self.model_name='RegDiffusion'


    @torch.no_grad()
    def forward_pass(self, x_0, t):
        """
        Forward diffusion process
        
        Args:
            x_0 (torch.FloatTensor): Torch tensor for expression data. Rows are 
            cells and columns are genes
            t (torch.LongTensor): Torch tensor for diffusion time steps.
        """
        noise = torch.randn_like(x_0)
        mean_coef = self.mean_schedule.gather(dim=-1, index=t)
        std_coef = self.std_schedule.gather(dim=-1, index=t)
        x_t = mean_coef.unsqueeze(-1) * x_0 + std_coef.unsqueeze(-1) * noise
        return x_t, noise

    def train(self, n_steps=None):
        """
        Train the initialized model for a number of steps. 

        Args:
            n_steps (int): Number of steps to train. If not provided, it will 
                train the model by the n_steps sepcified in class 
                initialization. Please read our paper to see how to identify
                the converge point. 
        """
        start_time = datetime.now()
        eval_steps = self.hp['eval_on_n_steps']
        if n_steps is None:
            n_steps = self.hp['n_steps']
        sampled_adj = self.model.get_sampled_adj_()
        with tqdm(range(n_steps)) as pbar:
            for epoch in pbar: 
                epoch_loss = []
                for step, batch in enumerate(self.train_dataloader):
                    x_0, ct = batch
                    x_0 = x_0.to(self.device)
                    ct = ct.to(self.device)
                    self.opt.zero_grad()
                    t = torch.randint(
                        0, self.hp['T'], (x_0.shape[0],), 
                        device=self.device
                    ).long()
        
                    x_noisy, noise = self.forward_pass(x_0, t)
                    z = self.model(x_noisy, t, ct)
                    loss_ = F.mse_loss(noise, z, reduction='none')
                    loss = loss_.mean()
        
                    adj_m = self.model.get_adj_()
                    loss_sparse = adj_m.mean() * self.hp['sparse_loss_coef']
                            
                    if epoch > 10:
                        loss = loss + loss_sparse
                    loss.backward()
                    self.opt.step()
                    epoch_loss.append(loss.item())
                train_loss = np.mean(epoch_loss)
                sampled_adj_new = self.model.get_sampled_adj_()
                adj_diff = (
                    sampled_adj_new - sampled_adj
                    ).mean().item()*(self.n_gene-1)
                sampled_adj = sampled_adj_new
                pbar.set_description(
                    f'Training loss: {train_loss:.3f}, Change on Adj: {adj_diff:.3f}')
                epoch_log = {'train_loss': train_loss, 'adj_change': adj_diff}
                if epoch % eval_steps == eval_steps - 1:
                    if self.evaluator is not None:
                        eval_result = self.evaluator.evaluate(
                            self.model.get_adj()
                            )
                        for k in eval_result.keys():
                            epoch_log[k] = eval_result[k]
                    if self.hp['train_split'] < 1:
                        with torch.no_grad():
                            val_epoch_loss = []
                            for step, batch in enumerate(self.val_dataloader):
                                x_0, ct = batch
                                x_0 = x_0.to(self.device)
                                ct = ct.to(self.device)
                                t = torch.randint(
                                    0, self.hp['T'], (x_0.shape[0],), 
                                    device=self.device).long()
        
                                x_noisy, noise = self.forward_pass(x_0, t)
                                z = self.model(x_noisy, t, ct)
                                step_val_loss = F.mse_loss(
                                    noise, z, reduction='mean').item()
                                val_epoch_loss.append(step_val_loss)
                            epoch_log['val_loss'] = np.mean(val_epoch_loss)
                self.logger.log(epoch_log)
        self.losses_on_gene = loss_.detach().mean(0).cpu().numpy()
        self.total_time_cost += int(
            (datetime.now() - start_time).total_seconds())
        return None

    def training_curves(self):
        """
        Plot out the training curves on `train_loss` and `adj_change`. Check
        out our paper for how to use `adj_change` to identify the convergence 
        point. 
        """
        log_df = self.logger.to_df()
        if 'train_loss' in log_df:
            figure, axes = plt.subplots(1, 2, figsize=(8, 3))
            axes[0].plot(log_df['train_loss'])
            axes[0].set_xlabel('Steps')
            axes[0].set_ylabel('Training Loss')
            axes[1].plot(log_df['adj_change'][1:])
            axes[1].set_xlabel('Steps')
            axes[1].set_ylabel('Amount of Change in Adj. Matrix')
            plt.show()
        else:
            print(
                'Training log and Adj Change are not available. Train your ', 
                'model using the .train() method.')

    def get_grn(self, gene_names, tf_names=None, top_gene_percentile=None):
        """
        Obtain a GRN object. You need to provide the genes names. 

        Args:
            gene_names (np.ndarray): An array of names of all genes. The order 
                of genes should be the same as the order used in your expression
                table.
            tf_names (np.ndarray):An array of names of all transcriptional 
                factors. The order of genes should be the same as the order 
                used in your expression table.
            top_gene_percentile (int): If provided, we will set the value on 
                weak links to be zero. It is useful if you want to save the 
                regulatory relationship in a GRN object as a sparse matrix. 

        """
        adj = self.model.get_adj()
        return GRN(adj, gene_names, tf_names, top_gene_percentile)

    def get_adj(self):
        """
        Obtain the adjacency matrix. The values in this adjacency matix has 
        been scaled using regulatory norm. You may expect strong links to go
        beyond 5 or 10 in most cases. 
        """
        return self.model.get_adj()
        
