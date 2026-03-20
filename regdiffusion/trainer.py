import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import TensorDataset
from .models import RegDiffusion, RegDiffusionME
from tqdm import tqdm
from .logger import LightLogger
from datetime import datetime
from .grn import GRN
from .evaluator import GRNEvaluator
from .logger import LightLogger
import matplotlib.pyplot as plt
import warnings
import random

def set_seed(seed):
    """Set random seeds for reproducibility across PyTorch, NumPy, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

class _SparseExpressionDataset(Dataset):
    """Dataset that stores sparse expression data and normalizes on-the-fly.

    Instead of materializing the full dense normalized matrix (which can be
    100+ GB for 1M cells), this dataset keeps the original sparse matrix in
    memory and converts only the requested rows to dense during __getitem__.

    Normalization applied per sample:
        1. Min-max per cell:  (x - cell_min) / cell_range
        2. Z-score per gene:  (x - gene_mean) / gene_std
    """

    def __init__(self, sparse_X, cell_types, cell_min, cell_range,
                 gene_mean, gene_std, indices):
        self.X = sparse_X            # scipy CSR matrix (shared, read-only)
        self.cell_types = cell_types  # int array (full, not subset)
        self.cell_min = cell_min      # (n_cell,) float32
        self.cell_range = cell_range  # (n_cell,) float32
        self.gene_mean = gene_mean    # (n_gene,) float32
        self.gene_std = gene_std      # (n_gene,) float32
        self.indices = indices         # row indices for this split

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        row = self.X[real_idx].toarray().ravel().astype(np.float32)
        row = (row - self.cell_min[real_idx]) / self.cell_range[real_idx]
        row = (row - self.gene_mean) / self.gene_std
        ct = int(self.cell_types[real_idx])
        return torch.from_numpy(row), torch.tensor(ct, dtype=torch.long)


class RegDiffusionTrainer:
    """
    Initialize and Train a RegDiffusion model.

    For architecture and training details, please refer to our paper.

    > From noise to knowledge: probabilistic diffusion-based neural inference

    You can access the model through `RegDiffusionTrainer.model`. 
    
    Args:
        exp_array (np.ndarray or scipy.sparse matrix): 2D expression matrix.
            If used on single-cell RNAseq, the rows are cells and the columns
            are genes. Data should be log transformed. You may also want to
            remove all non expressed genes. Sparse matrices (e.g. from
            ``adata.X``) are supported and handled memory-efficiently — the
            full dense normalized matrix is never materialized.
        cell_types (np.ndarray): (Optional) 1D integer array for cell type. If 
            you have labels in your cell type, you need to convert them to 
            integer. Default is None.
        T (int): Total number of diffusion steps. Default: 5,000
        start_noise (float): Minimal noise level (beta) to be added. Default: 
            0.0001
        end_noise (float): Maximal noise level (beta) to be added. Default: 
            0.02
        time_dim (int): Dimension size for the time embedding. Default: 64.
        celltype_dim (int): Dimension size for the cell type embedding. 
            Default: 4.
        hidden_dims (list): Dimension sizes for the feature learning layers. We
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
            Compile the model is a good idea on large dataset and often improves
            inference speed when it works. For smaller dataset, eager execution
            is often good enough.
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
        gradient_accumulation (boolean): Whether to train with gradient
            accumulation. This is useful when number of genes are extremely
            large.
        use_amp (bool): Whether to use automatic mixed precision (bfloat16)
            during training. This reduces memory usage by storing activations
            in half precision while keeping model parameters in float32.
            Requires a GPU with bfloat16 support (Ampere or newer).
            Default: False.
        memory_efficient (bool): Whether to use the memory-efficient model
            variant (RegDiffusionME). This uses a custom autograd function
            for soft thresholding (saves boolean masks instead of float32
            tensors) and sampled sparse loss (avoids materializing full
            n_gene x n_gene matrix for L1 regularization). Default: False.
        seed (int): Random seed for full reproducibility. When set, all
            sources of randomness are seeded. Default: None (non-deterministic).
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
        evaluator=None, eval_on_n_steps=100, logger=None,
        gradient_accumulation=False, use_amp=False,
        memory_efficient=False,
        seed=None
    ):
        hp = locals()
        del hp['exp_array']
        del hp['cell_types']
        del hp['logger']
        self.hp = hp

        # Seed for reproducibility
        if seed is not None:
            set_seed(seed)
        self._sampler_generator = torch.Generator()
        if seed is not None:
            self._sampler_generator.manual_seed(seed)

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
        else:
            self.logger = logger
        self.note_id = self.logger.start()
        
        # Define diffusion schedule
        self.betas = linear_beta_schedule(T, start_noise, end_noise)
        self.alphas = 1. - self.betas
        alpha_bars = torch.cumprod(self.alphas, axis=0)
        self.mean_schedule = torch.sqrt(alpha_bars).to(device)
        self.std_schedule = torch.sqrt(1. - alpha_bars).to(device)
    
        # Prepare Data ---------------------------------------------------------
        if (exp_array.shape[0] < batch_size):
            warnings.warn(
                "Batch size needs to be smaller than the number of cells. "
            )
        if cell_types is None:
            cell_types = np.zeros(exp_array.shape[0], dtype=int)
        self.n_celltype = len(np.unique(cell_types))
        n_cell, n_gene = exp_array.shape
        self.n_cell = n_cell
        self.n_gene = n_gene

        self.evaluator = evaluator

        if sp.issparse(exp_array):
            self._prepare_sparse_data(
                exp_array, cell_types, batch_size,
                train_split, train_split_seed)
        else:
            self._prepare_dense_data(
                exp_array, cell_types, batch_size,
                train_split, train_split_seed)
    
        # Setup Model ----------------------------------------------------------
        gene_reg_norm = 1/(n_gene-1)
        ModelClass = RegDiffusionME if memory_efficient else RegDiffusion
        self.model = ModelClass(
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
        self.model_name='RegDiffusionME' if memory_efficient else 'RegDiffusion'

        # AMP setup
        self.use_amp = use_amp
        self.amp_device_type = 'cuda' if device.startswith('cuda') else 'cpu'


    def _prepare_dense_data(self, exp_array, cell_types, batch_size,
                            train_split, train_split_seed):
        """Normalize dense numpy array and build TensorDataset (original path)."""
        cell_min = exp_array.min(axis=1, keepdims=True)
        cell_max = exp_array.max(axis=1, keepdims=True)
        cell_range = cell_max - cell_min
        n_zero_cells = (cell_range == 0).sum()
        if n_zero_cells > 0:
            warnings.warn(
                f'{n_zero_cells} cells are removed from analysis where no genes are expressed.')
        normalized_X = (exp_array - cell_min) / cell_range
        normalized_X_std = normalized_X.std(0)
        n_zero_genes = (normalized_X_std == 0).sum()
        if n_zero_genes > 0:
            raise ValueError(
                f'{n_zero_genes} genes have 0 variance. Please remove these genes from your data.')
        normalized_X = (normalized_X - normalized_X.mean(0)) / normalized_X_std

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
        celltype_tensor_val = torch.tensor(cell_types[val_index], dtype=int)

        self.train_dataset = TensorDataset(
            x_tensor_train, celltype_tensor_train)
        train_sampler = torch.utils.data.RandomSampler(
            self.train_dataset, replacement=True, num_samples=batch_size,
            generator=self._sampler_generator)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=batch_size,
            drop_last=True)

        self.val_dataset = TensorDataset(
            x_tensor_val, celltype_tensor_val)
        self.val_dataloader = DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=batch_size,
            drop_last=False)

    def _prepare_sparse_data(self, exp_array, cell_types, batch_size,
                             train_split, train_split_seed):
        """Compute normalization stats from sparse matrix in chunks, then build
        a _SparseExpressionDataset that normalizes on-the-fly per sample.

        This never materializes the full dense normalized matrix, so memory
        usage stays proportional to (sparse matrix size + one batch of dense
        rows) rather than (n_cell * n_gene * 4 bytes).
        """
        n_cell, n_gene = exp_array.shape
        exp_array = sp.csr_matrix(exp_array)  # ensure CSR for fast row slicing

        # --- Per-cell min/max in chunks ---
        chunk_size = min(10000, n_cell)
        cell_min = np.empty(n_cell, dtype=np.float64)
        cell_max = np.empty(n_cell, dtype=np.float64)
        for start in range(0, n_cell, chunk_size):
            end = min(start + chunk_size, n_cell)
            chunk = exp_array[start:end]
            cell_min[start:end] = np.asarray(
                chunk.min(axis=1).todense()).ravel()
            cell_max[start:end] = np.asarray(
                chunk.max(axis=1).todense()).ravel()

        cell_range = cell_max - cell_min
        valid_mask = cell_range > 0
        n_zero_cells = (~valid_mask).sum()
        if n_zero_cells > 0:
            warnings.warn(
                f'{n_zero_cells} cells are removed from analysis where no '
                'genes are expressed.')

        # --- Per-gene mean/std of min-max-normalized data (chunked) ---
        n_valid = valid_mask.sum()
        gene_sum = np.zeros(n_gene, dtype=np.float64)
        gene_sum_sq = np.zeros(n_gene, dtype=np.float64)

        for start in range(0, n_cell, chunk_size):
            end = min(start + chunk_size, n_cell)
            chunk_valid = valid_mask[start:end]
            if not chunk_valid.any():
                continue
            # Only materialize valid rows in this chunk
            chunk_dense = exp_array[start:end][chunk_valid].toarray().astype(
                np.float64)
            c_min = cell_min[start:end][chunk_valid][:, None]
            c_range = cell_range[start:end][chunk_valid][:, None]
            chunk_norm = (chunk_dense - c_min) / c_range
            gene_sum += chunk_norm.sum(axis=0)
            gene_sum_sq += (chunk_norm ** 2).sum(axis=0)

        gene_mean = (gene_sum / n_valid).astype(np.float32)
        gene_var = (gene_sum_sq / n_valid) - gene_mean.astype(np.float64) ** 2
        gene_std = np.sqrt(np.maximum(gene_var, 0)).astype(np.float32)

        n_zero_genes = (gene_std == 0).sum()
        if n_zero_genes > 0:
            raise ValueError(
                f'{n_zero_genes} genes have 0 variance. Please remove these '
                'genes from your data.')

        # --- Train/validation split (only among valid cells) ---
        valid_indices = np.where(valid_mask)[0]
        random_state = np.random.RandomState(train_split_seed)
        split_vals = random_state.rand(len(valid_indices))
        train_indices = valid_indices[split_vals <= train_split]
        val_indices = valid_indices[split_vals > train_split]

        cell_min_f32 = cell_min.astype(np.float32)
        cell_range_f32 = cell_range.astype(np.float32)

        # --- Build datasets ---
        self.train_dataset = _SparseExpressionDataset(
            exp_array, cell_types,
            cell_min_f32, cell_range_f32, gene_mean, gene_std,
            train_indices)
        train_sampler = torch.utils.data.RandomSampler(
            self.train_dataset, replacement=True, num_samples=batch_size,
            generator=self._sampler_generator)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=batch_size,
            drop_last=True)

        self.val_dataset = _SparseExpressionDataset(
            exp_array, cell_types,
            cell_min_f32, cell_range_f32, gene_mean, gene_std,
            val_indices)
        self.val_dataloader = DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=batch_size,
            drop_last=False)

    @torch.no_grad()
    def forward_pass(self, x_0, t):
        """
        Forward diffusion process. Adds noise to the input according to the
        diffusion schedule at the given timesteps.

        Args:
            x_0 (torch.FloatTensor): Torch tensor for expression data. Rows
                are cells and columns are genes.
            t (torch.LongTensor): Torch tensor for diffusion time steps.

        Returns:
            tuple: (x_t, noise) where x_t is the noised input and noise is
                the sampled Gaussian noise that was added.
        """
        noise = torch.randn_like(x_0)
        mean_coef = self.mean_schedule.gather(dim=-1, index=t)
        std_coef = self.std_schedule.gather(dim=-1, index=t)
        x_t = mean_coef.unsqueeze(-1) * x_0 + std_coef.unsqueeze(-1) * noise
        return x_t, noise

    def train(self, n_steps=None):
        """
        Train the model. Dispatches to gradient accumulation or normal
        training based on the ``gradient_accumulation`` setting.

        Args:
            n_steps (int): Number of steps to train. If not provided, uses
                the n_steps specified during initialization.
        """
        if self.hp['gradient_accumulation']:
            self._train_with_gradient_accumulation(n_steps=None)
        else:
            self._train_normal(n_steps=None)
        
        
    def _train_normal(self, n_steps=None):
        """
        Train the initialized model for a number of steps. 

        Args:
            n_steps (int): Number of steps to train. If not provided, it will 
                train the model by the n_steps specified in class
                initialization. Please read our paper to see how to identify
                the convergence point.
        """
        start_time = datetime.now()
        eval_steps = self.hp['eval_on_n_steps']
        if n_steps is None:
            n_steps = self.hp['n_steps']
        sampled_adj = self.model.get_sampled_adj_()
        with tqdm(range(n_steps)) as pbar:
            for epoch in pbar:
                if self.hp['seed'] is not None:
                    torch.manual_seed(self.hp['seed'] + epoch)
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
                    with torch.autocast(
                        device_type=self.amp_device_type,
                        dtype=torch.bfloat16,
                        enabled=self.use_amp
                    ):
                        z = self.model(x_noisy, t, ct)
                        loss_ = F.mse_loss(noise, z, reduction='none')
                        loss = loss_.mean()

                        if hasattr(self.model, 'get_sampled_sparse_loss'):
                            loss_sparse = self.model.get_sampled_sparse_loss() * self.hp['sparse_loss_coef']
                        else:
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
                        if self.hp['seed'] is not None:
                            torch.manual_seed(self.hp['seed'] + n_steps + epoch)
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
                                with torch.autocast(
                                    device_type=self.amp_device_type,
                                    dtype=torch.bfloat16,
                                    enabled=self.use_amp
                                ):
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

    def _train_with_gradient_accumulation(self, n_steps=None):
        """
        Train the initialized model using gradient accumulation. For each batch,
        the gradient is computed for each sample individually and accumulated
        before the optimizer step. This reduces memory but increases training time.

        Args:
            n_steps (int): Number of steps to train. If not provided, it will 
                train the model by the n_steps specified in class 
                initialization.
        """
        start_time = datetime.now()
        eval_steps = self.hp['eval_on_n_steps']
        if n_steps is None:
            n_steps = self.hp['n_steps']
        
        batch_size = self.hp['batch_size']
        sampled_adj = self.model.get_sampled_adj_()
        
        # This will hold the per-gene losses from the final step
        final_losses_on_gene = torch.zeros(self.n_gene, device=self.device)

        with tqdm(range(n_steps)) as pbar:
            for step_idx in pbar:
                if self.hp['seed'] is not None:
                    torch.manual_seed(self.hp['seed'] + step_idx)
                # Fetch a single batch for the current step
                x_0_batch, ct_batch = next(iter(self.train_dataloader))
                x_0_batch = x_0_batch.to(self.device)
                ct_batch = ct_batch.to(self.device)
                
                # Zero gradients before starting the accumulation for the batch
                self.opt.zero_grad()
                
                batch_mse_loss = 0.0
                accumulated_loss_on_gene = torch.zeros(self.n_gene, device=self.device)

                # Loop over each sample in the batch to accumulate gradients
                for i in range(batch_size):
                    x_0 = x_0_batch[i:i+1]
                    ct = ct_batch[i:i+1]
                    
                    # Generate timestep for the single sample
                    t = torch.randint(0, self.hp['T'], (1,), device=self.device).long()

                    x_noisy, noise = self.forward_pass(x_0, t)
                    with torch.autocast(
                        device_type=self.amp_device_type,
                        dtype=torch.bfloat16,
                        enabled=self.use_amp
                    ):
                        z = self.model(x_noisy, t, ct)

                        # Calculate MSE loss, get per-gene loss with reduction='none'
                        loss_ = F.mse_loss(noise, z, reduction='none')

                    # Accumulate per-gene loss values for the last step
                    with torch.no_grad():
                        accumulated_loss_on_gene += loss_.float().squeeze()

                    # Calculate mean loss for the sample
                    loss_sample = loss_.mean()
                    batch_mse_loss += loss_sample.item()

                    # Scale loss for gradient accumulation
                    scaled_loss = loss_sample / batch_size
                    scaled_loss.backward()

                # Handle sparse loss once per effective batch
                with torch.autocast(
                    device_type=self.amp_device_type,
                    dtype=torch.bfloat16,
                    enabled=self.use_amp
                ):
                    if hasattr(self.model, 'get_sampled_sparse_loss'):
                        loss_sparse = self.model.get_sampled_sparse_loss() * self.hp['sparse_loss_coef']
                    else:
                        adj_m = self.model.get_adj_()
                        loss_sparse = adj_m.mean() * self.hp['sparse_loss_coef']

                if step_idx > 10:
                    loss_sparse.backward()

                # Perform the optimizer step after accumulating all gradients
                self.opt.step()
                
                # Logging and evaluation logic
                train_loss = (batch_mse_loss / batch_size)
                if step_idx > 10:
                    train_loss += loss_sparse.item()
                    
                sampled_adj_new = self.model.get_sampled_adj_()
                adj_diff = (sampled_adj_new - sampled_adj).mean().item() * (self.n_gene - 1)
                sampled_adj = sampled_adj_new

                pbar.set_description(
                    f'Training loss: {train_loss:.3f}, Change on Adj: {adj_diff:.3f}')
                
                epoch_log = {'train_loss': train_loss, 'adj_change': adj_diff}
                
                if step_idx % eval_steps == eval_steps - 1:
                    if self.evaluator is not None:
                        eval_result = self.evaluator.evaluate(self.model.get_adj())
                        for k in eval_result.keys():
                            epoch_log[k] = eval_result[k]
                    
                    if self.hp['train_split'] < 1:
                        if self.hp['seed'] is not None:
                            torch.manual_seed(self.hp['seed'] + n_steps + step_idx)
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
                                with torch.autocast(
                                    device_type=self.amp_device_type,
                                    dtype=torch.bfloat16,
                                    enabled=self.use_amp
                                ):
                                    z = self.model(x_noisy, t, ct)
                                    step_val_loss = F.mse_loss(
                                        noise, z, reduction='mean').item()
                                val_epoch_loss.append(step_val_loss)
                            epoch_log['val_loss'] = np.mean(val_epoch_loss)

                self.logger.log(epoch_log)
                final_losses_on_gene = accumulated_loss_on_gene # Save from the last step

        # Set the final per-gene losses, averaged over the batch
        self.losses_on_gene = (final_losses_on_gene / batch_size).detach().cpu().numpy()
        self.total_time_cost += int((datetime.now() - start_time).total_seconds())
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
        Obtain a GRN object. You need to provide the gene names.

        Args:
            gene_names (np.ndarray): An array of names of all genes. The order
                of genes should be the same as the order used in your expression
                table.
            tf_names (np.ndarray): An array of names of all transcriptional
                factors. The order should be the same as the order used in
                your expression table.
            top_gene_percentile (int): If provided, we will set the value on
                weak links to be zero. It is useful if you want to save the
                regulatory relationship in a GRN object as a sparse matrix.

        Returns:
            GRN: A GRN object containing the inferred regulatory network.
        """
        adj = self.model.get_adj()
        return GRN(adj, gene_names, tf_names, top_gene_percentile)

    def get_adj(self):
        """
        Obtain the adjacency matrix. The values in this adjacency matrix have
        been scaled using regulatory norm. You may expect strong links to go
        beyond 5 or 10 in most cases.

        Returns:
            np.ndarray: 2D adjacency matrix of shape (n_gene, n_gene).
        """
        return self.model.get_adj()
        
