import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from .models import RegDiffusion
from .evaluate import get_metrics
from tqdm import tqdm
from .logger import LightLogger

DEFAULT_REGDIFFUSION_CONFIGS = {
    'T': 5000,
    'start_noise': 0.0001,
    'end_noise': 0.02,
    'lr_nn': 1e-3,
    'lr_adj': 2e-5,
    'weight_decay_nn': 0.1,
    'weight_decay_adj': 0.01,
    'sparse_loss_coef': 0.25,
    'adj_dropout': 0.3, 
    'batch_size': 128,
    'time_dim': 64,
    'celltype_dim': 32,
    'hidden_dims': [16, 16, 16],
    'n_epoch': 250,
    'device': 'cuda',
    'verbose': False,
    'train_split': 1.0,
    'train_split_seed': 123,
    'eval_on_n_steps': 10
}

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

@torch.no_grad()
def forward_pass(x_0, t, mean_schedule, std_schedule):
    noise = torch.randn_like(x_0)
    mean_coef = mean_schedule.gather(dim=-1, index=t)
    std_coef = std_schedule.gather(dim=-1, index=t)
    x_t = mean_coef.unsqueeze(-1) * x_0 + std_coef.unsqueeze(-1) * noise
    return x_t, noise

def runRegDiffusion(
    exp_array, configs, 
    is_count=False, cell_types=None, 
    ground_truth=None, 
    logger=None, progress_bar=True):
    '''
    Initialize and Train a RegDiffusion model with configs
    
    Parameters
    ----------
    exp_array: np.array
        Expression data with cells on rows and genes on columns. 
    configs: dict
        A dictionary defining various hyperparameters of the 
        model. See Hyperparameters include `train_split`, 
        `train_split_seed`, `batch_size`, `hidden_dim`, `z_dim`,
        `train_on_non_zero`, `dropout_augmentation`, `cuda`,
        `alpha`, `beta`, `delayed_steps_on_sparse`, `n_epochs`, 
        `eval_on_n_steps`, `lr_nn`, `lr_adj`, `K1`, and `K2`. 
    ground_truth: tuple or None
        (Optional, only for BEELINE evaluation) You don't need 
        to define this parameter when you execute DAZZLE on real 
        datasets when the ground truth network is unknown. For 
        evaluations on BEELINE, 
        BEELINE ground truth object exported by 
        data.load_beeline_ground_truth. The first element of this
        tuple is eval_flat_mask, the boolean mask on the flatten
        adjacency matrix to identify TFs and target genes. The
        second element is the lable values y_true after flatten.
    logger: LightLogger or None
        Either a predefined logger or None to start a new one. This 
        logger contains metric information logged during training. 
    progress_bar: bool
        Whether to display a progress bar on epochs. 
        
    Returns
    -------
    (torch.Module, List)
        This function returns a tuple of the trained model and a list of 
        adjacency matrix at all evaluation points. 
    '''
        
    # Logger -------------------------------------------------------------------
    if logger is None:
        logger = LightLogger()
    logger.set_configs(configs)
    note_id = logger.start()
    
    # Define diffusion schedule
    betas = linear_beta_schedule(
        configs['T'], configs['start_noise'], configs['end_noise']
    )
    alphas = 1. - betas
    alpha_bars = torch.cumprod(alphas, axis=0)
    mean_schedule = torch.sqrt(alpha_bars).to(configs['device'])
    std_schedule = torch.sqrt(1. - alpha_bars).to(configs['device'])

    # Preprocess data
    if is_count:
        exp_array = np.log2(exp_array + 1)
    if cell_types is None:
        cell_types = np.zeros(exp_array.shape[0], dtype=int)
    n_celltype = len(np.unique(cell_types))
    n_cell, n_gene = exp_array.shape
    
    # Normalize data
    cell_min = exp_array.min(axis=1, keepdims=True)
    cell_max = exp_array.max(axis=1, keepdims=True)
    normalized_X = (exp_array - cell_min) / (cell_max - cell_min)
    normalized_X = (normalized_X - normalized_X.mean(0))/normalized_X.std(0)

    # Train/test split
    random_state = np.random.RandomState(configs['train_split_seed'])
    train_test_split = random_state.rand(normalized_X.shape[0])
    train_index = train_test_split <= configs['train_split']
    test_index = train_test_split > configs['train_split']

    x_tensor_train = torch.tensor(normalized_X[train_index, ], dtype=torch.float32)
    celltype_tensor_train = torch.tensor(cell_types[train_index], dtype=int)
    x_tensor_test = torch.tensor(normalized_X[test_index, ], dtype=torch.float32)
    celltype_tensor_test = torch.tensor(cell_types[test_index],dtype=int)

    train_dataset = torch.utils.data.TensorDataset(
        x_tensor_train, celltype_tensor_train
    )
    train_sampler = torch.utils.data.RandomSampler(
        train_dataset, replacement=True, num_samples=normalized_X.shape[0]*1)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        sampler=train_sampler,
        batch_size = configs['batch_size'], 
        drop_last=True)
    
    test_dataset = torch.utils.data.TensorDataset(
        x_tensor_test, celltype_tensor_test
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        shuffle=False,
        batch_size = configs['batch_size'], 
        drop_last=False)
    
    model = RegDiffusion(
        n_gene=n_gene, 
        time_dim=configs['time_dim'], 
        n_celltype=n_celltype,
        celltype_dim = configs['celltype_dim'],
        hidden_dims=configs['hidden_dims'],
        adj_dropout=configs['adj_dropout']
    )

    param_all_but_adj = [p for i, p in enumerate(model.parameters()) if i != 0]
    param_adj = [model.adj_A]
    opt = torch.optim.Adam(
        [{'params': param_all_but_adj}, {'params': param_adj}], 
        lr=configs['lr_nn'], 
        weight_decay=configs['weight_decay_nn'], betas=[0.9, 0.99]
    )
    opt.param_groups[0]['lr'] = configs['lr_nn']
    opt.param_groups[1]['lr'] = configs['lr_adj']
    opt.param_groups[1]['weight_decay'] = configs['weight_decay_adj']

    model.to(configs['device'])

    for epoch in tqdm(range(configs['n_epoch'])):
        epoch_loss = []
        for step, batch in enumerate(train_dataloader):
            x_0, ct = batch
            x_0 = x_0.to(configs['device'])
            ct = ct.to(configs['device'])
            opt.zero_grad()
            t = torch.randint(
                0, configs['T'], (x_0.shape[0],), 
                device=configs['device']).long()

            x_noisy, noise = forward_pass(x_0, t, mean_schedule, std_schedule)
            z = model(x_noisy, t, ct)
            loss = F.mse_loss(noise, z, reduction='mean')

            adj_m = model.get_adj_()
            loss_sparse = adj_m.mean() * configs['sparse_loss_coef']

            if epoch > 3:
                loss = loss + loss_sparse 
            loss.backward()
            opt.step()
            epoch_loss.append(loss.item())
        if epoch % configs['eval_on_n_steps'] == configs['eval_on_n_steps'] - 1:
            if ground_truth is None:
                eval_result = {}
            else:
                eval_result = get_metrics(model.get_adj(), ground_truth)
            if configs['train_split'] < 1:
                with torch.no_grad():
                    test_epoch_loss = []
                    for step, batch in enumerate(test_dataloader):
                        x_0, ct = batch
                        x_0 = x_0.to(configs['device'])
                        ct = ct.to(configs['device'])
                        t = torch.randint(
                            0, configs['T'], (x_0.shape[0],), 
                            device=configs['device']).long()

                        x_noisy, noise = forward_pass(x_0, t, mean_schedule, std_schedule)
                        z = model(x_noisy, t, ct)
                        step_test_loss = F.mse_loss(noise, z, reduction='mean').item()
                        test_epoch_loss.append(step_test_loss)
                    eval_result[f'test_loss'] = np.mean(test_epoch_loss)
            eval_result['train_loss'] = np.mean(epoch_loss)
            if configs['verbose']:
                print(eval_result)
            logger.log(eval_result)
    logger.finish()

    return model
