import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from .models import DAZZLE
from .evaluate import get_metrics
from tqdm import tqdm
from .logger import LightLogger

DEFAULT_DEEPSEM_CONFIGS = {
    # Train/Test split
    'train_split': 1.0,
    'train_split_seed': None, 
    
    # Neural Net Definition
    'hidden_dim': 128,
    'z_dim': 1,
    'A_dim': 1,
    'train_on_non_zero': False,
    'dropout_augmentation_p': 0.0,
    'dropout_augmentation_type': 'all',
    'cuda': True,
    
    # Loss
    'alpha': 100,
    'beta': 1,
    'chi': 0,
    'h_scale': 0,
    'delayed_steps_on_sparse': 0,
    
    # Neural Net Training
    'number_of_opt': 2,
    'batch_size': 64,
    'n_epochs': 120,
    'schedule': [1000, 2000],
    'eval_on_n_steps': 10,
    'early_stopping': 0,
    'lr_nn': 1e-4,
    'lr_adj': 2e-5,
    'K1': 1,
    'K2': 2
}

DEFAULT_DAZZLE_CONFIGS = {
    # Train/Test split
    'train_split': 1.0,
    'train_split_seed': None, 
    
    # Neural Net Definition
    'hidden_dim': 128,
    'z_dim': 1,
    'A_dim': 1,
    'train_on_non_zero': True,
    'dropout_augmentation_p': 0.1,
    'dropout_augmentation_type': 'all',
    'cuda': True,
    
    # Loss
    'alpha': 100,
    'beta': 1,
    'chi': 0.5,
    'h_scale': 0,
    'delayed_steps_on_sparse': 5,
    
    # Neural Net Training
    'number_of_opt': 1,
    'batch_size': 64,
    'n_epochs': 120,
    'schedule': [1000, 2000],
    'eval_on_n_steps': 10,
    'early_stopping': 0,
    'lr_nn': 1e-4,
    'lr_adj': 2e-5,
    'K1': 1,
    'K2': 2
}
def one_hot(x):
    x_unique = np.unique(x)
    n_obs = x.shape[0]
    n_classes = x_unique.shape[0]
    
    label_dict = {label: i for i, label in enumerate(x_unique)}
    
    one_hot_matrix = np.zeros([n_obs, n_classes])
    for i, label in enumerate(x):
        one_hot_matrix[i, label_dict[label]] = 1.0
    return one_hot_matrix, x_unique

def runDAZZLE(exp_array, configs, 
              ground_truth=None, logger=None, progress_bar=True):
    '''
    Initialize and Train a DAZZLE model with configs
    
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
    if configs['cuda']:
        if not torch.cuda.is_available():
            print('Cuda is not available for torch. Proceed on CPU instead.')
            configs['cuda'] = False
    
    if configs['early_stopping'] != 0 and configs['train_split'] == 1.0:
        raise Exception(
            "You indicate early stopping but you have not specified any ", 
            "validation data. Consider decrease your train_split. ")
    es = configs['early_stopping']
    
    n_obs, n_gene = exp_array.shape
        
    # Logger -------------------------------------------------------------------
    if logger is None:
        logger = LightLogger()
    logger.set_configs(configs)
    note_id = logger.start()

    # cell_min = exp_array.min(1, keepdims=True)
    # cell_max = exp_array.max(1, keepdims=True)
    # exp_array = (exp_array - cell_min) / (cell_max - cell_min)
    
    # Global Mean/Std ----------------------------------------------------------
    global_mean = torch.FloatTensor(exp_array.mean(0))
    global_std = torch.FloatTensor(exp_array.std(0))

    # Train/Test split if requested --------------------------------------------
    assert configs['train_split']>0 and configs['train_split']<=1, \
        f'Expect 0<configs["train_split"]<=1'
    has_train_test_split = (configs['train_split'] != 1.0)
    
    if configs['train_split_seed'] is None:
        train_mask = np.random.rand(n_obs)
    else:
        rs = np.random.RandomState(seed=configs['train_split_seed'])
        train_mask = rs.rand(n_obs)
        
    train_dt = TensorDataset(
        torch.FloatTensor(exp_array[train_mask <= configs['train_split'], ]),
    )
    train_loader = DataLoader(
        train_dt, batch_size=configs['batch_size'], shuffle=True)
    if has_train_test_split:
        val_dt = TensorDataset(
            torch.FloatTensor(exp_array[train_mask > configs['train_split'], ]),
        )
        val_loader = DataLoader(
            val_dt, batch_size=configs['batch_size'], shuffle=True)

    # Defining Model -----------------------------------------------------------
    vae = DAZZLE(
        n_gene = n_gene, 
        hidden_dim=configs['hidden_dim'], z_dim=configs['z_dim'], 
        A_dim = configs['A_dim'],
        train_on_non_zero=configs['train_on_non_zero'], 
        dropout_augmentation_p=configs['dropout_augmentation_p'],
        dropout_augmentation_type=configs['dropout_augmentation_type']
        # A_dim=configs['A_dim']
    )
    
    # Move things to cuda if necessary -----------------------------------------
    if configs['cuda']:
        global_mean = global_mean.cuda()
        global_std = global_std.cuda()
        vae = vae.cuda()
    
    if configs['number_of_opt'] == 2:
        opt_nn = torch.optim.RMSprop(vae.parameters(), lr=configs['lr_nn'])
        opt_adj = torch.optim.RMSprop([vae.adj_A], lr=configs['lr_adj'])
        scheduler_nn = torch.optim.lr_scheduler.StepLR(
            opt_nn, step_size=configs['schedule'][0], gamma=0.5)
    else:
        param_all_but_adj = [p for i, p in enumerate(vae.parameters()) if i != 0]
        param_adj = [vae.adj_A]
        opt = torch.optim.Adam([{'params': param_all_but_adj}, 
                                   {'params': param_adj}], 
                                  lr=configs['lr_nn'], 
                               weight_decay=0.00, betas=[0.9, 0.9]
                              )
        # opt = torch.optim.RMSprop([{'params': param_all_but_adj}, 
        #                            {'params': param_adj}], 
        #                           lr=configs['lr_nn'], 
        #                        weight_decay=0.00
        #                       )
        opt.param_groups[0]['lr'] = configs['lr_nn']
        opt.param_groups[1]['lr'] = configs['lr_adj']
        opt.param_groups[1]['weight_decay'] = 0
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=configs['schedule'], gamma=0.5)
        
    # Training loops -----------------------------------------------------------
    es_tracks = []
    adjs = []
    disable_tqdm = (progress_bar == False)
    for epoch in tqdm(range(configs['n_epochs']), disable=disable_tqdm):
        if configs['number_of_opt'] == 2:
            vae.train(True)
            iteration_for_A = epoch%(configs['K1']+configs['K2'])>=configs['K1']
            vae.adj_A.requires_grad = iteration_for_A
            evaluation_turn = (epoch % configs['eval_on_n_steps'] == 0)

            # go through training samples 
            eval_log = {
                'train_loss_rec': 0, 'train_loss_kl': 0, 'train_loss_sparse': 0, 
                'train_loss_da': 0, 'train_loss': 0
            }
            for i, batch in enumerate(train_loader):
                exp = batch[0]
                if configs['cuda']:
                    exp = exp.cuda()

                if iteration_for_A:
                    opt_adj.zero_grad()
                    out = vae(exp, global_mean, global_std)
                else:
                    opt_nn.zero_grad()
                    out = vae(exp, global_mean, global_std)
                loss = out['loss_rec'] + configs['beta'] * out['loss_kl'] 
                adj_m = vae.get_adj_()
                loss_sparse = torch.norm(adj_m, 1) / n_gene / n_gene
                if epoch >= configs['delayed_steps_on_sparse']:
                    loss += configs['alpha'] * loss_sparse
                loss.backward()
                if iteration_for_A:
                    opt_adj.step()
                else:
                    opt_nn.step()
                scheduler_nn.step()
                eval_log['train_loss_rec'] += out['loss_rec'].detach().cpu().item()
                eval_log['train_loss_kl'] += out['loss_kl'].detach().cpu().item()
                eval_log['train_loss_sparse'] += loss_sparse.detach().cpu().item()
                eval_log['train_loss'] += loss.detach().cpu().item()
        else:
            vae.train(True)
            evaluation_turn = (epoch % configs['eval_on_n_steps'] == 0)

            # go through training samples 
            eval_log = {
                'train_loss_rec': 0, 'train_loss_kl': 0, 'train_loss_sparse': 0,
                'train_loss_da': 0, 'train_loss': 0, 
            }
            for i, batch in enumerate(train_loader):
                exp = batch[0]
                if configs['cuda']:
                    exp = exp.cuda()

                opt.zero_grad()
                out = vae(exp, global_mean, global_std)

                loss = out['loss_rec'] + configs['beta'] * out['loss_kl'] 
                adj_m = vae.get_adj_()
                loss_sparse = torch.norm(adj_m, 1) / n_gene / n_gene
                
                if epoch >= configs['delayed_steps_on_sparse']:
                    loss += configs['alpha'] * loss_sparse
                if configs['dropout_augmentation_p'] != 0:
                    loss += configs['chi'] * out['loss_da']
                loss.backward()
                opt.step()
                scheduler.step()
                eval_log['train_loss_rec'] += out['loss_rec'].detach().cpu().item()
                eval_log['train_loss_kl'] += out['loss_kl'].detach().cpu().item()
                eval_log['train_loss_da'] += out['loss_da'].detach().cpu().item()
                eval_log['train_loss_sparse'] += loss_sparse.detach().cpu().item()
                eval_log['train_loss'] += loss.detach().cpu().item()
        
        for log_item in eval_log.keys():
            eval_log[log_item] /= (i+1)
        
        # go through val samples
        if evaluation_turn:
            adj_matrix = adj_m.cpu().detach().numpy()
            adjs.append(adj_matrix)
            eval_log['negative_adj'] = int(np.sum(adj_matrix < -1e-5))
            if ground_truth is not None:
                epoch_perf = get_metrics(adj_matrix, ground_truth)
                for k in epoch_perf.keys():
                    eval_log[k] = epoch_perf[k]
            
            if has_train_test_split:
                eval_log['val_loss_rec'] = 0
                eval_log['val_loss_kl'] = 0
                eval_log['val_loss_sparse'] = 0
                vae.train(False)
                for batch in val_loader:
                    x = batch[0]
                    if configs['cuda']:
                        x = x.cuda()
                    out = vae(x, global_mean, global_std)
                    eval_log['val_loss_rec'] += out['loss_rec'].detach().cpu().item()
                    eval_log['val_loss_kl'] += out['loss_kl'].detach().cpu().item()
                    eval_log['val_loss_sparse'] += out['loss_sparse'].detach().cpu().item()
                if epoch >= configs['delayed_steps_on_sparse']:
                    es_tracks.append(eval_log['val_loss_rec'])
            
            logger.log(eval_log)
            # early stopping
            if (es > 0) and (len(es_tracks) > (es + 2)):
                if min(es_tracks[(-es-1):]) < min(es_tracks[(-es):]):
                    print('Early stopping triggered')
                    break
    logger.finish()
    vae = vae.cpu()
    vae.classifier_pos_weight = vae.classifier_pos_weight.cpu()
    return vae, adjs

def runDAZZLE_ensemble(exp_array, configs,
                       ground_truth=None, logger=None, rep_times=10):
    trained_models = []
    final_adjs = []
    for _ in tqdm(range(rep_times)):
        vae, adjs = runDAZZLE(exp_array, configs, ground_truth, logger)
        trained_models.append(vae)
        final_adjs.append(vae.get_adj())
    ensembled_adj = sum(final_adjs)
    return trained_models, ensembled_adj
