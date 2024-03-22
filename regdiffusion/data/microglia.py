import numpy as np
import scanpy as sc
import pandas as pd
from tqdm import tqdm
import zipfile
import os
from .utils import download_file


def load_atlas_microglia(data_dir='data'):
    ''' Load single cell for microglia from SCP795

    Data Source: https://singlecell.broadinstitute.org/single_cell/study/SCP795/a-transcriptomic-atlas-of-the-mouse-cerebellum#study-summary

    Data is just count data and has been log 
    transformed at the end of the loading step
    '''
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    file_dir = f'{data_dir}/scp795_microglia/'
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
        download_regdiffusion_data(file_dir, 'atlas_microglia.zip')
    ann_dt = sc.read_h5ad(f'{file_dir}scp795_microglia.h5ad')
    ann_dt.X = ann_dt.X.toarray()
    ann_dt = ann_dt.transpose()
    sc.pp.filter_genes(ann_dt, min_counts=1)
    ann_dt = ann_dt[:, ~ann_dt.var_names.str.startswith('Gm')]
    ann_dt = ann_dt[:, ~ann_dt.var_names.str.startswith('mt')]
    ann_dt = ann_dt[:, ~ann_dt.var_names.str.startswith('Rpl')]
    ann_dt = ann_dt[:, ~ann_dt.var_names.str.startswith('Rps')]
    ann_dt = sc.pp.log1p(ann_dt, copy=True)

    return ann_dt

def load_hammond_microglia(data_dir='data'):
    ''' Load single cell for hammond microglia

    We selected the 4 P100 male mice data. Data has been log transformed. 
    '''
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    file_dir = f'{data_dir}/hammond_microglia/'
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
        download_regdiffusion_data(file_dir, 'hammond_microglia.zip')
    ann_dt = sc.read_csv(f'{file_dir}/hammond_male_p100_microglia.csv')
    ann_dt = ann_dt.transpose()
    sc.pp.filter_genes(ann_dt, min_counts=0.0001)
    ann_dt = ann_dt[:, ~ann_dt.var_names.str.startswith('Gm')]
    ann_dt = ann_dt[:, ~ann_dt.var_names.str.startswith('mt')]
    ann_dt = ann_dt[:, ~ann_dt.var_names.str.startswith('Rpl')]
    ann_dt = ann_dt[:, ~ann_dt.var_names.str.startswith('Rps')]

    return ann_dt

def download_regdiffusion_data(save_dir, file_name, remove_zip=True):
    if not os.path.exists(save_dir):
        raise Exception("save_dir does not exist")
    zip_path = os.path.join(save_dir, file_name)
    download_file(
        f'https://bcb.cs.tufts.edu/regdiffusion/{file_name}', 
        zip_path)
    with zipfile.ZipFile(zip_path,"r") as zip_ref:
        for file in tqdm(desc='Extracting', iterable=zip_ref.namelist(), 
                         total=len(zip_ref.namelist())):
            zip_ref.extract(member=file, path=save_dir)
    if remove_zip:
        os.remove(zip_path)