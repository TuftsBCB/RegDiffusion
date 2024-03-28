import numpy as np
import scanpy as sc
import pandas as pd
from tqdm import tqdm
import zipfile
import os
from .utils import download_file


def load_atlas_microglia(data_dir='data') -> sc.AnnData:
    """
    Load single cell for microglia from Broad Institute SCP795

    Data Source: https://singlecell.broadinstitute.org/single_cell/study/SCP795/a-transcriptomic-atlas-of-the-mouse-cerebellum#study-summary

    Paper: A transcriptomic atlas of mouse cerebellar cortex comprehensively 
    defines cell types

    Paper Link: https://www.nature.com/articles/s41586-021-03220-z

    Raw data is count data. We select all genes that have non-zero expression. 
    We also removed all gene models, Mitochondrial genes, and ribosome genes. 
    We used log-plus-one to transform the count data. 

    The output is an AnnData object where rows are cells and columns are genes.

    Args:
        data_dir (str): Parent directory to save and load the data. If the path
        does not exist, it will be created. Data will be saved in a 
        subdirectory under the provided path. 
    """
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
    """
    Load single cell for microglia from Hammond Microglia dataset

    Data Source: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE121654

    Paper: Single-Cell RNA Sequencing of Microglia throughout the Mouse Lifespan
    and in the Injured Brain Reveals Complex Cell-State Changes
    Paper Link: https://www.cell.com/immunity/fulltext/S1074-7613(18)30485-0

    IMPORTANT! This is not the complete data from the study. We only selected
    data from the 4 adult male mouses at P100. Here are their accession IDs. 

    GSM3442026	P100 male no 1
    GSM3442027	P100 male no 2
    GSM3442030	P100 male no 3
    GSM3442031	P100 male no 4
    
    Raw data has already been log transformed. We select all genes that have 
    non-zero expression. We also removed all gene models, Mitochondrial genes, 
    and ribosome genes. 

    The output is an AnnData object where rows are cells and columns are genes.

    Args:
        data_dir (str): Parent directory to save and load the data. If the path
        does not exist, it will be created. Data will be saved in a
        subdirectory under the provided path. 
    
    """
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