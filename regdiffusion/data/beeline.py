import numpy as np
import scanpy as sc
import pandas as pd
from tqdm import tqdm
import zipfile
import os
from .utils import download_file

# Read ground truth
def load_beeline_ground_truth(data_dir):
    ground_truth = pd.read_csv(f'{data_dir}/label.csv')
    return ground_truth.values

beeline_cell_type_dict = {
    'hESC': [1],
    'hHep': [1],
    'mDC': [1],
    'mESC': [2],
    'mHSC-E': [0],
    'mHSC-GM': [0],
    'mHSC-L': [0]
}

def cell_type_separator(sc_data, cell_type_element_indices=[0], sep='_'):
    cell_types = []
    for x in sc_data.obs_names:
        x_elements = x.split(sep)
        cell_type = [x_elements[i] for i in cell_type_element_indices]
        cell_types.append(sep.join(cell_type))        
    cell_type_set = set(cell_types)
    cell_type_dict = {ct:i for i, ct in enumerate(cell_type_set)}
    cell_type_indices = [cell_type_dict[x] for x in cell_types]
    return cell_types, cell_type_indices

def load_beeline(data_dir='data', benchmark_data='hESC', 
                 benchmark_setting='500_STRING'):
    ''' Load BEELINE
    
    Load BEELINE data into memory (download if necessary).
    
    Parameters
    ----------
    data_dir: str
        Root folder where the BEELINE data is/will be located. 
    benchmark_data: str
        Benchmark datasets. Choose among `hESC`, `hHep`, `mDC`, 
        `mESC`, `mHSC`, `mHSC-GM`, and `mHSC-L`.
    benchmark_setting: str
        Benchmark settings. Choose among `500_STRING`, 
        `1000_STRING`, `500_Non-ChIP`, `1000_Non-ChIP`, 
        `500_ChIP-seq`, `1000_ChIP-seq`, `500_lofgof`,
        and `1000_lofgof`. If either of the `lofgof` settings
        is choosed, only `mESC` data is available.  
        
    Returns
    -------
    tuple
        First element is a scanpy data with cells on rows and 
        genes on columns. Second element is the corresponding 
        BEELINE ground truth data 
    '''
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(f'{data_dir}/BEELINE/'):
        download_beeline(data_dir)
    data_dir = f'{data_dir}/BEELINE/{benchmark_setting}_{benchmark_data}'
    data = sc.read(f'{data_dir}/data.csv')
    # We do need to transpose the data to have cells on rows and genes on columns
    data = data.transpose()
    cell_types, cell_type_indices = cell_type_separator(
        data, beeline_cell_type_dict[benchmark_data])
    data.obs['cell_type'] = cell_types
    data.obs['cell_type_index'] = cell_type_indices
    
    ground_truth = load_beeline_ground_truth(data_dir)
    return data, ground_truth

def download_beeline(save_dir, remove_zip=True):
    if not os.path.exists(save_dir):
        raise Exception("save_dir does not exist")
    zip_path = os.path.join(save_dir, 'BEELINE.zip')
    download_file('https://bcb.cs.tufts.edu/DAZZLE/BEELINE.zip', 
                  zip_path)
    with zipfile.ZipFile(zip_path,"r") as zip_ref:
        for file in tqdm(desc='Extracting', iterable=zip_ref.namelist(), 
                         total=len(zip_ref.namelist())):
            zip_ref.extract(member=file, path=save_dir)
    if remove_zip:
        os.remove(zip_path)