import numpy as np
import scanpy as sc
import pandas as pd
import urllib
from tqdm import tqdm
import zipfile
import os

# Read ground truth
def load_beeline_ground_truth(data_dir, gene_names):
    n_gene = len(gene_names)
    ground_truth = pd.read_csv(f'{data_dir}/label.csv')
    TF = set(ground_truth['Gene1'])
    All_gene = set(ground_truth['Gene1']) | set(ground_truth['Gene2'])

    tf_mask = (np.zeros(n_gene) == 1)
    gene_mask = (np.zeros(n_gene) == 1)
    tf_map = {}
    gene_map = {}
    for i, item in enumerate(gene_names):
        if item in TF:
            tf_mask[i] = True
            tf_map[item] = len(tf_map)
        if item in All_gene:
            gene_mask[i] = True
            gene_map[item] = len(gene_map)
    
    y_true = np.zeros([len(TF), len(All_gene)])
    truth_edges = []
    for i, r in ground_truth.iterrows():
        y_true[tf_map[r['Gene1']], gene_map[r['Gene2']]] = 1.0
        truth_edges.append([r['Gene1'], r['Gene2']])
    y_true = y_true.flatten()
    
    output = {
        'tf_mask': tf_mask,
        'gene_mask': gene_mask,
        'truth_edges': truth_edges,
        'y_true': y_true
    }
    return output

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
    
    ground_truth = load_beeline_ground_truth(data_dir, data.var_names)
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

# Modified from https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
def download_file(url, file_path, chunk_size=1024):
    req = urllib.request.urlopen(url)
    with open(file_path, 'wb') as f, tqdm(
        desc=f'Downloading {file_path}', total=req.length, unit='iB', 
        unit_scale=True, unit_divisor=1024
    ) as bar:
        for _ in range(req.length // chunk_size + 1):
            chunk = req.read(chunk_size)
            if not chunk: break
            size = f.write(chunk)
            bar.update(size)
