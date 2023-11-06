# RegDiffusion: Probabilistic Diffusion-Based Neural Inference of Gene Regulatory Networks

Welcome to the official repository for RegDiffusion, an innovative and high-performance approach for inferring Gene Regulatory Networks (GRNs) from single-cell RNA sequencing data. RegDiffusion leverages the principles of Denoising Diffusion Probabilistic Models to predict regulatory interactions with lifted performance and efficiency.

The details of RegDiffusion is described in the following paper. 

```
From Noise to Knowledge: Probabilistic Diffusion-Based Neural Inference of Gene Regulatory Networks
Hao Zhu, Donna K. Slonim
bioRxiv 2023.11.05.565675; doi: https://doi.org/10.1101/2023.11.05.565675
```

## Installation

RegDiffusion is on pypi.

```
pip install regdiffusion
```


## Getting Started

To run a quick example, you can try to run the one of the BEELINE benchmarking datasets. You can also check out [this Google colab](https://colab.research.google.com/drive/1Fa6o-0_-bXiZjhcDWhrt_XMjZfLqrkbx?usp=sharing).

```
import regdiffusion as rd

bl_data, bl_gt = rd.load_beeline(
    benchmark_data='mESC', 
    benchmark_setting='1000_STRING')
    
configs = rd.DEFAULT_REGDIFFUSION_CONFIGS

model = rd.runRegDiffusion(bl_data.X, configs)
adj = model.get_adj()
print(rd.get_metrics(adj, bl_gt))

# extract edges
tf_mask = bl_gt['tf_mask']
rd.extract_edges(adj, bl_data.var_names, 
                 TFmask=tf_mask, threshold=0.001)
```
