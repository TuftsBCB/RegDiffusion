# Getting Startted with GRN inference using diffusion model

Diffusion model has been widely used in generative AI, especially in the vision domain. In our paper, we proposed RegDiffusion, a diffusion based model for GRN inference. Compared with previous model, RegDiffusion completes inference within a fraction of time and yield better benchmarking results. 

In this tutorial, we provide an example of running GRN inference using RegDiffusion and generating biological insights from the inferred network. 

## Requirements

We will need the python package `regdiffusion` for GRN inference. For accelerated inference speed, you may want to run `regdiffusion` on GPU devices with the latest CUDA installation.

```
import regdiffusion as rd
import numpy as numpy
```

## Data Loading

The input of `regdiffusion` is simply a single-cell gene expression matrix, where the columns are genes and rows are cells. We expect you to log transform your data. RegDiffusion is capable to infer GRNs among 10,000+ genes (depending on GPU hardware) within minutes so there is no need to apply heavy gene filtering. The only genes you may want to remove are genes that are not expressed at all (total raw count on all cells == 0). 

The `regdiffusion` package comes with a set of preprocessed data, including the [BEELINE benchmarks](https://pubmed.ncbi.nlm.nih.gov/31907445/), [Hammond microglia](https://pubmed.ncbi.nlm.nih.gov/30471926/) in male adult mice, and another labelled microglia subset from a [mice cerebellum atlas project](https://singlecell.broadinstitute.org/single_cell/study/SCP795/a-transcriptomic-atlas-of-the-mouse-cerebellum#study-summary). 

Here we use the `mESC` data from the BEELINE benchmark. The `mESC` data comes from [Mouse embryonic stem cells](https://www.nature.com/articles/s41467-018-02866-0). It has 421 cells and 1,620 genes. 

If you want to see the inference on a larger network with 14,000+ genes and 8,000+ cells, check out the other example. 

```
bl_dt, bl_gt = rd.data.load_beeline(
    benchmark_data='mESC', benchmark_setting='1000_STRING'
)
```

Here, `load_beeline` gives you a tuple, where the first element is an anndata of the single cell experession data and the second element is an array of all the ground truth links (based on the STRING network in this case). 

```
bl_dt
```
AnnData object with n_obs × n_vars = 421 × 1620
    obs: 'cell_type', 'cell_type_index'

## GRN Inference

You are recommended to use the provided trainer to train a RegDiffusion Model. You need to provide the expression data in a numpy array to the trainer. 

During the training process, the training loss and the average amount of change on the adjacency matrix are provided on the progress bar. The model converges when the step change n the adjacency matrix is near-zero. By default, the `train` method will train the model for 1,000 iterations. It should be sufficient in most cases. If you want to keep training the model afterwards, you can simply call the `train` methods again with the desired number of iterations. 

```
rd_trainer = rd.RegDiffusionTrainer(bl_dt.X)
rd_trainer.train()
```

We run this experiment on an A100 card and the inference finishes within 8 seconds. 

When ground truth links are avaiable, you can test the inference performance by setting up an evaluator. You need to provide both the ground truth links and the gene names. Note that the order of the provided gene names here should be the same as the column order in the expression table (and the inferred adjacency matrix). 

```
evaluator = rd.evaluator.GRNEvaluator(bl_gt, bl_dt.var_names)
inferred_adj = rd_trainer.get_adj()
evaluator.evaluate(inferred_adj)
```

## GRN object

In order to facilitate the downstream analyses on GRN, we defined an `GRN` object in the `regdiffusion` package. You need to provide the gene names in the same order as in your expression table.