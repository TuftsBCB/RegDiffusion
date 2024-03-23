RegDiffusion
=============

RegDiffusion is a very fast regulatory network inference algorithm based on 
probabilistic diffusion model. It works well on genes and is capable to rapidly
predict biologically verifiable links from large single cell RNA-seq data with 
10,000+ genes.

Installation
------------

``regdiffusion`` is available on pypi:

    pip install regdiffusion

Quick Tour
----------
This package `regdiffusion` provides the official implementation of the
RegDiffusion algorithm and a set of easy-to-use companion tools to evaluate,
analyze, and visualize the inferred network. We also provide access tools to 
GRN benchmarks and preprocessed single cell datasets for evaluation. 

We tried to keep the top level interface straightforward. Right now, it only 
consists of 4 components: the `RegDiffusionTrainer` class, the `GRN` class, the 
`GRNEvaluator` class, and the `data` module. 

- `RegDiffusionTrainer`: You can use it to train a `RegDiffusion` model by 
  providing log transformed expression data in a `numpy` array. The training
  process could be either started or continued using the `.train()` method. You 
  can export the inferred `GRN` using the `.get_grn()` method.
- `GRN`: The `GRN` class provides a container to save the inferred adjacency
  matrix and the corresponding gene names. You can save the `GRN` object to 
  a local `HDF5` file using the `.to_hdf5()` method and reload the saved file 
  using the `read_hdf5()` function. It also comes with functionalities to 
  export or visualize local regions. 
- `GRNEvaluator`: The ground truth of regulatory relationship often exist as 
  list of edges but the values to be evaluated are often in adjacency matrix. 
  The `GRNEvaluator` class is designed to fill the gap. Right now it supports
  common metrics such as AUROC, AUPR, AUPR Ratio, EP, and EPR. 
- `data` module: Right now, the `data` module includes quick access to BEELINE 
  benchmarks and our preprocessed single cell datasets on mouse microglia. 

Inference Speed
---------------
Inference on networks with 15,000 genes takes under 5 minutes with modern 
accelerated computing. In contrast, previous VAE based models would take more 
than 4 hours on the same device. Even if you don't have access to those fancy 
GPU cards, RegDiffusion still works. Inference on the same large network takes 
roughly 3 hours on a mid-range 12-core CPU. It does run slower but it's still 
faster than previous algorithm on GPU thanks to the improvement on algorithm. 

Citation
--------
Please consider cite our work if you found it useful for your work:



.. toctree::
   :caption: Get Started:
   :hidden:

   quick_tour

.. toctree::
   :caption: References:
   :hidden:

   modules
