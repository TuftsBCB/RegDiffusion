RegDiffusion: Gene Regulatory Network Inference
===============================================

RegDiffusion is an open-source Python package for **gene regulatory network
(GRN) inference from single-cell RNA-seq data** using probabilistic diffusion
models. It learns candidate regulatory relationships from gene expression data
without requiring a ground-truth network for training, and includes tools to
evaluate, export, and visualize inferred networks.

Start with the :doc:`quick_tour`, explore :doc:`large_networks`, or use inferred
edges for :doc:`downstream_with_pyscenic`. The :doc:`faq` explains input formats,
hardware requirements, and how to interpret the results.

`Source code <https://github.com/TuftsBCB/RegDiffusion>`_ ·
`PyPI package <https://pypi.org/project/regdiffusion/>`_ ·
`Research paper <https://doi.org/10.1089/cmb.2024.0607>`_

Installation
------------

``regdiffusion`` is available on PyPI:

.. code-block:: bash

    pip install regdiffusion

Quick Tour
----------
This package ``regdiffusion`` provides the official implementation of the
RegDiffusion algorithm and a set of easy-to-use companion tools to evaluate,
analyze, and visualize the inferred network. We also provide access tools to 
GRN benchmarks and preprocessed single cell datasets for evaluation. 

We tried to keep the top level interface straightforward. Right now, it only 
consists of 4 components: the ``RegDiffusionTrainer`` class, the ``GRN`` class, the 
``GRNEvaluator`` class, and the ``data`` module. 

- ``RegDiffusionTrainer``: You can use it to train a ``RegDiffusion`` model by 
  providing log-transformed expression data in a NumPy array or SciPy sparse
  matrix, with cells as rows and genes as columns. The training
  process could be either started or continued using the ``.train()`` method. You 
  can export the inferred ``GRN`` using the ``.get_grn()`` method.
- ``GRN``: The ``GRN`` class provides a container to save the inferred adjacency
  matrix and the corresponding gene names. You can save the ``GRN`` object to 
  a local ``HDF5`` file using the ``.to_hdf5()`` method and reload the saved file 
  using the ``read_hdf5()`` function. It also comes with functionalities to 
  export or visualize local regions. For example, you can use the 
  ``.visualize_local_neighborhood()`` to generate a similar plot as used in 
  the RegDiffusion paper. You can also extract the underlying adjacency list 
  using the ``.extract_local_neighborhood()`` method.
- ``GRNEvaluator``: The ground truth of regulatory relationship often exist as 
  list of edges but the values to be evaluated are often in adjacency matrix. 
  The ``GRNEvaluator`` class is designed to fill the gap. Right now it supports
  common metrics such as AUROC, AUPR, AUPR Ratio, EP, and EPR. 
- ``data`` module: Right now, the ``data`` module includes quick access to BEELINE 
  benchmarks and our preprocessed single cell datasets on mouse microglia. 

Model Structure
---------------

RegDiffusion includes an innovative model structure to estimate the added noise. Here is an high level illustraction. Please refer to our paper for details. 

.. image:: https://github.com/TuftsBCB/RegDiffusion/blob/master/resources/regdiffusion_structure.png?raw=true
    :width: 700
    :alt: RegDiffusion Structure

Understanding the Inferred Networks
-----------------------------------
After the ``RegDiffusion`` model converges, what you get is simply an 
`adjacency` matrix. When you have thousands or tens of thousands of genes, 
it's getting difficult to analyze matrix at that scale. In our paper, we 
propose a way to analyze the local network by focusing on the genes you care 
the most. Check out the tutorials on the left side for how to perform a similar 
network analysis like the one we did in the paper. The :doc:`visualizing_grn`
tutorial demonstrates interactive network exploration with ``lightgraph``.

.. image:: https://github.com/TuftsBCB/RegDiffusion/blob/master/resources/apoe_net.png?raw=true
    :width: 700
    :alt: Inferred network around ApoE

Inference Speed
---------------
The project reports inference on a 15,000-gene network in under five minutes on
an NVIDIA A100 GPU, or roughly three hours on a 12-core CPU. Runtime depends on
dataset size, hardware, and training settings. See :doc:`large_networks` for
memory benchmarks and the :doc:`quick_tour` for a worked inference example.


Citation
--------
If you use RegDiffusion, please cite:

Zhu H, Slonim D. **From Noise to Knowledge: Diffusion Probabilistic Model-Based
Neural Inference of Gene Regulatory Networks.** *Journal of Computational
Biology*. 2024;31(11):1087–1103.
`doi:10.1089/cmb.2024.0607 <https://doi.org/10.1089/cmb.2024.0607>`_.

The paper describes the method and evaluation. Inferred edges are hypotheses
for follow-up analysis, not experimental proof of regulation.



.. toctree::
   :caption: Get Started:
   :hidden:

   quick_tour
   visualizing_grn
   large_networks
   downstream_with_pyscenic
   faq

.. toctree::
   :caption: References:
   :hidden:

   main_api
   models
   data_module

.. toctree::
   :caption: Paper Supplements
   :hidden:

   supplements
