RegDiffusion
=============

RegDiffusion is a very fast regulatory network inference algorithm based on 
probabilistic diffusion model. It works well on genes and is capable to predict 
biologically verifiable links from single cell RNA-seq data.

Inference on networks with 15,000 genes takes under 5 minutes with modern 
accelerated computing. In contrast, previous VAE based models would take more 
than 4 hours on the same device. Even if you don't have access to those fancy 
GPU cards, RegDiffusion still works. Inference on the same large network takes 
roughly 3 hours on a mid-range 12-core CPU. It does run slower but it's still 
faster than previous algorithm on GPU thanks to the improvement on algorithm. 

Installation
------------

``regdiffusion`` is available on pypi:

    pip install regdiffusion

.. toctree::
   :caption: Get Started:

   quick_tour

.. toctree::
   :caption: References:
   :hidden:

   modules
