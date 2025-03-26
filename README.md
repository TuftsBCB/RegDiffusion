# RegDiffusion <a href="https://tuftsbcb.github.io/RegDiffusion/"><img src="https://raw.githubusercontent.com/TuftsBCB/RegDiffusion/master/docs/_static/rd_logo_horizontal.png" align="right" alt="logo" width="200" height = "56" style = "border: none; float: right;"></a>

[![Downloads](https://static.pepy.tech/badge/regdiffusion)](https://pepy.tech/project/regdiffusion)
[![Downloads](https://static.pepy.tech/badge/regdiffusion/month)](https://pepy.tech/project/regdiffusion)
![PyPI - Version](https://img.shields.io/pypi/v/regdiffusion)

RegDiffusion is a very fast unsupervised regulatory network inference algorithm (just like GENIE3 and GRNBoost2), based on probabilistic diffusion model. It works well on genes and is capable to rapidly (<5min) predict biologically verifiable links from large single cell RNA-seq data with 14,000+ genes.

```
Zhu H, Slonim D. From Noise to Knowledge: Diffusion Probabilistic Model-Based Neural Inference of Gene Regulatory Networks. J Comput Biol. 2024 Nov;31(11):1087-1103. doi: 10.1089/cmb.2024.0607. Epub 2024 Oct 10. PMID: 39387266; PMCID: PMC11698671.
```

![](https://raw.githubusercontent.com/TuftsBCB/RegDiffusion/master/resources/regdiffusion_structure.png)

## Installation

RegDiffusion is on pypi.

```
pip install regdiffusion
```

Check out the [this tutorial](https://tuftsbcb.github.io/RegDiffusion/quick_tour.html) for a quick tour of how to use RegDiffusion! If you would like to integrate results from RegDiffusion into the SCENIC pipeline, checkout [this tutorial](https://tuftsbcb.github.io/RegDiffusion/downstream_with_pyscenic.html). 

## Inferred Networks from RegDiffusion
Here are two examples of inferred networks from regdiffusion. The networks are coherent with existing literature and across datasets. 

![](https://raw.githubusercontent.com/TuftsBCB/RegDiffusion/master/resources/apoe_net.png)

## Inference Speed
Inference on networks with 15,000 genes takes under 5 minutes on an A100 GPU. 
In contrast, previous VAE based models would take more than 4 hours on the same 
device. Even if you don't have access to those fancy GPU cards, RegDiffusion 
still works. Inference on the same large network takes roughly 3 hours on a 
mid-range 12-core CPU. 

## Citation 

If you find our package useful, consider cite our paper! =)

```
@article{zhu2024noise,
  title={From Noise to Knowledge: Diffusion Probabilistic Model-Based Neural Inference of Gene Regulatory Networks},
  author={Zhu, Hao and Slonim, Donna},
  journal={Journal of Computational Biology},
  volume={31},
  number={11},
  pages={1087--1103},
  year={2024},
  publisher={Mary Ann Liebert, Inc., publishers 140 Huguenot Street, 3rd Floor New~…}
}
```