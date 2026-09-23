# RegDiffusion <a href="https://tuftsbcb.github.io/RegDiffusion/"><img src="https://raw.githubusercontent.com/TuftsBCB/RegDiffusion/master/docs/_static/rd_logo_horizontal.png" align="right" alt="logo" width="200" height = "56" style = "border: none; float: right;"></a>

[![Downloads](https://static.pepy.tech/badge/regdiffusion)](https://pepy.tech/project/regdiffusion)
[![Downloads](https://static.pepy.tech/badge/regdiffusion/month)](https://pepy.tech/project/regdiffusion)
![PyPI - Version](https://img.shields.io/pypi/v/regdiffusion)

RegDiffusion is an open-source Python package for **gene regulatory network (GRN) inference from single-cell RNA-seq data** using probabilistic diffusion models. It learns candidate regulatory relationships from gene expression data without requiring a ground-truth network for training, and includes tools to evaluate, export, and visualize inferred networks.

[Documentation](https://tuftsbcb.github.io/RegDiffusion/) · [Quick start](https://tuftsbcb.github.io/RegDiffusion/quick_tour.html) · [FAQ](https://tuftsbcb.github.io/RegDiffusion/faq.html) · [PyPI](https://pypi.org/project/regdiffusion/) · [Paper](https://doi.org/10.1089/cmb.2024.0607)

## What can RegDiffusion do?

- Infer GRNs from a cells-by-genes expression matrix with GPU acceleration or on CPU.
- Accept log-transformed NumPy arrays or SciPy sparse matrices through the Python API. The CLI accepts raw counts in CSV or H5AD files and performs the log transformation.
- Work with large gene sets using [memory-efficient training](https://tuftsbcb.github.io/RegDiffusion/large_networks.html).
- Export inferred edges for [downstream pySCENIC analysis](https://tuftsbcb.github.io/RegDiffusion/downstream_with_pyscenic.html).

The project reports inference on a 15,000-gene network in under five minutes on an NVIDIA A100 GPU. Runtime and memory use depend on dataset size, hardware, and training settings; see the [large-network guide](https://tuftsbcb.github.io/RegDiffusion/large_networks.html) for memory benchmarks. Inferred edges are hypotheses for follow-up analysis, not experimental proof of regulation.

```
Zhu H, Slonim D. From Noise to Knowledge: Diffusion Probabilistic Model-Based Neural Inference of Gene Regulatory Networks. J Comput Biol. 2024 Nov;31(11):1087-1103. doi: 10.1089/cmb.2024.0607. Epub 2024 Oct 10. PMID: 39387266; PMCID: PMC11698671.
```

## Installation

RegDiffusion is on pypi.

```
pip install regdiffusion
```

Check out [this tutorial](https://tuftsbcb.github.io/RegDiffusion/quick_tour.html) for a quick tour of how to use RegDiffusion! If you would like to integrate results from RegDiffusion into the SCENIC pipeline, checkout [this tutorial](https://tuftsbcb.github.io/RegDiffusion/downstream_with_pyscenic.html).

## New in v0.2
- **Memory-efficient mode**: Set `memory_efficient=True` in `RegDiffusionTrainer` to reduce peak GPU memory by ~45%, making it easier to work with large gene sets on consumer GPUs (You can now fit 20k genes on a 16GB GPU).
- **Sparse matrix support**: `RegDiffusionTrainer` now accepts scipy sparse matrices directly (e.g., `adata.X`), enabling training on datasets with 1M+ cells without excessive memory usage.

## Inferred Networks from RegDiffusion
Here are two examples of inferred networks from regdiffusion. The networks are coherent with existing literature and across datasets. 

![Inferred gene regulatory networks around APOE](https://raw.githubusercontent.com/TuftsBCB/RegDiffusion/master/resources/apoe_net.png)

## Inference Speed
The project reports inference on a 15,000-gene network in under five minutes on
an NVIDIA A100 GPU, or roughly three hours on a 12-core CPU. These timings depend
on the dataset and training settings. See the [paper](https://doi.org/10.1089/cmb.2024.0607)
for the method's evaluation and the [large-network guide](https://tuftsbcb.github.io/RegDiffusion/large_networks.html)
for memory benchmarks.

## CLI tool
regdiffusion has a CLI tool now! It takes a count matrix as the input (different from the main API, which needs the data to be log transformed) and returns a table of inferred edges. 

```
usage: regdiffusion [-h] [--output OUTPUT] [--top_gene_percentile TOP_GENE_PERCENTILE] [--k K] [--workers WORKERS] input

Infer a gene regulatory network (GRN) from a single-cell count dataset.

positional arguments:
  input                 Input single-cell count dataset file (CSV or H5AD format).

options:
  -h, --help            show this help message and exit
  --output OUTPUT       Output file path for the edgelist (CSV). Default: rd_grn.csv
  --top_gene_percentile TOP_GENE_PERCENTILE
                        Percentile cutoff to filter weak edges (e.g., 50 for the top 50%). Default: 50
  --k K                 Number of edges per gene to extract (-1 for all edges). Default: -1
  --workers WORKERS     Number of workers to use for edgelist extraction. Default: 4
```

## Citation 

If you find our package useful, consider citing our paper! =)

```
@article{zhu2024noise,
  title={From Noise to Knowledge: Diffusion Probabilistic Model-Based Neural Inference of Gene Regulatory Networks},
  author={Zhu, Hao and Slonim, Donna},
  journal={Journal of Computational Biology},
  volume={31},
  number={11},
  pages={1087--1103},
  year={2024},
  doi={10.1089/cmb.2024.0607},
  url={https://doi.org/10.1089/cmb.2024.0607}
}
```
