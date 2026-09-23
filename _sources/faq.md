# RegDiffusion FAQ: single-cell gene regulatory network inference

## What is RegDiffusion?

RegDiffusion is an open-source Python package that uses probabilistic diffusion
models to infer gene regulatory networks (GRNs) from single-cell RNA-seq gene
expression data. Training does not require a ground-truth network. The package
also provides network evaluation, export, and visualization tools.

The method is described in [Zhu and Slonim (2024), *From Noise to Knowledge*](https://doi.org/10.1089/cmb.2024.0607).

## What input data does RegDiffusion need?

The Python `RegDiffusionTrainer` API expects a **log-transformed expression
matrix with cells as rows and genes as columns**. It accepts NumPy arrays and
SciPy sparse matrices, including a suitably preprocessed `adata.X`. Remove
genes that are not expressed before training, and preserve the column order
when providing gene names for the inferred network.

The command-line interface instead accepts **raw counts** in CSV or H5AD
files and applies the log transformation itself. Do not pass already
log-transformed data to the CLI. See the [quick tour](quick_tour.md) and
[API reference](main_api.rst) for Python usage.

## Does RegDiffusion require a GPU?

No. Set `device='cpu'` in `RegDiffusionTrainer` to use a CPU. A CUDA-capable
NVIDIA GPU accelerates training. The project reports under five minutes on an
A100 GPU and roughly three hours on a 12-core CPU for a 15,000-gene network;
these are example timings, not guarantees for every dataset or configuration.

## Can RegDiffusion handle large or sparse single-cell datasets?

Yes. Sparse expression input avoids materializing the entire normalized
cells-by-genes matrix. The inferred gene-by-gene adjacency matrix is still
dense, so its memory cost grows quadratically with the number of genes.
Set `memory_efficient=True` to reduce training memory use. The
[large-network guide](large_networks.md) reports benchmark conditions, memory
measurements, and gene-filtering options.

## Can I use RegDiffusion with pySCENIC?

Yes. RegDiffusion can supply inferred network edges for downstream pySCENIC
analysis. Follow the [pySCENIC tutorial](downstream_with_pyscenic.md) for the
workflow. RegDiffusion performs network inference; downstream motif analysis
and regulon activity scoring remain separate steps.

## How does RegDiffusion relate to GENIE3 and GRNBoost2?

All three methods address gene regulatory network inference from expression
data. RegDiffusion uses probabilistic diffusion models. For a method choice,
consider your dataset, hardware, runtime budget, and evaluation against a
relevant reference network. The [RegDiffusion paper](https://doi.org/10.1089/cmb.2024.0607)
provides the method's evaluation; the package includes a `GRNEvaluator` and
[BEELINE dataset loaders](data_module.rst) for your own comparisons.

## Are inferred edges experimentally validated regulatory interactions?

An inferred edge is a model prediction for follow-up analysis. It does not by
itself establish causality or experimental validation. Use reference networks,
biological context, and independent experiments to assess specific edges.

## How should I cite RegDiffusion?

Zhu H, Slonim D. **From Noise to Knowledge: Diffusion Probabilistic Model-Based
Neural Inference of Gene Regulatory Networks.** *Journal of Computational
Biology*. 2024;31(11):1087–1103.
[doi:10.1089/cmb.2024.0607](https://doi.org/10.1089/cmb.2024.0607).

The [GitHub repository](https://github.com/TuftsBCB/RegDiffusion) includes a
`CITATION.cff` file for citation tools.
