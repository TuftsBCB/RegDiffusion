# Working with Large Gene Networks

RegDiffusion can infer GRNs among 10,000+ genes within minutes on a modern GPU. However, as the number of genes grows, GPU memory becomes the primary bottleneck. This guide covers practical strategies to reduce memory usage and scale to larger networks.

## Why Memory Grows Quickly

The core of RegDiffusion is a dense adjacency matrix of shape `(n_gene, n_gene)`. For a network with `n` genes, the memory cost scales quadratically. Beyond the adjacency matrix itself, PyTorch's autograd retains multiple `(n_gene, n_gene)` intermediate tensors during each training step for the backward pass — these are the dominant memory cost.

## Strategy 1: Memory-Efficient Mode (Recommended)

The most effective way to reduce memory is to enable the memory-efficient model variant. This cuts peak GPU memory by approximately **45%** with no impact on accuracy and no slowdown.

```python
trainer = rd.RegDiffusionTrainer(
    exp_array,
    memory_efficient=True,
    device='cuda'
)
trainer.train()
```

**What it does:**
- Uses a custom autograd function for soft thresholding that saves boolean masks (1 byte per entry) instead of float32 tensors (4 bytes per entry), reducing autograd memory by ~8x for these operations
- Computes sparse regularization loss from random samples of the adjacency matrix instead of materializing the full `(n_gene, n_gene)` matrix, eliminating 3 large intermediate tensors

**Impact on accuracy:** Minimal. We benchmarked RegDiffusion and RegDiffusionME on all 7 BEELINE datasets (1000_STRING, 10 repeats each). The ME version tends to be slightly lower but the results are statistically indistinguishable:

| Dataset | RegDiffusion AUROC | RegDiffusionME AUROC | RegDiffusion EPR | RegDiffusionME EPR |
|---------|-------------------|---------------------|-----------------|-------------------|
| hESC | 0.6531 +/- 0.0009 | 0.6526 +/- 0.0009 | 5.35 +/- 0.06 | 5.35 +/- 0.05 |
| hHep | 0.6542 +/- 0.0004 | 0.6534 +/- 0.0007 | 3.49 +/- 0.04 | 3.48 +/- 0.04 |
| mDC | 0.5739 +/- 0.0010 | 0.5734 +/- 0.0011 | 2.25 +/- 0.05 | 2.25 +/- 0.05 |
| mESC | 0.6127 +/- 0.0005 | 0.6124 +/- 0.0007 | 4.20 +/- 0.05 | 4.20 +/- 0.04 |
| mHSC-E | 0.7115 +/- 0.0021 | 0.7109 +/- 0.0025 | 8.37 +/- 0.15 | 8.33 +/- 0.15 |
| mHSC-GM | 0.7424 +/- 0.0023 | 0.7411 +/- 0.0022 | 9.17 +/- 0.09 | 9.12 +/- 0.08 |
| mHSC-L | 0.6924 +/- 0.0048 | 0.6921 +/- 0.0057 | 7.35 +/- 0.16 | 7.35 +/- 0.18 |

## GPU Memory Benchmark

We measured peak GPU memory during training with synthetic data (200 cells, 20 training steps) across different gene counts:

| Genes | Default | ME | ME Savings | ME + AMP |
|-------|---------|------|-----------|----------|
| 5,000 | 2.1 GB | 1.2 GB | 41% | 1.1 GB |
| 10,000 | 7.3 GB | 4.0 GB | 45% | 3.8 GB |
| 15,000 | 15.7 GB | 8.5 GB | 46% | 8.1 GB |
| 20,000 | 27.3 GB | 14.6 GB | 47% | 14.1 GB |
| 25,000 | 42.1 GB | 22.4 GB | 47% | 21.8 GB |
| 30,000 | 60.2 GB | 31.8 GB | 47% | 31.1 GB |

The memory-efficient mode also runs slightly faster than default (about 15% faster), since fewer intermediate tensors need to be allocated and freed.

## Strategy 2: Mixed Precision Training

Automatic mixed precision (AMP) provides a modest additional memory reduction (about 5%) on top of memory-efficient mode. It runs the forward pass in bfloat16. However, AMP increases training time by roughly 2x due to dtype casting overhead, so it is only recommended if you need the last few percent of memory savings.

```python
trainer = rd.RegDiffusionTrainer(
    exp_array,
    memory_efficient=True,
    use_amp=True,
    device='cuda'
)
trainer.train()
```

**Requirements:** A GPU with bfloat16 support (NVIDIA Ampere architecture or newer, e.g., A100, RTX 3090, RTX 4090).

## Strategy 3: Gene Filtering

Reducing the number of genes is always effective since memory scales quadratically. RegDiffusion does not require heavy gene filtering, but removing clearly uninformative genes can save substantial memory with minimal impact on results.

**Remove non-expressed genes:** Genes with zero total counts across all cells carry no information and should always be removed.

```python
import scanpy as sc

# Remove genes expressed in fewer than 10 cells
sc.pp.filter_genes(adata, min_cells=10)
```

**Consider highly variable gene selection:** If your dataset has 25,000+ genes and you are memory constrained, selecting the top 15,000-20,000 highly variable genes is a reasonable compromise.

```python
sc.pp.highly_variable_genes(adata, n_top_genes=15000)
adata = adata[:, adata.var.highly_variable]
```

## Running on CPU

If you don't have access to a GPU with enough memory, RegDiffusion also runs on CPU. It will be slower (but still much faster than GRNBoost2) but has no memory constraint from GPU VRAM.

```python
trainer = rd.RegDiffusionTrainer(
    exp_array,
    device='cpu',
    n_steps=1000
)
trainer.train()
```

For reference, inference on a 15,000-gene network takes roughly 3 hours on a mid-range 12-core CPU versus under 5 minutes on an A100 GPU.
