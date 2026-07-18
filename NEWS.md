# RegDiffusion News / Changelog

## 0.2.1 [unreleased]

### New Features
- Added sparse matrix support for `RegDiffusionTrainer`. The `exp_array` argument now accepts scipy sparse matrices (e.g. `adata.X`) directly. Normalization statistics are computed in chunks and each sample is normalized on-the-fly during training, so the full dense matrix is never materialized. This enables training on datasets with 1M+ cells without excessive memory usage.
- Updated CLI (`regdiffusion` command) to handle sparse `adata.X` from H5AD files, including sparse-safe data validation and log-transform via `log1p` on non-zero entries.

### Bug Fixes
- Fixed `RegDiffusionTrainer.train(n_steps=...)` silently ignoring its argument. Both the normal and gradient-accumulation branches hardcoded `n_steps=None`, so training always ran for the `n_steps` given at initialization.
- Fixed the L1 sparsity penalty being computed on *signed* adjacency values in the base `RegDiffusion` model. Soft thresholding preserves sign, so the penalty could in principle be reduced by making inhibitory (negative) edges more negative. Both models now share a single `get_sparse_loss()` that takes the mean absolute value over off-diagonal entries. In practice this does not change results — negative entries land in the soft-threshold dead zone, where both the output and its gradient are zero — and BEELINE metrics are unchanged; the penalty is now correct by construction rather than by accident.
- `RegDiffusionME.get_sampled_sparse_loss()` is renamed to `get_sparse_loss()` and now excludes diagonal samples, matching the base model, whose `get_adj_()` masks the diagonal.
- Registered the sampled adjacency index tensors (`sampled_adj_row/col_nonparam`) as buffers instead of `nn.Parameter`. These are integer indices, not learnable weights, and they no longer appear in `model.parameters()`. `state_dict` keys are unchanged, so existing checkpoints still load.
- The sparse loss is no longer computed during the warmup window, where it is discarded.


## 0.2.0

### Memory Optimization
- Removed three `(n_gene, n_gene)` helper matrices (`zeros_nonparam`, `eye_nonparam`, `mask_nonparam`) from `RegDiffusion`, replacing them with inline computations. Saves ~4.8 GB persistent GPU memory for 20K genes with no impact on model accuracy.

### New Features
- Added `RegDiffusionME`, a memory-efficient model variant enabled via `memory_efficient=True` in `RegDiffusionTrainer`. Reduces peak GPU memory by ~45% with no impact on accuracy. Uses a custom autograd function for soft thresholding (boolean masks instead of float32 tensors) and sampled sparse loss (avoids materializing full adjacency matrix for L1 regularization). Benchmarked on all 7 BEELINE datasets with identical AUROC/AUPRR/EPR.
- Added automatic mixed precision (AMP) support via `use_amp=True` in `RegDiffusionTrainer`. Uses bfloat16 for forward pass and loss computation, reducing memory for autograd-saved activations while keeping model parameters in float32. Requires Ampere or newer GPU.

### Bug Fixes
- Fixed `I_minus_A()` using `self.train` instead of `self.training`, causing dropout to always apply during inference
- Fixed missing `self` parameter in `GRN.remove_weak_edges()`
- Fixed `GRN.get_edgelist()` incorrectly passing `self` to `extract_edgelist()`
- Fixed external logger not being assigned in `RegDiffusionTrainer.__init__()`
- Fixed `forward()` crash when `n_celltype=None` by adding conditional cell type embedding