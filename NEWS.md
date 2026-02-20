# RegDiffusion News / Changelog

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