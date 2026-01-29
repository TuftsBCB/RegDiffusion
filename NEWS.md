# RegDiffusion News / Changelog

## [Unreleased]

### Bug Fixes

#### Critical: Fixed training mode check in `RegDiffusion.I_minus_A()`
- **File**: `regdiffusion/models/regdiffusion.py`
- **Issue**: The method incorrectly used `self.train` (a method inherited from `nn.Module`) instead of `self.training` (a boolean property) to check if the model is in training mode.
- **Impact**: Adjacency matrix dropout was always applied regardless of training/eval mode, potentially affecting inference results.
- **Fix**: Changed `if self.train:` to `if self.training:`.

#### Fixed missing `self` parameter in `GRN.remove_weak_edges()`
- **File**: `regdiffusion/grn.py`
- **Issue**: The method definition was missing the `self` parameter, causing a `TypeError` when called.
- **Fix**: Added `self` as the first parameter and improved the method with proper null checking and documentation.

#### Fixed incorrect method call in `GRN.get_edgelist()`
- **File**: `regdiffusion/grn.py`
- **Issue**: The deprecated `get_edgelist()` method incorrectly passed `self` as an argument to `extract_edgelist()`, causing a `TypeError`.
- **Fix**: Removed the extra `self` argument and added a deprecation warning.

#### Fixed logger assignment in `RegDiffusionTrainer.__init__()`
- **File**: `regdiffusion/trainer.py`
- **Issue**: When an external `logger` was provided, it was not assigned to `self.logger`, causing an `AttributeError` on subsequent logger access.
- **Fix**: Added `else: self.logger = logger` branch to properly assign external loggers.

#### Fixed `n_celltype=None` handling in `RegDiffusion.forward()`
- **File**: `regdiffusion/models/regdiffusion.py`
- **Issue**: When `n_celltype=None` was passed during initialization, `self.celltype_emb` was not created, but `forward()` still attempted to call it, causing an `AttributeError`.
- **Fix**: Added conditional check in `forward()` to use a zero tensor for cell type embedding when `n_celltype` is `None`. Also stored `celltype_dim` and `n_celltype` as instance attributes.
