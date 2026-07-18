"""Adjacency EMA variants of the RegDiffusion models.

Motivation
----------
``adj_A`` does not settle by the end of training. Snapshotting the adjacency
during the last 500 steps of a single seeded run on the BEELINE hESC data, the
top-10,000 edges of the step-500 and step-1000 snapshots overlap by only ~0.73
(Jaccard); the top-100 overlap by ~0.49. The reported network is therefore one
arbitrary point on a still-moving trajectory.

These classes keep an exponential moving average of ``adj_A`` alongside the
live parameter and report the averaged adjacency at inference time. This is
Polyak/SWA-style averaging applied to the adjacency only.

Training is untouched. ``get_adj_()`` -- the method the trainer uses for the
sparsity penalty -- still returns the live adjacency, so gradients, the loss,
and the optimizer trajectory are bit-identical to the corresponding non-EMA
model. Only ``get_adj()`` / ``get_adj_ema_()`` differ, which means the EMA is a
pure post-processing improvement and cannot destabilize training.

The averaging is applied to the *raw* ``adj_A`` parameter and thresholding is
applied afterwards, which is the standard formulation.

Measured effect
---------------
Averaged over all 7 BEELINE 1000_STRING datasets, 5 seeds each, comparing the
EMA against the final snapshot of the *same* runs (stability = mean pairwise
Jaccard of the top-10,000 edges across seeds)::

    ema_decay   window     dAUROC     dEPR   dStability
       0.95     ~20 steps  +0.0001   -0.004    +0.004
       0.99     ~100 steps +0.0007   -0.026    +0.019
       0.999    ~1000 st.  -0.0019   -0.342    -0.014

The gain is a modest stability improvement, not an accuracy improvement --
AUROC is essentially unchanged and EPR is very slightly worse. At 0.99 the
stability gain held on 7/7 datasets. Decay 0.999 averages over the whole run,
including unconverged early iterates, and is worse on every metric; it is not
a safe default despite being the conventional EMA value.
"""
import numpy as np
import torch

from .regdiffusion import RegDiffusion
from .regdiffusion_me import RegDiffusionME


class AdjEMAMixin:
    """Maintains an EMA of ``adj_A``. Mix in before a RegDiffusion model."""

    def init_adj_ema(self, ema_decay=0.99, ema_start=100):
        """
        Set up the EMA buffer. Called by the model subclasses at construction.

        Args:
            ema_decay (float): Decay for the moving average, in [0, 1). Higher
                values average over a longer window. The effective window is
                roughly ``1 / (1 - ema_decay)`` steps, so 0.999 averages over
                ~1000 steps and 0.99 over ~100.
            ema_start (int): Number of optimizer steps to skip before the EMA
                starts accumulating. Early training moves fast and those
                iterates are not worth averaging.
        """
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError(f'ema_decay must be in [0, 1), got {ema_decay}')
        if ema_start < 0:
            raise ValueError(f'ema_start must be >= 0, got {ema_start}')
        self.ema_decay = ema_decay
        self.ema_start = ema_start
        self.register_buffer('adj_A_ema', self.adj_A.detach().clone())
        self.register_buffer(
            'ema_num_updates', torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update_adj_ema(self, step):
        """
        Fold the current ``adj_A`` into the moving average.

        Called once per optimizer step by ``RegDiffusionEMATrainer``. The first
        accumulated step seeds the average with the current value rather than
        blending into the initialization, which removes the usual EMA bias
        correction.

        Args:
            step (int): 1-based optimizer step index.
        """
        if step <= self.ema_start:
            return
        if self.ema_num_updates.item() == 0:
            self.adj_A_ema.copy_(self.adj_A.detach())
        else:
            self.adj_A_ema.mul_(self.ema_decay).add_(
                self.adj_A.detach(), alpha=1.0 - self.ema_decay)
        self.ema_num_updates += 1

    @property
    def ema_is_active(self):
        """True once at least one EMA update has been accumulated."""
        return bool(self.ema_num_updates.item() > 0)

    def get_adj_ema_(self):
        """
        Thresholded, diagonal-masked EMA adjacency (unscaled).

        Mirrors ``get_adj_`` but reads the moving average. Falls back to the
        live adjacency if the EMA has not started yet, so a run shorter than
        ``ema_start`` still returns a usable network.
        """
        if not self.ema_is_active:
            return self.get_adj_()
        mask = 1 - torch.eye(self.n_gene, device=self.adj_A_ema.device)
        return self.soft_thresholding(
            self.adj_A_ema, self.gene_reg_norm / 2) * mask

    def get_adj(self):
        """
        Adjacency matrix scaled by the regulatory norm, from the EMA.

        This overrides the base implementation so that everything downstream of
        the model -- ``trainer.get_adj()``, ``trainer.get_grn()``, evaluation --
        sees the averaged network.
        """
        adj = self.get_adj_ema_().detach().cpu().numpy() / self.gene_reg_norm
        return adj.astype(np.float16)

    def get_adj_live(self):
        """
        Adjacency from the live (non-averaged) ``adj_A``.

        Provided so the EMA and the raw final snapshot can be compared from a
        single trained model.
        """
        adj = self.get_adj_().detach().cpu().numpy() / self.gene_reg_norm
        return adj.astype(np.float16)


class RegDiffusionEMA(AdjEMAMixin, RegDiffusion):
    """:class:`RegDiffusion` reporting an EMA-averaged adjacency."""

    def __init__(self, *args, ema_decay=0.99, ema_start=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_adj_ema(ema_decay=ema_decay, ema_start=ema_start)


class RegDiffusionMEEMA(AdjEMAMixin, RegDiffusionME):
    """:class:`RegDiffusionME` reporting an EMA-averaged adjacency."""

    def __init__(self, *args, ema_decay=0.99, ema_start=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_adj_ema(ema_decay=ema_decay, ema_start=ema_start)
