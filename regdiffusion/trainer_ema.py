"""Trainer for the adjacency-EMA RegDiffusion variants."""
import torch

from .models.regdiffusion_ema import RegDiffusionEMA, RegDiffusionMEEMA
from .trainer import RegDiffusionTrainer
from .grn import GRN


class RegDiffusionEMATrainer(RegDiffusionTrainer):
    """
    Train a RegDiffusion model and report an EMA-averaged adjacency.

    ``adj_A`` has not converged when training stops -- on BEELINE hESC the
    top-10,000 edges move by ~0.27 Jaccard over the final 500 steps of a single
    seeded run -- so the usual final snapshot is one arbitrary point on a
    drifting trajectory. This trainer keeps an exponential moving average of
    ``adj_A`` and reports that instead.

    The averaging happens after each optimizer step and feeds nothing back into
    training: the loss, gradients, and parameter trajectory are identical to
    :class:`RegDiffusionTrainer` with the same seed. Only the reported
    adjacency differs. ``trainer.get_adj()`` and ``trainer.get_grn()`` return
    the averaged network; ``get_adj_live()`` returns the final snapshot, so
    both can be compared from one run.

    Benchmarked on all 7 BEELINE datasets (5 seeds each), the EMA buys a modest
    stability gain rather than accuracy: cross-seed top-10,000 edge overlap
    improves by ~0.019 Jaccard (7/7 datasets) while AUROC is unchanged
    (+0.0007) and EPR is very slightly worse (-0.026). It costs no extra
    training. See ``models/regdiffusion_ema.py`` for the full decay sweep.

    Args:
        ema_decay (float): Decay for the moving average, in [0, 1). The
            effective averaging window is about ``1 / (1 - ema_decay)`` steps,
            so the default 0.99 averages over roughly the last 100 steps.
            0.999 (~the whole run) was measured to be worse on every metric --
            raise this only with care. Default: 0.99.
        ema_start (int): Optimizer steps to skip before the average starts
            accumulating. Default: 100.
        **kwargs: Everything accepted by :class:`RegDiffusionTrainer`.

    Example:
        >>> trainer = RegDiffusionEMATrainer(bl_dt.X, seed=42)
        >>> trainer.train()
        >>> adj_ema = trainer.get_adj()        # averaged (recommended)
        >>> adj_final = trainer.get_adj_live() # final snapshot
    """

    MODEL_CLASSES = {False: RegDiffusionEMA, True: RegDiffusionMEEMA}

    def __init__(self, *args, ema_decay=0.99, ema_start=100, **kwargs):
        super().__init__(*args, **kwargs)
        self._configure_ema(ema_decay, ema_start)
        self._install_ema_hook()
        self.model_name = self.model_name + 'EMA'

    @property
    def _ema_model(self):
        """The underlying model, unwrapping ``torch.compile`` if present."""
        return getattr(self, 'original_model', self.model)

    def _configure_ema(self, ema_decay, ema_start):
        model = self._ema_model
        model.init_adj_ema(ema_decay=ema_decay, ema_start=ema_start)
        model.adj_A_ema = model.adj_A_ema.to(self.device)
        model.ema_num_updates = model.ema_num_updates.to(self.device)
        self.hp['ema_decay'] = ema_decay
        self.hp['ema_start'] = ema_start

    def _install_ema_hook(self):
        """
        Update the EMA after every optimizer step.

        Wrapping ``opt.step`` rather than overriding the training loops keeps
        this working for both the normal and gradient-accumulation paths, and
        avoids duplicating the loop bodies.
        """
        model = self._ema_model
        original_step = self.opt.step
        state = {'step': 0}

        def step_with_ema(*args, **kwargs):
            out = original_step(*args, **kwargs)
            state['step'] += 1
            model.update_adj_ema(state['step'])
            return out

        self.opt.step = step_with_ema
        self._ema_step_state = state

    def get_adj_live(self):
        """Adjacency from the live ``adj_A`` (the pre-EMA final snapshot)."""
        return self._ema_model.get_adj_live()

    def get_grn_live(self, gene_names, tf_names=None, top_gene_percentile=None):
        """GRN built from the live adjacency instead of the EMA."""
        return GRN(self.get_adj_live(), gene_names, tf_names,
                   top_gene_percentile)
