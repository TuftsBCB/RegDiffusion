"""Device handling and MPS numerical/training regression tests.

Run with: python -m unittest discover -s test_scripts -p test_mps_support.py -v
GPU tests are skipped when MPS is unavailable; no CPU fallback is enabled.
"""
import copy
import functools
import itertools
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

import regdiffusion as rd
import regdiffusion.trainer as trainer_module
from regdiffusion.models import RegDiffusion, RegDiffusionME


class DeviceTestCase(unittest.TestCase):
    def setUp(self):
        self.logs = tempfile.TemporaryDirectory()
        self.addCleanup(self.logs.cleanup)
        quiet = patch.object(
            trainer_module, 'tqdm',
            functools.partial(trainer_module.tqdm, disable=True))
        quiet.start()
        self.addCleanup(quiet.stop)
        rng = np.random.default_rng(21)
        self.x = np.log1p(rng.poisson(2, size=(64, 32))).astype(np.float32)
        self.cell_types = np.arange(64) % 2

    def trainer(self, sparse=False, **kwargs):
        options = dict(
            batch_size=8, n_steps=14, time_dim=16, hidden_dims=[8, 8, 8],
            seed=42, logger=rd.LightLogger(self.logs.name))
        options.update(kwargs)
        return rd.RegDiffusionTrainer(
            sp.csr_matrix(self.x) if sparse else self.x,
            cell_types=self.cell_types, **options)


class TestDeviceSelection(DeviceTestCase):
    def test_cpu_device_object_and_fp32_without_autocast(self):
        trainer = self.trainer(device=torch.device('cpu'), n_steps=1)
        with patch('torch.autocast', side_effect=AssertionError('Unexpected AMP')):
            trainer.train()
        self.assertTrue(np.isfinite(trainer.model.get_adj()).all())

    def test_unavailable_mps_is_reported_for_all_device_forms(self):
        with patch('torch.backends.mps.is_available', return_value=False):
            for device in ('mps', 'mps:0', torch.device('mps')):
                with self.subTest(device=device):
                    with self.assertRaisesRegex(RuntimeError, 'MPS.*not available'):
                        self.trainer(device=device)

    def test_cuda_device_object_still_falls_back_to_cpu(self):
        with patch('torch.cuda.is_available', return_value=False):
            trainer = self.trainer(device=torch.device('cuda:1'))
        self.assertEqual(trainer.model.adj_A.device.type, 'cpu')
        self.assertEqual(trainer.hp['device'], 'cpu')

    def test_cpu_amp_still_uses_bfloat16(self):
        trainer = self.trainer(device='cpu', use_amp=True, n_steps=1)
        x, ct = next(iter(trainer.train_dataloader))
        with trainer._autocast_context():
            prediction = trainer.model(x, torch.zeros(8, dtype=torch.long), ct)
        self.assertEqual(prediction.dtype, torch.bfloat16)
        trainer.train()
        self.assertTrue(np.isfinite(trainer.losses_on_gene).all())


@unittest.skipUnless(torch.backends.mps.is_available(), 'MPS GPU unavailable')
class TestMPS(DeviceTestCase):
    def test_training_modes_including_regularization_and_validation(self):
        for me, sparse, accumulation, amp in itertools.product((False, True), repeat=4):
            with self.subTest(me=me, sparse=sparse, accumulation=accumulation, amp=amp):
                trainer = self.trainer(
                    device=torch.device('mps'), memory_efficient=me,
                    sparse=sparse, gradient_accumulation=accumulation,
                    use_amp=amp, train_split=.75, eval_on_n_steps=7)
                self.assertEqual(trainer.model.adj_A.device.type, 'mps')
                if amp:
                    x, ct = next(iter(trainer.train_dataloader))
                    with trainer._autocast_context():
                        prediction = trainer.model(
                            x.to('mps'), torch.zeros(8, dtype=torch.long, device='mps'),
                            ct.to('mps'))
                    self.assertEqual(prediction.dtype, torch.bfloat16)
                initial = trainer.model.adj_A.detach().cpu().clone()
                with patch.object(trainer.model, 'get_sparse_loss',
                                  wraps=trainer.model.get_sparse_loss) as sparse_loss:
                    trainer.train()
                    self.assertGreater(sparse_loss.call_count, 0)
                logs = trainer.logger.to_df()
                self.assertEqual(len(logs), 14)
                self.assertEqual(logs.val_loss.notna().sum(), 2)
                for column in ('train_loss', 'val_loss'):
                    self.assertTrue(np.isfinite(logs[column].dropna().astype(float)).all())
                self.assertFalse(torch.equal(initial, trainer.model.adj_A.detach().cpu()))
                self.assertTrue(np.isfinite(trainer.losses_on_gene).all())
                grn = trainer.get_grn(np.array([f'gene_{i}' for i in range(32)]))
                self.assertEqual(grn.adj_matrix.shape, (32, 32))
                self.assertTrue(np.isfinite(grn.adj_matrix).all())
                np.testing.assert_array_equal(np.diag(grn.adj_matrix), 0)

    def test_cpu_mps_forward_backward_and_optimizer_agree(self):
        # Shared inputs/weights and disabled dropout isolate numerical errors
        # from the different CPU and MPS random-number streams.
        for model_class, n_gene in itertools.product((RegDiffusion, RegDiffusionME), (32, 256)):
            with self.subTest(model=model_class.__name__, n_gene=n_gene):
                torch.manual_seed(42)
                cpu = model_class(
                    n_gene=n_gene, time_dim=16, n_celltype=2,
                    hidden_dims=[8, 8, 8]).eval()
                with torch.no_grad():
                    cpu.adj_A.copy_(torch.randn_like(cpu.adj_A) * cpu.gene_reg_norm)
                mps = copy.deepcopy(cpu).to('mps')
                opt_cpu = torch.optim.Adam(cpu.parameters(), lr=1e-3)
                opt_mps = torch.optim.Adam(mps.parameters(), lr=1e-3)
                for step in range(20):
                    x = torch.randn(8, n_gene)
                    t = torch.randint(0, 5000, (8,))
                    ct = torch.arange(8) % 2
                    target = torch.randn_like(x)
                    opt_cpu.zero_grad()
                    opt_mps.zero_grad()
                    a = cpu(x, t, ct)
                    b = mps(x.to('mps'), t.to('mps'), ct.to('mps'))
                    torch.testing.assert_close(b.cpu(), a, rtol=5e-4, atol=3e-5)
                    loss_a = (a - target).square().mean() + .25 * cpu.get_sparse_loss()
                    loss_b = (b - target.to('mps')).square().mean() + .25 * mps.get_sparse_loss()
                    loss_a.backward()
                    loss_b.backward()
                    for (name, p), (_, q) in zip(cpu.named_parameters(), mps.named_parameters()):
                        torch.testing.assert_close(
                            q.grad.cpu(), p.grad, rtol=1e-3, atol=3e-5,
                            msg=lambda msg: f'{name} at step {step}: {msg}')
                    opt_cpu.step()
                    opt_mps.step()
                torch.testing.assert_close(mps.adj_A.cpu(), cpu.adj_A, rtol=1e-3, atol=3e-5)

    def test_seeded_training_is_repeatable_on_same_mps_device(self):
        for me in (False, True):
            with self.subTest(memory_efficient=me):
                outputs = []
                for _ in range(2):
                    trainer = self.trainer(device='mps', memory_efficient=me)
                    trainer.train()
                    outputs.append(trainer.model.adj_A.detach().cpu().numpy())
                np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-5, atol=1e-6)

    def test_compile_request_warns_and_still_trains(self):
        with self.assertWarnsRegex(UserWarning, 'MPS will use eager execution'):
            trainer = self.trainer(device='mps:0', compile=True, n_steps=1)
        trainer.train()
        self.assertEqual(trainer.model.adj_A.device.type, 'mps')


if __name__ == '__main__':
    unittest.main()
