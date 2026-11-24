"""Compare production projection-first models against the original contraction."""
import itertools
import unittest

import torch

from regdiffusion.models import RegDiffusion, RegDiffusionME


def original_forward(model, x, t, ct):
    """Pre-0.2.5 forward, including the original placement of the final bias."""
    h_time = model.time_mlp(t)
    if model.n_celltype is not None:
        h_celltype = model.celltype_emb(ct)
    else:
        h_celltype = torch.zeros(x.shape[0], model.celltype_dim,
                                device=x.device, dtype=x.dtype)
    h = model.gene_emb(x)
    for i, block in enumerate(model.blocks):
        if i:
            h = torch.cat((h, x.unsqueeze(-1)), dim=-1)
        h = block(h, h_time, h_celltype)
    mixed = torch.einsum('ogd,gh->ohd', h, model.I_minus_A())
    return model.final(mixed).squeeze(-1)


class TestProjectionFirst(unittest.TestCase):
    def check_device(self, device):
        for cls, training, conditioned, batch, bias in itertools.product(
                (RegDiffusion, RegDiffusionME), (False, True), (False, True), (1, 4), (False, True)):
            with self.subTest(device=device, model=cls.__name__, training=training,
                              conditioned=conditioned, batch=batch, bias=bias):
                torch.manual_seed(42)
                # A non-default, varying feature width catches shape assumptions.
                kwargs = dict(n_gene=32, time_dim=16, n_celltype=2 if conditioned else None,
                              hidden_dims=[8, 12, 5], adj_dropout=.3)
                reference = cls(**kwargs).to(device)
                optimized = cls(**kwargs).to(device)
                with torch.no_grad():
                    reference.adj_A.copy_(torch.randn_like(reference.adj_A) * reference.gene_reg_norm)
                    reference.final.bias.fill_(.37)
                if not bias:
                    reference.final.bias = None
                    optimized.final.bias = None
                optimized.load_state_dict(reference.state_dict(), strict=True)
                reference.train(training)
                optimized.train(training)
                x = torch.randn(batch, 32, device=device, requires_grad=True)
                x_new = x.detach().clone().requires_grad_()
                t = torch.randint(0, 5000, (batch,), device=device)
                ct = torch.arange(batch, device=device) % 2
                target = torch.randn(batch, 32, device=device)
                # Identical dropout masks on the same device for both paths.
                torch.manual_seed(123)
                expected = original_forward(reference, x, t, ct)
                torch.manual_seed(123)
                actual = optimized(x_new, t, ct)
                self.assertEqual(actual.shape, (batch, 32))
                torch.testing.assert_close(actual, expected, rtol=2e-4, atol=3e-5)
                ((expected - target).square().mean() + .25 * reference.get_sparse_loss()).backward()
                ((actual - target).square().mean() + .25 * optimized.get_sparse_loss()).backward()
                torch.testing.assert_close(x_new.grad, x.grad, rtol=5e-4, atol=3e-5)
                expected_params = dict(reference.named_parameters())
                for name, parameter in optimized.named_parameters():
                    torch.testing.assert_close(
                        parameter.grad, expected_params[name].grad, rtol=5e-4, atol=3e-5,
                        msg=lambda msg: name + ': ' + msg)

    def test_cpu_forward_and_gradients(self):
        self.check_device('cpu')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS GPU unavailable')
    def test_mps_forward_and_gradients(self):
        self.check_device('mps')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU unavailable')
    def test_cuda_forward_and_gradients(self):
        # Check reassociation independently of TF32's reduced internal precision.
        precision = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision('highest')
            self.check_device('cuda')
        finally:
            torch.set_float32_matmul_precision(precision)


if __name__ == '__main__':
    unittest.main()
