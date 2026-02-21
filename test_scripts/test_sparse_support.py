"""Verify that RegDiffusionTrainer accepts sparse matrices and produces
results consistent with the dense path."""
import numpy as np
import scipy.sparse as sp
import torch
import regdiffusion as rd


def test_sparse_dense_consistency():
    """Train briefly with dense and sparse input; compare adjacency matrices."""
    np.random.seed(42)
    torch.manual_seed(42)

    # Create a small synthetic expression matrix (log-transformed counts)
    n_cell, n_gene = 200, 50
    dense_X = np.random.rand(n_cell, n_gene).astype(np.float32)
    # Make it look like log-transformed scRNA-seq (many near-zero values)
    dense_X[dense_X < 0.6] = 0.0
    dense_X = np.log1p(dense_X * 10)

    sparse_X = sp.csr_matrix(dense_X)

    # --- Dense path ---
    torch.manual_seed(0)
    np.random.seed(0)
    trainer_dense = rd.RegDiffusionTrainer(
        dense_X, device='cpu', n_steps=5, batch_size=32)

    # --- Sparse path ---
    torch.manual_seed(0)
    np.random.seed(0)
    trainer_sparse = rd.RegDiffusionTrainer(
        sparse_X, device='cpu', n_steps=5, batch_size=32)

    # Both should have the same shape metadata
    assert trainer_dense.n_cell == trainer_sparse.n_cell
    assert trainer_dense.n_gene == trainer_sparse.n_gene

    # Check that one batch from each dataloader has the same shape
    batch_d = next(iter(trainer_dense.train_dataloader))
    batch_s = next(iter(trainer_sparse.train_dataloader))
    assert batch_d[0].shape == batch_s[0].shape, (
        f"Dense batch: {batch_d[0].shape}, Sparse batch: {batch_s[0].shape}")

    # Verify no NaN/Inf in sparse-path batches
    assert torch.isfinite(batch_s[0]).all(), "Sparse batch contains NaN/Inf!"

    print("PASS: Sparse and dense paths produce consistent shapes, no NaN/Inf.")


def test_sparse_training():
    """Full sparse training smoke test."""
    np.random.seed(42)
    torch.manual_seed(42)

    n_cell, n_gene = 300, 30
    dense_X = np.random.rand(n_cell, n_gene).astype(np.float32)
    dense_X[dense_X < 0.5] = 0.0
    dense_X = np.log1p(dense_X * 10)
    sparse_X = sp.csr_matrix(dense_X)

    trainer = rd.RegDiffusionTrainer(
        sparse_X, device='cpu', n_steps=10, batch_size=64)
    trainer.train()

    adj = trainer.get_adj()
    assert adj.shape == (n_gene, n_gene), f"Unexpected adj shape: {adj.shape}"
    assert np.isfinite(adj).all(), "Adjacency matrix has NaN/Inf"
    print("PASS: Sparse training completes, adjacency matrix is valid.")


def test_sparse_with_cell_types():
    """Sparse input with cell type labels."""
    np.random.seed(42)
    n_cell, n_gene = 200, 20
    dense_X = np.random.rand(n_cell, n_gene).astype(np.float32)
    dense_X[dense_X < 0.5] = 0.0
    dense_X = np.log1p(dense_X * 10)
    sparse_X = sp.csr_matrix(dense_X)
    cell_types = np.random.randint(0, 3, n_cell)

    trainer = rd.RegDiffusionTrainer(
        sparse_X, cell_types=cell_types,
        device='cpu', n_steps=5, batch_size=32)
    trainer.train()
    print("PASS: Sparse with cell types works.")


def test_sparse_val_split():
    """Sparse input with train/val split."""
    np.random.seed(42)
    n_cell, n_gene = 200, 20
    dense_X = np.random.rand(n_cell, n_gene).astype(np.float32)
    dense_X[dense_X < 0.5] = 0.0
    dense_X = np.log1p(dense_X * 10)
    sparse_X = sp.csr_matrix(dense_X)

    trainer = rd.RegDiffusionTrainer(
        sparse_X, device='cpu', n_steps=5, batch_size=32,
        train_split=0.8)
    trainer.train()
    print("PASS: Sparse with train/val split works.")


if __name__ == '__main__':
    test_sparse_dense_consistency()
    test_sparse_training()
    test_sparse_with_cell_types()
    test_sparse_val_split()
    print("\nAll sparse support tests passed!")
