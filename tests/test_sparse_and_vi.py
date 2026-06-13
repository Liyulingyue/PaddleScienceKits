"""Smoke tests for SparseCoding and BayesianLinearVI.

Run with:
    .venv/bin/python tests/test_sparse_and_vi.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paddle  # noqa: E402
import numpy as np  # noqa: E402

from PaddleScienceKits.ClassicalML import SparseCoding, BayesianLinearVI  # noqa: E402


def _ok(cond, msg):
    assert cond, msg
    print(f"  ok  {msg}")


# ---------------------------------------------------------- SparseCoding
def test_sparse_coding_shapes():
    sc = SparseCoding(n_atoms=10, n_features=8, lmbda=0.1, n_iter=30)
    x = paddle.randn([5, 8])
    z, x_hat = sc(x)
    _ok(z.shape == [5, 10], f"code shape {z.shape}")
    _ok(x_hat.shape == [5, 8], f"reconstruction shape {x_hat.shape}")


def test_sparse_coding_finds_sparse_codes():
    """Data generated from exactly 3 active atoms out of 16 should
    yield a learned dictionary whose codes are mostly zero."""
    paddle.seed(0)
    n_atoms, n_features = 16, 12
    D_true = paddle.randn([n_atoms, n_features])
    D_true = D_true / paddle.norm(D_true, axis=1, keepdim=True)
    # Each sample: pick 3 active atoms at random.
    N = 200
    Z = paddle.zeros([N, n_atoms])
    for i in range(N):
        active = np.random.choice(n_atoms, 3, replace=False)
        Z[i, active] = paddle.rand([3]) * 2 - 1
    X = Z @ D_true + 0.01 * paddle.randn([N, n_features])

    sc = SparseCoding(n_atoms=n_atoms, n_features=n_features,
                      lmbda=0.05, n_iter=200, encoder="fista")
    sc.fit(X, n_outer=200, lr=5e-2)
    z, x_hat = sc(X)
    # Sparsity: fraction of codes with |z| > 0.1
    active_frac = float((paddle.abs(z) > 0.1).astype("float32").mean().numpy().item())
    _ok(active_frac < 0.5,
        f"active code fraction = {active_frac:.3f} (expected < 0.5)")
    # Reconstruction quality
    rel_err = float(paddle.mean(paddle.sum((X - x_hat) ** 2, axis=1)) /
                    paddle.mean(paddle.sum(X ** 2, axis=1)))
    _ok(rel_err < 0.2, f"reconstruction relative err = {rel_err:.4f}")


def test_sparse_coding_gradient_flows():
    sc = SparseCoding(n_atoms=5, n_features=4, lmbda=0.1, n_iter=10)
    x = paddle.randn([3, 4])
    loss = sc.sparse_loss(x)
    loss.backward()
    has_grad = sc.D.grad is not None and float(paddle.sum(paddle.abs(sc.D.grad))) > 0
    _ok(has_grad, "Dictionary receives non-zero gradients")


# ------------------------------------------------------ BayesianLinearVI
def test_bayesian_vi_neg_elbo_decreases():
    paddle.seed(0)
    n, d = 100, 3
    x = paddle.randn([n, d])
    w_true = paddle.to_tensor([[1.0], [-2.0], [0.5]])
    y = x @ w_true + 0.05 * paddle.randn([n, 1])
    blr = BayesianLinearVI(n_features=d, n_outputs=1,
                            prior_std=1.0, noise_std=0.1)
    opt = paddle.optimizer.Adam(parameters=blr.parameters(), learning_rate=5e-2)
    initial = float(blr.neg_elbo(x, y, n_samples=1).numpy().item())
    for _ in range(100):
        loss = blr.neg_elbo(x, y, n_samples=1)
        opt.clear_grad()
        loss.backward()
        opt.step()
    final = float(blr.neg_elbo(x, y, n_samples=1).numpy().item())
    _ok(final < initial, f"VI neg-ELBO {initial:.3f} -> {final:.3f}")


def test_bayesian_vi_predictive_std_grows_out_of_distribution():
    paddle.seed(0)
    n, d = 50, 2
    x = paddle.randn([n, d])
    y = paddle.sum(x, axis=1, keepdim=True) + 0.1 * paddle.randn([n, 1])
    blr = BayesianLinearVI(n_features=d, n_outputs=1, prior_std=1.0, noise_std=0.1)
    opt = paddle.optimizer.Adam(parameters=blr.parameters(), learning_rate=5e-2)
    for _ in range(200):
        loss = blr.neg_elbo(x, y, n_samples=1)
        opt.clear_grad()
        loss.backward()
        opt.step()
    _, std_in = blr.predict(paddle.to_tensor([[0.0, 0.0]]), n_samples=50)
    _, std_out = blr.predict(paddle.to_tensor([[10.0, 10.0]]), n_samples=50)
    si = float(std_in.numpy().item())
    so = float(std_out.numpy().item())
    _ok(so > si,
        f"out-of-range predictive std ({so:.3f}) > in-range ({si:.3f})")


def test_bayesian_vi_posterior_std_shrinks_with_more_data():
    """Training on more data should reduce the posterior std."""
    paddle.seed(0)
    d = 3
    blr_small = BayesianLinearVI(n_features=d, n_outputs=1, noise_std=0.1)
    blr_big = BayesianLinearVI(n_features=d, n_outputs=1, noise_std=0.1)
    opt = paddle.optimizer.Adam
    for blr, N in [(blr_small, 20), (blr_big, 500)]:
        x = paddle.randn([N, d])
        y = x @ paddle.to_tensor([[1.0], [-2.0], [0.5]]) + 0.05 * paddle.randn([N, 1])
        o = opt(parameters=blr.parameters(), learning_rate=5e-2)
        for _ in range(100):
            loss = blr.neg_elbo(x, y, n_samples=1)
            o.clear_grad()
            loss.backward()
            o.step()
    sig_small = float(paddle.exp(blr_small.log_sigma).mean().numpy().item())
    sig_big = float(paddle.exp(blr_big.log_sigma).mean().numpy().item())
    _ok(sig_big < sig_small,
        f"more data shrinks posterior: sigma {sig_small:.4f} -> {sig_big:.4f}")


if __name__ == "__main__":
    test_sparse_coding_shapes()
    test_sparse_coding_finds_sparse_codes()
    test_sparse_coding_gradient_flows()
    test_bayesian_vi_neg_elbo_decreases()
    test_bayesian_vi_predictive_std_grows_out_of_distribution()
    test_bayesian_vi_posterior_std_shrinks_with_more_data()
    print("All SparseCoding + BayesianLinearVI tests passed.")
