"""Smoke tests for GaussianProcess and GaussianHMM.

Run with:
    .venv/bin/python tests/test_gp_and_hmm.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paddle  # noqa: E402
import numpy as np  # noqa: E402

from PaddleScienceKits.ClassicalML import GaussianProcess, GaussianHMM  # noqa: E402


def _ok(cond, msg):
    assert cond, msg
    print(f"  ok  {msg}")


# ----------------------------------------------------------------- GP
def test_gp_fits_noisy_sine():
    paddle.seed(0)
    n = 50
    x = paddle.linspace(-3.0, 3.0, n).unsqueeze(-1)
    y = paddle.sin(x).squeeze(-1) + 0.1 * paddle.randn([n])
    gp = GaussianProcess(dim=1, n_train=n, kernel="rbf",
                         log_lengthscale=0.0, log_outputscale=0.0, log_noise=-3.0)
    # tune the lengthscale a little: the true sine has lengthscale ~1,
    # so l^2 = 1 → log_lengthscale = 0 already works.
    gp.fit(x, y)
    pred = gp(x)
    mse = float(paddle.mean((pred.squeeze() - y) ** 2))
    _ok(mse < 0.2, f"GP RBF sin-fit MSE = {mse:.4f}")


def test_gp_predictive_std_increases_outside_training_data():
    paddle.seed(0)
    x = paddle.linspace(-3.0, 3.0, 30).unsqueeze(-1)
    y = paddle.sin(x).squeeze(-1)
    gp = GaussianProcess(dim=1, n_train=30, kernel="rbf").fit(x, y)
    x_in = paddle.to_tensor([[0.0]])
    x_out = paddle.to_tensor([[10.0]])
    _, std_in = gp.forward_with_std(x_in)
    _, std_out = gp.forward_with_std(x_out)
    si = float(std_in.numpy().item())
    so = float(std_out.numpy().item())
    _ok(so > si * 2,
        f"out-of-range std ({so:.3f}) > 2 * in-range std ({si:.3f})")


def test_gp_kernel_hyperparams_tune_via_grad():
    """Optimising neg_log_marginal_likelihood should reduce it."""
    paddle.seed(0)
    n = 30
    x = paddle.linspace(-2.0, 2.0, n).unsqueeze(-1)
    y = paddle.sin(x).squeeze(-1) + 0.1 * paddle.randn([n])
    gp = GaussianProcess(dim=1, n_train=n, kernel="rbf",
                         log_lengthscale=-1.0, log_outputscale=-1.0, log_noise=-1.0)
    gp.fit(x, y)
    initial = float(gp.neg_log_marginal_likelihood().numpy().item())
    opt = paddle.optimizer.Adam(
        parameters=[gp.log_lengthscale, gp.log_outputscale, gp.log_noise],
        learning_rate=5e-2,
    )
    for _ in range(80):
        loss = gp.neg_log_marginal_likelihood()
        opt.clear_grad()
        loss.backward()
        opt.step()
    final = float(gp.neg_log_marginal_likelihood().numpy().item())
    _ok(final < initial,
        f"NLML decreased from {initial:.2f} to {final:.2f}")


# ----------------------------------------------------------------- HMM
def test_hmm_recovers_two_state_regime():
    """Generate a sequence from a 2-state HMM with very different
    emission distributions, then re-fit and check that
    responsibilities recover the true states."""
    paddle.seed(0)
    # True HMM parameters
    log_start = paddle.log(paddle.to_tensor([0.6, 0.4]))
    trans = paddle.to_tensor([[0.95, 0.05], [0.10, 0.90]])
    emit = paddle.to_tensor([[0.90, 0.05, 0.05],
                             [0.05, 0.05, 0.90]])

    def sample(T):
        z = []
        x = []
        s = 0 if np.random.rand() < float(log_start[0]) else 1
        z.append(s)
        x.append(int(np.random.choice(3, p=emit[s].numpy())))
        for _ in range(T - 1):
            s = int(np.random.choice(2, p=trans[s].numpy()))
            z.append(s)
            x.append(int(np.random.choice(3, p=emit[s].numpy())))
        return z, x

    np.random.seed(0)
    z_true, x_seq = sample(200)
    x = paddle.to_tensor(x_seq, dtype="int64")
    hmm = GaussianHMM(n_states=2, n_emissions=3)
    hmm.fit_em(x, n_iter=100)
    # Responsibilities should be near one-hot on the correct state.
    gamma = hmm(x).numpy()
    pred = gamma.argmax(axis=1)
    # Compute the agreement modulo state-label swap.
    same = (np.array(z_true) == pred).sum()
    swap = (np.array(z_true) == (1 - pred)).sum()
    accuracy = max(same, swap) / len(z_true)
    _ok(accuracy >= 0.8, f"HMM 2-state recovery accuracy (mod swap) = {accuracy:.3f}")


def test_hmm_viterbi_returns_int_sequence():
    paddle.seed(0)
    np.random.seed(0)
    trans = paddle.to_tensor([[0.9, 0.1], [0.2, 0.8]])
    emit = paddle.to_tensor([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
    seq = []
    s = 0
    for _ in range(40):
        seq.append(int(np.random.choice(3, p=emit[s].numpy())))
        s = int(np.random.choice(2, p=trans[s].numpy()))
    hmm = GaussianHMM(n_states=2, n_emissions=3)
    hmm.fit_em(paddle.to_tensor(seq, dtype="int64"), n_iter=60)
    path = hmm.viterbi(paddle.to_tensor(seq, dtype="int64"))
    _ok(path.shape == [40] and path.dtype == paddle.int64,
        f"viterbi path shape={path.shape}, dtype={path.dtype}")


def test_hmm_responsibilities_sum_to_one():
    hmm = GaussianHMM(n_states=3, n_emissions=4)
    x = paddle.randint(0, 4, [25])
    gamma = hmm(x)
    _ok(paddle.allclose(gamma.sum(-1), paddle.ones([25]), atol=1e-4),
        "HMM responsibilities sum to 1")


if __name__ == "__main__":
    test_gp_fits_noisy_sine()
    test_gp_predictive_std_increases_outside_training_data()
    test_gp_kernel_hyperparams_tune_via_grad()
    test_hmm_recovers_two_state_regime()
    test_hmm_viterbi_returns_int_sequence()
    test_hmm_responsibilities_sum_to_one()
    print("All GP + HMM tests passed.")
