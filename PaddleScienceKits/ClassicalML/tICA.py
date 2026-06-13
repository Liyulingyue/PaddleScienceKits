"""Time-lagged Independent Component Analysis (tICA, Molgedey &
Schuster 1994 / Schwantes & Pande 2015) re-implemented as a
``paddle.nn.Layer``.

Given a centred time series ``X = [x_1, ..., x_T]`` with each
``x_t ∈ R^D``, tICA finds the projection directions that maximise
the autocorrelation at a fixed lag ``τ`` while minimising the
instantaneous variance. This is the closed-form generalisation
of PCA to time series and corresponds to the slowest linear
relaxation modes of the underlying dynamics.

The projection matrix ``V: [n_components, dim]`` is stored as a
learnable parameter; at every forward call it is re-orthonormalised
on the Stiefel manifold (QR) so the modes are well-conditioned.
``fit`` solves the closed-form generalised eigen problem
``C(τ) v = λ C(0) v`` and stores the ``n_components`` eigenvectors
with the smallest ``|λ|`` (slowest modes).
"""

import paddle

from .utils import _to_2d


def _orthonormalise(matrix: paddle.Tensor) -> paddle.Tensor:
    q, r = paddle.linalg.qr(matrix.T, mode="reduced")
    sign = paddle.sign(paddle.diag(r))
    sign = paddle.where(sign == 0, paddle.ones_like(sign), sign)
    q = q * sign.unsqueeze(0)
    return q.T


class tICA(paddle.nn.Layer):
    """Time-lagged ICA / slow-modes projection.

    Parameters
    ----------
    n_components : int
        Number of slow modes to keep.
    dim : int
        Input feature dimension.
    lag : int, default 1
        Number of timesteps between the two snapshots whose
        covariance is maximised.
    """

    def __init__(self, n_components: int, dim: int, lag: int = 1) -> None:
        super().__init__()
        if not (0 < n_components <= dim):
            raise ValueError(
                f"0 < n_components={n_components} <= dim={dim} required"
            )
        if lag < 1:
            raise ValueError(f"lag must be >= 1, got {lag}")
        self.n_components = n_components
        self.dim = dim
        self.lag = lag

        weight = paddle.create_parameter(
            shape=[n_components, dim], dtype="float32",
            default_initializer=paddle.nn.initializer.Uniform(-0.5, 0.5),
        )
        self.add_parameter("components", weight)
        with paddle.no_grad():
            self.components.set_value(_orthonormalise(weight))
        self.register_buffer(
            "mean", paddle.zeros([dim], dtype="float32")
        )
        self._is_fitted = False

    def _basis(self) -> paddle.Tensor:
        return _orthonormalise(self.components)

    # ----------------------------------------------------------------- fit
    @paddle.no_grad()
    def fit(self, x: paddle.Tensor) -> "tICA":
        """Solve the generalised eigen problem on ``x``."""
        x = _to_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} features, got {x.shape[1]}"
            )
        if x.shape[0] <= self.lag + 1:
            raise ValueError(
                f"Need at least lag+1 = {self.lag + 1} samples, got {x.shape[0]}"
            )
        self.mean.set_value(paddle.mean(x, axis=0))
        xc = x - self.mean
        # Time-lagged covariance: C_tau = (1/(T-lag)) Σ x_t x_{t+lag}^T
        x0 = xc[:-self.lag]
        x1 = xc[self.lag:]
        C_tau = (x0.T @ x1) / x0.shape[0]
        C_0 = (xc.T @ xc) / xc.shape[0] + 1e-4 * paddle.eye(self.dim, dtype=xc.dtype)
        # Generalised eigen problem C_tau v = λ C_0 v.  We
        # symmetrise C_0^{-1} C_tau so a standard eigh gives
        # the right eigenvalues.
        C_0_inv_C_tau = paddle.linalg.solve(C_0, C_tau)
        M = 0.5 * (C_0_inv_C_tau + C_0_inv_C_tau.T)
        eigvals, eigvecs = paddle.linalg.eigh(M)               # ascending
        # Slowest modes are the ones with |λ| closest to 1 (i.e.
        # near 1 for very slow modes). We sort by |1 - λ| ascending
        # and take the first ``n_components``.
        slowness = paddle.abs(1.0 - eigvals)
        order = paddle.argsort(slowness)[: self.n_components]
        basis = eigvecs[:, order].T                            # [k, dim]
        self.components.set_value(basis)
        self._is_fitted = True
        return self

    # ----------------------------------------------------------------- ops
    def transform(self, x: paddle.Tensor) -> paddle.Tensor:
        """Project centred inputs onto the slow-mode subspace."""
        x = _to_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} features, got {x.shape[1]}"
            )
        if not self._is_fitted:
            raise RuntimeError("tICA is not fitted; call fit() first.")
        return (x - self.mean) @ self._basis().T

    def reconstruct(self, coords: paddle.Tensor) -> paddle.Tensor:
        coords = _to_2d(coords)
        if coords.shape[1] != self.n_components:
            raise ValueError(
                f"Expected {self.n_components} coords, got {coords.shape[1]}"
            )
        return coords @ self._basis() + self.mean

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.reconstruct(self.transform(x))

    def extra_repr(self) -> str:
        return (
            f"n_components={self.n_components}, dim={self.dim}, "
            f"lag={self.lag}, fitted={self._is_fitted}"
        )
