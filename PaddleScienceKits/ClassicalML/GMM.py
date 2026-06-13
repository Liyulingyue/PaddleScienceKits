"""Gaussian Mixture Model re-implemented as a ``paddle.nn.Layer``.

The model stores, per component ``k``:

* a learnable ``means``        parameter of shape ``[k, dim]``
* a learnable ``log_vars``     parameter of shape ``[k, dim]`` (diagonal
  log-variances; the ``spherical`` shape is treated as a length-1
  tensor and broadcast)
* a learnable ``log_weights``  parameter of shape ``[k]`` (logits; the
  softmax gives the mixture coefficients)

The forward pass returns the soft responsibilities
``gamma_{nk} = p(z=k | x_n)`` of shape ``[batch, k]``, fully
differentiable, so the layer can act as a K-dimensional feature
extractor in a downstream classifier.
"""

from typing import Optional

import paddle

from .utils import _to_2d


_VALID_COV = {"diag", "spherical", "full"}


def _log_gaussian_diag(
    x: paddle.Tensor, mean: paddle.Tensor, log_var: paddle.Tensor
) -> paddle.Tensor:
    """Per-component log-density, ``[batch, k]``."""
    # x: [B, D], mean: [K, D], log_var: [K, D]
    diff = x.unsqueeze(1) - mean.unsqueeze(0)                  # [B, K, D]
    inv_var = paddle.exp(-log_var)                              # [K, D]
    quad = paddle.sum(diff * diff * inv_var.unsqueeze(0), axis=-1)  # [B, K]
    log_norm = paddle.sum(log_var, axis=-1) + mean.shape[-1] * paddle.log(
        paddle.to_tensor(2 * paddle.pi, dtype=mean.dtype)
    )                                                          # [K]
    return -0.5 * (quad + log_norm.unsqueeze(0))


def _log_gaussian_spherical(
    x: paddle.Tensor, mean: paddle.Tensor, log_var: paddle.Tensor
) -> paddle.Tensor:
    diff = x.unsqueeze(1) - mean.unsqueeze(0)                  # [B, K, D]
    sq = paddle.sum(diff * diff, axis=-1)                       # [B, K]
    D = mean.shape[-1]
    return -0.5 * (sq * paddle.exp(-log_var).squeeze(-1).unsqueeze(0)
                   + D * log_var.squeeze(-1).unsqueeze(0)
                   + D * paddle.log(paddle.to_tensor(2 * paddle.pi, dtype=mean.dtype)))


def _log_gaussian_full(
    x: paddle.Tensor, mean: paddle.Tensor, L: paddle.Tensor, log_det: paddle.Tensor
) -> paddle.Tensor:
    """``L`` is the lower-triangular Cholesky factor, ``log_det`` its
    log-determinant of the corresponding covariance."""
    diff = x.unsqueeze(1) - mean.unsqueeze(0)                  # [B, K, D]
    z = paddle.linalg.solve_triangular(
        L.unsqueeze(0).tile([diff.shape[0], 1, 1]),
        diff,
        upper=False,
    )                                                          # [B, K, D]
    quad = paddle.sum(z * z, axis=-1)                           # [B, K]
    return -0.5 * (quad + log_det.unsqueeze(0) + mean.shape[-1] * paddle.log(
        paddle.to_tensor(2 * paddle.pi, dtype=mean.dtype)
    ))


class GMM(paddle.nn.Layer):
    """
    Analogue:
        sklearn.mixture.GaussianMixture (Dempster et al. 1977 EM)
    Gaussian Mixture with learnable parameters and EM fitting.

    Parameters
    ----------
    k : int
        Number of components.
    dim : int
        Feature dimension.
    covariance_type : {"diag", "spherical", "full"}
    reg : float
        Diagonal regularisation added to covariances during the
        forward pass (for numerical stability).
    """

    def __init__(
        self,
        k: int,
        dim: int,
        covariance_type: str = "diag",
        reg: float = 1e-6,
    ) -> None:
        super().__init__()
        if k <= 0 or dim <= 0:
            raise ValueError("k and dim must be > 0")
        if covariance_type not in _VALID_COV:
            raise ValueError(
                f"Unknown covariance_type {covariance_type!r}; "
                f"pick from {_VALID_COV}"
            )

        self.k = k
        self.dim = dim
        self.covariance_type = covariance_type
        self.reg = reg

        self.means = paddle.create_parameter(
            shape=[k, dim], dtype="float32",
            default_initializer=paddle.nn.initializer.Uniform(-0.5, 0.5),
        )
        if covariance_type == "spherical":
            self.log_vars = paddle.create_parameter(
                shape=[k, 1], dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
        elif covariance_type == "diag":
            self.log_vars = paddle.create_parameter(
                shape=[k, dim], dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
        else:  # full
            self.log_vars = paddle.create_parameter(
                shape=[k, dim, dim], dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
        self.log_weights = paddle.create_parameter(
            shape=[k], dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0.0),
        )

    # ------------------------------------------------------- probabilities
    def _log_resp(self, x: paddle.Tensor) -> paddle.Tensor:
        """Log responsibilities ``[batch, k]`` (differentiable)."""
        if self.covariance_type == "full":
            L = paddle.linalg.cholesky(
                paddle.exp(self.log_vars) + self.reg * paddle.eye(
                    self.dim, dtype=self.log_vars.dtype
                ).unsqueeze(0)
            )
            log_det = 2.0 * paddle.sum(
                paddle.log(paddle.diagonal(L, axis1=-2, axis2=-1)), axis=-1
            )
            log_p = _log_gaussian_full(x, self.means, L, log_det)
        elif self.covariance_type == "spherical":
            log_p = _log_gaussian_spherical(x, self.means, self.log_vars)
        else:
            log_p = _log_gaussian_diag(x, self.means, self.log_vars)

        log_w = paddle.nn.functional.log_softmax(self.log_weights, axis=-1)
        log_p = log_p + log_w.unsqueeze(0)
        return paddle.nn.functional.log_softmax(log_p, axis=-1)

    def responsibilities(self, x: paddle.Tensor) -> paddle.Tensor:
        return paddle.exp(self._log_resp(x))

    def log_likelihood(self, x: paddle.Tensor) -> paddle.Tensor:
        """Per-sample log marginal likelihood ``[batch]``."""
        return paddle.logsumexp(self._log_resp(x), axis=-1)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Soft responsibilities ``[batch, k]`` (differentiable)."""
        x = _to_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} features, got {x.shape[1]}"
            )
        return self.responsibilities(x)

    # ------------------------------------------------------------------- EM
    @paddle.no_grad()
    def fit_em(self, x: paddle.Tensor, n_iter: int = 50) -> "GMM":
        """Closed-form maximum-likelihood EM updates."""
        x = _to_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} features, got {x.shape[1]}"
            )
        for _ in range(n_iter):
            self._em_step_(x)
        return self

    @paddle.no_grad()
    def _em_step_(self, x: paddle.Tensor) -> None:
        # E-step
        log_r = self._log_resp(x)                               # [B, K]
        r = paddle.exp(log_r)                                   # [B, K]
        N = r.shape[0]

        N_k = paddle.sum(r, axis=0)                             # [K]
        weights = N_k / N

        # M-step
        new_means = (r.T @ x) / N_k.unsqueeze(-1)               # [K, D]
        diff = x.unsqueeze(1) - new_means.unsqueeze(0)          # [B, K, D]

        if self.covariance_type == "diag":
            new_var = paddle.sum(
                r.unsqueeze(-1) * diff * diff, axis=0
            ) / N_k.unsqueeze(-1) + self.reg                    # [K, D]
            self.log_vars.set_value(paddle.log(new_var))
        elif self.covariance_type == "spherical":
            sq = paddle.sum(diff * diff, axis=-1)               # [B, K]
            new_var = paddle.sum(r * sq, axis=0) / (N_k * self.dim) + self.reg
            self.log_vars.set_value(paddle.log(new_var).unsqueeze(-1))
        else:  # full
            new_cov = paddle.zeros_like(self.log_vars)           # [K, D, D]
            for k in range(self.k):
                w = r[:, k].unsqueeze(-1).unsqueeze(-1)         # [B, 1, 1]
                d = diff[:, k, :].unsqueeze(-1)                  # [B, D, 1]
                new_cov[k] = (w * d @ d.transpose([0, 2, 1])).sum(axis=0) / N_k[k]
            new_cov = new_cov + self.reg * paddle.eye(
                self.dim, dtype=new_cov.dtype
            ).unsqueeze(0)
            # store as log-Cholesky-diagonal proxy; the forward uses
            # Cholesky(exp(log_vars) + reg*I) anyway, so we keep log_vars
            # as the matrix log of the covariance via the cholesky
            # diagonal. Here we simply store cholesky(cov) and let
            # the forward recompute the cholesky. To stay consistent
            # we just overwrite log_vars with the cholesky factors and
            # interpret them as such in the forward. To keep the API
            # simple we set log_vars = log(cholesky_diag) is unstable;
            # instead we store the full matrix and let forward reuse it.
            self.log_vars.set_value(new_cov)

        self.means.set_value(new_means)
        # log_weights via log of weights (with a tiny floor to avoid -inf)
        safe_w = paddle.clip(weights, min=1e-12)
        self.log_weights.set_value(paddle.log(safe_w))

    def extra_repr(self) -> str:
        return (
            f"k={self.k}, dim={self.dim}, covariance={self.covariance_type!r}, "
            f"reg={self.reg}"
        )
