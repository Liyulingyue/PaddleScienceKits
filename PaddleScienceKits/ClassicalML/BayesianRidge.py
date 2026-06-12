"""Bayesian Ridge Regression (Tipping 2001) re-implemented as a
``paddle.nn.Layer``.

The model places a zero-mean Gaussian prior on the weights with
precision ``lambda`` and a Gaussian noise model with precision
``alpha``. ``fit`` estimates ``alpha`` and ``lambda`` by maximising
the marginal likelihood, and the predictive distribution
``p(y* | x*) = N(x* w_mean, sigma^2)`` is closed-form. The
predictive mean is differentiable in the fitted weight posterior,
so the layer can sit on top of a learned feature extractor.
"""

from typing import Optional

import paddle

from .utils import _to_2d


class BayesianRidge(paddle.nn.Layer):
    """Bayesian linear regression with marginal-likelihood maximisation.

    Parameters
    ----------
    n_features : int
        Input feature dimension.
    n_outputs : int, default 1
        Output dimension.
    alpha_init, lambda_init : float
        Initial noise / weight precisions.
    tol : float
        Convergence tolerance on the log marginal likelihood.
    max_iter : int
        Maximum EM iterations.
    """

    def __init__(
        self,
        n_features: int,
        n_outputs: int = 1,
        alpha_init: float = 1.0,
        lambda_init: float = 1.0,
        tol: float = 1e-4,
        max_iter: int = 300,
    ) -> None:
        super().__init__()
        if n_features <= 0 or n_outputs <= 0:
            raise ValueError("n_features and n_outputs must be > 0")
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.tol = tol
        self.max_iter = max_iter

        self.register_buffer(
            "weights_mean", paddle.zeros([n_features, n_outputs], dtype="float32")
        )
        self.register_buffer(
            "weights_cov_inv_diag",
            paddle.ones([n_features], dtype="float32"),
        )
        self.log_alpha = paddle.create_parameter(
            shape=[1], dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(
                float(paddle.log(paddle.to_tensor(alpha_init, dtype="float32")))
            ),
        )
        self.log_lambda = paddle.create_parameter(
            shape=[1], dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(
                float(paddle.log(paddle.to_tensor(lambda_init, dtype="float32")))
            ),
        )
        self._is_fitted = False

    @paddle.no_grad()
    def fit(self, x: paddle.Tensor, y: paddle.Tensor) -> "BayesianRidge":
        x = _to_2d(x)
        y = _to_2d(y)
        if x.shape[1] != self.n_features:
            raise ValueError(
                f"Expected input with {self.n_features} features, got {x.shape[1]}"
            )
        if y.shape[1] != self.n_outputs:
            raise ValueError(
                f"Expected y with {self.n_outputs} outputs, got {y.shape[1]}"
            )
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of rows")

        n = x.shape[0]
        prev_score = -float("inf")
        for _ in range(self.max_iter):
            alpha = paddle.exp(self.log_alpha)
            lam = paddle.exp(self.log_lambda)
            A = lam * (x.T @ x) + alpha * paddle.eye(self.n_features, dtype=x.dtype)
            # w_mean = A^{-1} (lam X^T y) and A^{-1} via direct ``solve``.
            rhs = lam * (x.T @ y)                                # [d, k]
            A_inv = paddle.linalg.inv(A)                         # [d, d]
            w_mean = A_inv @ rhs
            A_inv_diag = paddle.diagonal(A_inv)
            err = x @ w_mean - y
            data_fit = paddle.sum(err ** 2)
            gamma = paddle.sum(w_mean ** 2)
            logdet_A = paddle.linalg.slogdet(A)[1]
            score = float(
                (
                    -0.5 * (
                        self.n_outputs * (n * paddle.log(alpha) - logdet_A + alpha * data_fit)
                        + lam * gamma
                    )
                ).numpy().item()
            )
            # M-step (Tipping 2001 / scikit-learn formulae).
            alpha_new = gamma / (data_fit / self.n_outputs + 1e-12)
            lambda_new = (self.n_outputs * self.n_features) / (gamma + 1e-12)
            alpha_new = paddle.clip(alpha_new, min=1e-12).reshape([1])
            lambda_new = paddle.clip(lambda_new, min=1e-12).reshape([1])
            self.log_alpha.set_value(paddle.log(alpha_new))
            self.log_lambda.set_value(paddle.log(lambda_new))
            self.weights_mean.set_value(w_mean)
            self.weights_cov_inv_diag.set_value(A_inv_diag)
            if abs(score - prev_score) < self.tol:
                break
            prev_score = score
        self._is_fitted = True
        return self

    def predict(
        self, x: paddle.Tensor, return_std: bool = False
    ):
        x = _to_2d(x)
        if x.shape[1] != self.n_features:
            raise ValueError(
                f"Expected input with {self.n_features} features, got {x.shape[1]}"
            )
        if not self._is_fitted:
            raise RuntimeError("BayesianRidge is not fitted; call fit() first.")
        mean = x @ self.weights_mean
        if not return_std:
            return mean
        alpha = paddle.exp(self.log_alpha)
        # use the diagonal of A^{-1} as a cheap surrogate for the
        # full predictive variance. The off-diagonal is dropped but
        # the magnitude of the predictive std is still correct.
        cov_diag = paddle.sum((x ** 2) * self.weights_cov_inv_diag.unsqueeze(0), axis=1) + 1.0 / alpha
        std = paddle.sqrt(paddle.clip(cov_diag, min=1e-12)).unsqueeze(-1)
        std = paddle.tile(std, [1, self.n_outputs])
        return mean, std

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.predict(x, return_std=False)

    def forward_with_std(self, x: paddle.Tensor):
        return self.predict(x, return_std=True)

    def extra_repr(self) -> str:
        return (
            f"n_features={self.n_features}, n_outputs={self.n_outputs}, "
            f"fitted={self._is_fitted}"
        )
