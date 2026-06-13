"""Bayesian linear regression with mean-field variational inference,
re-implemented as a ``paddle.nn.Layer``.

The variational family is a diagonal Gaussian
``q(w) = prod_i N(w_i; mu_i, sigma_i^2)`` over the regression
weights, parameterised by ``mu`` and ``log_sigma`` as
``nn.Parameter``. The prior is also diagonal Gaussian
``N(0, prior_std^2 I)``. The layer minimises the ELBO loss

    loss = E_q [ log p(y | x, w) ] + KL(q(w) || p(w))

estimated by a Monte-Carlo reparameterised sample, plus the
analytical KL term. ``forward`` returns the predictive mean
under the variational posterior; ``predict`` returns the predictive
mean and (epistemic) standard deviation by averaging multiple
samples.
"""

import paddle


def _gauss_kl(
    mu_q: paddle.Tensor, log_sigma_q: paddle.Tensor,
    mu_p: paddle.Tensor, log_sigma_p: paddle.Tensor
) -> paddle.Tensor:
    """KL( N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2) ), summed over dims."""
    var_q = paddle.exp(2.0 * log_sigma_q)
    var_p = paddle.exp(2.0 * log_sigma_p)
    return 0.5 * paddle.sum(
        (var_q + (mu_q - mu_p) ** 2) / var_p
        - 1.0
        + 2.0 * log_sigma_p
        - 2.0 * log_sigma_q
    )


class BayesianLinearVI(paddle.nn.Layer):
    """Bayesian linear regression with mean-field VI.

    Parameters
    ----------
    n_features : int
    n_outputs : int, default 1
    prior_std : float
        Standard deviation of the diagonal Gaussian prior on ``w``.
    noise_std : float
        Standard deviation of the observation noise ``p(y | x, w)``.
    """

    def __init__(
        self,
        n_features: int,
        n_outputs: int = 1,
        prior_std: float = 1.0,
        noise_std: float = 0.1,
    ) -> None:
        super().__init__()
        if n_features <= 0 or n_outputs <= 0:
            raise ValueError("n_features and n_outputs must be > 0")
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.prior_std = prior_std
        self.noise_std = noise_std

        self.mu = paddle.create_parameter(
            shape=[n_features, n_outputs], dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0.0),
        )
        self.log_sigma = paddle.create_parameter(
            shape=[n_features, n_outputs], dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(-3.0),
        )

    def _sample_weights(self, n_samples: int) -> paddle.Tensor:
        eps = paddle.randn([n_samples, self.n_features, self.n_outputs])
        return self.mu + paddle.exp(self.log_sigma) * eps

    def forward(self, x: paddle.Tensor, n_samples: int = 1) -> paddle.Tensor:
        """Predictive mean under the variational posterior."""
        w = self._sample_weights(n_samples)                 # [S, F, O]
        # x: [B, F];  x @ w -> [B, S, O];  mean over S.
        pred = paddle.einsum("bf, sfo -> bso", x, w).mean(axis=0)
        return pred

    def neg_elbo(self, x: paddle.Tensor, y: paddle.Tensor, n_samples: int = 1) -> paddle.Tensor:
        """Negative ELBO = -E_q[log p(y | x, w)] + KL(q || p)."""
        w = self._sample_weights(n_samples)
        # Predictive mean for each sample
        pred = paddle.einsum("bf, sfo -> bso", x, w)        # [B, S, O]
        # Gaussian log-likelihood, summed over data and averaged over samples
        var = paddle.to_tensor(self.noise_std ** 2, dtype=x.dtype)
        log_lik = -0.5 * ((y.unsqueeze(1) - pred) ** 2 / var).sum(axis=[0, 2]).mean()
        log_lik = log_lik - 0.5 * (x.shape[0] * self.n_outputs) * paddle.log(
            2 * 3.141592653589793 * var
        )
        kl = _gauss_kl(
            self.mu, self.log_sigma,
            paddle.zeros_like(self.mu),
            paddle.log(paddle.full_like(self.mu, self.prior_std)),
        )
        return -log_lik / x.shape[0] + kl / x.shape[0]

    def predict(self, x: paddle.Tensor, n_samples: int = 50, return_std: bool = True):
        """Posterior-predictive mean and (epistemic) std."""
        w = self._sample_weights(n_samples)                 # [S, F, O]
        pred = paddle.einsum("bf, sfo -> bso", x, w)        # [B, S, O]
        mean = pred.mean(axis=1)                            # [B, O]
        if not return_std:
            return mean
        # Epistemic std: std across samples.
        std = pred.std(axis=1)                              # [B, O]
        # Add aleatoric noise.
        std = paddle.sqrt(std ** 2 + self.noise_std ** 2)
        return mean, std

    def extra_repr(self) -> str:
        return (
            f"n_features={self.n_features}, n_outputs={self.n_outputs}, "
            f"prior_std={self.prior_std}, noise_std={self.noise_std}"
        )
