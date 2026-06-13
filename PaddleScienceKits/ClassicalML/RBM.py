"""Restricted Boltzmann Machine (Hinton 2002) with CD-k training
re-implemented as a ``paddle.nn.Layer``.

Analogue:
    sklearn.neural_network.BernoulliRBM (Hinton 2002, CD-k).
    This is the Bernoulli-binary variant; Gaussian-binary RBMs
    (for continuous data) are not implemented.

Parameters
----------
n_visible, n_hidden : int
lr : float
    Constant learning rate (no momentum / weight decay).
k : int, default 1
    Number of Gibbs steps for contrastive divergence.
"""

import paddle


def _sample_bernoulli(p: paddle.Tensor) -> paddle.Tensor:
    return paddle.cast(paddle.rand(p.shape) < p, p.dtype)


class RBM(paddle.nn.Layer):
    """Bernoulli RBM with CD-k training and free-energy evaluation.

    Analogue:
        sklearn.neural_network.BernoulliRBM (Hinton 2002, CD-k).
    """

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        lr: float = 0.01,
        k: int = 1,
    ) -> None:
        super().__init__()
        if n_visible <= 0 or n_hidden <= 0:
            raise ValueError("n_visible and n_hidden must be > 0")
        if lr <= 0:
            raise ValueError("lr must be > 0")
        if k < 1:
            raise ValueError("k must be >= 1")
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = lr
        self.k = k

        self.W = paddle.create_parameter(
            shape=[n_visible, n_hidden], dtype="float32",
            default_initializer=paddle.nn.initializer.Uniform(-0.1, 0.1),
        )
        self.vbias = paddle.create_parameter(
            shape=[n_visible], dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0.0),
        )
        self.hbias = paddle.create_parameter(
            shape=[n_hidden], dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0.0),
        )

    def _sigmoid(self, x: paddle.Tensor) -> paddle.Tensor:
        return paddle.nn.functional.sigmoid(x)

    def sample_h_given_v(self, v: paddle.Tensor, return_prob: bool = False):
        """p(h | v) is independent Bernoulli with prob sigmoid(W^T v + hbias)."""
        prob = self._sigmoid(v @ self.W + self.hbias.unsqueeze(0))
        if return_prob:
            return prob
        return prob, _sample_bernoulli(prob)

    def sample_v_given_h(self, h: paddle.Tensor, return_prob: bool = False):
        prob = self._sigmoid(h @ self.W.T + self.vbias.unsqueeze(0))
        if return_prob:
            return prob
        return prob, _sample_bernoulli(prob)

    def free_energy(self, v: paddle.Tensor) -> paddle.Tensor:
        """Per-sample free energy ``F(v) = -v^T b - sum log(1 + exp(c + W^T v))``."""
        vbias_term = -v @ self.vbias
        wx_b = v @ self.W + self.hbias.unsqueeze(0)               # [N, H]
        # log(1 + exp(x)) computed stably.
        hidden_term = -paddle.sum(
            paddle.nn.functional.softplus(wx_b), axis=1
        )
        return vbias_term + hidden_term

    def contrastive_divergence(self, v0: paddle.Tensor) -> paddle.Tensor:
        """One step of CD-k. Returns the per-sample weight gradient
        accumulated by the positive and negative phases."""
        # Positive phase
        h0_prob, h0 = self.sample_h_given_v(v0, return_prob=False)
        # Negative phase: run k Gibbs steps starting from h0
        vk = v0
        hk = h0
        for _ in range(self.k):
            vk_prob, vk = self.sample_v_given_h(hk, return_prob=False)
            hk_prob, hk = self.sample_h_given_v(vk, return_prob=False)
        # For the parameter update we only need
        #   dW += h0_prob.T @ v0 - hk_prob.T @ vk
        # and the biases.
        dW = v0.T @ h0_prob - vk.T @ hk_prob
        dvbias = paddle.sum(v0 - vk, axis=0)
        dhbias = paddle.sum(h0_prob - hk_prob, axis=0)
        return dW, dvbias, dhbias

    def fit(self, x: paddle.Tensor, n_epochs: int = 50) -> "RBM":
        """Vanilla CD-k training (no momentum, no weight decay)."""
        n = x.shape[0]
        for _ in range(n_epochs):
            dW, dvbias, dhbias = self.contrastive_divergence(x)
            self.W.set_value(self.W + self.lr * dW / n)
            self.vbias.set_value(self.vbias + self.lr * dvbias / n)
            self.hbias.set_value(self.hbias + self.lr * dhbias / n)
        return self

    def sample(self, n_steps: int = 100, n_chains: int = 1) -> paddle.Tensor:
        """Run a Gibbs chain for ``n_steps`` and return the last
        visible sample for each of ``n_chains`` parallel chains."""
        v = paddle.cast(paddle.rand([n_chains, self.n_visible]) < 0.5, "float32")
        for _ in range(n_steps):
            _, h = self.sample_h_given_v(v)
            _, v = self.sample_v_given_h(h)
        return v

    def transform(self, x: paddle.Tensor) -> paddle.Tensor:
        """Hidden activation probabilities ``p(h | v)``, a deterministic
        feature extractor."""
        return self.sample_h_given_v(x, return_prob=True)

    def extra_repr(self) -> str:
        return f"n_visible={self.n_visible}, n_hidden={self.n_hidden}, lr={self.lr}, k={self.k}"
