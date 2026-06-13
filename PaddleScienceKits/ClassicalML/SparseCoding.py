"""Sparse coding / dictionary learning re-implemented as a
``paddle.nn.Layer``.

The dictionary ``D: [n_atoms, n_features]`` is a learnable
:class:`paddle.nn.Parameter` whose columns are unit-norm to break
the scaling degeneracy with the codes. The forward path runs
ISTA (or FISTA) for a fixed number of iterations to obtain sparse
codes ``z`` for the input batch, then the layer returns
``D.T @ z`` as the reconstruction.

The standard training objective is

    loss = (1/2) ||x - D^T z||_2^2 + lambda ||z||_1

which ``sparse_loss`` returns directly. ``fit`` wraps the loss in
an Adam update so the dictionary learns end-to-end with
differentiable ISTA.
"""

import paddle


def _soft_threshold(x: paddle.Tensor, t: paddle.Tensor) -> paddle.Tensor:
    return paddle.sign(x) * paddle.maximum(paddle.abs(x) - t, paddle.zeros_like(x))


class SparseCoding(paddle.nn.Layer):
    """
    Analogue:
        sklearn.decomposition.DictionaryLearning (Olshausen & Field 1996; Lee & Seung 2001 ISTA/FISTA)
    Sparse coding with ISTA / FISTA encoder.

    Parameters
    ----------
    n_atoms : int
        Number of dictionary atoms (code dimension).
    n_features : int
        Data dimension.
    lmbda : float
        Sparsity penalty coefficient.
    encoder : {"ista", "fista"}
    n_iter : int
        Number of encoder iterations inside ``forward``.
    lr : float
        Step size for ISTA. Should satisfy ``lr < 2 / ||D D^T||_2``;
        the constructor uses the safe default ``1 / n_atoms``.
    """

    def __init__(
        self,
        n_atoms: int,
        n_features: int,
        lmbda: float = 0.1,
        encoder: str = "ista",
        n_iter: int = 50,
        lr: float = 0.0,
    ) -> None:
        super().__init__()
        if encoder not in {"ista", "fista"}:
            raise ValueError(f"Unknown encoder {encoder!r}")
        if n_atoms <= 0 or n_features <= 0:
            raise ValueError("n_atoms and n_features must be > 0")
        if n_iter <= 0:
            raise ValueError("n_iter must be > 0")

        self.n_atoms = n_atoms
        self.n_features = n_features
        self.lmbda = lmbda
        self.encoder = encoder
        self.n_iter = n_iter
        self.lr = lr if lr > 0 else 1.0 / n_atoms

        D = paddle.randn([n_atoms, n_features], dtype="float32")
        D = D / paddle.clip(paddle.norm(D, axis=1, keepdim=True), min=1e-12)
        self.D = paddle.create_parameter(
            shape=D.shape, dtype="float32",
            default_initializer=paddle.nn.initializer.Assign(D),
        )

    def _renormalise(self) -> None:
        """Re-normalise dictionary columns after each step to break
        the scale degeneracy with the codes."""
        with paddle.no_grad():
            D = self.D / paddle.clip(paddle.norm(self.D, axis=1, keepdim=True), min=1e-12)
            self.D.set_value(D)

    def encode(self, x: paddle.Tensor) -> paddle.Tensor:
        """Run ISTA / FISTA on a batch ``x: [batch, n_features]`` and
        return sparse codes ``[batch, n_atoms]``."""
        D = self.D                                          # [k, d]
        threshold = self.lmbda * self.lr
        if self.encoder == "ista":
            z = paddle.zeros([x.shape[0], self.n_atoms], dtype=x.dtype)
            for _ in range(self.n_iter):
                # grad of (1/2)||x - zD||^2 wrt z is (zD - x) D^T
                residual = z @ D - x                         # [batch, d]
                grad = residual @ D.T                       # [batch, k]
                z = _soft_threshold(z - self.lr * grad, threshold)
        else:  # fista
            z = paddle.zeros([x.shape[0], self.n_atoms], dtype=x.dtype)
            z_prev = z.clone()
            t = paddle.to_tensor(1.0, dtype=x.dtype)
            for _ in range(self.n_iter):
                y = z + ((t - 1.0) / (t + 1.0)) * (z - z_prev)
                residual = y @ D - x
                grad = residual @ D.T
                z_new = _soft_threshold(y - self.lr * grad, threshold)
                z_prev = z
                z = z_new
                t = (1.0 + (5.0 * t * t + 1.0).sqrt()) / 2.0
        return z

    def reconstruct(self, z: paddle.Tensor) -> paddle.Tensor:
        return z @ self.D

    def forward(self, x: paddle.Tensor) -> tuple:
        """Return (z, x_hat) where x_hat is the reconstruction."""
        z = self.encode(x)
        return z, self.reconstruct(z)

    def sparse_loss(self, x: paddle.Tensor) -> paddle.Tensor:
        z, x_hat = self.forward(x)
        return 0.5 * paddle.mean(paddle.sum((x - x_hat) ** 2, axis=1)) + \
            self.lmbda * paddle.mean(paddle.sum(paddle.abs(z), axis=1))

    @paddle.no_grad()
    def fit(self, x: paddle.Tensor, n_outer: int = 50, lr: float = 5e-2) -> "SparseCoding":
        """Train the dictionary with Adam on the sparse coding loss."""
        opt = paddle.optimizer.Adam(parameters=[self.D], learning_rate=lr)
        for _ in range(n_outer):
            loss = self.sparse_loss(x)
            opt.clear_grad()
            loss.backward()
            opt.step()
            self._renormalise()
        return self

    def extra_repr(self) -> str:
        return (
            f"n_atoms={self.n_atoms}, n_features={self.n_features}, "
            f"lmbda={self.lmbda}, encoder={self.encoder!r}, n_iter={self.n_iter}"
        )
