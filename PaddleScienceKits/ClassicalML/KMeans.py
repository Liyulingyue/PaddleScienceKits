"""KMeans re-implemented as a ``paddle.nn.Layer``.

The layer holds ``k`` centroids as a learnable :class:`paddle.nn.Parameter`
of shape ``[k, dim]``. In ``train()`` mode a single Lloyd-style update
moves the centroids toward the input batch; in ``eval()`` mode the
centroids are frozen and the layer behaves as a quantizer that turns
an input vector into either a soft assignment distribution
(shape ``[batch, k]``) or a hard integer label.

The soft-assignment path is differentiable, so ``KMeans`` can be
plugged in front of a downstream classifier and trained end-to-end
(e.g. for representation clustering as a regularizer).
"""

from typing import Optional

import paddle

from .utils import _to_2d


class KMeans(paddle.nn.Layer):
    """K-means quantizer with learnable centroids.

    Parameters
    ----------
    k : int
        Number of clusters.
    dim : int
        Input feature dimension.
    temperature : float, default 1.0
        Softmax temperature for the soft-assignment path. Smaller
        values produce sharper distributions.
    init : {"random", "kmeans++"}, default "kmeans++"
        How to initialize the centroids at construction time.
    """

    def __init__(
        self,
        k: int,
        dim: int,
        temperature: float = 1.0,
        init: str = "kmeans++",
    ) -> None:
        super().__init__()
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.k = k
        self.dim = dim
        self.temperature = temperature

        weight = paddle.create_parameter(
            shape=[k, dim],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Uniform(-0.5, 0.5),
        )
        self.add_parameter("centroids", weight)

        if init == "kmeans++":
            self._kmeanspp_init_()
        elif init != "random":
            raise ValueError(f"Unknown init {init!r}; use 'random' or 'kmeans++'.")

    # ------------------------------------------------------------------ init
    @paddle.no_grad()
    def _kmeanspp_init_(self) -> None:
        """Proper k-means++ seeding on a Gaussian sample.

        Without any data the layer can't do a meaningful k-means++, so it
        draws ``max(k * 10, 100)`` unit-Gaussian points and seeds the
        centroids one by one, each new centroid picked with probability
        proportional to the squared distance to the closest existing
        centroid. This avoids the all-cluster-collapse pathology of a
        naive Lloyd step on tight initial data.
        """
        sample = paddle.randn([max(self.k * 10, 100), self.dim])
        # First centroid: pick the point with the largest norm (good
        # diversity in expectation for a zero-mean sample).
        norms = paddle.sum(sample * sample, axis=1)
        first = int(paddle.argmax(norms))
        centroids = [sample[first].clone()]
        for _ in range(1, self.k):
            cur = paddle.stack(centroids, axis=0)        # [m, dim]
            cur_sq = paddle.sum(cur * cur, axis=1)       # [m]
            x_sq = paddle.sum(sample * sample, axis=1)   # [n]
            cross = sample @ cur.T                       # [n, m]
            d = paddle.maximum(
                x_sq.unsqueeze(1) + cur_sq.unsqueeze(0) - 2.0 * cross,
                paddle.zeros_like(cross),
            )
            min_d = paddle.min(d, axis=1)                # [n]
            idx = int(paddle.argmax(min_d))
            centroids.append(sample[idx].clone())
        self.centroids.set_value(paddle.stack(centroids, axis=0))

    # ----------------------------------------------------------------- assign
    def soft_assignment(self, x: paddle.Tensor) -> paddle.Tensor:
        """Differentiable soft assignment, shape ``[batch, k]``."""
        x = _to_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} features, got {x.shape[1]}"
            )
        # squared euclidean distance via ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        sq_x = paddle.sum(x * x, axis=1, keepdim=True)        # [B, 1]
        sq_c = paddle.sum(self.centroids * self.centroids, axis=1)  # [k]
        cross = x @ self.centroids.T                          # [B, k]
        dist = sq_x + sq_c - 2.0 * cross                      # [B, k]
        # clamp tiny negatives from float roundoff
        dist = paddle.maximum(dist, paddle.zeros_like(dist))
        return paddle.nn.functional.softmax(-dist / self.temperature, axis=-1)

    def hard_assignment(self, x: paddle.Tensor) -> paddle.Tensor:
        """Argmin cluster index, shape ``[batch]`` (int64)."""
        x = _to_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} features, got {x.shape[1]}"
            )
        sq_x = paddle.sum(x * x, axis=1, keepdim=True)
        sq_c = paddle.sum(self.centroids * self.centroids, axis=1)
        cross = x @ self.centroids.T
        dist = sq_x + sq_c - 2.0 * cross
        return paddle.argmin(dist, axis=-1)

    def forward(
        self,
        x: paddle.Tensor,
        hard: bool = False,
    ) -> paddle.Tensor:
        if hard or not self.training:
            return self.hard_assignment(x)
        return self.soft_assignment(x)

    # ------------------------------------------------------------------- fit
    @paddle.no_grad()
    def fit(self, x: paddle.Tensor, n_iter: int = 10) -> "KMeans":
        """Run Lloyd iterations on ``x`` to refine the centroids."""
        x = _to_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} features, got {x.shape[1]}"
            )
        for _ in range(n_iter):
            self._lloyd_step_(x)
        return self

    @paddle.no_grad()
    def fit_kmeanspp(self, x: paddle.Tensor) -> "KMeans":
        """Re-seed the centroids with k-means++ on the supplied data.

        Useful when ``__init__`` was called without a real data sample
        (e.g. in unit tests that probe a fresh layer).
        """
        x = _to_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} features, got {x.shape[1]}"
            )
        n = x.shape[0]
        if n == 0:
            return self
        # pick the point with the largest norm as the first centroid
        norms = paddle.sum(x * x, axis=1)
        first = int(paddle.argmax(norms))
        centroids = [x[first].clone()]
        for _ in range(1, self.k):
            cur = paddle.stack(centroids, axis=0)
            cur_sq = paddle.sum(cur * cur, axis=1)
            x_sq = paddle.sum(x * x, axis=1)
            cross = x @ cur.T
            d = paddle.maximum(
                x_sq.unsqueeze(1) + cur_sq.unsqueeze(0) - 2.0 * cross,
                paddle.zeros_like(cross),
            )
            min_d = paddle.min(d, axis=1)
            idx = int(paddle.argmax(min_d))
            centroids.append(x[idx].clone())
        self.centroids.set_value(paddle.stack(centroids, axis=0))
        return self

    @paddle.no_grad()
    def _lloyd_step_(self, x: paddle.Tensor) -> None:
        """One Lloyd assignment+update step, in-place on ``self.centroids``.

        Empty clusters are *re-seeded* at the point farthest from any
        current centroid; this prevents degenerate collapses when the
        initial centroids are poorly placed.
        """
        sq_x = paddle.sum(x * x, axis=1, keepdim=True)
        sq_c = paddle.sum(self.centroids * self.centroids, axis=1)
        cross = x @ self.centroids.T
        dist = paddle.maximum(sq_x + sq_c - 2.0 * cross, paddle.zeros_like(cross))

        labels = paddle.argmin(dist, axis=-1)              # [B]
        oh = paddle.nn.functional.one_hot(labels, num_classes=self.k).astype("float32")
        counts = paddle.sum(oh, axis=0)                     # [k]
        new_centroids = oh.T @ x                            # [k, dim]
        safe_counts = paddle.where(counts > 0, counts, paddle.ones_like(counts))
        new_centroids = new_centroids / safe_counts.unsqueeze(-1)
        empty = (counts == 0)
        new_centroids = paddle.where(
            empty.unsqueeze(-1), self.centroids, new_centroids
        )
        self.centroids.set_value(new_centroids)

        # Re-seed any cluster that is still empty after the safe update.
        if bool((counts == 0).any()):
            cur = self.centroids
            cur_sq = paddle.sum(cur * cur, axis=1, keepdim=True)
            x_sq = paddle.sum(x * x, axis=1, keepdim=True)
            cross = x @ cur.T
            d = paddle.maximum(x_sq + cur_sq - 2.0 * cross, paddle.zeros_like(cross))
            farthest_idx = paddle.argmax(d, axis=0)        # [k]
            replacement = x[farthest_idx]                  # [k, dim]
            reseed = empty
            self.centroids.set_value(
                paddle.where(reseed.unsqueeze(-1), replacement, self.centroids)
            )

    # --------------------------------------------------------------- extras
    def extra_repr(self) -> str:
        return f"k={self.k}, dim={self.dim}, temperature={self.temperature}"
