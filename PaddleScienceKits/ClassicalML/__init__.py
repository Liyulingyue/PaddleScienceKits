"""ClassicalML submodule: classical ML models re-implemented as
``paddle.nn.Layer`` so they can be composed with deep networks.

Currently hosts:

* :class:`KMeans`       — centroids as learnable parameters; soft or
  hard assignment; standard Lloyd updates during ``fit``.
* :class:`KNN`          — non-parametric memory layer; top-k retrieval
  and average-pool reconstruction.
* :class:`PCA`          — orthonormal basis as a learnable parameter;
  project / reconstruct; SVD-based fitting.
* :class:`KernelRidge`  — dual-form linear-in-memory model with
  RBF / linear / polynomial kernels.
* :class:`GMM`          — Gaussian mixture with learnable means,
  log-variances, and weights; EM fitting; differentiable soft
  responsibilities.
"""

from .KMeans import KMeans
from .KNN import KNN
from .PCA import PCA
from .KernelRidge import KernelRidge
from .GMM import GMM

__all__ = ["KMeans", "KNN", "PCA", "KernelRidge", "GMM"]
