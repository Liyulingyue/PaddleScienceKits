"""ClassicalML submodule: classical ML models re-implemented as
``paddle.nn.Layer`` so they can be composed with deep networks.

Currently hosts:

* :class:`KMeans`            — centroids as learnable parameters; soft or
  hard assignment; standard Lloyd updates during ``fit``.
* :class:`KNN`               — non-parametric memory layer; top-k
  retrieval and average-pool reconstruction.
* :class:`PCA`               — orthonormal basis as a learnable
  parameter; project / reconstruct; SVD-based fitting.
* :class:`KernelRidge`       — dual-form linear-in-memory model with
  RBF / linear / polynomial kernels.
* :class:`GMM`               — Gaussian mixture with learnable means,
  log-variances, and weights; EM fitting; differentiable soft
  responsibilities.
* :class:`LDA`               — Fisher linear discriminant analysis with
  closed-form eigen solve.
* :class:`GaussianNB`        — per-class Gaussian Naive Bayes with
  MLE-estimated mean / variance.
* :class:`MultinomialNB`     — multinomial Naive Bayes for count
  features with Laplace smoothing.
* :class:`ICA`               — FastICA fixed-point source separation
  with whitening.
* :class:`SoftDecisionTree`  — Frosst 2017 differentiable decision
  tree with sigmoid inner nodes and per-leaf class distributions.
"""

from .KMeans import KMeans
from .KNN import KNN
from .PCA import PCA
from .KernelRidge import KernelRidge
from .GMM import GMM
from .LDA import LDA
from .NaiveBayes import GaussianNB, MultinomialNB
from .ICA import ICA
from .SoftDecisionTree import SoftDecisionTree

__all__ = [
    "KMeans", "KNN", "PCA", "KernelRidge", "GMM",
    "LDA", "GaussianNB", "MultinomialNB", "ICA", "SoftDecisionTree",
]
