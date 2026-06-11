"""ClassicalML submodule: classical ML models re-implemented as
``paddle.nn.Layer`` so they can be composed with deep networks.

Currently hosts:

* :class:`KMeans`  — centroids as learnable parameters; soft or hard
  assignment; standard Lloyd updates during ``train()`` mode.
* :class:`KNN`     — non-parametric memory layer; top-k retrieval and
  average-pool reconstruction.
"""

from .KMeans import KMeans
from .KNN import KNN

__all__ = ["KMeans", "KNN"]
