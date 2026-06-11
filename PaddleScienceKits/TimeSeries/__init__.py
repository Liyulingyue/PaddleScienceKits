"""Time-series classical models implemented as ``paddle.nn.Layer``.

All models share a single underlying building block,
:class:`PaddleScienceKits.TimeSeries.Autoregressive.Autoregressive`,
which encodes the generic relation

    A(p) y(k) = B(q) u(k) + C(o) v(k)

with ``y`` the dependent variable, ``u`` the exogenous input(s),
and ``v`` the disturbance / moving-average noise.
"""

from .Autoregressive import Autoregressive
from .AR import AR
from .ARMA import ARMA
from .FIR import FIR

__all__ = ["Autoregressive", "AR", "ARMA", "FIR"]
