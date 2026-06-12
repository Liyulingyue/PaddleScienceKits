"""PaddleScienceKits
A machine learning kits based on PaddlePaddle, re-implementing classical
models as ``paddle.nn.Layer`` so they can be composed with deep networks.
"""

from . import TimeSeries, ClassicalML, SignalProcessing

__version__ = "0.1.0"
__all__ = ["TimeSeries", "ClassicalML", "SignalProcessing", "__version__"]
