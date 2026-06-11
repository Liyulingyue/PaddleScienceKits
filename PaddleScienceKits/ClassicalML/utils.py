"""Shared utilities for the ClassicalML submodule."""

from typing import List

import paddle


def _to_2d(tensor: paddle.Tensor) -> paddle.Tensor:
    """Ensure ``tensor`` is 2D ``[batch, features]``."""
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(
            f"Expected a 1D or 2D tensor [batch, features], got shape {tensor.shape}"
        )
    return tensor


def _check_shapes(tensors: List[paddle.Tensor], names: List[str]) -> None:
    if not tensors:
        return
    batch = tensors[0].shape[0]
    for t, n in zip(tensors, names):
        if t.shape[0] != batch:
            raise ValueError(
                f"Batch size mismatch: expected {batch} for {n!r}, got {t.shape[0]}"
            )
