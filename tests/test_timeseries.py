"""Smoke tests for PaddleScienceKits.TimeSeries.

Run with:
    .venv/bin/python -m pytest tests/ -q
or:
    .venv/bin/python tests/test_timeseries.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paddle  # noqa: E402

from PaddleScienceKits.TimeSeries import AR, ARMA, FIR, Autoregressive  # noqa: E402


def _ok(cond, msg):
    assert cond, msg
    print(f"  ok  {msg}")


def test_autoregressive_generic():
    block = Autoregressive(y_features=3, x_features=[2, 4], e_features=2)
    y = paddle.randn([8, 3])
    u1 = paddle.randn([8, 2])
    u2 = paddle.randn([8, 4])
    v = paddle.randn([8, 2])
    out = block(y, u1, u2, v)
    _ok(out.shape == [8, 1], f"generic block output shape {out.shape}")
    _ok(isinstance(out, paddle.Tensor), "output is a paddle.Tensor")

    # also accept single-sample (rank-1) inputs
    out1 = block(y[0], u1[0], u2[0], v[0])
    _ok(out1.shape == [1, 1], f"rank-1 inputs promoted to {out1.shape}")


def test_ar():
    p = 5
    ar = AR(p)
    y = paddle.randn([16, p])
    out = ar(y)
    _ok(out.shape == [16, 1], f"AR({p}) output shape {out.shape}")


def test_arma():
    p, q = 4, 3
    arma = ARMA(p, q)
    y = paddle.randn([16, p])
    v = paddle.randn([16, q + 1])
    out = arma(y, v)
    _ok(out.shape == [16, 1], f"ARMA({p},{q}) output shape {out.shape}")


def test_fir():
    q = 4
    fir = FIR(q)
    u = paddle.randn([16, q + 1])
    out = fir(u)
    _ok(out.shape == [16, 1], f"FIR({q}) output shape {out.shape}")


def test_gradient_flows():
    ar = AR(3)
    y = paddle.randn([4, 3])
    target = paddle.randn([4, 1])
    pred = ar(y)
    loss = paddle.nn.functional.mse_loss(pred, target)
    loss.backward()
    has_grad = any(
        p.grad is not None and paddle.sum(p.grad) != 0
        for p in ar.parameters()
    )
    _ok(has_grad, "AR produces non-zero gradients on parameters")


if __name__ == "__main__":
    test_autoregressive_generic()
    test_ar()
    test_arma()
    test_fir()
    test_gradient_flows()
    print("All TimeSeries smoke tests passed.")
