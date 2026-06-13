"""Short-Time Fourier Transform layers.

Both the analysis window and the optional centre padding are exposed
as :class:`paddle.nn.Layer` so the layers participate in autograd.
The forward path is implemented with the standard frame / window /
rFFT pipeline; the inverse path uses overlap-add.
"""

import math
from typing import Optional, Tuple

import paddle
import paddle.nn.functional as F


def _hann_window(win_length: int) -> paddle.Tensor:
    n = paddle.arange(win_length, dtype="float32")
    return 0.5 - 0.5 * paddle.cos(2.0 * math.pi * n / win_length)


def _pad_centred(x: paddle.Tensor, pad: int, n_fft: int) -> paddle.Tensor:
    """Pad ``x`` with reflect-padding of length ``n_fft // 2`` on each
    side so the first frame is centred on sample 0."""
    if pad <= 0:
        return x
    # reflect_pad is not in paddle by name; emulate with F.pad mode "reflect".
    if x.ndim == 1:
        return F.pad(x.unsqueeze(0).unsqueeze(0), [pad, pad], mode="reflect").squeeze(0).squeeze(0)
    if x.ndim == 2:
        return F.pad(x.unsqueeze(1), [pad, pad], mode="reflect").squeeze(1)
    raise ValueError("Expected 1D or 2D input")


class STFT(paddle.nn.Layer):
    """
    Analogue:
        librosa STFT; Oppenheim & Schafer 'Discrete-Time Signal Processing'
    Forward STFT.

    Parameters
    ----------
    win_length : int
        Analysis window length.
    hop_length : int
        Hop between consecutive frames.
    n_fft : int, default ``win_length``
        FFT size; pads the window with zeros to this length.
    learnable_window : bool, default False
        If True, the window is a learnable parameter; otherwise a
        Hann window is used as a buffer.
    center : bool, default True
        Pad the input so the first and last frames are centred on the
        signal endpoints.
    return_complex : bool, default True
        If True, return complex ``[B, n_fft//2+1, T]``. If False,
        return a 2-channel tensor ``[B, 2, n_fft//2+1, T]`` so the
        output is real and the layer can be wired into a non-complex
        loss.
    """

    def __init__(
        self,
        win_length: int,
        hop_length: int,
        n_fft: Optional[int] = None,
        learnable_window: bool = False,
        center: bool = True,
        return_complex: bool = True,
    ) -> None:
        super().__init__()
        if win_length <= 0 or hop_length <= 0:
            raise ValueError("win_length and hop_length must be > 0")
        n_fft = n_fft or win_length
        if n_fft < win_length:
            raise ValueError("n_fft must be >= win_length")

        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.center = center
        self.return_complex = return_complex

        win = _hann_window(win_length)
        if learnable_window:
            self.window = paddle.create_parameter(
                shape=win.shape, dtype="float32",
                default_initializer=paddle.nn.initializer.Assign(win),
            )
        else:
            self.register_buffer("window", win)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2:
            raise ValueError(f"Expected 1D or 2D input, got shape {x.shape}")
        pad = self.n_fft // 2 if self.center else 0
        x = _pad_centred(x, pad, self.n_fft)
        # Frame: [B, n_frames, n_fft]
        frames = x.unfold(-1, self.n_fft, self.hop_length)
        # Window: broadcast over frames
        win = paddle.zeros([self.n_fft], dtype=x.dtype)
        win[: self.win_length] = self.window
        frames = frames * win
        spec = paddle.fft.rfft(frames, n=self.n_fft, axis=-1)
        spec = spec.transpose([0, 2, 1])                          # [B, F, T]
        if self.return_complex:
            return spec
        # stack real/imag as 2 channels
        real = spec.real().unsqueeze(1)
        imag = spec.imag().unsqueeze(1)
        return paddle.concat([real, imag], axis=1)

    def extra_repr(self) -> str:
        return (
            f"win_length={self.win_length}, hop_length={self.hop_length}, "
            f"n_fft={self.n_fft}, center={self.center}"
        )


class ISTFT(paddle.nn.Layer):
    """
    Analogue:
        librosa.istft; Griffin-Lim algorithm context
    Inverse STFT using overlap-add.

    Mirrors :class:`STFT` defaults: same ``win_length``, ``hop_length``,
    ``n_fft``, and ``center`` setting.
    """

    def __init__(
        self,
        win_length: int,
        hop_length: int,
        n_fft: Optional[int] = None,
        center: bool = True,
    ) -> None:
        super().__init__()
        if win_length <= 0 or hop_length <= 0:
            raise ValueError("win_length and hop_length must be > 0")
        n_fft = n_fft or win_length
        if n_fft < win_length:
            raise ValueError("n_fft must be >= win_length")

        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.center = center

        win = _hann_window(win_length)
        self.register_buffer("window", win)

    def forward(self, spec: paddle.Tensor) -> paddle.Tensor:
        """Reconstruct a real signal. ``spec`` is ``[B, F, T]`` complex
        or ``[B, 2, F, T]`` real (the latter from
        :class:`STFT` with ``return_complex=False``)."""
        if spec.ndim == 4 and spec.shape[1] == 2:
            real = spec[:, 0]
            imag = spec[:, 1]
            spec = paddle.complex(real, imag)
        if spec.ndim != 3:
            raise ValueError(f"Expected [B, F, T] complex spec, got {spec.shape}")
        B, F_, T = spec.shape
        spec = spec.transpose([0, 2, 1])                          # [B, T, F]
        frames = paddle.fft.irfft(spec, n=self.n_fft, axis=-1)    # [B, T, n_fft]
        win = paddle.zeros([self.n_fft], dtype=frames.dtype)
        win[: self.win_length] = self.window
        # Apply synthesis window = analysis window to make the
        # OLA constant-summation. For Hann with hop=win/4 the
        # per-sample sum-of-windows^2 is a constant C; we then
        # divide by C to recover the original sample.
        frames = frames * win
        n_samples = (T - 1) * self.hop_length + self.n_fft
        out = paddle.zeros([B, n_samples], dtype=frames.dtype)
        wsum = paddle.zeros([B, n_samples], dtype=frames.dtype)
        for t in range(T):
            start = t * self.hop_length
            out[:, start : start + self.n_fft] += frames[:, t]
            wsum[:, start : start + self.n_fft] += win ** 2
        out = out / paddle.maximum(wsum, paddle.full_like(wsum, 1e-8))
        if self.center:
            pad = self.n_fft // 2
            out = out[:, pad:-pad] if out.shape[1] > 2 * pad else out
        return out

    def extra_repr(self) -> str:
        return (
            f"win_length={self.win_length}, hop_length={self.hop_length}, "
            f"n_fft={self.n_fft}, center={self.center}"
        )
