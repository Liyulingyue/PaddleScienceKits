"""End-to-end demo: classify 1-D audio-like signals by their
log-Mel-spectrogram followed by a linear head.

The two "classes" are pure sine waves of different frequencies; the
log-Mel features make the frequency discrimination trivially linear
in the mel basis. The MelSpectrogram's STFT window is left as a
Hann buffer (non-learnable) so the run is fast.
"""

import paddle
import numpy as np
from PaddleScienceKits.SignalProcessing import MelSpectrogram


def synthesize(freqs, sr, duration):
    t = paddle.arange(int(sr * duration), dtype="float32") / sr
    sigs = [paddle.sin(2 * np.pi * f * t) for f in freqs]
    return paddle.stack(sigs, axis=0)  # [B, samples]


def main():
    paddle.seed(0)
    sr = 16000
    duration = 0.5
    freqs_low = [200.0, 250.0, 300.0]
    freqs_high = [800.0, 1000.0, 1200.0]
    X = paddle.concat([synthesize(freqs_low, sr, duration),
                       synthesize(freqs_high, sr, duration)], axis=0)
    Y = paddle.concat([paddle.zeros([3], dtype="int64"),
                       paddle.ones([3], dtype="int64")])

    mel = MelSpectrogram(n_mels=32, sample_rate=sr,
                         win_length=400, hop_length=160, n_fft=512)
    feats = mel(X)                                  # [B, n_mels, T]
    feats_flat = feats.reshape([feats.shape[0], -1]) # [B, n_mels * T]

    head = paddle.nn.Linear(feats_flat.shape[1], 2)
    opt = paddle.optimizer.Adam(parameters=head.parameters(), learning_rate=1e-2)
    for epoch in range(200):
        feats = mel(X)
        feats_flat = feats.reshape([feats.shape[0], -1])
        logits = head(feats_flat)
        loss = paddle.nn.functional.cross_entropy(logits, Y)
        opt.clear_grad()
        loss.backward()
        opt.step()
        if epoch % 50 == 0:
            acc = float(paddle.mean((paddle.argmax(logits, -1) == Y).astype("float32")))
            print(f"epoch {epoch:3d}  loss={loss.item():.4f}  acc={acc:.3f}")
    final_acc = float(paddle.mean(
        (paddle.argmax(head(mel(X).reshape([X.shape[0], -1])), -1) == Y).astype("float32")
    ))
    print(f"final acc = {final_acc:.3f}")


if __name__ == "__main__":
    main()
