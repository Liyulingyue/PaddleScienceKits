"""Two demos for the new ClassicalML additions.

1. tICA on a synthetic 2-D diffusion with one slow and one fast
   direction. We project the trajectory onto the recovered slow
   mode and inspect the autocorrelation at the requested lag.

2. NMF on a synthetic 3-topic document-term matrix. Each
   "document" is a random mixture of the 3 topics; we verify that
   the recovered dictionary is close to the true topics up to
   permutation and report the reconstruction error.
"""

import paddle
import numpy as np
from PaddleScienceKits.ClassicalML import tICA, NMF


def tica_demo():
    paddle.seed(0)
    T = 4000
    slow = paddle.cumsum(0.05 * paddle.randn([T]))
    fast = paddle.cumsum(1.0 * paddle.randn([T]))
    theta = 0.5
    R = paddle.to_tensor(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    X = paddle.stack([slow, fast], axis=1) @ R.T
    tica = tICA(n_components=1, dim=2, lag=20)
    tica.fit(X)
    proj = tica.transform(X).squeeze().numpy()
    # Compute autocorrelation of the projected slow mode at lag=20.
    proj_d = paddle.to_tensor(proj)
    a = proj_d[:-20]
    b = proj_d[20:]
    corr = float((a - a.mean()) @ (b - b.mean()) /
                 (a.shape[0] * a.std() * b.std()))
    print(f"tICA slow-mode autocorrelation at lag=20: {corr:.3f}")
    # Compare to the *fast* mode (any vector orthogonal to slow).
    fast_axis = R @ paddle.to_tensor([0.0, 1.0])
    other = X @ fast_axis.unsqueeze(-1)
    other = other.squeeze().numpy()
    other_d = paddle.to_tensor(other)
    a2 = other_d[:-20]
    b2 = other_d[20:]
    corr_fast = float((a2 - a2.mean()) @ (b2 - b2.mean()) /
                      (a2.shape[0] * a2.std() * b2.std()))
    print(f"fast mode autocorrelation at lag=20  : {corr_fast:.3f}")


def nmf_demo():
    paddle.seed(0)
    np.random.seed(0)
    n_features = 12
    topic_a = paddle.to_tensor([3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    topic_b = paddle.to_tensor([0.0, 0.0, 0.0, 2.0, 3.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    topic_c = paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
    W_true = paddle.stack([topic_a, topic_b, topic_c], axis=0)
    N = 200
    H_true = paddle.abs(paddle.randn([N, 3])) + 0.1
    X = H_true @ W_true + 0.01 * paddle.abs(paddle.randn([N, n_features]))

    nmf = NMF(n_components=3, n_features=n_features, init="nndsvd")
    nmf.fit(X, n_iter=500)
    rec = nmf.reconstruct(nmf.H)
    rel_err = float(paddle.mean(paddle.sum((X - rec) ** 2, axis=1)) /
                    paddle.mean(paddle.sum(X ** 2, axis=1)))
    # Per-topic cosine (best over permutations)
    W = nmf.W.numpy()
    best = []
    for i in range(3):
        c = []
        for j in range(3):
            cos = float(W[i] @ W_true[j].numpy()) / (
                np.linalg.norm(W[i]) * np.linalg.norm(W_true[j].numpy())
            )
            c.append(abs(cos))
        best.append(max(c))
    print(f"NMF 3-topic rel reconstruction err: {rel_err:.4f}")
    print(f"NMF per-topic best cosines       : {[round(c, 3) for c in best]}")


if __name__ == "__main__":
    print("=== tICA demo ===")
    tica_demo()
    print()
    print("=== NMF demo ===")
    nmf_demo()
