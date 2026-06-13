"""Two end-to-end demos for the new ClassicalML additions.

1. KalmanFilter 2-D tracking: simulate a constant-velocity target
   observed through a noisy linear measurement, then run the
   smoother and print the recovered trajectory.
2. LinearChainCRF toy "POS" tagging: each token gets a small random
   feature vector; tag sequences follow a simple "color -> color"
   bias. Train the CRF for a few hundred steps and report the Viterbi
   decode accuracy on held-out data.
3. tSNE on a 3-cluster Gaussian mixture in 5-D; visualise the
   resulting 2-D embedding by reporting the cluster centroid
   distances.
"""

import paddle
import numpy as np
from PaddleScienceKits.ClassicalML import KalmanFilter, LinearChainCRF, tSNE


def kalman_demo():
    paddle.seed(0)
    T = 80
    A = paddle.to_tensor([[1.0, 0.1], [0.0, 1.0]])
    C = paddle.to_tensor([[1.0, 0.0], [0.0, 1.0]])
    x_true = paddle.zeros([T, 2])
    for t in range(1, T):
        x_true[t] = A @ x_true[t - 1] + 0.1 * paddle.randn([2])
    y = x_true @ C.T + 0.05 * paddle.randn([T, 2])

    kf = KalmanFilter(state_dim=2, obs_dim=2)
    kf.A.set_value(paddle.eye(2))
    kf.C.set_value(paddle.eye(2))
    pred = kf(y)
    err = float(paddle.mean((pred - x_true) ** 2))
    print(f"Kalman 2-D tracking MSE = {err:.4f}")
    print("first 5 smoother outputs (vs truth):")
    for t in range(5):
        print(f"  t={t}  pred={pred[t].numpy()}  truth={x_true[t].numpy()}")


def crf_demo():
    paddle.seed(0)
    np.random.seed(0)
    n_tags = 3
    crf = LinearChainCRF(n_features=4, n_tags=n_tags)
    opt = paddle.optimizer.Adam(parameters=crf.parameters(), learning_rate=5e-2)

    def make_example():
        T = 8
        # Tags follow: prefer staying in the same tag (85%).
        tags = [int(np.random.randint(0, n_tags))]
        for _ in range(T - 1):
            if np.random.rand() < 0.85:
                tags.append(tags[-1])
            else:
                tags.append(int(np.random.randint(0, n_tags)))
        # Make the emission features correlated with the tag: each
        # tag has a different mean feature vector so the emitter
        # has signal to lock onto.
        tag_means = paddle.to_tensor(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0]], dtype="float32"
        )
        feats = tag_means[tags] + 0.1 * paddle.randn([T, 4])
        return feats, paddle.to_tensor(tags, dtype="int64")

    for step in range(400):
        feats, tags = make_example()
        loss = crf.nll(feats, tags)
        opt.clear_grad()
        loss.backward()
        opt.step()
    n_correct, n_total = 0, 0
    for _ in range(50):
        feats, tags = make_example()
        pred = crf.decode(feats)
        n_correct += int((pred == tags).sum())
        n_total += int(tags.shape[0])
    print(f"CRF Viterbi accuracy = {n_correct / n_total:.3f}")


def tsne_demo():
    paddle.seed(0)
    np.random.seed(0)
    parts = []
    for c in paddle.to_tensor([[0.0, 0.0, 0.0, 0.0, 0.0],
                               [5.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 5.0, 0.0, 0.0]]):
        parts.append(0.1 * paddle.randn([25, 5]) + c)
    X = paddle.concat(parts, axis=0)
    tsne = tSNE(n_components=2, perplexity=15.0, n_iter=120, learning_rate=5.0)
    Y = tsne.fit_transform(X).numpy()
    cent = [Y[i * 25:(i + 1) * 25].mean(axis=0) for i in range(3)]
    print("tSNE 2-D cluster centroids:")
    for i, c in enumerate(cent):
        print(f"  cluster {i}: {c}")
    # Distances
    d01 = float(np.linalg.norm(np.array(cent[0]) - np.array(cent[1])))
    d02 = float(np.linalg.norm(np.array(cent[0]) - np.array(cent[2])))
    d12 = float(np.linalg.norm(np.array(cent[1]) - np.array(cent[2])))
    print(f"  pairwise centroid distances: {d01:.2f} {d02:.2f} {d12:.2f}")


if __name__ == "__main__":
    print("=== KalmanFilter demo ===")
    kalman_demo()
    print()
    print("=== LinearChainCRF demo ===")
    crf_demo()
    print()
    print("=== tSNE demo ===")
    tsne_demo()
