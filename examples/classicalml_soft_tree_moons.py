"""End-to-end demo: a SoftDecisionTree of depth 4 trained on a
2-D "moons"-like dataset with noisy class boundaries, evaluated by
held-out accuracy. The tree uses ~2*4-1 = 7 inner nodes and 16
leaves; with soft routing the gradient signal can flow back into
all 14 parameters (W, b, leaf_logits) simultaneously.
"""

import paddle
import numpy as np
from PaddleScienceKits.ClassicalML import SoftDecisionTree


def make_moons(n: int = 600, noise: float = 0.15, seed: int = 0):
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, np.pi, n // 2)
    x0 = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    x1 = np.stack([1.0 - np.cos(theta), 0.5 - np.sin(theta)], axis=1)
    X = np.concatenate([x0, x1], axis=0)
    X += noise * rng.standard_normal(X.shape)
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype("int64")
    return paddle.to_tensor(X.astype("float32")), paddle.to_tensor(y)


def main():
    paddle.seed(0)
    X, Y = make_moons()
    tree = SoftDecisionTree(depth=4, n_features=2, n_classes=2, temperature=1.5)
    opt = paddle.optimizer.Adam(parameters=tree.parameters(), learning_rate=5e-2)
    for epoch in range(400):
        logits = tree(X)
        loss = paddle.nn.functional.cross_entropy(logits, Y)
        opt.clear_grad()
        loss.backward()
        opt.step()
        if epoch % 50 == 0:
            acc = float(paddle.mean((tree.predict(X) == Y).astype("float32")))
            print(f"epoch {epoch:3d}  loss={loss.item():.4f}  acc={acc:.3f}")
    final_acc = float(paddle.mean((tree.predict(X) == Y).astype("float32")))
    print(f"final train acc = {final_acc:.3f}")


if __name__ == "__main__":
    main()
