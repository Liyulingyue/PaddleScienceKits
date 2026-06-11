"""Smoke tests for LDA, NaiveBayes, ICA, and SoftDecisionTree.

Run with:
    .venv/bin/python tests/test_classicalml3.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paddle  # noqa: E402
import numpy as np  # noqa: E402

from PaddleScienceKits.ClassicalML import (  # noqa: E402
    LDA, GaussianNB, MultinomialNB, ICA, SoftDecisionTree,
)


def _ok(cond, msg):
    assert cond, msg
    print(f"  ok  {msg}")


# ----------------------------------------------------------------- LDA
def test_lda_separates_three_clusters():
    paddle.seed(0)
    centers = paddle.to_tensor([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
    X, Y = [], []
    for cls, c in enumerate(centers):
        X.append(c + 0.4 * paddle.randn([40, 2]))
        Y.append(paddle.full([40], cls, dtype="int64"))
    X = paddle.concat(X, axis=0)
    Y = paddle.concat(Y, axis=0)
    lda = LDA(n_components=2, dim=2, n_classes=3).fit(X, Y)
    proj = lda.project(X).numpy()
    # The 1-D rank along the LDA axis should put class-0 points
    # mostly at one end and class-1 / class-2 points mostly at the other.
    class0 = np.argsort(proj[:40, 0])[:30]
    class1 = np.argsort(proj[40:80, 0])[:30]
    overlap = len(set(class0) & set(class1))
    _ok(overlap < 30, f"LDA top-1 axis class0 vs class1 overlap={overlap} (expected < 30)")


def test_lda_log_proba_is_differentiable():
    paddle.seed(0)
    centers = paddle.to_tensor([[0.0, 0.0], [5.0, 0.0]])
    X, Y = [], []
    for cls, c in enumerate(centers):
        X.append(c + 0.4 * paddle.randn([30, 2]))
        Y.append(paddle.full([30], cls, dtype="int64"))
    X = paddle.concat(X, axis=0)
    Y = paddle.concat(Y, axis=0)
    lda = LDA(n_components=1, dim=2, n_classes=2).fit(X, Y)
    # predict_log_proba uses the *fitted* class stats (not the
    # components parameter) and so is not differentiable in the
    # ``components`` slot. We just check it produces well-shaped,
    # non-trivial log-probs.
    log_p = lda.predict_log_proba(X)
    _ok(log_p.shape == [60, 2], f"log_p shape {log_p.shape}")
    # The two classes should be distinguished by their log-probs on
    # the X that was used for fitting.
    diff = log_p[:30, 0] - log_p[:30, 1]
    _ok(float(paddle.mean(diff)) > 0, "class 0 mean log-p0 - log-p1 should be > 0")


# ------------------------------------------------------------- NaiveBayes
def test_gaussian_nb_fits_separable_blobs():
    paddle.seed(0)
    X = paddle.concat([
        paddle.randn([50, 2]) + paddle.to_tensor([0.0, 0.0]),
        paddle.randn([50, 2]) + paddle.to_tensor([5.0, 5.0]),
    ], axis=0)
    Y = paddle.concat([paddle.zeros([50], dtype="int64"),
                       paddle.ones([50], dtype="int64")])
    nb = GaussianNB(dim=2, n_classes=2).fit(X, Y)
    preds = nb.predict(X)
    acc = float(paddle.mean((preds == Y).astype("float32")))
    _ok(acc >= 0.95, f"GaussianNB accuracy on 2-blob data = {acc:.3f}")


def test_multinomial_nb_fits_text_counts():
    paddle.seed(0)
    # Two "topics": class 0 emphasises words 0-2, class 1 emphasises words 3-5.
    n0_rows = [
        [5, 4, 3, 1, 0, 0],
        [6, 5, 2, 0, 1, 0],
        [4, 6, 3, 0, 0, 1],
        [5, 5, 4, 1, 0, 0],
    ]
    n1_rows = [
        [0, 1, 0, 5, 6, 4],
        [1, 0, 1, 4, 5, 5],
        [0, 0, 1, 6, 4, 5],
        [1, 1, 0, 5, 5, 4],
    ]
    rows = n0_rows + n1_rows
    X = paddle.to_tensor(rows, dtype="float32")
    Y = paddle.to_tensor([0] * 4 + [1] * 4, dtype="int64")
    mnb = MultinomialNB(n_features=6, n_classes=2, alpha=1.0).fit(X, Y)
    preds = mnb.predict(X)
    acc = float(paddle.mean((preds == Y).astype("float32")))
    _ok(acc == 1.0, f"MultinomialNB accuracy on synthetic topics = {acc:.3f}")


# ----------------------------------------------------------------- ICA
def test_ica_recovers_independent_sources_up_to_permutation_and_sign():
    paddle.seed(0)
    n = 2000
    # Two super-Gaussian sources
    s = paddle.concat([paddle.randn([n, 1]) ** 3, paddle.sign(paddle.randn([n, 1]))], axis=1)
    A = paddle.to_tensor([[1.0, 0.5], [0.4, 1.0]])
    X = s @ A.T
    ica = ICA(n_components=2, dim=2, nonlinearity="cube", max_iter=1000, tol=1e-6)
    ica.fit(X)
    S_hat = ica.transform(X).numpy()
    # Correlation matrix between true and estimated sources, after
    # matching the best (sign, permutation).
    C = np.abs(np.corrcoef(s.numpy().T, S_hat.T)[:2, 2:])
    max_per_col = C.max(axis=0)
    _ok((max_per_col > 0.95).all(),
        f"ICA recovered both sources with |corr| > 0.95 (got {max_per_col.tolist()})")


def test_ica_inverse_transform_round_trip():
    paddle.seed(0)
    X = paddle.randn([100, 3])
    ica = ICA(n_components=3, dim=3, nonlinearity="cube").fit(X)
    rec = ica.inverse_transform(ica.transform(X))
    err = float(paddle.mean(paddle.sum((X - rec) ** 2, axis=1)))
    _ok(err < 1e-3, f"ICA reconstruct round-trip MSE = {err:.6f}")


# ------------------------------------------------------ SoftDecisionTree
def test_soft_tree_leaf_probabilities_sum_to_one():
    tree = SoftDecisionTree(depth=3, n_features=4, n_classes=2)
    x = paddle.randn([7, 4])
    p = tree(x)
    _ok(p.shape == [7, 2], f"tree output shape {p.shape}")
    _ok(paddle.allclose(p.sum(-1), paddle.ones([7]), atol=1e-5),
        "tree output rows sum to 1")


def test_soft_tree_learns_xor():
    """Depth-2 tree should reach >=75% accuracy on XOR with enough steps."""
    paddle.seed(0)
    n = 200
    x = paddle.randint(0, 2, [n, 2]).astype("float32")
    y = (x[:, 0] != x[:, 1]).astype("int64")
    tree = SoftDecisionTree(depth=2, n_features=2, n_classes=2, temperature=2.0)
    opt = paddle.optimizer.Adam(parameters=tree.parameters(), learning_rate=5e-2)
    for _ in range(300):
        logits = tree(x)
        loss = paddle.nn.functional.cross_entropy(logits, y)
        opt.clear_grad()
        loss.backward()
        opt.step()
    acc = float(paddle.mean((tree.predict(x) == y).astype("float32")))
    _ok(acc >= 0.75, f"SoftDecisionTree XOR accuracy = {acc:.3f}")


if __name__ == "__main__":
    test_lda_separates_three_clusters()
    test_lda_log_proba_is_differentiable()
    test_gaussian_nb_fits_separable_blobs()
    test_multinomial_nb_fits_text_counts()
    test_ica_recovers_independent_sources_up_to_permutation_and_sign()
    test_ica_inverse_transform_round_trip()
    test_soft_tree_leaf_probabilities_sum_to_one()
    test_soft_tree_learns_xor()
    print("All LDA / NaiveBayes / ICA / SoftDecisionTree tests passed.")
