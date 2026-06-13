"""Two end-to-end demos for the new ClassicalML additions.

1. SparseCoding on a synthetic dictionary-sparse dataset.  We
   sample data x = z D, with z having exactly 3 active atoms
   per sample, and ask the model to recover a dictionary that
   explains the data sparsely.
2. BayesianLinearVI on a noisy 1-D sine.  After training we
   compare the predictive mean and std against frequentist
   BayesianRidge to show that the variational posterior also
   contracts in regions of dense data.
"""

import paddle
import numpy as np

from PaddleScienceKits.ClassicalML import SparseCoding, BayesianLinearVI


def sparse_coding_demo():
    paddle.seed(0)
    np.random.seed(0)
    n_atoms, n_features = 20, 16
    D_true = paddle.randn([n_atoms, n_features])
    D_true = D_true / paddle.norm(D_true, axis=1, keepdim=True)
    N = 300
    Z = paddle.zeros([N, n_atoms])
    for i in range(N):
        active = np.random.choice(n_atoms, 3, replace=False)
        Z[i, active] = paddle.rand([3]) * 2 - 1
    X = Z @ D_true + 0.01 * paddle.randn([N, n_features])

    sc = SparseCoding(n_atoms=n_atoms, n_features=n_features,
                      lmbda=0.05, encoder="fista", n_iter=200)
    sc.fit(X, n_outer=200, lr=5e-2)
    z, x_hat = sc(X)
    rel_err = float(paddle.mean(paddle.sum((X - x_hat) ** 2, axis=1)) /
                    paddle.mean(paddle.sum(X ** 2, axis=1)))
    active_frac = float((paddle.abs(z) > 0.1).astype("float32").mean().numpy().item())
    print(f"SparseCoding on dictionary-sparse data:")
    print(f"  rel reconstruction err = {rel_err:.3f}")
    print(f"  active code fraction    = {active_frac:.3f}")
    print(f"  (true: 3/20 = 0.150)")


def bayesian_vi_demo():
    paddle.seed(0)
    n, d = 60, 3
    x = paddle.linspace(-3.0, 3.0, n).unsqueeze(-1)
    x = paddle.concat([x, x ** 2, paddle.sin(x)], axis=1)
    w_true = paddle.to_tensor([[1.0], [-0.3], [0.7]])
    y = x @ w_true + 0.1 * paddle.randn([n, 1])

    blr = BayesianLinearVI(n_features=d, n_outputs=1, prior_std=1.0, noise_std=0.2)
    opt = paddle.optimizer.Adam(parameters=blr.parameters(), learning_rate=5e-2)
    for step in range(200):
        loss = blr.neg_elbo(x, y, n_samples=1)
        opt.clear_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"  VI step {step:3d}  neg-ELBO = {float(loss.numpy().item()):.3f}")

    x_test = paddle.linspace(-6.0, 6.0, 25).unsqueeze(-1)
    x_test = paddle.concat([x_test, x_test ** 2, paddle.sin(x_test)], axis=1)
    mean, std = blr.predict(x_test, n_samples=80, return_std=True)
    print("BayesianLinearVI on 1-D noisy sine (with cubic-ish features):")
    print("  x_test   mean      std")
    for xi, m, s in zip(x_test[:, 0].numpy(), mean.squeeze().numpy(), std.squeeze().numpy()):
        print(f"  {xi:+5.2f}    {m:+5.2f}    {s:5.2f}")


if __name__ == "__main__":
    print("=== SparseCoding demo ===")
    sparse_coding_demo()
    print()
    print("=== BayesianLinearVI demo ===")
    bayesian_vi_demo()
