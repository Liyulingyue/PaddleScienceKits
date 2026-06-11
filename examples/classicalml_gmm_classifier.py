"""End-to-end demo: GMM produces K-dim soft responsibilities that feed
a linear classifier. The GMM parameters are *not* frozen, so they get
nudged toward class-separable positions during training, while still
behaving as a probabilistic mixture model.
"""

import paddle
from PaddleScienceKits.ClassicalML import GMM


def main():
    paddle.seed(0)
    # three Gaussian clusters, each assigned to its own class
    centers = paddle.to_tensor([[0.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
    X, Y = [], []
    for cls, c in enumerate(centers):
        pts = c + 0.3 * paddle.randn([80, 2])
        X.append(pts)
        Y.append(paddle.full([80, 1], cls, dtype="int64"))
    X = paddle.concat(X, axis=0)
    Y = paddle.concat(Y, axis=0)

    gmm = GMM(k=3, dim=2, covariance_type="diag", reg=1e-3)
    # warm-start means with the true centres, log-variances = log(0.3^2)
    gmm.means.set_value(centers.clone())
    gmm.log_vars.set_value(paddle.log(paddle.ones([3, 2]) * 0.09))
    classifier = paddle.nn.Linear(3, 3)

    opt = paddle.optimizer.Adam(
        parameters=list(gmm.parameters()) + list(classifier.parameters()),
        learning_rate=1e-2,
    )

    for epoch in range(200):
        feats = gmm(X)                # [240, 3] differentiable responsibilities
        logits = classifier(feats)
        loss = paddle.nn.functional.cross_entropy(logits, Y)
        opt.clear_grad()
        loss.backward()
        opt.step()
        if epoch % 50 == 0:
            acc = paddle.argmax(logits, axis=-1)
            print(f"epoch {epoch:3d}  loss={loss.item():.4f}  acc={(acc==Y.squeeze()).astype('float32').mean().item():.3f}")

    print("final means:\n", gmm.means.numpy())


if __name__ == "__main__":
    main()
