"""End-to-end demo: quantize a 2-D embedding with KMeans and feed the
soft cluster-assignment vector into a linear classifier. Both pieces
train jointly with backprop — a tiny example of "classical model +
deep model in one autograd graph".
"""

import paddle
from PaddleScienceKits.ClassicalML import KMeans


def main():
    paddle.seed(0)
    centers = paddle.to_tensor([[0.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
    X, Y = [], []
    for cls, c in enumerate(centers):
        pts = c + 0.2 * paddle.randn([80, 2])
        X.append(pts)
        Y.append(paddle.full([80, 1], cls, dtype="int64"))
    X = paddle.concat(X, axis=0)
    Y = paddle.concat(Y, axis=0)

    km = KMeans(k=3, dim=2, temperature=0.5, init="kmeans++")
    km.fit(X, n_iter=15)                       # warm-start centroids
    classifier = paddle.nn.Linear(3, 3)        # 3 cluster-soft features -> 3 classes

    opt = paddle.optimizer.Adam(
        parameters=list(km.parameters()) + list(classifier.parameters()),
        learning_rate=1e-2,
    )

    for epoch in range(200):
        km.train()
        feats = km(X, hard=False)              # [240, 3] differentiable
        logits = classifier(feats)
        loss = paddle.nn.functional.cross_entropy(logits, Y)
        opt.clear_grad()
        loss.backward()
        opt.step()
        if epoch % 50 == 0:
            acc = paddle.argmax(logits, axis=-1)
            print(f"epoch {epoch:3d}  loss={loss.item():.4f}  acc={(acc==Y.squeeze()).astype('float32').mean().item():.3f}")

    print("final centroids:\n", km.centroids.numpy())


if __name__ == "__main__":
    main()
