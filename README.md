# PaddleScienceKits

一个基于飞桨的**科学/机器学习工具包**。

> 核心定位：把传统机器学习、信号处理、统计建模方法，**重写为 `paddle.nn.Layer`**，
> 从而可以像普通深度学习算子一样参与组网、梯度反传、GPU 加速。

This is **not** just a deep-learning toolkit. The whole point of
`PaddleScienceKits` is: classical models are also parameters + structure,
and structure can be expressed with `paddle.nn.Layer` so it composes with
neural networks.

## 子模块

| Submodule | Status | Contents |
| --- | --- | --- |
| `PaddleScienceKits.TimeSeries` | ✅ | `AR`, `ARMA`, `FIR`, 通用 `Autoregressive` |
| `PaddleScienceKits.ClassicalML` | ✅ | `KMeans`, `KNN`, `PCA`, `KernelRidge`, `GMM`, `LDA`, `GaussianNB`, `MultinomialNB`, `ICA`, `SoftDecisionTree` |

## 安装

```bash
git clone https://github.com/Liyulingyue/PaddleScienceKits.git
cd PaddleScienceKits
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 快速上手

### TimeSeries —— AR / ARMA / FIR

```python
import paddle
from PaddleScienceKits.TimeSeries import AR, ARMA, FIR, Autoregressive

# 通用 A(p) y = B(q) u + C(o) v
model = Autoregressive(y_features=3, x_features=[2, 4], e_features=2)
y = paddle.randn([8, 3])
u1, u2, v = paddle.randn([8, 2]), paddle.randn([8, 4]), paddle.randn([8, 2])
out = model(y, u1, u2, v)                      # [8, 1]

# AR(p)
ar = AR(p=5)
y_seq = paddle.randn([8, 5])
print(ar(y_seq).shape)                         # [8, 1]
```

### ClassicalML —— KMeans / KNN

```python
import paddle
from PaddleScienceKits.ClassicalML import KMeans, KNN

# KMeans: 质心是 nn.Parameter，软分配路径可微
km = KMeans(k=4, dim=3, temperature=0.5)
km.fit_kmeanspp(paddle.randn([200, 3]))         # k-means++ 种子
km.fit(paddle.randn([200, 3]), n_iter=10)        # Lloyd
soft = km(paddle.randn([8, 3]))                 # [8, 4]   软分配
hard = km(paddle.randn([8, 3]), hard=True)      # [8]      整数标签

# KNN: 记忆层
nn = KNN(k=3, dim=4)
nn.update_memory(paddle.randn([50, 4]))         # 注册样本
nn.update_memory(paddle.randn([50, 4]),
                 values=paddle.randn([50, 4]))  # 可选 value_bank
idx = nn(paddle.randn([5, 4]), mode="indices")  # [5, 3]
avg = nn(paddle.randn([5, 4]), mode="average")  # [5, 4]   邻居均值
```

### ClassicalML —— PCA / KernelRidge / GMM

```python
import paddle
from PaddleScienceKits.ClassicalML import PCA, KernelRidge, GMM

# PCA: 正交基是 nn.Parameter，每步 QR 重新正交化
pca = PCA(n_components=2, dim=5)
pca.fit(paddle.randn([200, 5]))
coords = pca.project(paddle.randn([8, 5]))     # [8, 2]
rec    = pca.reconstruct(coords)               # [8, 5]

# KernelRidge: 对偶形式 + 闭式解
kr = KernelRidge(n_support=200, dim_in=1, dim_out=1,
                 kernel="rbf", gamma=0.5, alpha=1e-2)
kr.fit(paddle.linspace(-3, 3, 200).unsqueeze(-1),
       paddle.sin(paddle.linspace(-3, 3, 200).unsqueeze(-1)))
pred = kr(paddle.linspace(-3, 3, 5).unsqueeze(-1))   # [5, 1]

# GMM: 软责任可微，EM 闭式更新
gmm = GMM(k=3, dim=2, covariance_type="diag")
gmm.fit_em(paddle.randn([300, 2]), n_iter=20)
resp = gmm(paddle.randn([8, 2]))                # [8, 3]   软责任
```

### ClassicalML —— LDA / NaiveBayes / ICA / SoftDecisionTree

```python
import paddle
from PaddleScienceKits.ClassicalML import (
    LDA, GaussianNB, MultinomialNB, ICA, SoftDecisionTree,
)

# LDA: 闭式特征分解；既能投影也能输出高斯判别 log p(y|x)
lda = LDA(n_components=2, dim=2, n_classes=3).fit(X, Y)
logits = lda(X)                    # [N, 3]   可微的高斯判别 log-prob

# GaussianNB / MultinomialNB: 闭式 MLE，可作分类头
gnb = GaussianNB(dim=2, n_classes=2).fit(X, Y)
mnb = MultinomialNB(n_features=6, n_classes=2, alpha=1.0).fit(counts, Y)

# ICA: PCA 白化 + 对称 FastICA 定点迭代
ica = ICA(n_components=2, dim=2, nonlinearity="cube").fit(X)
sources = ica.transform(X)          # [N, 2]
recon   = ica.inverse_transform(sources)

# SoftDecisionTree: 可微决策树（Frosst 2017）
tree = SoftDecisionTree(depth=4, n_features=2, n_classes=2, temperature=1.5)
# then optimise with Adam like any paddle layer
```

## 设计理念

* **可微**：`KMeans` 在训练模式下输出软分配（带 temperature 的 softmax），
  质心作为 `nn.Parameter` 接收梯度。`KNN.soft_retrieval` 也提供
  基于距离的 softmax 邻居权重。
* **可组合**：`KMeans` 输出一个 K 维软分配向量，可以直接喂给
  `nn.Linear`；`KNN.soft_retrieval` 的权重可与注意力分数等价使用。
* **可降级**：所有可微路径都对应一个不可微的"硬"等价物（`hard=True`、
  `mode="indices"` / `"average"` / `"values"`），保证推理阶段行为
  与传统算法一致。

完整示例见 `examples/`：

* `examples/timeseries_ar.py` —— AR(2) 端到端回归
* `examples/classicalml_kmeans_classifier.py` —— KMeans 软特征 + 线性分类器联合训练
* `examples/classicalml_gmm_classifier.py` —— GMM 软责任 + 线性分类器联合训练
* `examples/classicalml_soft_tree_moons.py` —— 软决策树在 moons 数据集上训练

## 旧项目说明

`PaddleAutoregressive` 已被废弃，所有内容并入本仓库的
`PaddleScienceKits.TimeSeries` 子模块。新公式更紧凑、批处理友好、参数量更少。

## 许可

Apache-2.0
