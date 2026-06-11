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
| `PaddleScienceKits.ClassicalML` | ✅ | `KMeans`（质心可学，可微软分配）、`KNN`（记忆层） |

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

## 旧项目说明

`PaddleAutoregressive` 已被废弃，所有内容并入本仓库的
`PaddleScienceKits.TimeSeries` 子模块。新公式更紧凑、批处理友好、参数量更少。

## 许可

Apache-2.0
