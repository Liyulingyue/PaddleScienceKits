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
| `PaddleScienceKits.ClassicalML` | 🚧 | 占位（计划承载 KMeans、KNN、PCA、…的可微/参数化版本） |

## 安装

```bash
git clone https://github.com/Liyulingyue/PaddleScienceKits.git
cd PaddleScienceKits
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 快速上手

```python
import paddle
from PaddleScienceKits.TimeSeries import AR, ARMA, FIR, Autoregressive

# 通用 A(p) y = B(q) u + C(o) v
model = Autoregressive(y_features=3, x_features=[2, 4], e_features=2)
y = paddle.randn([8, 3])      # [batch, y_features]
u1 = paddle.randn([8, 2])
u2 = paddle.randn([8, 4])
v = paddle.randn([8, 2])
out = model(y, u1, u2, v)     # [batch, 1]
print(out.shape)              # [8, 1]

# AR(p)
ar = AR(p=5)
y_seq = paddle.randn([8, 5])  # 5 步历史
print(ar(y_seq).shape)        # [8, 1]
```

## 旧项目说明

`PaddleAutoregressive` 已被废弃，所有内容并入本仓库的
`PaddleScienceKits.TimeSeries` 子模块。新公式更紧凑、批处理友好、参数量更少。

## 许可

Apache-2.0
