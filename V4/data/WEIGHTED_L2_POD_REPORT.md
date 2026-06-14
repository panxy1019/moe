# Weighted L2 POD Report

日期：2026-06-14

## 目的

为了避免圆柱壁面附近高网格密度区域在普通欧氏内积 SVD 中被点数过度放大，本次重新执行了带体积权重的全局 POD。新的 POD 不再直接对原始点值快照做普通 SVD，而是在空间维度乘以网格控制体积的平方根。

对于当前数据集，原始流场保存为 VTK `pointData`：

```text
u[t, i], v[t, i], p[t, i]
```

其中 `i` 是网格点索引。因此本次构造的是长度为 `n_points = 97368` 的点控制体积权重向量 `V_i`。

## 权重提取

权重来源于同一套 `blockMesh` 网格：

```text
/home/ray/base_case_cycles/system/blockMeshDict
```

执行流程：

```text
1. 复制 base_case_cycles 到临时目录。
2. 运行 blockMesh 重新生成网格。
3. 运行 foamToVTK -constant -noZero -noFaceZones 导出纯网格 VTK。
4. 使用 VTK 读取单元连通性和每个单元体积。
5. 将每个单元体积平均分摊到该单元的角点，得到点控制体积 V_i。
```

数学形式：

```text
V_i = sum_{cell c contains point i} Volume(c) / N_points(c)
```

本次网格检查结果：

```text
n_cells = 48128
n_points = 97368
sum(cell_volumes)  = 59.92146953
sum(point_volumes) = 59.92147006
min(point_volume)  = 1.0424e-06
max(point_volume)  = 1.3195e-03
mesh point max delta vs dataset points = 0.0
```

说明：当前数据是 `pointData`，所以这里实现的是与当前点值数据严格一致的 lumped nodal L2 内积。如果将来改为保存 OpenFOAM 原生 cell-centered 场，则可以直接使用 cell volume 对 cell field 做有限体积 L2 内积。

## 加权快照组装

每个 Re 仍沿用之前标准：

```text
Re = 500 到 1000，共 30 个等距点
每个 Re 原始帧数 = 241
丢弃前 2 个估计脱落周期
每个 Re 保留帧数 = 201
总快照数 = 6030
```

速度快照采用 `[u, v]` 拼接：

```text
x_raw = [u_1, ..., u_N, v_1, ..., v_N]
W_uv = diag(V_1, ..., V_N, V_1, ..., V_N)
x_weighted = x_raw * sqrt(W_uv)
```

压力快照：

```text
y_raw = [p_1, ..., p_N]
W_p = diag(V_1, ..., V_N)
y_weighted = y_raw * sqrt(W_p)
```

因此任意两个速度快照满足：

```text
x_weighted(a) dot x_weighted(b)
= x_raw(a)^T W_uv x_raw(b)
= sum_i V_i * (u_a,i u_b,i + v_a,i v_b,i)
```

压力满足：

```text
y_weighted(a) dot y_weighted(b)
= y_raw(a)^T W_p y_raw(b)
= sum_i V_i * p_a,i p_b,i
```

这就是当前点值数据上的离散 L2 能量内积。

## 输出文件

Weighted POD 输出目录：

```text
/home/ray/Desktop/Cylinder_Results_Re500_1000_30Re_POD/Global_POD_Weighted_L2
```

主要文件：

```text
mesh_l2_point_weights.npz
global_velocity_pod_weighted_l2.npz
global_pressure_pod_weighted_l2.npz
pod_snapshot_index.csv
pod_weighted_l2_metadata.json
README_weighted_l2.txt
```

权重文件：

```text
mesh_l2_point_weights.npz
  points                shape = (97368, 3)
  cell_volumes          shape = (48128,)
  point_volumes         shape = (97368,)
  sqrt_point_volumes    shape = (97368,)
```

速度 POD：

```text
global_velocity_pod_weighted_l2.npz
  phi_uv                shape = (80, 194736)
  phi_uv_weighted       shape = (80, 194736)
  coeff_uv              shape = (6030, 80)
  mean_uv_by_Re         shape = (30, 194736)
  cumulative_energy_uv  shape = available in file
```

压力 POD：

```text
global_pressure_pod_weighted_l2.npz
  phi_p                 shape = (80, 97368)
  phi_p_weighted        shape = (80, 97368)
  coeff_p               shape = (6030, 80)
  mean_p_by_Re          shape = (30, 97368)
  cumulative_energy_p   shape = available in file
```

## POD 结果

保留最大阶数仍设为 `80`。

```text
Velocity weighted POD:
  retained rank = 80
  cumulative weighted energy at 80 modes = 0.9922074313

Pressure weighted POD:
  retained rank = 80
  cumulative weighted energy at 80 modes = 0.9967036458
```

对比之前的 unweighted POD，weighted 后 80 阶累计能量下降是合理现象：原先高密度小单元区域按点数被放大，现在按实际控制体积计权后，能量分布更接近物理 L2 意义。

## 正交性核验

保存了两套模态：

```text
phi_*_weighted : 加权坐标中的模态
phi_*          : 原始物理空间中的模态
```

原始空间模态通过下式恢复：

```text
phi_raw = phi_weighted / sqrt(V)
```

因此原始空间模态满足质量矩阵正交：

```text
phi_raw M phi_raw^T = I
```

数值核验：

```text
velocity weighted-coordinate orthonormal max_abs_err = 2.62e-06
pressure weighted-coordinate orthonormal max_abs_err = 9.36e-07

velocity raw M-orthonormal max_abs_err = 2.65e-06
pressure raw M-orthonormal max_abs_err = 9.40e-07
```

误差处于 float32 存储和压缩输出可接受范围内。

## 重构方式

速度重构：

```python
import numpy as np

pod = np.load("global_velocity_pod_weighted_l2.npz")

phi_uv = pod["phi_uv"]
coeff_uv = pod["coeff_uv"]
mean_uv_by_Re = pod["mean_uv_by_Re"]
n_points = int(pod["n_points"])

snapshot_id = 0
re_index = 0
r = 80

uv = mean_uv_by_Re[re_index] + coeff_uv[snapshot_id, :r] @ phi_uv[:r]
u = uv[:n_points]
v = uv[n_points:]
```

压力重构：

```python
pod = np.load("global_pressure_pod_weighted_l2.npz")

phi_p = pod["phi_p"]
coeff_p = pod["coeff_p"]
mean_p_by_Re = pod["mean_p_by_Re"]

snapshot_id = 0
re_index = 0
r = 80

p = mean_p_by_Re[re_index] + coeff_p[snapshot_id, :r] @ phi_p[:r]
```

如果要计算物理 L2 误差，应使用同一个权重：

```python
w = np.load("mesh_l2_point_weights.npz")["point_volumes"]

rel_u = np.sqrt(np.sum(w * (u_true - u_pred)**2) / np.sum(w * u_true**2))
rel_p = np.sqrt(np.sum(w * (p_true - p_pred)**2) / np.sum(w * p_true**2))
```

速度双分量误差应写成：

```python
num = np.sum(w * ((u_true - u_pred)**2 + (v_true - v_pred)**2))
den = np.sum(w * (u_true**2 + v_true**2))
rel_uv = np.sqrt(num / den)
```

## 脚本

执行脚本：

```text
/home/ray/compute_global_pod_weighted_l2.py
```

本次脚本已经执行完成，临时矩阵目录 `_tmp` 已自动删除。
