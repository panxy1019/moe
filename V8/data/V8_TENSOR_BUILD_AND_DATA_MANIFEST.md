# V8: Re=50-300 100Re POD-Galerkin Tensor Build Manifest

生成日期：2026-06-18  
数据目录：`/home/ray/Desktop/Cylinder_Results_Re50_300_100Re_POD`

## 数据集

- Reynolds 数范围：`Re = 50-300`
- Reynolds 样本数：`100`
- POD 目录：`Global_POD_Weighted_L2`
- 网格点数：`97368`
- 单元数：`48128`
- 速度 POD 模态：`r_u = 80`
- 压力 POD 模态：`r_p = 80`
- 运动粘度：`nu = 1e-3`
- 质量权重：`Global_POD_Weighted_L2/mesh_l2_point_weights.npz`
- PyVista/VTK 导数模板：`/home/ray/Desktop/Cylinder_Results_Re500_1000_VTU/Re_1000_VTU/flow_60.vtu`

VTU 模板与 V8 POD 点坐标逐点一致，最大坐标差为 `0.0`。因此本次继续使用该同网格 VTU 文件提供拓扑和 `compute_derivative()` 的非结构网格导数。

## 速度场半侵入式 Galerkin 张量

脚本：

```text
compute_weighted_l2_semi_intrusive_galerkin_tensors.py
```

正式命令：

```bash
python3 compute_weighted_l2_semi_intrusive_galerkin_tensors.py \
  --all-re --ru 0 --rp 0 --chunk-size 2048 --skip-raw \
  --output semi_intrusive_galerkin_tensors_allRe100_weightedL2_ru80_rp80_compact.npz \
  --report SEMI_INTRUSIVE_GALERKIN_TENSORS_allRe100_weightedL2_ru80_rp80.md
```

输出：

- `semi_intrusive_galerkin_tensors_allRe100_weightedL2_ru80_rp80_compact.npz`
  - 完整兼容旧键名布局。
  - 对每个 Re 保存 `G_u, c, A, H, P`。
  - 文件较大，因为 `H/P/G_u` 在 100 个 Re 下重复保存。
- `semi_intrusive_galerkin_tensors_allRe100_weightedL2_ru80_rp80_slim.npz`
  - GitHub 友好的等价布局。
  - 共享保存 `G_u, H, P`。
  - 每个 Re 保存 `c_all[q]` 和 `A_all[q]`。
  - 对标签 `Re_labels_computed[q]`，最终方程使用：

```text
da/dt = c_all[q] + A_all[q] a + H(a,a) + P b
```

速度张量校验：

- `n_Re = 100`
- `c_all.shape = (100, 80)`
- `A_all.shape = (100, 80, 80)`
- `H.shape = (80, 80, 80)`
- `P.shape = (80, 80)`
- `G_u.shape = (80, 80)`
- 从完整布局检查到 `H/P/G_u` 在所有 Re 上完全共享：
  - `max_shared_H_delta = 0.0`
  - `max_shared_P_delta = 0.0`
  - `max_shared_G_delta = 0.0`

代表性范数：

| Re label | `||c||_2` | `||A||_F` | `||H||_F` | `||P||_F` |
|---|---:|---:|---:|---:|
| `Re_50p000000` | `6.261138e-05` | `5.888564e-01` | `2.427398e+01` | `2.374923e-01` |
| `Re_105p983150` | `3.740223e-04` | `1.264243e+00` | `2.427398e+01` | `2.374923e-01` |
| `Re_300p000000` | `6.069518e-03` | `4.072739e+00` | `2.427398e+01` | `2.374923e-01` |

## 压力 Poisson 代数代理张量

脚本：

```text
compute_pressure_poisson_surrogate_tensors.py
```

正式命令：

```bash
python3 compute_pressure_poisson_surrogate_tensors.py \
  --all-re --ru 0 --rp 0 --chunk-size 2048 \
  --output pressure_poisson_surrogate_tensors_allRe100_weightedL2_ru80_rp80.npz \
  --report PRESSURE_POISSON_SURROGATE_TENSORS_allRe100_weightedL2_ru80_rp80.md
```

输出：

- `pressure_poisson_surrogate_tensors_allRe100_weightedL2_ru80_rp80.npz`
- `PRESSURE_POISSON_SURROGATE_TENSORS_allRe100_weightedL2_ru80_rp80.md`

压力代理系统：

```text
L b = c^p + A^p a + H^p(a,a)
b = c_tilde + A_tilde a + H_tilde(a,a)
```

压力张量校验：

- `L.shape = (80, 80)`
- `H_p.shape = (80, 80, 80)`
- `H_tilde.shape = (80, 80, 80)`
- `L_pinv_rank = 80`
- `all finite = True`
- 最大回代残差：
  - `max ||L c_tilde - c^p|| / ||c^p|| = 3.299905e-14`
  - `max ||L A_tilde - A^p|| / ||A^p|| = 8.059069e-15`
  - `||L H_tilde - H^p|| / ||H^p|| = 7.142616e-15`

代表性范数：

| Re label | `||c^p||_2` | `||A^p||_F` | `||c_tilde||_2` | `||A_tilde||_F` |
|---|---:|---:|---:|---:|
| `Re_50p000000` | `7.880179e-03` | `4.829869e-01` | `3.013816e-04` | `9.136833e-02` |
| `Re_105p983150` | `2.080586e-03` | `1.096974e+00` | `1.199876e-03` | `2.146878e-01` |
| `Re_300p000000` | `6.658929e-02` | `2.177638e+00` | `2.441458e-02` | `4.121682e-01` |

## 推荐使用文件

集群 `/root/moe/V8` 保存完整数据与全量兼容张量：

- `Global_POD_Weighted_L2/`
- `semi_intrusive_galerkin_tensors_allRe100_weightedL2_ru80_rp80_compact.npz`
- `semi_intrusive_galerkin_tensors_allRe100_weightedL2_ru80_rp80_slim.npz`
- `pressure_poisson_surrogate_tensors_allRe100_weightedL2_ru80_rp80.npz`
- 技术报告、脚本、运行日志和数据说明文档

GitHub `V8/data` 保存轻量可追踪文件：

- weighted POD 的三个核心 `.npz`
- velocity slim 张量 `.npz`
- pressure surrogate 张量 `.npz`
- 技术报告、脚本和说明文档

由于 GitHub 普通仓库单文件限制约为 100MB，`409MB` 的旧键名兼容速度张量归档只上传到集群，不上传到 GitHub。GitHub 中的 slim 速度张量与其数学内容等价。

