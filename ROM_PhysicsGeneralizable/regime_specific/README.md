# Physics-Generalizable Regime-specific ROM Operator Library

## 目标

本次构建把 Re=20-200 Physics-Generalizable 圆柱绕流数据库按流动动力学阶段拆分为三套互相独立的 ROM 离线算子库，用于 V15/V16 后续实验直接替换统一全局 ROM。

## Regime 划分

- `steady`: `steady_wake` + `pre_hopf_steady`，对应稳定定常尾迹与 Hopf 前稳定尾迹。
- `hopf`: `hopf_transition`，对应临界分岔与弱周期振荡阶段。
- `periodic`: `developing_periodic_shedding` + `mature_periodic_shedding` + `high_re_2d_periodic_near_modeA`，对应发展/成熟卡门涡街与高 Re 二维周期段。

划分直接采用数据集自带物理 Regime 标签；这些标签来自 lift 振荡、尾迹周期性、limit-cycle 诊断和 Re 物理区间设计。

## 数学形式与接口兼容性

三套速度 ROM 均保持当前 V15 形式：

```text
da/dt = c(Re) + A(Re)a + H(a,a) + P b
```

三套压力代理均保持当前 Pressure Poisson Surrogate 形式：

```text
L b = c^p(Re) + A^p(Re)a + H^p(a,a)
b = c_tilde(Re) + A_tilde(Re)a + H_tilde(a,a)
```

所有内积继续采用 `point_areas` 的 lumped finite-volume L2 权重。POD 文件键名、ROM `.npz` 张量键名、`c_all/A_all/H/P` 和 `c_tilde/A_tilde/H_tilde` 接口均与上一版统一 ROM 保持兼容。

## POD 构建说明

- 每个 Regime 子库从原始 `Re_*_uvp_pointData.npz` 快照重新构建。
- 对每个 Re 分别计算均值场，并对快照做逐 Re 去均值。
- 使用面积加权 randomized block SVD：`rank=80`, `oversampling=32`, `power_iter=2`, seed=`20260705`。
- 该方法不复用统一全局 ROM 张量；只复用相同网格、相同 `point_areas` 与同一 PyVista 导数模板。

## 产物与校验

### steady

- Re 数：`20`，快照数：`1277`，标签范围：`Re_20p000000` -> `Re_46p440072`。
- Source regimes：`['pre_hopf_steady', 'steady_wake']`。
- Velocity POD rank-80 captured energy：`9.999999996581e-01`；Pressure POD：`9.999999994277e-01`。
- POD 正交误差：velocity `max|Phi^T M Phi-I|=4.295e-08`，pressure `2.918e-08`。
- Velocity ROM：`velocity_rom_steady.npz`，shapes `{'G_u': [80, 80], 'c_all': [20, 80], 'A_all': [20, 80, 80], 'H': [80, 80, 80], 'P': [80, 80]}`，`cond(G_u)=1.000000e+00`。
- Pressure surrogate：`pressure_poisson_surrogate_steady.npz`，shapes `{'L': [80, 80], 'H_p': [80, 80, 80], 'H_tilde': [80, 80, 80]}`，`rank(L)=80/80`。
- Pressure residual：`rel ||L H_tilde-H_p||=9.664e-15`，first/last `c,A` residuals `4.064e-17/3.509e-14` and `1.312e-16/8.249e-14`。
- 数值 finite：velocity `True`，pressure `True`。

### hopf

- Re 数：`17`，快照数：`2740`，标签范围：`Re_47p081355` -> `Re_59p201432`。
- Source regimes：`['hopf_transition']`。
- Velocity POD rank-80 captured energy：`9.999999991620e-01`；Pressure POD：`9.999999960741e-01`。
- POD 正交误差：velocity `max|Phi^T M Phi-I|=5.446e-08`，pressure `5.311e-08`。
- Velocity ROM：`velocity_rom_hopf.npz`，shapes `{'G_u': [80, 80], 'c_all': [17, 80], 'A_all': [17, 80, 80], 'H': [80, 80, 80], 'P': [80, 80]}`，`cond(G_u)=1.000000e+00`。
- Pressure surrogate：`pressure_poisson_surrogate_hopf.npz`，shapes `{'L': [80, 80], 'H_p': [80, 80, 80], 'H_tilde': [80, 80, 80]}`，`rank(L)=80/80`。
- Pressure residual：`rel ||L H_tilde-H_p||=6.849e-15`，first/last `c,A` residuals `9.902e-17/3.537e-15` and `1.417e-16/4.592e-15`。
- 数值 finite：velocity `True`，pressure `True`。

### periodic

- Re 数：`63`，快照数：`10150`，标签范围：`Re_60p307745` -> `Re_200p000000`。
- Source regimes：`['developing_periodic_shedding', 'high_re_2d_periodic_near_modeA', 'mature_periodic_shedding']`。
- Velocity POD rank-80 captured energy：`9.999720490877e-01`；Pressure POD：`9.999819190345e-01`。
- POD 正交误差：velocity `max|Phi^T M Phi-I|=5.485e-08`，pressure `5.206e-08`。
- Velocity ROM：`velocity_rom_periodic.npz`，shapes `{'G_u': [80, 80], 'c_all': [63, 80], 'A_all': [63, 80, 80], 'H': [80, 80, 80], 'P': [80, 80]}`，`cond(G_u)=1.000000e+00`。
- Pressure surrogate：`pressure_poisson_surrogate_periodic.npz`，shapes `{'L': [80, 80], 'H_p': [80, 80, 80], 'H_tilde': [80, 80, 80]}`，`rank(L)=80/80`。
- Pressure residual：`rel ||L H_tilde-H_p||=4.088e-15`，first/last `c,A` residuals `2.822e-17/3.829e-15` and `1.169e-15/8.111e-15`。
- 数值 finite：velocity `True`，pressure `True`。

## 文件组织

```text
Regime_ROM_Library/
  steady/Global_POD_AreaWeighted_L2/
  hopf/Global_POD_AreaWeighted_L2/
  periodic/Global_POD_AreaWeighted_L2/
  velocity_rom_steady.npz
  velocity_rom_hopf.npz
  velocity_rom_periodic.npz
  pressure_poisson_surrogate_steady.npz
  pressure_poisson_surrogate_hopf.npz
  pressure_poisson_surrogate_periodic.npz
  REGIME_SPECIFIC_ROM_LIBRARY_manifest.json
```
