# Area-weighted L2 Semi-intrusive Galerkin Projection Tensors

## 数据来源

- 数据目录：`/home/ray/Desktop/Cylinder_ROM_PhysicsGeneralizable_Re20_200_100Re/Regime_ROM_Library/steady`
- 网格模板：`/home/ray/Desktop/Cylinder_Results/Re_100_VTK/run_Re_100_7107.vtk`
- POD 目录：`/home/ray/Desktop/Cylinder_ROM_PhysicsGeneralizable_Re20_200_100Re/Regime_ROM_Library/steady/Global_POD_AreaWeighted_L2`
- 计算 Re 数量：`20`
- 计算 Re 标签：`['Re_20p000000', 'Re_22p535676', 'Re_24p630436', 'Re_26p667332', 'Re_28p695138', 'Re_30p720428', 'Re_32p740068', 'Re_34p737570', 'Re_36p657767', 'Re_38p357249', 'Re_39p685479', 'Re_40p711525', 'Re_41p576575', 'Re_42p359071', 'Re_43p093925', 'Re_43p797402', 'Re_44p478353', 'Re_45p142703', 'Re_45p795194', 'Re_46p440072']`
- 运动粘度：`nu = 0.001`
- 输出张量文件：`/home/ray/Desktop/Cylinder_ROM_PhysicsGeneralizable_Re20_200_100Re/Regime_ROM_Library/velocity_rom_steady.npz`

## 数组形状

- `N = 97368`
- `r_u = 80`，`phi_u.shape = (97368, 3, 80)`
- `r_p = 80`，`phi_p.shape = (97368, 80)`
- `mass_weights.shape = (97368,)`
- `sum(mass_weights) = 5.992146803695e+02`
- `min/max(mass_weights) = 1.042356689140e-05 / 1.319450698793e-02`

## 离散内积与质量权重

离散内积采用 lumped mass 形式：

```text
<f, g>_M = sum_n mass_weights[n] * sum_c f[n,c] * g[n,c]
```

本数据集已经提供 `Global_POD_AreaWeighted_L2/mesh_l2_point_area_weights.npz`，脚本直接读取其中的 `point_areas` 作为 `mass_weights`。这些权重是 2D 计算域单元面积按节点 lumping 后的 nodal control areas，与 area-weighted L2 POD 报告中的内积一致。

与上一版 unweighted 数据集不同，本次 `phi_uv/phi_p` 是 raw 物理空间 area-weighted POD 模态，并满足质量矩阵正交：`Phi^T M Phi ≈ I`。脚本仍保存 `G_u` 和 Gram 修正后的 `c/A/H/P`，用于消除 float32 存储和导出误差带来的小偏差。

## PyVista 导数计算

脚本通过 `pyvista.UnstructuredGrid.compute_derivative()` 在点数据上计算导数，并封装成 NumPy 接口：

```python
grad_q = mesh.compute_derivative(scalars='q', gradient='grad_q', preference='point')
lap_q = grad_q.compute_derivative(scalars='grad_q', divergence='lap_q', preference='point')
grad_u = mesh.compute_derivative(scalars='u', gradient='grad_u', preference='point')
```
向量梯度的 9 个分量按 `[du/dx, du/dy, du/dz, dv/dx, ..., dw/dz]` 排列，脚本 reshape 为 `(N, 3, 3)`，轴含义为 `[速度分量, 空间导数方向]`。

## 投影张量定义

原始投影张量为：

```text
G_ij     = <phi_i, phi_j>_M
c_raw_i  = <phi_i, -(ubar · grad) ubar + nu Lap(ubar) - grad(pbar)>_M
A_raw_ij = <phi_i, -(ubar · grad) phi_j - (phi_j · grad) ubar + nu Lap(phi_j)>_M
H_raw_ijk= <phi_i, -(phi_j · grad) phi_k>_M
P_raw_im = <phi_i, -grad(psi_m)>_M
```

对 area-weighted L2 raw-space 模态，理论上 `G≈I`。为保持数值严格性，脚本先装配：

```text
G da/dt = c_raw + A_raw a + H_raw(a,a) + P_raw b
```

脚本同时保存用于显式 ROM 的 Gram 修正张量：

```text
c = G^{-1} c_raw
A = G^{-1} A_raw
H[:,j,k] = G^{-1} H_raw[:,j,k]
P = G^{-1} P_raw
da/dt = c + A a + H(a,a) + P b
```

## `np.einsum` 收缩实现

关键对流项没有在模态维度上使用 Python 原生循环。核心收缩为：

```python
cross_1 = -np.einsum('na,ncaj->ncj', u_bar, grad_phi_u)
cross_2 = -np.einsum('naj,nca->ncj', phi_u, grad_u_bar)
H_raw  -=  np.einsum('nci,naj,ncak,n->ijk', phi_u_blk, phi_u_blk, grad_phi_u_blk, w_blk)
P_raw   = -np.einsum('nci,ncm,n->im', phi_u, grad_phi_p, weights)
```

## 本次运行结果

### Re_20p000000  (`Re = 20`)

- `G_u.shape = (80, 80)`
- `c.shape = (80,)`
- `A.shape = (80, 80)`
- `H.shape = (80, 80, 80)`
- `P.shape = (80, 80)`
- `cond(G_u) = 1.000000e+00`
- `max_abs(G_u - I) = 4.294780e-08`
- `||c||_2 = 5.496756e-05`
- `||A||_F = 6.614332e-01`
- `||H||_F = 4.511319e+01`
- `||P||_F = 3.447347e+01`

### Re_22p535676  (`Re = 22.5356758074`)

- `G_u.shape = (80, 80)`
- `c.shape = (80,)`
- `A.shape = (80, 80)`
- `H.shape = (80, 80, 80)`
- `P.shape = (80, 80)`
- `cond(G_u) = 1.000000e+00`
- `max_abs(G_u - I) = 4.294780e-08`
- `||c||_2 = 6.721272e-05`
- `||A||_F = 6.867884e-01`
- `||H||_F = 4.511319e+01`
- `||P||_F = 3.447347e+01`

### Re_24p630436  (`Re = 24.6304355074`)

- `G_u.shape = (80, 80)`
- `c.shape = (80,)`
- `A.shape = (80, 80)`
- `H.shape = (80, 80, 80)`
- `P.shape = (80, 80)`
- `cond(G_u) = 1.000000e+00`
- `max_abs(G_u - I) = 4.294780e-08`
- `||c||_2 = 7.812482e-05`
- `||A||_F = 7.092550e-01`
- `||H||_F = 4.511319e+01`
- `||P||_F = 3.447347e+01`

### Re_26p667332  (`Re = 26.6673319278`)

- `G_u.shape = (80, 80)`
- `c.shape = (80,)`
- `A.shape = (80, 80)`
- `H.shape = (80, 80, 80)`
- `P.shape = (80, 80)`
- `cond(G_u) = 1.000000e+00`
- `max_abs(G_u - I) = 4.294780e-08`
- `||c||_2 = 8.941583e-05`
- `||A||_F = 7.322969e-01`
- `||H||_F = 4.511319e+01`
- `||P||_F = 3.447347e+01`

### Re_28p695138  (`Re = 28.695137758`)

- `G_u.shape = (80, 80)`
- `c.shape = (80,)`
- `A.shape = (80, 80)`
- `H.shape = (80, 80, 80)`
- `P.shape = (80, 80)`
- `cond(G_u) = 1.000000e+00`
- `max_abs(G_u - I) = 4.294780e-08`
- `||c||_2 = 9.695475e-05`
- `||A||_F = 7.551926e-01`
- `||H||_F = 4.511319e+01`
- `||P||_F = 3.447347e+01`

### Re_30p720428  (`Re = 30.7204283434`)

- `G_u.shape = (80, 80)`
- `c.shape = (80,)`
- `A.shape = (80, 80)`
- `H.shape = (80, 80, 80)`
- `P.shape = (80, 80)`
- `cond(G_u) = 1.000000e+00`
- `max_abs(G_u - I) = 4.294780e-08`
- `||c||_2 = 1.075951e-04`
- `||A||_F = 7.796433e-01`
- `||H||_F = 4.511319e+01`
- `||P||_F = 3.447347e+01`

### Re_32p740068  (`Re = 32.740068162`)

- `G_u.shape = (80, 80)`
- `c.shape = (80,)`
- `A.shape = (80, 80)`
- `H.shape = (80, 80, 80)`
- `P.shape = (80, 80)`
- `cond(G_u) = 1.000000e+00`
- `max_abs(G_u - I) = 4.294780e-08`
- `||c||_2 = 1.178398e-04`
- `||A||_F = 8.048059e-01`
- `||H||_F = 4.511319e+01`
- `||P||_F = 3.447347e+01`

### Re_34p737570  (`Re = 34.7375697369`)

- `G_u.shape = (80, 80)`
- `c.shape = (80,)`
- `A.shape = (80, 80)`
- `H.shape = (80, 80, 80)`
- `P.shape = (80, 80)`
- `cond(G_u) = 1.000000e+00`
- `max_abs(G_u - I) = 4.294780e-08`
- `||c||_2 = 1.285303e-04`
- `||A||_F = 8.304874e-01`
- `||H||_F = 4.511319e+01`
- `||P||_F = 3.447347e+01`

### Re_36p657767  (`Re = 36.6577674843`)

- `G_u.shape = (80, 80)`
- `c.shape = (80,)`
- `A.shape = (80, 80)`
- `H.shape = (80, 80, 80)`
- `P.shape = (80, 80)`
- `cond(G_u) = 1.000000e+00`
- `max_abs(G_u - I) = 4.294780e-08`
- `||c||_2 = 1.389835e-04`
- `||A||_F = 8.557377e-01`
- `||H||_F = 4.511319e+01`
- `||P||_F = 3.447347e+01`

### Re_38p357249  (`Re = 38.357249335`)

- `G_u.shape = (80, 80)`
- `c.shape = (80,)`
- `A.shape = (80, 80)`
- `H.shape = (80, 80, 80)`
- `P.shape = (80, 80)`
- `cond(G_u) = 1.000000e+00`
- `max_abs(G_u - I) = 4.294780e-08`
- `||c||_2 = 1.486067e-04`
- `||A||_F = 8.785911e-01`
- `||H||_F = 4.511319e+01`
- `||P||_F = 3.447347e+01`

### Re_39p685479  (`Re = 39.6854792076`)

- `G_u.shape = (80, 80)`
- `c.shape = (80,)`
- `A.shape = (80, 80)`
- `H.shape = (80, 80, 80)`
- `P.shape = (80, 80)`
- `cond(G_u) = 1.000000e+00`
- `max_abs(G_u - I) = 4.294780e-08`
- `||c||_2 = 1.561744e-04`
- `||A||_F = 8.966905e-01`
- `||H||_F = 4.511319e+01`
- `||P||_F = 3.447347e+01`

### Re_40p711525  (`Re = 40.7115247449`)

- `G_u.shape = (80, 80)`
- `c.shape = (80,)`
- `A.shape = (80, 80)`
- `H.shape = (80, 80, 80)`
- `P.shape = (80, 80)`
- `cond(G_u) = 1.000000e+00`
- `max_abs(G_u - I) = 4.294780e-08`
- `||c||_2 = 1.623116e-04`
- `||A||_F = 9.108793e-01`
- `||H||_F = 4.511319e+01`
- `||P||_F = 3.447347e+01`

### Re_41p576575  (`Re = 41.5765746111`)

- `G_u.shape = (80, 80)`
- `c.shape = (80,)`
- `A.shape = (80, 80)`
- `H.shape = (80, 80, 80)`
- `P.shape = (80, 80)`
- `cond(G_u) = 1.000000e+00`
- `max_abs(G_u - I) = 4.294780e-08`
- `||c||_2 = 1.675050e-04`
- `||A||_F = 9.229275e-01`
- `||H||_F = 4.511319e+01`
- `||P||_F = 3.447347e+01`

### Re_42p359071  (`Re = 42.3590705675`)

- `G_u.shape = (80, 80)`
- `c.shape = (80,)`
- `A.shape = (80, 80)`
- `H.shape = (80, 80, 80)`
- `P.shape = (80, 80)`
- `cond(G_u) = 1.000000e+00`
- `max_abs(G_u - I) = 4.294780e-08`
- `||c||_2 = 1.723407e-04`
- `||A||_F = 9.339300e-01`
- `||H||_F = 4.511319e+01`
- `||P||_F = 3.447347e+01`

### Re_43p093925  (`Re = 43.0939251612`)

- `G_u.shape = (80, 80)`
- `c.shape = (80,)`
- `A.shape = (80, 80)`
- `H.shape = (80, 80, 80)`
- `P.shape = (80, 80)`
- `cond(G_u) = 1.000000e+00`
- `max_abs(G_u - I) = 4.294780e-08`
- `||c||_2 = 1.767194e-04`
- `||A||_F = 9.442790e-01`
- `||H||_F = 4.511319e+01`
- `||P||_F = 3.447347e+01`

### Re_43p797402  (`Re = 43.7974019747`)

- `G_u.shape = (80, 80)`
- `c.shape = (80,)`
- `A.shape = (80, 80)`
- `H.shape = (80, 80, 80)`
- `P.shape = (80, 80)`
- `cond(G_u) = 1.000000e+00`
- `max_abs(G_u - I) = 4.294780e-08`
- `||c||_2 = 1.810575e-04`
- `||A||_F = 9.542674e-01`
- `||H||_F = 4.511319e+01`
- `||P||_F = 3.447347e+01`

### Re_44p478353  (`Re = 44.4783532118`)

- `G_u.shape = (80, 80)`
- `c.shape = (80,)`
- `A.shape = (80, 80)`
- `H.shape = (80, 80, 80)`
- `P.shape = (80, 80)`
- `cond(G_u) = 1.000000e+00`
- `max_abs(G_u - I) = 4.294780e-08`
- `||c||_2 = 1.852302e-04`
- `||A||_F = 9.639766e-01`
- `||H||_F = 4.511319e+01`
- `||P||_F = 3.447347e+01`

### Re_45p142703  (`Re = 45.142702578`)

- `G_u.shape = (80, 80)`
- `c.shape = (80,)`
- `A.shape = (80, 80)`
- `H.shape = (80, 80, 80)`
- `P.shape = (80, 80)`
- `cond(G_u) = 1.000000e+00`
- `max_abs(G_u - I) = 4.294780e-08`
- `||c||_2 = 1.894423e-04`
- `||A||_F = 9.735143e-01`
- `||H||_F = 4.511319e+01`
- `||P||_F = 3.447347e+01`

### Re_45p795194  (`Re = 45.7951936696`)

- `G_u.shape = (80, 80)`
- `c.shape = (80,)`
- `A.shape = (80, 80)`
- `H.shape = (80, 80, 80)`
- `P.shape = (80, 80)`
- `cond(G_u) = 1.000000e+00`
- `max_abs(G_u - I) = 4.294780e-08`
- `||c||_2 = 1.935982e-04`
- `||A||_F = 9.829343e-01`
- `||H||_F = 4.511319e+01`
- `||P||_F = 3.447347e+01`

### Re_46p440072  (`Re = 46.440071584`)

- `G_u.shape = (80, 80)`
- `c.shape = (80,)`
- `A.shape = (80, 80)`
- `H.shape = (80, 80, 80)`
- `P.shape = (80, 80)`
- `cond(G_u) = 1.000000e+00`
- `max_abs(G_u - I) = 4.294780e-08`
- `||c||_2 = 1.977418e-04`
- `||A||_F = 9.922716e-01`
- `||H||_F = 4.511319e+01`
- `||P||_F = 3.447347e+01`

## 最终侵入式 ROM 方程

对本次输出文件中的每个 `Re_xxx`，使用 Gram 修正后的张量：

```text
da_i/dt = c_i + sum_j A_ij a_j + sum_j sum_k H_ijk a_j a_k + sum_m P_im b_m
```

其中 `a` 是速度模态系数，`b` 是压力模态系数；`c/A` 对应 `.npz` 中的 `Re_xxx_c`、`Re_xxx_A`。在全 Re compact 输出中，`G_u/H/P` 是共享张量；单 Re 或非 compact raw 输出中也可能包含 `Re_xxx_H`、`Re_xxx_P` 形式的兼容键。

总运行时间：`1158.8 s`。
