# Pressure Poisson Surrogate Galerkin Tensors

## 数据来源

- 数据目录：`/home/ray/Desktop/Cylinder_ROM_PhysicsGeneralizable_Re20_200_100Re/Regime_ROM_Library/hopf`
- POD 目录：`/home/ray/Desktop/Cylinder_ROM_PhysicsGeneralizable_Re20_200_100Re/Regime_ROM_Library/hopf/Global_POD_AreaWeighted_L2`
- 网格模板：`/home/ray/Desktop/Cylinder_Results/Re_100_VTK/run_Re_100_7107.vtk`
- 输出文件：`/home/ray/Desktop/Cylinder_ROM_PhysicsGeneralizable_Re20_200_100Re/Regime_ROM_Library/pressure_poisson_surrogate_hopf.npz`
- 计算 Re 数量：`17`
- 计算 Re 标签：`['Re_47p081355', 'Re_47p722947', 'Re_48p368688', 'Re_49p022357', 'Re_49p687640', 'Re_50p368054', 'Re_51p066785', 'Re_51p786450', 'Re_52p528767', 'Re_53p294175', 'Re_54p081508', 'Re_54p887950', 'Re_55p709610', 'Re_56p543246', 'Re_57p389970', 'Re_58p262636', 'Re_59p201432']`

## 弱形式与符号约定

不可压缩动量方程取散度后采用压力泊松形式：

```text
Delta p = - div((u dot grad) u)
```

用压力基函数 `psi_m` 测试并忽略边界项：

```text
- int grad(psi_m) dot grad(p) dOmega = int grad(psi_m) dot ((u dot grad)u) dOmega
```

令 `p = p_bar + sum_k psi_k b_k`，`u = u_bar + sum_j phi_j a_j`，得到：

```text
L b = c^p + A^p a + H^p(a,a)
L_mk = - int grad(psi_m) dot grad(psi_k) dOmega
c^p_m = int grad(psi_m) dot ((u_bar dot grad)u_bar + grad(p_bar)) dOmega
A^p_mj = int grad(psi_m) dot ((u_bar dot grad)phi_j + (phi_j dot grad)u_bar) dOmega
H^p_mjk = int grad(psi_m) dot ((phi_j dot grad)phi_k) dOmega
```

## 数值实现

- 导数由 `pyvista.UnstructuredGrid.compute_derivative()` 在非结构 VTU 网格上计算。
- 积分权重使用 area-weighted L2 POD 的 `point_areas`。
- 向量梯度 reshape 为 `(N, 3, 3)`，轴含义是 `[速度分量, 空间导数方向]`。
- `H^p` 使用节点分块和 `np.einsum('ncm,naj,ncak,n->mjk', ...)` 装配。

## 输出张量

- `L.shape = (80, 80)`
- `H_p.shape = (80, 80, 80)`
- `H_tilde.shape = (80, 80, 80)`
- `mass_weights.shape = (97368,)`
- `sum(mass_weights) = 5.992146803695e+02`
- `L` SVD rank = `80` / `80` with `rcond=1e-10`
- `L` singular cutoff = `3.370297e-08`
- `L` condition estimate = `8.655241e+04`
- `||H_p||_F = 3.582439e+01`
- `||H_tilde||_F = 5.137709e+00`

## 等效代数代理系统

脚本保存 `L_pinv`，并已左乘得到最终等效张量：

```text
c_tilde = L_pinv c^p
A_tilde = L_pinv A^p
H_tilde[:,j,k] = L_pinv H^p[:,j,k]
b(t) = c_tilde + A_tilde a(t) + H_tilde(a(t),a(t))
```

## 本次运行结果

### Re_47p081355 (`Re = 47.0813545644`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 1.857771e-03`
- `||A^p||_F = 6.398146e-01`
- `||c_tilde||_2 = 2.780933e-03`
- `||A_tilde||_F = 6.543968e-02`

### Re_47p722947 (`Re = 47.7229474482`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 1.904089e-03`
- `||A^p||_F = 6.502669e-01`
- `||c_tilde||_2 = 2.837036e-03`
- `||A_tilde||_F = 6.681457e-02`

### Re_48p368688 (`Re = 48.3686884481`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 1.951720e-03`
- `||A^p||_F = 6.607758e-01`
- `||c_tilde||_2 = 2.894772e-03`
- `||A_tilde||_F = 6.789714e-02`

### Re_49p022357 (`Re = 49.0223566571`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.000750e-03`
- `||A^p||_F = 6.713931e-01`
- `||c_tilde||_2 = 2.954129e-03`
- `||A_tilde||_F = 6.874135e-02`

### Re_49p687640 (`Re = 49.6876404962`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.050564e-03`
- `||A^p||_F = 6.823157e-01`
- `||c_tilde||_2 = 3.013687e-03`
- `||A_tilde||_F = 7.014589e-02`

### Re_50p368054 (`Re = 50.3680543703`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.102789e-03`
- `||A^p||_F = 6.934558e-01`
- `||c_tilde||_2 = 3.075675e-03`
- `||A_tilde||_F = 7.123990e-02`

### Re_51p066785 (`Re = 51.066784903`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.156675e-03`
- `||A^p||_F = 7.049595e-01`
- `||c_tilde||_2 = 3.139275e-03`
- `||A_tilde||_F = 7.256616e-02`

### Re_51p786450 (`Re = 51.7864496836`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.212733e-03`
- `||A^p||_F = 7.168401e-01`
- `||c_tilde||_2 = 3.205018e-03`
- `||A_tilde||_F = 7.395096e-02`

### Re_52p528767 (`Re = 52.5287670834`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.272707e-03`
- `||A^p||_F = 7.289032e-01`
- `||c_tilde||_2 = 3.275507e-03`
- `||A_tilde||_F = 7.436341e-02`

### Re_53p294175 (`Re = 53.2941749584`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.333263e-03`
- `||A^p||_F = 7.417622e-01`
- `||c_tilde||_2 = 3.345201e-03`
- `||A_tilde||_F = 7.650584e-02`

### Re_54p081508 (`Re = 54.0815080552`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.397182e-03`
- `||A^p||_F = 7.548630e-01`
- `||c_tilde||_2 = 3.418526e-03`
- `||A_tilde||_F = 7.802486e-02`

### Re_54p887950 (`Re = 54.887950068`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.464027e-03`
- `||A^p||_F = 7.682311e-01`
- `||c_tilde||_2 = 3.495068e-03`
- `||A_tilde||_F = 7.918627e-02`

### Re_55p709610 (`Re = 55.709610114`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.532514e-03`
- `||A^p||_F = 7.819529e-01`
- `||c_tilde||_2 = 3.572225e-03`
- `||A_tilde||_F = 8.093424e-02`

### Re_56p543246 (`Re = 56.5432463134`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.603326e-03`
- `||A^p||_F = 7.958189e-01`
- `||c_tilde||_2 = 3.652325e-03`
- `||A_tilde||_F = 8.211370e-02`

### Re_57p389970 (`Re = 57.3899704283`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.674879e-03`
- `||A^p||_F = 8.097668e-01`
- `||c_tilde||_2 = 3.733006e-03`
- `||A_tilde||_F = 8.354626e-02`

### Re_58p262636 (`Re = 58.2626356101`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.748447e-03`
- `||A^p||_F = 8.190333e-01`
- `||c_tilde||_2 = 3.802538e-03`
- `||A_tilde||_F = 8.414523e-02`

### Re_59p201432 (`Re = 59.2014322659`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.830250e-03`
- `||A^p||_F = 8.373118e-01`
- `||c_tilde||_2 = 3.901611e-03`
- `||A_tilde||_F = 8.609382e-02`

总运行时间：`985.8 s`。
