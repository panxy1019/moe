# Pressure Poisson Surrogate Galerkin Tensors

## 数据来源

- 数据目录：`/home/ray/Desktop/Cylinder_Results_Re500_1000_30Re_POD`
- POD 目录：`/home/ray/Desktop/Cylinder_Results_Re500_1000_30Re_POD/Global_POD_Weighted_L2`
- 网格模板：`/home/ray/Desktop/Cylinder_Results_Re500_1000_VTU/Re_1000_VTU/flow_60.vtu`
- 输出文件：`/home/ray/Desktop/Cylinder_Results_Re500_1000_30Re_POD/pressure_poisson_surrogate_tensors_allRe30_weightedL2_ru80_rp80.npz`
- 计算 Re 数量：`30`
- 计算 Re 标签：`['Re_500p000000', 'Re_517p241379', 'Re_534p482759', 'Re_551p724138', 'Re_568p965517', 'Re_586p206897', 'Re_603p448276', 'Re_620p689655', 'Re_637p931034', 'Re_655p172414', 'Re_672p413793', 'Re_689p655172', 'Re_706p896552', 'Re_724p137931', 'Re_741p379310', 'Re_758p620690', 'Re_775p862069', 'Re_793p103448', 'Re_810p344828', 'Re_827p586207', 'Re_844p827586', 'Re_862p068966', 'Re_879p310345', 'Re_896p551724', 'Re_913p793103', 'Re_931p034483', 'Re_948p275862', 'Re_965p517241', 'Re_982p758621', 'Re_1000p000000']`

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
- 体积分权重使用 weighted L2 POD 的 `point_volumes`。
- 向量梯度 reshape 为 `(N, 3, 3)`，轴含义是 `[速度分量, 空间导数方向]`。
- `H^p` 使用节点分块和 `np.einsum('ncm,naj,ncak,n->mjk', ...)` 装配。

## 输出张量

- `L.shape = (80, 80)`
- `H_p.shape = (80, 80, 80)`
- `H_tilde.shape = (80, 80, 80)`
- `mass_weights.shape = (97368,)`
- `sum(mass_weights) = 5.992147006350e+01`
- `L` SVD rank = `80` / `80` with `rcond=1e-10`
- `L` singular cutoff = `4.977021e-09`
- `L` condition estimate = `1.996307e+03`
- `||H_p||_F = 5.450937e+01`
- `||H_tilde||_F = 1.069562e+01`

## 等效代数代理系统

脚本保存 `L_pinv`，并已左乘得到最终等效张量：

```text
c_tilde = L_pinv c^p
A_tilde = L_pinv A^p
H_tilde[:,j,k] = L_pinv H^p[:,j,k]
b(t) = c_tilde + A_tilde a(t) + H_tilde(a(t),a(t))
```

## 本次运行结果

### Re_500p000000 (`Re = 500`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 1.146517e-01`
- `||A^p||_F = 5.600578e+00`
- `||c_tilde||_2 = 5.316325e-02`
- `||A_tilde||_F = 7.914566e-01`

### Re_517p241379 (`Re = 517.24137931`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 1.248577e-01`
- `||A^p||_F = 5.792048e+00`
- `||c_tilde||_2 = 5.600689e-02`
- `||A_tilde||_F = 8.172198e-01`

### Re_534p482759 (`Re = 534.482758621`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 1.354628e-01`
- `||A^p||_F = 5.987583e+00`
- `||c_tilde||_2 = 6.004583e-02`
- `||A_tilde||_F = 8.435377e-01`

### Re_551p724138 (`Re = 551.724137931`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 1.465363e-01`
- `||A^p||_F = 6.184314e+00`
- `||c_tilde||_2 = 6.439689e-02`
- `||A_tilde||_F = 8.700964e-01`

### Re_568p965517 (`Re = 568.965517241`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 1.589390e-01`
- `||A^p||_F = 6.373550e+00`
- `||c_tilde||_2 = 6.827332e-02`
- `||A_tilde||_F = 8.960048e-01`

### Re_586p206897 (`Re = 586.206896552`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 1.711114e-01`
- `||A^p||_F = 6.579934e+00`
- `||c_tilde||_2 = 7.219582e-02`
- `||A_tilde||_F = 9.245126e-01`

### Re_603p448276 (`Re = 603.448275862`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 1.838241e-01`
- `||A^p||_F = 6.782751e+00`
- `||c_tilde||_2 = 7.711578e-02`
- `||A_tilde||_F = 9.530413e-01`

### Re_620p689655 (`Re = 620.689655172`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 1.973392e-01`
- `||A^p||_F = 6.969188e+00`
- `||c_tilde||_2 = 8.405664e-02`
- `||A_tilde||_F = 9.807720e-01`

### Re_637p931034 (`Re = 637.931034483`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.101950e-01`
- `||A^p||_F = 7.177957e+00`
- `||c_tilde||_2 = 9.018896e-02`
- `||A_tilde||_F = 1.010480e+00`

### Re_655p172414 (`Re = 655.172413793`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.236717e-01`
- `||A^p||_F = 7.377593e+00`
- `||c_tilde||_2 = 9.741815e-02`
- `||A_tilde||_F = 1.039863e+00`

### Re_672p413793 (`Re = 672.413793103`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.374074e-01`
- `||A^p||_F = 7.582469e+00`
- `||c_tilde||_2 = 1.045276e-01`
- `||A_tilde||_F = 1.069793e+00`

### Re_689p655172 (`Re = 689.655172414`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.519366e-01`
- `||A^p||_F = 7.778569e+00`
- `||c_tilde||_2 = 1.128466e-01`
- `||A_tilde||_F = 1.099815e+00`

### Re_706p896552 (`Re = 706.896551724`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.667896e-01`
- `||A^p||_F = 7.983328e+00`
- `||c_tilde||_2 = 1.203138e-01`
- `||A_tilde||_F = 1.130034e+00`

### Re_724p137931 (`Re = 724.137931034`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.829943e-01`
- `||A^p||_F = 8.174317e+00`
- `||c_tilde||_2 = 1.294979e-01`
- `||A_tilde||_F = 1.159880e+00`

### Re_741p379310 (`Re = 741.379310345`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.992288e-01`
- `||A^p||_F = 8.375896e+00`
- `||c_tilde||_2 = 1.375423e-01`
- `||A_tilde||_F = 1.189667e+00`

### Re_758p620690 (`Re = 758.620689655`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 3.414230e-01`
- `||A^p||_F = 8.351253e+00`
- `||c_tilde||_2 = 1.583420e-01`
- `||A_tilde||_F = 1.198084e+00`

### Re_775p862069 (`Re = 775.862068966`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 3.614553e-01`
- `||A^p||_F = 8.548308e+00`
- `||c_tilde||_2 = 1.669115e-01`
- `||A_tilde||_F = 1.225445e+00`

### Re_793p103448 (`Re = 793.103448276`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 3.812874e-01`
- `||A^p||_F = 8.752158e+00`
- `||c_tilde||_2 = 1.758090e-01`
- `||A_tilde||_F = 1.254615e+00`

### Re_810p344828 (`Re = 810.344827586`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.025168e-01`
- `||A^p||_F = 8.955442e+00`
- `||c_tilde||_2 = 1.862595e-01`
- `||A_tilde||_F = 1.284603e+00`

### Re_827p586207 (`Re = 827.586206897`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.256005e-01`
- `||A^p||_F = 9.147611e+00`
- `||c_tilde||_2 = 1.998247e-01`
- `||A_tilde||_F = 1.314465e+00`

### Re_844p827586 (`Re = 844.827586207`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.315302e-01`
- `||A^p||_F = 9.458356e+00`
- `||c_tilde||_2 = 1.864458e-01`
- `||A_tilde||_F = 1.364880e+00`

### Re_862p068966 (`Re = 862.068965517`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.525210e-01`
- `||A^p||_F = 9.659586e+00`
- `||c_tilde||_2 = 1.947694e-01`
- `||A_tilde||_F = 1.388352e+00`

### Re_879p310345 (`Re = 879.310344828`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.730517e-01`
- `||A^p||_F = 9.878557e+00`
- `||c_tilde||_2 = 1.994813e-01`
- `||A_tilde||_F = 1.419174e+00`

### Re_896p551724 (`Re = 896.551724138`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.956612e-01`
- `||A^p||_F = 1.007339e+01`
- `||c_tilde||_2 = 2.090944e-01`
- `||A_tilde||_F = 1.442165e+00`

### Re_913p793103 (`Re = 913.793103448`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 5.179994e-01`
- `||A^p||_F = 1.028039e+01`
- `||c_tilde||_2 = 2.175606e-01`
- `||A_tilde||_F = 1.469176e+00`

### Re_931p034483 (`Re = 931.034482759`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 5.398015e-01`
- `||A^p||_F = 1.050378e+01`
- `||c_tilde||_2 = 2.215352e-01`
- `||A_tilde||_F = 1.501018e+00`

### Re_948p275862 (`Re = 948.275862069`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 5.635874e-01`
- `||A^p||_F = 1.069904e+01`
- `||c_tilde||_2 = 2.347573e-01`
- `||A_tilde||_F = 1.526097e+00`

### Re_965p517241 (`Re = 965.517241379`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 5.869423e-01`
- `||A^p||_F = 1.091224e+01`
- `||c_tilde||_2 = 2.457634e-01`
- `||A_tilde||_F = 1.555027e+00`

### Re_982p758621 (`Re = 982.75862069`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 6.093498e-01`
- `||A^p||_F = 1.114615e+01`
- `||c_tilde||_2 = 2.438692e-01`
- `||A_tilde||_F = 1.589051e+00`

### Re_1000p000000 (`Re = 1000`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 6.339841e-01`
- `||A^p||_F = 1.134934e+01`
- `||c_tilde||_2 = 2.595255e-01`
- `||A_tilde||_F = 1.616041e+00`

总运行时间：`1877.0 s`。
