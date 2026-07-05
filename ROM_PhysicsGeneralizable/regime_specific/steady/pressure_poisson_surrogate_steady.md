# Pressure Poisson Surrogate Galerkin Tensors

## 数据来源

- 数据目录：`/home/ray/Desktop/Cylinder_ROM_PhysicsGeneralizable_Re20_200_100Re/Regime_ROM_Library/steady`
- POD 目录：`/home/ray/Desktop/Cylinder_ROM_PhysicsGeneralizable_Re20_200_100Re/Regime_ROM_Library/steady/Global_POD_AreaWeighted_L2`
- 网格模板：`/home/ray/Desktop/Cylinder_Results/Re_100_VTK/run_Re_100_7107.vtk`
- 输出文件：`/home/ray/Desktop/Cylinder_ROM_PhysicsGeneralizable_Re20_200_100Re/Regime_ROM_Library/pressure_poisson_surrogate_steady.npz`
- 计算 Re 数量：`20`
- 计算 Re 标签：`['Re_20p000000', 'Re_22p535676', 'Re_24p630436', 'Re_26p667332', 'Re_28p695138', 'Re_30p720428', 'Re_32p740068', 'Re_34p737570', 'Re_36p657767', 'Re_38p357249', 'Re_39p685479', 'Re_40p711525', 'Re_41p576575', 'Re_42p359071', 'Re_43p093925', 'Re_43p797402', 'Re_44p478353', 'Re_45p142703', 'Re_45p795194', 'Re_46p440072']`

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
- `L` singular cutoff = `1.122937e-07`
- `L` condition estimate = `3.679324e+05`
- `||H_p||_F = 1.437318e+03`
- `||H_tilde||_F = 1.471903e+01`

## 等效代数代理系统

脚本保存 `L_pinv`，并已左乘得到最终等效张量：

```text
c_tilde = L_pinv c^p
A_tilde = L_pinv A^p
H_tilde[:,j,k] = L_pinv H^p[:,j,k]
b(t) = c_tilde + A_tilde a(t) + H_tilde(a(t),a(t))
```

## 本次运行结果

### Re_20p000000 (`Re = 20`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 1.347122e-03`
- `||A^p||_F = 3.084745e+00`
- `||c_tilde||_2 = 1.745250e-04`
- `||A_tilde||_F = 7.854036e-02`

### Re_22p535676 (`Re = 22.5356758074`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 1.637997e-03`
- `||A^p||_F = 3.535906e+00`
- `||c_tilde||_2 = 2.027185e-04`
- `||A_tilde||_F = 9.014871e-02`

### Re_24p630436 (`Re = 24.6304355074`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 1.896807e-03`
- `||A^p||_F = 3.915906e+00`
- `||c_tilde||_2 = 2.263294e-04`
- `||A_tilde||_F = 9.986191e-02`

### Re_26p667332 (`Re = 26.6673319278`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.164811e-03`
- `||A^p||_F = 4.290935e+00`
- `||c_tilde||_2 = 2.498929e-04`
- `||A_tilde||_F = 1.093945e-01`

### Re_28p695138 (`Re = 28.695137758`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.447516e-03`
- `||A^p||_F = 4.620241e+00`
- `||c_tilde||_2 = 2.774652e-04`
- `||A_tilde||_F = 1.185172e-01`

### Re_30p720428 (`Re = 30.7204283434`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.750391e-03`
- `||A^p||_F = 4.980204e+00`
- `||c_tilde||_2 = 3.026143e-04`
- `||A_tilde||_F = 1.279697e-01`

### Re_32p740068 (`Re = 32.740068162`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 3.031836e-03`
- `||A^p||_F = 5.338091e+00`
- `||c_tilde||_2 = 3.295382e-04`
- `||A_tilde||_F = 1.374203e-01`

### Re_34p737570 (`Re = 34.7375697369`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 3.324386e-03`
- `||A^p||_F = 5.695697e+00`
- `||c_tilde||_2 = 3.555350e-04`
- `||A_tilde||_F = 1.468289e-01`

### Re_36p657767 (`Re = 36.6577674843`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 3.619518e-03`
- `||A^p||_F = 6.037481e+00`
- `||c_tilde||_2 = 3.809756e-04`
- `||A_tilde||_F = 1.559057e-01`

### Re_38p357249 (`Re = 38.357249335`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 3.893655e-03`
- `||A^p||_F = 6.342086e+00`
- `||c_tilde||_2 = 4.045838e-04`
- `||A_tilde||_F = 1.639752e-01`

### Re_39p685479 (`Re = 39.6854792076`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.113073e-03`
- `||A^p||_F = 6.578576e+00`
- `||c_tilde||_2 = 4.228162e-04`
- `||A_tilde||_F = 1.702984e-01`

### Re_40p711525 (`Re = 40.7115247449`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.287840e-03`
- `||A^p||_F = 6.763099e+00`
- `||c_tilde||_2 = 4.379570e-04`
- `||A_tilde||_F = 1.751935e-01`

### Re_41p576575 (`Re = 41.5765746111`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.435645e-03`
- `||A^p||_F = 6.918038e+00`
- `||c_tilde||_2 = 4.492995e-04`
- `||A_tilde||_F = 1.793269e-01`

### Re_42p359071 (`Re = 42.3590705675`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.573638e-03`
- `||A^p||_F = 7.059280e+00`
- `||c_tilde||_2 = 4.607299e-04`
- `||A_tilde||_F = 1.830715e-01`

### Re_43p093925 (`Re = 43.0939251612`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.702869e-03`
- `||A^p||_F = 7.190104e+00`
- `||c_tilde||_2 = 4.721941e-04`
- `||A_tilde||_F = 1.865939e-01`

### Re_43p797402 (`Re = 43.7974019747`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.826687e-03`
- `||A^p||_F = 7.316189e+00`
- `||c_tilde||_2 = 4.811914e-04`
- `||A_tilde||_F = 1.899677e-01`

### Re_44p478353 (`Re = 44.4783532118`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.949651e-03`
- `||A^p||_F = 7.437995e+00`
- `||c_tilde||_2 = 4.916726e-04`
- `||A_tilde||_F = 1.932396e-01`

### Re_45p142703 (`Re = 45.142702578`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 5.071014e-03`
- `||A^p||_F = 7.557373e+00`
- `||c_tilde||_2 = 5.017776e-04`
- `||A_tilde||_F = 1.964317e-01`

### Re_45p795194 (`Re = 45.7951936696`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 5.189682e-03`
- `||A^p||_F = 7.674995e+00`
- `||c_tilde||_2 = 5.102926e-04`
- `||A_tilde||_F = 1.995697e-01`

### Re_46p440072 (`Re = 46.440071584`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 5.309650e-03`
- `||A^p||_F = 7.790668e+00`
- `||c_tilde||_2 = 5.201069e-04`
- `||A_tilde||_F = 2.026745e-01`

总运行时间：`1032.3 s`。
