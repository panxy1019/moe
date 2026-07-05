# Pressure Poisson Surrogate Galerkin Tensors

## 数据来源

- 数据目录：`/home/ray/Desktop/Cylinder_ROM_PhysicsGeneralizable_Re20_200_100Re/Regime_ROM_Library/periodic`
- POD 目录：`/home/ray/Desktop/Cylinder_ROM_PhysicsGeneralizable_Re20_200_100Re/Regime_ROM_Library/periodic/Global_POD_AreaWeighted_L2`
- 网格模板：`/home/ray/Desktop/Cylinder_Results/Re_100_VTK/run_Re_100_7107.vtk`
- 输出文件：`/home/ray/Desktop/Cylinder_ROM_PhysicsGeneralizable_Re20_200_100Re/Regime_ROM_Library/pressure_poisson_surrogate_periodic.npz`
- 计算 Re 数量：`63`
- 计算 Re 标签：`['Re_60p307745', 'Re_61p755954', 'Re_63p499817', 'Re_65p259829', 'Re_66p970112', 'Re_68p649714', 'Re_70p314635', 'Re_71p972931', 'Re_73p628287', 'Re_75p282340', 'Re_76p935803', 'Re_78p588971', 'Re_80p241943', 'Re_81p894708', 'Re_83p547164', 'Re_85p199103', 'Re_86p850168', 'Re_88p499815', 'Re_90p147341', 'Re_91p792204', 'Re_93p435204', 'Re_95p081752', 'Re_96p749308', 'Re_98p480345', 'Re_100p352251', 'Re_102p440042', 'Re_104p710911', 'Re_107p050737', 'Re_109p395985', 'Re_111p734011', 'Re_114p066308', 'Re_116p395488', 'Re_118p723173', 'Re_121p050171', 'Re_123p376833', 'Re_125p703274', 'Re_128p029461', 'Re_130p355225', 'Re_132p680203', 'Re_135p003744', 'Re_137p324830', 'Re_139p642302', 'Re_141p956319', 'Re_144p273459', 'Re_146p619578', 'Re_149p059229', 'Re_151p686208', 'Re_154p520852', 'Re_157p459588', 'Re_160p415176', 'Re_163p364123', 'Re_166p306373', 'Re_169p244893', 'Re_172p181708', 'Re_175p117940', 'Re_178p054368', 'Re_180p992055', 'Re_183p933395', 'Re_186p884600', 'Re_189p862278', 'Re_192p911664', 'Re_196p160723', 'Re_200p000000']`

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
- `L` singular cutoff = `7.097875e-09`
- `L` condition estimate = `3.903254e+03`
- `||H_p||_F = 5.514492e+01`
- `||H_tilde||_F = 7.912246e+00`

## 等效代数代理系统

脚本保存 `L_pinv`，并已左乘得到最终等效张量：

```text
c_tilde = L_pinv c^p
A_tilde = L_pinv A^p
H_tilde[:,j,k] = L_pinv H^p[:,j,k]
b(t) = c_tilde + A_tilde a(t) + H_tilde(a(t),a(t))
```

## 本次运行结果

### Re_60p307745 (`Re = 60.3077454666`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 7.312116e-04`
- `||A^p||_F = 1.075489e+00`
- `||c_tilde||_2 = 1.474983e-03`
- `||A_tilde||_F = 1.267559e-01`

### Re_61p755954 (`Re = 61.7559536282`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 8.573373e-04`
- `||A^p||_F = 1.038900e+00`
- `||c_tilde||_2 = 1.404951e-03`
- `||A_tilde||_F = 1.245148e-01`

### Re_63p499817 (`Re = 63.499816815`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 8.479101e-04`
- `||A^p||_F = 1.109725e+00`
- `||c_tilde||_2 = 1.486155e-03`
- `||A_tilde||_F = 1.315250e-01`

### Re_65p259829 (`Re = 65.2598292316`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 1.150363e-03`
- `||A^p||_F = 1.023515e+00`
- `||c_tilde||_2 = 1.407612e-03`
- `||A_tilde||_F = 1.266636e-01`

### Re_66p970112 (`Re = 66.9701121204`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 1.064731e-03`
- `||A^p||_F = 1.118170e+00`
- `||c_tilde||_2 = 1.466933e-03`
- `||A_tilde||_F = 1.352402e-01`

### Re_68p649714 (`Re = 68.6497139288`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 1.509026e-03`
- `||A^p||_F = 1.012765e+00`
- `||c_tilde||_2 = 1.538061e-03`
- `||A_tilde||_F = 1.297447e-01`

### Re_70p314635 (`Re = 70.3146353337`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 1.709996e-03`
- `||A^p||_F = 1.008887e+00`
- `||c_tilde||_2 = 1.653667e-03`
- `||A_tilde||_F = 1.314927e-01`

### Re_71p972931 (`Re = 71.9729308789`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 1.926019e-03`
- `||A^p||_F = 1.006073e+00`
- `||c_tilde||_2 = 1.801855e-03`
- `||A_tilde||_F = 1.333633e-01`

### Re_73p628287 (`Re = 73.6282869994`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.084764e-03`
- `||A^p||_F = 1.021797e+00`
- `||c_tilde||_2 = 1.918014e-03`
- `||A_tilde||_F = 1.364312e-01`

### Re_75p282340 (`Re = 75.2823402819`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.326194e-03`
- `||A^p||_F = 1.021348e+00`
- `||c_tilde||_2 = 2.111336e-03`
- `||A_tilde||_F = 1.384593e-01`

### Re_76p935803 (`Re = 76.9358029334`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.585015e-03`
- `||A^p||_F = 1.021715e+00`
- `||c_tilde||_2 = 2.332112e-03`
- `||A_tilde||_F = 1.405509e-01`

### Re_78p588971 (`Re = 78.5889708164`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.860904e-03`
- `||A^p||_F = 1.023001e+00`
- `||c_tilde||_2 = 2.578223e-03`
- `||A_tilde||_F = 1.427150e-01`

### Re_80p241943 (`Re = 80.2419430177`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 3.256346e-03`
- `||A^p||_F = 1.007312e+00`
- `||c_tilde||_2 = 2.953272e-03`
- `||A_tilde||_F = 1.439327e-01`

### Re_81p894708 (`Re = 81.8947080155`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 3.576484e-03`
- `||A^p||_F = 1.010688e+00`
- `||c_tilde||_2 = 3.247292e-03`
- `||A_tilde||_F = 1.461974e-01`

### Re_83p547164 (`Re = 83.5471642516`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 3.797722e-03`
- `||A^p||_F = 1.031718e+00`
- `||c_tilde||_2 = 3.442798e-03`
- `||A_tilde||_F = 1.494837e-01`

### Re_85p199103 (`Re = 85.1991031321`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.147778e-03`
- `||A^p||_F = 1.036434e+00`
- `||c_tilde||_2 = 3.764818e-03`
- `||A_tilde||_F = 1.518290e-01`

### Re_86p850168 (`Re = 86.8501684903`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.516540e-03`
- `||A^p||_F = 1.042196e+00`
- `||c_tilde||_2 = 4.099505e-03`
- `||A_tilde||_F = 1.542135e-01`

### Re_88p499815 (`Re = 88.4998153335`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 5.029125e-03`
- `||A^p||_F = 1.035763e+00`
- `||c_tilde||_2 = 4.546339e-03`
- `||A_tilde||_F = 1.556719e-01`

### Re_90p147341 (`Re = 90.1473406471`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 5.312734e-03`
- `||A^p||_F = 1.056451e+00`
- `||c_tilde||_2 = 4.802620e-03`
- `||A_tilde||_F = 1.590464e-01`

### Re_91p792204 (`Re = 91.792204085`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 5.742095e-03`
- `||A^p||_F = 1.064575e+00`
- `||c_tilde||_2 = 5.170527e-03`
- `||A_tilde||_F = 1.614470e-01`

### Re_93p435204 (`Re = 93.4352043333`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 6.189890e-03`
- `||A^p||_F = 1.073482e+00`
- `||c_tilde||_2 = 5.548076e-03`
- `||A_tilde||_F = 1.638433e-01`

### Re_95p081752 (`Re = 95.0817519005`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 6.788492e-03`
- `||A^p||_F = 1.073651e+00`
- `||c_tilde||_2 = 6.001338e-03`
- `||A_tilde||_F = 1.655308e-01`

### Re_96p749308 (`Re = 96.749307569`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 7.292718e-03`
- `||A^p||_F = 1.084446e+00`
- `||c_tilde||_2 = 6.395620e-03`
- `||A_tilde||_F = 1.680408e-01`

### Re_98p480345 (`Re = 98.4803451202`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 7.843310e-03`
- `||A^p||_F = 1.095924e+00`
- `||c_tilde||_2 = 6.817589e-03`
- `||A_tilde||_F = 1.706093e-01`

### Re_100p352251 (`Re = 100.352251335`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 8.467135e-03`
- `||A^p||_F = 1.108792e+00`
- `||c_tilde||_2 = 7.287285e-03`
- `||A_tilde||_F = 1.733546e-01`

### Re_102p440042 (`Re = 102.440041809`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 9.195450e-03`
- `||A^p||_F = 1.123690e+00`
- `||c_tilde||_2 = 7.826430e-03`
- `||A_tilde||_F = 1.763635e-01`

### Re_104p710911 (`Re = 104.710910987`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 1.001438e-02`
- `||A^p||_F = 1.140966e+00`
- `||c_tilde||_2 = 8.429873e-03`
- `||A_tilde||_F = 1.796432e-01`

### Re_107p050737 (`Re = 107.050736557`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 1.090909e-02`
- `||A^p||_F = 1.158889e+00`
- `||c_tilde||_2 = 9.070787e-03`
- `||A_tilde||_F = 1.829553e-01`

### Re_109p395985 (`Re = 109.395985153`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 1.167272e-02`
- `||A^p||_F = 1.183311e+00`
- `||c_tilde||_2 = 9.694540e-03`
- `||A_tilde||_F = 1.872681e-01`

### Re_111p734011 (`Re = 111.734011486`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 1.283804e-02`
- `||A^p||_F = 1.196016e+00`
- `||c_tilde||_2 = 1.041368e-02`
- `||A_tilde||_F = 1.895141e-01`

### Re_114p066308 (`Re = 114.066307867`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 1.388088e-02`
- `||A^p||_F = 1.214642e+00`
- `||c_tilde||_2 = 1.111048e-02`
- `||A_tilde||_F = 1.927016e-01`

### Re_116p395488 (`Re = 116.395488165`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 1.484681e-02`
- `||A^p||_F = 1.237137e+00`
- `||c_tilde||_2 = 1.180607e-02`
- `||A_tilde||_F = 1.966884e-01`

### Re_118p723173 (`Re = 118.723173369`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 1.607505e-02`
- `||A^p||_F = 1.254633e+00`
- `||c_tilde||_2 = 1.254588e-02`
- `||A_tilde||_F = 1.994717e-01`

### Re_121p050171 (`Re = 121.05017082`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 1.708126e-02`
- `||A^p||_F = 1.278046e+00`
- `||c_tilde||_2 = 1.328473e-02`
- `||A_tilde||_F = 2.034001e-01`

### Re_123p376833 (`Re = 123.376832843`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 1.844282e-02`
- `||A^p||_F = 1.295087e+00`
- `||c_tilde||_2 = 1.404357e-02`
- `||A_tilde||_F = 2.064506e-01`

### Re_125p703274 (`Re = 125.703273692`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 1.977041e-02`
- `||A^p||_F = 1.314141e+00`
- `||c_tilde||_2 = 1.485364e-02`
- `||A_tilde||_F = 2.087472e-01`

### Re_128p029461 (`Re = 128.029461168`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.093077e-02`
- `||A^p||_F = 1.337368e+00`
- `||c_tilde||_2 = 1.565624e-02`
- `||A_tilde||_F = 2.126929e-01`

### Re_130p355225 (`Re = 130.355224828`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.246655e-02`
- `||A^p||_F = 1.355588e+00`
- `||c_tilde||_2 = 1.647337e-02`
- `||A_tilde||_F = 2.154242e-01`

### Re_132p680203 (`Re = 132.680202796`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.397685e-02`
- `||A^p||_F = 1.374467e+00`
- `||c_tilde||_2 = 1.733798e-02`
- `||A_tilde||_F = 2.178328e-01`

### Re_135p003744 (`Re = 135.003743604`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.557082e-02`
- `||A^p||_F = 1.393700e+00`
- `||c_tilde||_2 = 1.818251e-02`
- `||A_tilde||_F = 2.203928e-01`

### Re_137p324830 (`Re = 137.324829556`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.684993e-02`
- `||A^p||_F = 1.418789e+00`
- `||c_tilde||_2 = 1.909167e-02`
- `||A_tilde||_F = 2.245415e-01`

### Re_139p642302 (`Re = 139.642302099`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 2.860956e-02`
- `||A^p||_F = 1.436709e+00`
- `||c_tilde||_2 = 1.998990e-02`
- `||A_tilde||_F = 2.268362e-01`

### Re_141p956319 (`Re = 141.956318757`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 3.033994e-02`
- `||A^p||_F = 1.456831e+00`
- `||c_tilde||_2 = 2.092424e-02`
- `||A_tilde||_F = 2.292691e-01`

### Re_144p273459 (`Re = 144.273459297`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 3.215523e-02`
- `||A^p||_F = 1.475758e+00`
- `||c_tilde||_2 = 2.187021e-02`
- `||A_tilde||_F = 2.313407e-01`

### Re_146p619578 (`Re = 146.619578296`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 3.366592e-02`
- `||A^p||_F = 1.501017e+00`
- `||c_tilde||_2 = 2.286546e-02`
- `||A_tilde||_F = 2.359780e-01`

### Re_149p059229 (`Re = 149.059229449`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 3.570757e-02`
- `||A^p||_F = 1.520828e+00`
- `||c_tilde||_2 = 2.394162e-02`
- `||A_tilde||_F = 2.375070e-01`

### Re_151p686208 (`Re = 151.686208001`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 3.799229e-02`
- `||A^p||_F = 1.543928e+00`
- `||c_tilde||_2 = 2.506952e-02`
- `||A_tilde||_F = 2.404240e-01`

### Re_154p520852 (`Re = 154.520851959`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.060602e-02`
- `||A^p||_F = 1.565720e+00`
- `||c_tilde||_2 = 2.638802e-02`
- `||A_tilde||_F = 2.424874e-01`

### Re_157p459588 (`Re = 157.45958766`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.276360e-02`
- `||A^p||_F = 1.598020e+00`
- `||c_tilde||_2 = 2.775332e-02`
- `||A_tilde||_F = 2.476348e-01`

### Re_160p415176 (`Re = 160.415175616`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.570850e-02`
- `||A^p||_F = 1.621124e+00`
- `||c_tilde||_2 = 2.921420e-02`
- `||A_tilde||_F = 2.495010e-01`

### Re_163p364123 (`Re = 163.364122702`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.854858e-02`
- `||A^p||_F = 1.648246e+00`
- `||c_tilde||_2 = 3.065344e-02`
- `||A_tilde||_F = 2.522681e-01`

### Re_166p306373 (`Re = 166.306372744`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 5.171681e-02`
- `||A^p||_F = 1.671431e+00`
- `||c_tilde||_2 = 3.218194e-02`
- `||A_tilde||_F = 2.544826e-01`

### Re_169p244893 (`Re = 169.244893107`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 5.468611e-02`
- `||A^p||_F = 1.698361e+00`
- `||c_tilde||_2 = 3.374048e-02`
- `||A_tilde||_F = 2.569130e-01`

### Re_172p181708 (`Re = 172.181708206`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 5.729455e-02`
- `||A^p||_F = 1.729126e+00`
- `||c_tilde||_2 = 3.532370e-02`
- `||A_tilde||_F = 2.620478e-01`

### Re_175p117940 (`Re = 175.117940142`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 6.082157e-02`
- `||A^p||_F = 1.752768e+00`
- `||c_tilde||_2 = 3.697489e-02`
- `||A_tilde||_F = 2.635895e-01`

### Re_178p054368 (`Re = 178.05436806`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 6.423318e-02`
- `||A^p||_F = 1.777716e+00`
- `||c_tilde||_2 = 3.867678e-02`
- `||A_tilde||_F = 2.656465e-01`

### Re_180p992055 (`Re = 180.992054605`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 6.776778e-02`
- `||A^p||_F = 1.802661e+00`
- `||c_tilde||_2 = 4.044974e-02`
- `||A_tilde||_F = 2.676763e-01`

### Re_183p933395 (`Re = 183.933394636`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 7.146846e-02`
- `||A^p||_F = 1.827551e+00`
- `||c_tilde||_2 = 4.225840e-02`
- `||A_tilde||_F = 2.703346e-01`

### Re_186p884600 (`Re = 186.884600344`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 7.511616e-02`
- `||A^p||_F = 1.852198e+00`
- `||c_tilde||_2 = 4.414745e-02`
- `||A_tilde||_F = 2.720878e-01`

### Re_189p862278 (`Re = 189.86227838`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 7.836206e-02`
- `||A^p||_F = 1.885049e+00`
- `||c_tilde||_2 = 4.600361e-02`
- `||A_tilde||_F = 2.766790e-01`

### Re_192p911664 (`Re = 192.911663952`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 8.286766e-02`
- `||A^p||_F = 1.906616e+00`
- `||c_tilde||_2 = 4.806420e-02`
- `||A_tilde||_F = 2.778877e-01`

### Re_196p160723 (`Re = 196.160723205`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 8.692734e-02`
- `||A^p||_F = 1.940460e+00`
- `||c_tilde||_2 = 5.029279e-02`
- `||A_tilde||_F = 2.812782e-01`

### Re_200p000000 (`Re = 200`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 9.291165e-02`
- `||A^p||_F = 1.967642e+00`
- `||c_tilde||_2 = 5.307921e-02`
- `||A_tilde||_F = 2.839886e-01`

总运行时间：`1136.2 s`。
