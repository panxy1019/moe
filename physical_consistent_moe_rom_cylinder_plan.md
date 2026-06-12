# 面向圆柱绕流的物理一致 MoE-ROM 可执行路线

日期：2026-06-12  
基于数据索引：`D:/学习资料/GENERATED_DATA_INDEX.md`  
基于前置文档：`pod_moe_rom_proposal.md`

暂定题名：

**Physics-consistent Shared-Routed Mixture-of-Experts for POD Reduced-Order Modeling of Cylinder Wake Flows**

## 1. 这件事应该怎么定位

你现在的方向应从“用 MoE 预测 POD 系数”升级为：

> 用 **共享专家 + 稀疏路由专家** 学习圆柱绕流 POD 系数的低维动力学，并通过 Galerkin / Operator Inference 形式的 reduced RHS、重构误差、物理可观测量和路由解释性约束，使模型具有物理一致性。

圆柱绕流是一个很合适的案例，因为它有清楚的物理结构：

- Reynolds number 控制流动强度和涡脱落频率；
- 涡脱落具有近周期性，相位信息很关键；
- 低阶 POD mode 往往成对出现，对应主涡脱落振荡；
- 高阶 POD mode 常对应谐波、局部剪切层、尾迹细节；
- 不同 Re 的 wake regime 和相位演化有差异，适合让 MoE router 学出可解释分区。

因此，模型不要只输入 `(mu, t)`。对这个数据，最有价值的输入应该是：

```text
Re / U / 1/Re / normalized time / shedding phase
+ POD modal state and history
+ phase Fourier features
+ reduced physical quantities
+ Galerkin or OpInf reduced RHS features
+ mode identity and mode energy features
```

## 2. 数据条件总结

当前推荐数据集是：

```text
/home/ray/Desktop/Cylinder_Results_Re500_1000_VTU
```

核心设置：

```text
D = 1 m
nu = 1e-3 m^2/s
Re = 500, 600, 700, 800, 900, 1000
U = Re * nu / D
solver = pimpleFoam
turbulence model = laminar
2D mesh, front/back empty
```

时间序列：

```text
estimated St = 0.2
T = D / (St * U)
each case covers 12 estimated shedding periods
about 20 saved frames per period
each Re has 241 VTU frames
first 2 cycles are discarded for POD
each Re contributes 201 retained snapshots
total retained POD snapshots = 1206
```

POD 输出：

```text
velocity POD:
phi_uv:        (80, 194736)
coeff_uv:      (1206, 80)
mean_uv_by_Re: (6, 194736)

pressure POD:
phi_p:         (80, 97368)
coeff_p:       (1206, 80)
mean_p_by_Re:  (6, 97368)
```

重构方式：

```text
uv = mean_uv_by_Re[re_index] + coeff_uv[snapshot_id, :r] @ phi_uv[:r]
p  = mean_p_by_Re[re_index]  + coeff_p[snapshot_id, :r]  @ phi_p[:r]
```

这个数据天然适合做 **global POD + Re-conditioned ROM**。但有一个需要注意的点：

> 当前 POD 是 unweighted global snapshot POD。若要做严格 intrusive Galerkin projection，最好补一个 mass-weighted POD 或至少构造离散积分权重。否则 Galerkin 内积和 POD 正交性使用的是不同度量，物理残差会变得不够干净。

## 3. `R_r` 方程好构造吗？

短答：

> 理论形式标准，不难写；但基于你现在的 `u/v/p pointData npz` 直接构造严格 intrusive Galerkin RHS 并不算“一键简单”。更稳的路线是先做 **Operator Inference 版 reduced RHS**，再逐步升级到半侵入式 Galerkin projection。

### 3.1 连续方程形式

圆柱绕流对应不可压 Navier-Stokes：

```text
∂u/∂t + (u · ∇)u = -∇p + nu Δu
∇ · u = 0
```

也可以用无量纲形式：

```text
∂u/∂t + (u · ∇)u = -∇p + (1/Re) Δu
∇ · u = 0
```

对于 POD-ROM，写：

```text
u(x,t;Re) ≈ u_bar(x;Re) + sum_{j=1}^{r_u} a_j(t;Re) phi_j(x)
p(x,t;Re) ≈ p_bar(x;Re) + sum_{m=1}^{r_p} b_m(t;Re) psi_m(x)
```

投影后典型 reduced dynamics 是：

```text
da_i/dt = c_i(Re) + sum_j A_ij(Re) a_j
          + sum_{j,k} H_ijk a_j a_k
          + pressure/boundary correction
```

如果速度基函数严格散度自由、边界处理理想，压力项可在投影中消掉或变成边界项。但你的 POD 来自 OpenFOAM point data，且速度/压力分开做 POD，所以实际中最好不要假设压力项完全消失。

### 3.2 为什么直接 intrusive 不算简单

要真正构造 Galerkin RHS，需要：

- 网格 connectivity、cell/face 面积、体积或质量矩阵；
- 梯度、散度、拉普拉斯离散算子；
- 圆柱壁面、入口、出口边界条件处理；
- POD basis 在物理内积下的正交性；
- mean field `u_bar(Re)` 的 Re 依赖；
- 压力项或压力泊松约束；
- 非均匀网格上的数值积分权重。

当前 `.npz` 里有 `points, u, v, p`，但 pointData 本身不一定包含单元连接关系和 FV 离散算子。因此：

- 用 `.npz` 直接做严格 Galerkin：不推荐；
- 用 `.vtu` 读出 mesh connectivity 后做半侵入式投影：可做；
- 从 OpenFOAM case 里复用 mesh、face flux、离散 operator：最好但工程量最大；
- 用 POD coefficients 拟合 reduced RHS：最快，适合作为第一版物理一致模型。

### 3.3 推荐三阶段 `R_r` 路线

**阶段 A：Operator Inference RHS，最快可落地**

用已有 `coeff_uv` 和时间索引估计：

```text
da/dt ≈ finite_difference(a)
```

拟合：

```text
R_opinf(a, Re) = c(Re) + A(Re)a + H(a ⊗ a)
```

其中 `c(Re), A(Re)` 可以设成 Re 的低阶多项式或小网络输出：

```text
c(Re) = c0 + c1/Re + c2/Re^2
A(Re) = A0 + (1/Re) A1
```

好处：

- 只需要现有 POD coefficients；
- 能立刻提供 `L_dyn = ||d a/dt - R_r(a,Re)||^2`；
- 可以作为 MoE 的物理残差或 baseline；
- 论文里可称为 Galerkin-inspired / operator-inferred reduced dynamics。

**阶段 B：半侵入式 Galerkin projection，中等工程量**

从 VTU 读取网格和 basis，构造近似积分：

```text
<f,g>_M = f^T M g
```

计算：

```text
c_i(Re)   = <phi_i, - (u_bar · ∇)u_bar + nu Δu_bar - ∇p_bar>
A_ij(Re) = <phi_i, - (u_bar · ∇)phi_j - (phi_j · ∇)u_bar + nu Δphi_j>
H_ijk    = <phi_i, - (phi_j · ∇)phi_k>
P_im     = <phi_i, -∇psi_m>
```

然后：

```text
da/dt = c(Re) + A(Re)a + H(a,a) + P b
```

如果同时预测压力系数 `b`，可以把 `P b` 留在 RHS 里；如果不想让压力进动力学，可以把压力项作为 residual correction 交给 MoE 学。

**阶段 C：OpenFOAM/FV fully intrusive，最高质量但最费工程**

复用 OpenFOAM 网格、边界和离散算子，把投影写得更接近高保真求解器。这条路线适合后续冲论文时增强物理说服力，但不建议作为第一版 MVP。

## 4. 推荐模型：Physical Shared-Routed MoE-ROM

我建议模型不是单纯：

```text
(Re, t) -> alpha
```

而是状态空间形式：

```text
input:  z(t) = [Re features, phase features, reduced state/history, physics features, mode token]
output: alpha(t) and/or d alpha/dt correction
```

最推荐的核心动力学形式：

```text
d a/dt = R_r(a, Re) + C_MoE(a, b, Re, phase, history, mode)
```

其中：

- `R_r` 是 Galerkin / OpInf reduced RHS，提供物理骨架；
- `C_MoE` 是 MoE correction，补偿截断误差、压力处理误差、数值投影误差和复杂 Re 依赖；
- shared expert 学公共涡脱落动力学；
- routed experts 学不同 Re、phase、mode group、瞬态局部结构。

如果暂时不做 Neural ODE，也可以先用双头监督：

```text
alpha_hat = Head_alpha(h)
adot_hat  = R_r(alpha_hat, Re) + Head_corr(h)
```

训练时同时约束 `alpha` 和 `d alpha/dt`。

## 5. 输入特征设计

### 5.1 Tier 0：一定要有的物理参数特征

对每个 snapshot，从 `pod_snapshot_index.csv` 可得到：

```text
Re, time, period, cycle, phase, local_snapshot_index
```

建议构造：

```text
Re_norm = (Re - 750) / 250
inv_Re = 1 / Re
log_Re = log(Re)
U = Re * nu / D
U_norm = U / U_ref
tau = time / T_Re
cycle = floor(tau)
phase = tau mod 1
theta = 2π phase
```

因为你的每个 Re 大约 20 帧/周期，`phase` 比原始 `time` 更有意义。不同 Re 的物理时间长度不同，但涡脱落相位是可比较的。

### 5.2 Tier 1：相位 Fourier 特征

圆柱绕流的主动力学是涡脱落周期运动，强烈建议加 Fourier phase features：

```text
sin(k theta), cos(k theta), k = 1, ..., K
```

推荐：

```text
K = 4 or 6
```

理由：

- `k=1` 捕捉主涡脱落；
- `k=2` 捕捉二次谐波，常与阻力波动相关；
- 更高 `k` 捕捉剪切层和高阶 POD mode 的细节；
- 这比直接喂 `t` 更符合周期尾迹的物理。

### 5.3 Tier 2：POD 状态与历史特征

如果目标是做时间推进 ROM，而不是单点插值，输入应包含当前或历史 POD state：

```text
a(t), a(t-dt), ..., a(t-L dt)
b(t), b(t-dt), ..., b(t-L dt)
```

推荐历史长度：

```text
L = 8 or 16
```

因为约 20 帧/周期，8 帧覆盖约 0.4 个周期，16 帧覆盖约 0.8 个周期，足够识别涡脱落相位和局部趋势。

派生动态特征：

```text
da/dt
d2a/dt2
||a_low||_2
||a_high||_2
modal energy E_r = 0.5 * sum_j a_j^2
modal pair amplitude sqrt(a_1^2 + a_2^2)
modal pair phase atan2(a_2, a_1)
```

这里 `a_1, a_2` 是否刚好成对，要通过 POD mode 频谱确认；圆柱 wake 中低阶 mode 通常会成对，但不要在代码里硬编码死。

### 5.4 Tier 3：从 POD 或场重构得到的物理可观测量

这些特征比 `(Re,t)` 更像物理模型：

```text
Kinetic energy:
E_k = 0.5 ∫ (u^2 + v^2) dx

Fluctuation energy:
E'_k = 0.5 ∫ (u'^2 + v'^2) dx

Enstrophy:
Ω = 0.5 ∫ ω^2 dx,  ω = ∂v/∂x - ∂u/∂y

Wake centerline velocity probes:
u(x_probe, y=0), v(x_probe, y=0)

Pressure difference probes:
p_front - p_back

Approximate lift/drag proxies:
surface pressure integral or near-cylinder pressure probes

Recirculation length:
distance behind cylinder where centerline u crosses zero
```

可执行性分级：

- 只用 POD coefficients 估计能量：最容易；
- 用点云/VTU 估计 probe 值：容易；
- 用 VTU connectivity 估计涡量/enstrophy：中等；
- 精确 lift/drag 需要壁面法向、压力和剪切应力，最好从 OpenFOAM functionObject 直接输出，当前 pointData 不一定够。

建议第一版使用：

```text
modal energy
low/high mode energy ratio
phase portrait features
selected wake probes
pressure front/back proxy
```

后续再加：

```text
enstrophy
lift/drag
recirculation length
```

### 5.5 Tier 4：Galerkin / OpInf RHS 特征

如果已经有 `R_r(a,Re)`，不要只把它放在 loss 里，也可以把它作为输入上下文：

```text
R_r(a,Re)
||R_r(a,Re)||
a^T R_r(a,Re)
R_r low-mode norm
R_r high-mode norm
```

这些特征能告诉 router：

- 当前动力学变化快不快；
- 是否处于强非线性阶段；
- 高阶模式是否被激活；
- Galerkin 骨架在哪些区域可能不够准。

注意：

> 如果最终模型要做真正在线预测，输入特征必须能由当前 reduced state 和参数计算出来，不能依赖真实未来场。

### 5.6 Tier 5：mode token 特征

如果采用 mode-aware MoE，每个 mode token 输入：

```text
mode_id
field_type = velocity or pressure
mode_energy_ratio
cumulative_energy
singular_value
mode_group = low / middle / high
dominant_frequency
optional parity/pair id
```

对于 velocity 和 pressure，可以有两种设计：

1. **统一 token 池**  
   160 个 token：80 个 velocity coefficient token + 80 个 pressure coefficient token。

2. **双分支 token 池**  
   velocity branch 预测 `a`，pressure branch 预测 `b`，中间用 cross-attention 或 shared context 连接。

推荐第一版使用双分支，因为压力的物理角色和速度不同。

## 6. 多层 Shared-Routed MoE Layer 设计

### 6.1 为什么要同时有 shared expert 和 routed experts

只用 routed experts 容易出现：

- 专家塌缩；
- 小数据下路由不稳定；
- 每个专家只看到少量样本，泛化差；
- 不同 Re 之间公共涡脱落结构学不到。

shared expert 的作用是：

- 捕捉所有 Re 共享的 wake oscillator 动力学；
- 给模型一个稳定的 dense path；
- 保证即使 router 初期不准，模型也能学习；
- routed experts 只需要学习 regime-specific correction。

因此推荐每层都使用：

```text
h_{l+1} = h_l
        + SharedExpert_l(h_l, context)
        + RoutedMoE_l(h_l, context)
```

其中：

```text
RoutedMoE_l(h) = sum_{e in TopK(g_l(h, context))} w_e Expert_{l,e}(h)
```

### 6.2 一层 MoE block 的结构

建议：

```text
LayerNorm
Context-conditioned self/cross attention, optional
Shared FFN Expert, always active
Top-k Routed Experts, sparse active
Residual connection
LayerNorm
Small FFN or gated linear unit
```

router 输入：

```text
[token hidden, global physical context, mode features, R_r features]
```

router 输出：

```text
expert logits -> top-k -> normalized weights
```

推荐参数：

```text
num_routed_experts E = 8
top_k = 2
num_shared_experts = 1 or 2
num_moe_layers = 3
hidden_dim = 128 or 192
expert_hidden_dim = 128
router_temperature warmup: 2.0 -> 0.7
```

你的数据只有 1206 个 POD snapshots，模型不要太大。`E=8, top_k=2, 3 layers` 已经足够做论文级实验。

### 6.3 多层 MoE 应该怎么分工

可以把多层设计成不同抽象层：

```text
MoE Layer 1: mode-aware local feature routing
MoE Layer 2: Re/phase regime routing
MoE Layer 3: output correction routing
```

解释：

- 第一层让低阶/高阶/压力/速度 mode 进入不同专家组合；
- 第二层让不同 Re 和涡脱落相位进入不同专家组合；
- 第三层靠近输出，学习 Galerkin residual correction 或 coefficient refinement。

这样比只放一个 MoE head 更有表达力，也更容易做可解释性分析。

### 6.4 稀疏激活策略

训练策略：

```text
Epoch 0-20%: soft routing or high-temperature top-2
Epoch 20-80%: top-2 routing + load balancing
Epoch 80-100%: lower temperature, optional top-1 evaluation
```

推荐 loss：

```text
L_balance = coefficient of variation of expert load
L_importance = encourage balanced routing probabilities
L_router_smooth = ||g(Re, phase_t) - g(Re, phase_{t+1})||
L_entropy = avoid too-early hard collapse
```

圆柱绕流中 router 应该随相位平滑变化，除非经过某些强非线性事件。`router_smooth` 很有用。

## 7. 傅里叶层到底要不要？

结论：

> 不建议把完整 FNO 当主干，因为你的空间场已经被 POD 压成低维系数，而且网格是 unstructured point cloud。更合理的是使用 **Fourier phase features** 和可选的 **1D temporal Fourier mixing**。

### 7.1 一定建议保留：Fourier phase encoding

这是最便宜、最有效、最符合圆柱尾迹周期性的做法：

```text
gamma_phase(theta) =
[sin(theta), cos(theta), ..., sin(K theta), cos(K theta)]
```

这不是装饰性技巧，而是物理先验：涡脱落就是近周期系统。

### 7.2 可选：temporal Fourier mixing layer

如果输入历史窗口：

```text
[a(t-Ldt), ..., a(t)]
```

可以加一个轻量 1D spectral mixer：

```text
FFT over time window -> keep low temporal modes -> inverse FFT -> latent
```

适合捕捉周期和谐波。但第一版可以先不要，因为：

- 数据量小；
- 20 帧/周期已经不长；
- GRU/TCN/MLP 加 phase encoding 可能足够。

建议实验中作为 ablation：

```text
without temporal Fourier mixer
with temporal Fourier mixer
```

### 7.3 不建议第一版使用：spatial FNO

FNO 适合输入/输出规则网格或可规整化场变量。你的模型目标是 POD coefficients，空间结构已经在 `phi_uv, phi_p` 中。直接上 spatial FNO 会让故事变混乱：

- 它绕开了 POD-ROM 的核心设定；
- unstructured mesh 需要额外插值或 graph/FV operator；
- 训练数据只有 1206 snapshots；
- 评审会问为什么不用神经算子直接学场。

所以第一版保持：

```text
POD basis handles space
MoE handles reduced coefficient dynamics
Fourier features handle phase/time periodicity
Galerkin/OpInf handles physical RHS
```

## 8. 输出层怎么设计

### 8.1 方案 A：直接预测系数

```text
[context, mode token] -> alpha_j or beta_j
```

优点：

- 简单；
- 训练稳定；
- 适合 snapshot interpolation。

缺点：

- 时间推进物理性弱；
- 长期 rollout 不一定稳定；
- `R_r` 只能作为辅助 loss。

建议作为 baseline。

### 8.2 方案 B：预测 RHS correction

```text
da/dt = R_r(a,Re) + C_MoE(a,Re,phase,history)
```

优点：

- 最符合物理一致 ROM；
- Galerkin/OpInf 给出可解释骨架；
- MoE 只学习缺失项和复杂 regime correction；
- 可以做长期 rollout。

缺点：

- 需要数值积分；
- 对 `d a/dt` 估计质量敏感；
- 训练实现稍复杂。

这是最终最推荐的主模型。

### 8.3 方案 C：双头输出，第一版最稳

训练时同时输出：

```text
alpha_hat
adot_hat = R_r(alpha_hat, Re) + C_MoE(...)
```

loss 同时约束：

```text
L_alpha = ||alpha_hat - alpha||
L_adot  = ||adot_hat - finite_difference(alpha)||
L_cons  = ||d alpha_hat/dt - adot_hat||
```

优点：

- 不必一开始就完全依赖积分 rollout；
- 又能把物理 RHS 纳入模型；
- 适合从 direct prediction 过渡到 Neural ODE/RHS model。

推荐第一篇实验主模型采用方案 C。

### 8.4 速度与压力输出

建议：

```text
velocity branch: predict a_uv and/or da_uv/dt
pressure branch: predict b_p
```

压力可以这样处理：

- 第一版：压力系数作为额外监督输出；
- 第二版：用 velocity latent + Re + phase 预测 pressure coefficients；
- 第三版：加入 pressure Poisson residual 或投影压力项。

不要一开始就把压力强行塞进 velocity Galerkin RHS，除非已经能可靠计算 `-∇p` 投影项。

## 9. Loss 设计

总 loss：

```text
L = L_coeff
  + λ_rec L_rec
  + λ_dyn L_dyn
  + λ_roll L_rollout
  + λ_phys L_phys
  + λ_router L_router
```

### 9.1 系数损失

```text
L_coeff = ||a_hat - a||^2 + η ||b_hat - b||^2
```

建议对每个 mode 标准化后计算 MSE，避免低阶系数完全支配训练。

### 9.2 重构损失

```text
u_hat = u_bar(Re) + Phi_u a_hat
p_hat = p_bar(Re) + Phi_p b_hat

L_rec = ||u_hat - u||_M^2 + η_p ||p_hat - p||_M^2
```

如果暂时没有质量矩阵 `M`，先用 unweighted L2；但文档和论文中应说明当前 POD 是 unweighted。

### 9.3 动力学一致性损失

```text
L_dyn = ||D_t a - [R_r(a,Re) + C_MoE(a,Re,...)]||^2
```

其中 `D_t a` 用有限差分或 Savitzky-Golay 平滑差分计算。

### 9.4 Galerkin residual correction 正则

希望 MoE 不要完全覆盖 `R_r`，而是学习小修正：

```text
L_corr = ||C_MoE||^2
```

也可以做相对约束：

```text
||C_MoE|| / (||R_r|| + eps)
```

这能让论文叙事更稳：

> Galerkin/OpInf provides the dominant dynamics, while MoE learns regime-dependent closure corrections.

### 9.5 rollout loss

用模型积分 H 步：

```text
a_hat(t+h) = Integrator(a(t), R_r + C_MoE)
L_roll = sum_{h=1}^H ||a_hat(t+h) - a(t+h)||^2
```

推荐：

```text
H = 5, 10, 20
```

20 步约一个涡脱落周期，非常适合圆柱 wake。

### 9.6 router loss

```text
L_router = L_balance + L_importance + L_smooth + L_entropy
```

其中：

- `L_balance` 防止专家塌缩；
- `L_smooth` 防止相邻相位疯狂切换专家；
- `L_entropy` 前期防止 gate 过早变硬；
- 后期可降低 entropy 权重，让专家分工更清晰。

## 10. 具体可执行实验路线

### Step 1：整理训练表

从 `pod_snapshot_index.csv` 构造每行样本：

```text
snapshot_id
Re
time
period
cycle
phase
local_snapshot_index
coeff_uv[80]
coeff_p[80]
```

额外计算：

```text
U
1/Re
theta = 2π phase
Fourier phase features
normalized coefficient a_std, b_std
finite difference da/dt, db/dt
modal energy features
history windows
```

### Step 2：先做 baseline

必须有这些 baseline：

```text
POD + MLP(Re, phase)
POD + MLP(Re, phase, history)
POD + GRU/TCN(history)
POD + OpInf R_r rollout
POD + local/clustered model by Re or phase
```

这样才能证明 MoE 不是复杂但没必要。

### Step 3：构造 `R_r`

第一版：

```text
fit R_opinf(a, Re)
```

形式：

```text
R_opinf(a, Re) = c0 + c1/Re + (A0 + A1/Re)a + H(a⊗a)
```

如果二次项参数太多，可以：

- 只对前 `r_dyn = 20 or 40` 个 velocity modes 建二次 RHS；
- 对高阶 modes 用线性或 MoE correction；
- 使用 ridge/LASSO 正则；
- 对 `H` 做低秩分解。

第二版再做：

```text
semi-intrusive Galerkin tensors from VTU mesh
```

### Step 4：实现主模型

建议第一版配置：

```text
r_u = 40 or 80
r_p = 20 or 40
history_len = 8 or 16
phase_harmonics K = 6
hidden_dim = 128
num_moe_layers = 3
routed_experts = 8
shared_experts = 1
top_k = 2
```

模型结构：

```text
PhysicalContextEncoder
  input: Re features + phase Fourier + modal history + energy + R_r features

VelocityModeTokenizer
  input: mode_id + singular value + mode energy + field_type

PressureModeTokenizer
  input: mode_id + singular value + mode energy + field_type

Shared-Routed MoE Blocks x 3

Output heads
  alpha_head
  alpha_rhs_correction_head
  pressure_head
```

### Step 5：训练策略

训练阶段：

```text
Stage 1: train direct alpha/pressure prediction
Stage 2: add derivative loss with frozen or weak R_r
Stage 3: add rollout loss
Stage 4: reduce router temperature, analyze experts
```

建议不要一开始就全 loss 打满，否则很难调参。

### Step 6：测试划分

推荐几种划分：

```text
Interpolation in Re:
train Re = 500, 600, 800, 900, 1000
test  Re = 700

Harder interpolation:
train Re = 500, 700, 900, 1000
test  Re = 600, 800

Extrapolation:
train Re = 500, 600, 700, 800, 900
test  Re = 1000

Time generalization:
train cycles 3-10
test cycles 11-12
```

圆柱 wake 的周期性比较强，只做随机 snapshot split 会太容易。一定要做 leave-one-Re-out 和 long rollout。

### Step 7：评价指标

系数层面：

```text
relative L2 error of a
relative L2 error of b
mode-wise error
phase error of leading mode pair
frequency spectrum error
```

场层面：

```text
velocity reconstruction error
pressure reconstruction error
pointwise max error
wake probe time-series error
```

动力学层面：

```text
rollout error over 1, 3, 6 shedding periods
energy drift
limit-cycle amplitude error
dominant Strouhal frequency error
```

MoE 层面：

```text
expert utilization
router entropy
expert-Re heatmap
expert-phase heatmap
expert-mode heatmap
shared vs routed contribution ratio
```

## 11. 可解释性：这篇工作最该讲什么

MoE 的解释性可以围绕三个问题：

### 11.1 专家是否按 Re 分工？

画：

```text
Re -> expert usage
```

如果 Re=500/600 和 Re=900/1000 的专家分布不同，这说明 router 学到了流动强度差异。

### 11.2 专家是否按涡脱落相位分工？

画：

```text
phase -> expert usage
```

如果某些专家集中在涡生成、脱落、尾迹翻转阶段，很有物理意义。

### 11.3 专家是否按 POD mode 分工？

画：

```text
mode index -> expert usage
low/mid/high energy modes -> expert usage
velocity vs pressure modes -> expert usage
```

理想情况：

- shared expert 负责主周期；
- routed expert A 负责低阶 velocity pair；
- routed expert B 负责压力响应；
- routed expert C 负责高阶谐波；
- routed expert D 负责高 Re 的局部剪切层修正。

## 12. 推荐 ablation

必须做：

```text
MLP vs MoE
without R_r vs with R_r
routed only vs shared + routed
single MoE layer vs 3 MoE layers
without Fourier phase features vs with Fourier phase features
without router smoothness vs with router smoothness
top-1 vs top-2
E = 4, 8, 16
direct alpha head vs RHS correction head vs dual head
```

如果时间允许：

```text
unweighted POD vs mass-weighted POD
OpInf R_r vs semi-intrusive Galerkin R_r
velocity-only vs velocity-pressure coupled
history-free vs history-aware
```

## 13. 风险和处理

| 风险 | 原因 | 建议 |
|---|---|---|
| 数据量小，MoE 过拟合 | 只有 1206 个 POD snapshots | 小 expert、共享 expert、强正则、leave-one-Re-out |
| router 塌缩 | top-k 太早变硬 | soft warmup、load balance、temperature annealing |
| intrusive Galerkin 工程量大 | 需要网格、算子、边界、质量矩阵 | 先做 OpInf `R_r`，再升级半侵入式 |
| 压力项处理复杂 | velocity/pressure 分开 POD | 第一版压力监督输出，后续加压力投影或 Poisson residual |
| unweighted POD 与 Galerkin 内积不一致 | 当前 global POD 是 unweighted | 论文中说明，后续补 mass-weighted POD |
| 长期 rollout 发散 | 单步监督不保证稳定 | rollout loss、energy regularization、RHS correction 小范数 |
| Fourier 层过度复杂 | 数据量小，POD 已降维 | 先用 Fourier phase encoding，temporal Fourier mixer 做 ablation |

## 14. 我建议的最终主线

最稳的研究路线是：

1. 用当前 global POD 数据建立 velocity/pressure coefficient 数据表。
2. 先拟合一个 OpInf reduced RHS `R_r(a,Re)`，作为 Galerkin 物理骨架的第一版。
3. 设计一个 **PhysicalContextEncoder + 多层 Shared-Routed MoE + dual output heads**。
4. 输入不仅包含 `Re,t`，还包含 `phase Fourier features、history coefficients、modal energy、R_r features、mode token features`。
5. 输出同时预测 `alpha` 和 `RHS correction`，训练中加入 coefficient loss、reconstruction loss、dynamic residual loss、rollout loss 和 router loss。
6. 用 leave-one-Re-out 和 long rollout 验证泛化。
7. 用 expert usage heatmap 证明专家按 Re、phase、mode、velocity/pressure 形成物理可解释分工。
8. 后续把 `R_r` 从 OpInf 升级到半侵入式 Galerkin projection，提高物理一致性和论文说服力。

## 15. 一句话结论

这个课题最值得做的点不是“POD 后面接 MoE”，而是：

> 以 Galerkin/OpInf reduced dynamics 为物理骨架，用 shared expert 学全局 wake oscillator，用 routed experts 学 Re-phase-mode dependent closure correction，再用多层稀疏路由和可解释性分析证明专家分工对应真实圆柱尾迹物理。

这样做，模型有清楚的物理含义、可执行的训练路线，也比单纯 `(Re,t)->POD coefficients` 的黑箱回归更像一篇能投 JCP/CMAME/计算力学方向的工作。

