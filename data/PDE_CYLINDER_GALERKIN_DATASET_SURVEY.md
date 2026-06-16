# PDE / 二维圆柱绕流 / Galerkin 张量数据集调研

检索日期：2026-06-16  
目标：寻找适合顶会论文基准、二维圆柱绕流、并且适合提取半侵入式 POD-Galerkin 张量的数据集。

## 结论先行

如果目标是**精准提取 Galerkin 投影张量**，最重要的不是数据集名气，而是数据里是否同时有：

1. 速度场 `u, v`，最好可扩展成 `(N, 3)`。
2. 压力场 `p`，用于压力项或压力泊松代理。
3. 网格节点坐标和单元连接，或至少规则网格坐标。
4. 可构造质量权重 `mass_weights` 或面积/体积权重。
5. 足够长的时间快照，最好同一条轨迹内网格固定。
6. 明确的边界、黏度/Reynolds 数、时间步长。

综合“顶会认可度 + 圆柱绕流 + Galerkin 可用性”，优先级建议如下：

| 优先级 | 数据集 | 推荐用途 | Galerkin 适配判断 |
|---|---|---|---|
| S | MeshGraphNets `cylinder_flow` | 顶会/图网格基准、二维圆柱、速度压力预测 | 很强：官方 `meta.json` 明确有 `cells, mesh_pos, velocity, pressure` |
| S- | CFDBench `cylinder` raw | 大规模 CFD 泛化基准、参数变化 | 强：raw 数据含坐标、速度、压力；但需确认每个 case 的网格组织和单元连接 |
| A | Zenodo fixed cylinder Re=100 | 经典圆柱涡街、压力速度全场 | 中强：有节点坐标/速度/压力，但公开解析脚本未显示单元连接，需 Delaunay 或另找网格 |
| A | DatasetMeca oscillating cylinder Re=185 | 受迫振荡圆柱、DNS 快照 | 中强：规则/结构化网格、U/V/P 快照；外部访问需申请 |
| B+ | PDEBench | 顶会 SciML baseline | 论文对比强，但不是圆柱绕流；不建议作为 Galerkin 主数据 |
| B | Nonlinear Benchmark unsteady fluid mechanics | 系统辨识、受迫圆柱 | 31x31 wake 网格较粗，适合 ROM/控制对比，不适合精确空间积分 |
| B | Re=3900 cylinder DNS wake dataset | 3D DNS 圆柱尾迹 | 高保真但不是二维全域；可做 2D 切片/平均后的扩展实验 |
| C | Raissi PINNs `cylinder_nektar_wake.mat` | PINN toy baseline | 数据小且经典，但缺网格单元/质量矩阵；只适合快速 smoke test |

## 推荐 1：MeshGraphNets CylinderFlow

来源：

- Official site: https://sites.google.com/view/meshgraphnets
- Code/data loader: https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets
- Dataset metadata: https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/meta.json
- Paper: https://arxiv.org/abs/2010.03409

关键信息：

- 论文为 ICLR 2021，官方页面标注 Outstanding Paper。
- 任务是不可压流体绕圆柱，COMSOL 生成。
- 官方页面给出 `CylinderFlow` 平均约 1885 nodes、600 time steps。
- `meta.json` 中字段非常适合 Galerkin 抽取：
  - `cells`: static, triangular connectivity, shape `[1, -1, 3]`
  - `mesh_pos`: static node coordinates, shape `[1, -1, 2]`
  - `velocity`: dynamic, shape `[600, -1, 2]`
  - `pressure`: dynamic, shape `[600, -1, 1]`
  - `dt = 0.01`

Galerkin 适配：

- 这是本次检索中最像“即插即用”的公开圆柱绕流数据。
- 有三角单元连接，可直接用 PyVista/VTK 构造网格并计算 lumped mass、梯度、拉普拉斯。
- 速度是二维，需要补零成 `(u, v, 0)`。
- 注意：不同轨迹可能对应不同几何/网格。若要做全局 POD，需要先统一到参考网格，或每个几何单独提张量后训练参数化/MoE 组合。

建议下载：

```bash
git clone https://github.com/google-deepmind/deepmind-research.git
bash deepmind-research/meshgraphnets/download_dataset.sh cylinder_flow /path/to/data
```

推荐等级：**S，最建议优先试验**。

## 推荐 2：CFDBench Cylinder / Raw

来源：

- GitHub: https://github.com/luo-yining/CFDBench
- Paper: https://arxiv.org/abs/2310.05963
- Interpolated data: https://huggingface.co/datasets/chen-yingfa/CFDBench
- Raw data: https://huggingface.co/datasets/chen-yingfa/CFDBench-raw

关键信息：

- CFDBench 是大规模流体机器学习基准，包含 cavity、tube、dam、cylinder 四类问题。
- GitHub README 说明 cylinder 数据被拆成多个文件，并包含 `bc / geo / prop` 三类泛化子集。
- 插值版本约 13.4GB，原始版本约 460GB。
- raw Hugging Face 页面展示字段包括：
  - `nodenumber`
  - `x-coordinate`
  - `y-coordinate`
  - `y-velocity`
  - `x-velocity`
  - `absolute-pressure`

Galerkin 适配：

- 如果使用 interpolated 64x64 网格，张量可用有限差分/规则网格积分近似，但圆柱内部 mask 与边界处理要谨慎。
- 如果使用 raw 数据，速度、压力、坐标更完整；但需要确认原始 Fluent/ANSYS 输出是否保留单元连接。若只有节点表，则仍需重构网格。
- 优点是参数泛化非常好：边界条件、几何、物性变化都覆盖，适合 MoE-ROM 做专家划分。

推荐等级：**S-，适合大规模参数化实验，但前处理工作量明显大于 MeshGraphNets**。

## 推荐 3：Zenodo fixed circular cylinder Re=100

来源：

- Dataset: https://zenodo.org/records/5039610
- DOI: https://doi.org/10.5281/zenodo.5039610

关键信息：

- 标题：Numerical simulation data of a two-dimensional flow around a fixed circular cylinder。
- Re = 100，二维圆柱涡街，Cadyf 有限元求解器。
- Zenodo 描述明确：raw file 包含每个时间步的 flow velocity、pressure、node coordinates。
- 文件大小约 1.2GB，CC BY 4.0。
- 官方 `text_flow.py` 说明每行数据格式为：

```text
node_x node_y U(node) V(node) p(node)
```

Galerkin 适配：

- 有 `u, v, p, x, y`，非常适合构造 POD 数据矩阵。
- 公开解析脚本没有显示 element connectivity，因此精确 FEM 质量矩阵/梯度矩阵可能需要额外寻找原始网格，或用节点坐标重构三角网格。
- 如果目标是论文中的经典 Re=100 圆柱基准，这个数据很干净；如果目标是“严格半侵入式张量”，需要先解决网格连接和积分权重。

推荐等级：**A，适合经典圆柱 Re=100 对照和 ROM 展示**。

## 推荐 4：DatasetMeca oscillating cylinder Re_H=185

来源：

- https://datasetmeca.lisn.upsaclay.fr/doku.php?id=datasetmeca%3Aflow2dpastosccylinder1

关键信息：

- 二维受迫振荡圆柱 DNS，`Re_H = 185`。
- 数据大小约 7GB。
- 页面列出 `480 x 480` 空间分辨率。
- 2D snapshots 包含 `U, V, P, TRACE`，记录间隔为 `0.4` time unit，时间范围 `400` 到 `600`。
- 访问状态：实验室成员 free access，外部成员 on request。

Galerkin 适配：

- 规则/结构化网格下可以直接生成体积权重和有限差分导数。
- 因为圆柱运动，若直接在 Eulerian 网格上做 POD-Galerkin，需要处理 moving-body mask/phase function `TRACE`。
- 适合测试“带强迫/输入的 ROM”。

推荐等级：**A，但需要申请访问**。

## 推荐 5：PDEBench

来源：

- GitHub: https://github.com/pdebench/PDEBench
- Dataset: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986
- OpenReview: https://openreview.net/forum?id=dh_MkX0QfrK

关键信息：

- NeurIPS 2022 Datasets and Benchmarks 方向论文。
- 数据以 HDF5 为主，DaRUS 页面说明常用数组约定为 `[b,t,x1,...,xd,v]`。
- 包含 1D/2D/3D 多类 PDE，含可压缩 Navier-Stokes、Darcy、shallow water 等。
- GitHub 代码里有 `gen_ns_incomp.py` 用于 2D incompressible inhomogeneous Navier-Stokes 数据生成。

Galerkin 适配：

- 优点是顶会认可度高，适合论文 baseline 和模型泛化比较。
- 但 PDEBench 主数据不是圆柱绕流；官方 DaRUS measured variables 更偏可压缩 NS 的 `rho, velocity, pressure`，不是固定圆柱 wake。
- 对 Galerkin 张量来说，PDEBench 更适合做“规则网格 PDE baseline”，不适合作为圆柱 POD-Galerkin 主实验。

推荐等级：**B+，适合顶会基准引用，不适合替代圆柱数据**。

## 推荐 6：Nonlinear Benchmark - Unsteady Fluid Mechanics

来源：

- https://www.nonlinearbenchmark.org/benchmarks/unsteady-fluid-mechanics

关键信息：

- 受迫圆柱运动，包含 sinesweep、sine、multisine 等输入。
- 页面说明 wake 区域有 velocity、pressure、vorticity，网格为 `31 x 31`。
- 还提供圆柱表面的 kinematic pressure、drag/lift force。

Galerkin 适配：

- 非常适合非线性系统辨识、控制、输入输出 ROM。
- 但空间网格较粗，只覆盖尾迹局部，不适合高精度体积分和导数张量。
- 可以作为 MoE-ROM 的低维验证集，不建议作为主 Galerkin 张量来源。

推荐等级：**B**。

## 推荐 7：Re=3900 cylinder DNS wake dataset

来源：

- https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.15454/GLNRHK

关键信息：

- Data in Brief / Recherche Data Gouv 数据集。
- Re = 3900，smooth cylinder wake。
- 页面说明包含 Eulerian velocity and pressure fields，以及约 200,000 个 Lagrangian particle trajectories。
- DNS 代码为 Incompact3d。
- License 页面标注 Etalab Open License 2.0 / compatible CC-BY。

Galerkin 适配：

- 高保真、速度压力完整，适合湍流/尾迹机器学习。
- 但它是 3D wake/downstream sub-domain，不是完整二维圆柱全域。
- 可用于后续扩展：做 spanwise average、中心切片，或者 3D ROM/Galerkin 研究。

推荐等级：**B，作为扩展实验或高 Re 泛化，不作为当前 2D Galerkin 主数据**。

## 推荐 8：MegaFlow2D

来源：

- GitHub: https://github.com/cmudrc/MegaFlow2D
- ACM paper: https://dl.acm.org/doi/10.1145/3576914.3587552

关键信息：

- 超过 2 million snapshots，3000 个 external/internal 2D flow configurations。
- 面向 CFD super-resolution，多保真 low/high resolution。
- 基于 FEniCS/Oasis 流程。

Galerkin 适配：

- 如果数据包保留有限元网格、压力、速度和边界信息，会很适合大规模参数化 ROM。
- 当前公开摘要更强调 super-resolution 和多配置，不是专门圆柱 wake；需要先下载样例确认变量、网格连接和压力字段。

推荐等级：**B，作为参数化流动备选，不作为第一批 Galerkin 数据**。

## 推荐 9：Raissi / PINNs cylinder_nektar_wake.mat

来源：

- Data: https://github.com/maziarraissi/PINNs/blob/master/main/Data/cylinder_nektar_wake.mat
- HFM paper: https://arxiv.org/abs/1808.04327

关键信息：

- 经典 PINN/HFM 圆柱尾迹数据。
- `.mat` 文件约 23MB。
- 常见字段包括 `X_star, t, U_star, p_star`。

Galerkin 适配：

- 优点：小、经典、容易调通 pipeline。
- 缺点：没有公开单元连接/质量矩阵，通常是采样点数据；只覆盖局部 wake。
- 不建议作为正式 Galerkin 张量主数据，可作为单元测试或展示 PINN baseline。

推荐等级：**C**。

## 推荐 10：NVIDIA Modulus / PhysicsNeMo Cylinder Flow

来源：

- NGC entry: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/resources/modulus_datasets_cylinder-flow
- PhysicsNeMo example: https://docs.nvidia.com/physicsnemo/26.03/physicsnemo/examples/cfd/vortex_shedding_mesh_reduced/README.html
- Related paper: https://arxiv.org/abs/2201.09113

关键信息：

- 面向 transient vortex shedding / mesh-reduced temporal attention。
- NGC 搜索摘要显示数据为 `.npy` cylinder flow dataset。
- PhysicsNeMo 文档说明任务是 parameterized geometries 上的 transient vortex shedding。

Galerkin 适配：

- 如果 `.npy` 中包含 mesh/cells/velocity/pressure，则可转为 Galerkin 数据。
- 目前网页可读元信息有限，建议先作为备选，下载后检查字段。

推荐等级：**B-/待确认**。

## 数据集选择建议

若下一步要做“顶会级 MoE-ROM + 物理 Galerkin 张量”：

1. 第一阶段：用 **MeshGraphNets CylinderFlow** 做主公开数据集。
   - 优点是顶会背书、二维圆柱、网格连接明确、压力速度齐全。
   - 可直接验证从 TFRecord 到 PyVista/NumPy 再到 `c, A, H, P, L, c_tilde, A_tilde, H_tilde` 的完整 pipeline。

2. 第二阶段：用 **CFDBench cylinder raw/interpolated** 做参数泛化。
   - 适合训练 MoE，因为有 BC/geometry/property 三类变化。
   - 如果 raw 单元连接不足，则用 interpolated grid 先做规则网格 Galerkin 近似。

3. 第三阶段：用 **Zenodo Re=100 fixed cylinder** 做经典 Re=100 展示。
   - 用于和文献里的 POD-ROM 圆柱涡街结果对齐。
   - 需要优先解决网格连接/质量权重问题。

4. PDEBench 用作 SciML baseline 引用和辅助对照，不建议作为圆柱 Galerkin 主数据。

## Galerkin 前处理检查清单

拿到任一数据集后，先检查：

```text
必须：
  velocity: [T, N, 2] 或 [T, N, 3]
  pressure: [T, N, 1]
  coordinates: [N, 2] 或 [N, 3]
  topology: cells/connectivity 或规则网格 dx,dy

强烈建议：
  node_type / mask
  dt
  Re 或 nu
  boundary labels
  train/valid/test split
```

然后转换为本项目现有张量抽取格式：

```text
mass_weights: (N,)
u_bar:        (N, 3)
p_bar:        (N, 1)
phi_u:        (N, 3, r_u)
phi_p:        (N, 1, r_p)
mesh:         PyVista mesh with cells
```

对 MeshGraphNets `cylinder_flow` 的最短路线：

```text
TFRecord/meta.json
  -> parse cells, mesh_pos, velocity, pressure
  -> build triangular PyVista grid
  -> lumped triangle area mass_weights
  -> weighted POD
  -> pyvista.compute_derivative()
  -> Galerkin tensors
```

## 最终排序

1. **MeshGraphNets CylinderFlow**：最优先，公开、顶会、字段齐全、Galerkin 友好。
2. **CFDBench cylinder raw**：最适合 MoE 参数泛化，但要先确认 raw 网格拓扑。
3. **Zenodo Re=100 fixed cylinder**：最适合经典圆柱 ROM 展示，但需重构/确认单元连接。
4. **DatasetMeca oscillating cylinder**：适合受迫/输入 ROM，访问需申请。
5. **PDEBench**：适合顶会 baseline，不适合作为圆柱 Galerkin 主数据。

