# MoE 在具体领域中的应用调研

日期：2026-06-12  
用途：为“用稀疏混合专家模型预测 POD 系数”的降阶模型研究选题提供文献背景。

> 说明：这里把 MoE 理解为广义的 mixture-of-experts：通过门控/路由网络把不同样本、token、空间位置、时间阶段、任务或物理状态分配给不同专家模型。JCP、CMAME 中直接以现代稀疏 MoE 为主线的论文相对少，但它们在 PINN、局部 ROM、领域分解、非侵入式 POD 系数学习方面提供了非常接近的建模思想。

## 1. 总体脉络

MoE 的核心价值不是“多个网络拼在一起”，而是把模型容量拆成一组可条件激活的局部函数：

- 在 NLP/CV/多模态中，MoE 常用于扩大参数量而保持单次推理计算量可控。
- 在推荐、医学、交通、材料等应用中，MoE 常用于处理任务异质性、数据分布多峰、样本群体差异或区域差异。
- 在科学计算中，MoE 很自然对应参数空间分区、物理机制分区、空间/时间子域分区、局部低维流形和多尺度结构。
- 对 POD-ROM 而言，MoE 可以被理解为“对系数映射 alpha(mu,t) 的局部分片近似”，也可以进一步做成“对不同 POD mode 或 mode group 的条件路由”。

## 2. MoE 基础与通用机器学习顶会工作

| 方向 | 代表工作 | 场景 | 关键做法 | 对 ROM 的启发 |
|---|---|---|---|---|
| 稀疏门控 MoE | [Shazeer et al., ICLR 2017, Outrageously Large Neural Networks](https://openreview.net/forum?id=B1ckMDqlg) | 语言建模/机器翻译 | top-k 稀疏门控，只激活少数专家；配合负载均衡 | ROM 中可以让一个输入条件只调用少数“物理/参数专家”，减少推理代价 |
| 大规模稀疏 Transformer | [GShard, ICLR 2021](https://openreview.net/forum?id=qrwe7XHTmYb), [Switch Transformer, JMLR 2022](https://www.jmlr.org/papers/v23/21-0998.html) | 多语言翻译/语言模型 | token-level routing，top-1/top-2 expert，capacity 与 load balancing | sparse routing 的工程问题：专家塌缩、负载不均、token 丢弃，都会在回归型 ROM 中出现 |
| Expert Choice Routing | [Zhou et al., NeurIPS 2022](https://openreview.net/forum?id=jdJo1HIVinI) | 大模型训练 | 由专家选择 token，使负载更稳定 | 对连续回归任务很有价值，因为 ROM 不希望某些样本因为 capacity 被丢弃 |
| Soft MoE | [Puigcerver et al., ICLR 2024](https://openreview.net/forum?id=jxpsAj7ltE) | 视觉模型 | 用软分配替代硬 top-k，降低路由不稳定 | POD 系数回归如果数据量不大，soft/hybrid routing 可能比硬 top-1 更稳 |
| Vision MoE | [Riquelme et al., NeurIPS 2021, V-MoE](https://proceedings.neurips.cc/paper/2021/hash/48237d9f2dea8c74c2a72126cf63d933-Abstract.html) | 图像分类 | Transformer 中部分 MLP 层换成稀疏专家 | 可以把“token”换成 POD mode token 或空间子域 token |
| 多模态 MoE | [Mustafa et al., NeurIPS 2022, LIMoE](https://proceedings.neurips.cc/paper_files/paper/2022/hash/8651b704b0de9d7ffefb91b88d5fd0b1-Abstract-Conference.html) | 图文对齐 | 图像 token 和文本 token 共享专家池 | 对多物理场 ROM 有启发：不同变量、边界条件、几何描述可共享专家池 |
| 多任务推荐 | [Ma et al., KDD 2018, Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/10.1145/3219819.3220007) | YouTube 推荐多目标学习 | 每个任务有独立 gate，共享专家 | 多个 POD 系数/多个物理量可以看作多任务输出，每个 mode 或 mode group 使用不同 gate |

这些工作解决的是 MoE 的通用问题：专家容量、门控稳定性、负载均衡、专家退化、任务共享与任务冲突。它们不直接解决 PDE 或 ROM，但给出了可复用的架构部件。

## 3. 领域应用：MoE 如何落到具体问题

### 3.1 推荐系统与多任务学习

MMoE 是非常值得借鉴的应用型 MoE。它不追求“一个专家对应一个任务”的硬绑定，而是让每个任务通过自己的 gate 组合共享专家。这一点对 POD 系数预测尤其重要：POD 系数之间通常有耦合，强行让每个专家只服务某一个系数，可能损失共享动力学信息。更合理的做法可能是“共享专家池 + mode-aware gate”。

相关工作：

- [Multi-gate Mixture-of-Experts for Multi-task Learning, KDD 2018](https://dl.acm.org/doi/10.1145/3219819.3220007)
- [Progressive Layered Extraction, RecSys 2020](https://dl.acm.org/doi/10.1145/3383313.3412236)

### 3.2 计算机视觉、多模态与医学图像

视觉和多模态 MoE 说明了一个重要事实：专家不必只按“样本”分工，也可以按 token、patch、模态、尺度分工。医学图像 segmentation 中也有使用 patch/token 专家处理不同解剖结构或尺度特征的工作。

相关工作：

- [V-MoE, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/48237d9f2dea8c74c2a72126cf63d933-Abstract.html)
- [LIMoE, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/8651b704b0de9d7ffefb91b88d5fd0b1-Abstract-Conference.html)
- [Patcher: Patch Transformers with Mixture of Experts for Precise Medical Image Segmentation, MICCAI 2022](https://link.springer.com/chapter/10.1007/978-3-031-16443-9_21)

对 ROM 的启发：

- 把 POD mode 当作 token：每个 mode token 携带 mode index、能量占比、频率特征等信息。
- 把物理变量当作模态：例如速度、压力、温度、浓度可共享部分专家，同时有变量特定 gate。
- 把空间子域或局部基函数当作 patch：这与局部 POD、domain decomposition ROM 接近。

### 3.3 时间序列、交通与天气气候

时间序列、交通流和天气气候具有明显的时空异质性：高峰/低峰、不同区域、不同天气型、不同季节机制。MoE 常被用来让专家对应不同区域、尺度或动力学状态。

相关工作：

- [Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts, ICLR 2025](https://openreview.net/forum?id=e1wDDFmlVu)
- [Spatial Mixture-of-Experts, NeurIPS 2022](https://openreview.net/forum?id=Oc_gQJ3I5eK)
- [EWMoE: An Effective Model for Global Weather Forecasting with Mixture-of-Experts, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/31997)
- [ST-MoE / spatio-temporal MoE traffic forecasting 相关方向](https://arxiv.org/search/?query=spatio-temporal+mixture+of+experts+traffic&searchtype=all)

对 ROM 的启发：

- POD 系数 alpha_j(t,mu) 本质上是低维时间序列；如果存在 vortex shedding、shock motion、transition、bifurcation，不同时间段可能需要不同专家。
- 对参数化 PDE，可以把 gate 可视化到参数-时间平面，看专家是否自动学到 regime partition。
- 对外推任务，router 的置信度和专家分布可以作为不确定性/OOD 指标。

### 3.4 材料、化学、能源与 Nature 系列论文

Nature 系列中出现的 MoE 往往不是为了炫耀大模型容量，而是为了处理多源异构数据、不同材料族、不同物理机制或跨任务迁移。

| 工作 | 期刊/年份 | 应用场景 | MoE 用法 | 启发 |
|---|---:|---|---|---|
| [Towards overcoming data scarcity in materials science: unifying models and datasets with a mixture of experts framework](https://www.nature.com/articles/s41524-022-00929-x) | npj Computational Materials, 2022 | 材料性质预测 | 用 MoE 融合不同模型/数据集，缓解材料小数据问题 | 参数化 PDE 也有“不同工况/不同数据源”的异构性 |
| [iMOE: prediction of second-life battery degradation trajectory using interpretable mixture of experts](https://www.nature.com/articles/s41467-026-69369-1) | Nature Communications, 2026 | 电池退化预测 | 强调专家可解释性，把不同退化模式分配给不同专家 | 与 POD-ROM 的专家-regime 可解释性高度相似 |
| [Mixture-of-experts graph transformers for interpretable particle physics](https://www.nature.com/articles/s41598-025-12003-9) | Scientific Reports, 2025 | 高能物理事件/喷注相关分类 | 用 MoE graph transformer 提升表达并保留可解释分析 | 复杂物理事件分类/回归可利用 token-level expert routing |
| [Photonic Mixture-of-Experts for scalable multi-task on-chip optical neural networks](https://www.nature.com/articles/s41467-026-73983-4) | Nature Communications, 2026 | 光子/硬件加速 | 用并行光学专家扩展片上光神经网络容量 | 若 ROM 要在线控制/实时仿真，稀疏激活的硬件友好性是额外优势 |
| [Scaling neural machine translation to 200 languages](https://www.nature.com/articles/s41586-024-07335-x) | Nature, 2024 | 低资源多语言翻译 | 条件计算 MoE 支撑多语言、多任务共享 | 多参数 PDE 可以借鉴“共享底座 + 条件专家”的多域迁移思路 |

> 注：Nature 站内有关 MoE 的论文主题变化较快，建议正式写论文时重新核对 DOI、卷期和最终题名。

### 3.5 科学机器学习、PDE 与神经算子

这部分最接近你的想法。MoE 在 PDE/神经算子里开始出现，主要理由是：单一全局模型很难覆盖多个参数区间、边界条件、几何和物理 regime。

代表性方向：

- [Mixture-of-Experts Operator Transformer for Large-Scale PDE Pre-Training, NeurIPS 2025](https://openreview.net/forum?id=PNgG4H3q9D)  
  面向 PDE operator learning，把 MoE 引入 operator transformer，用稀疏专家提高多 PDE/多参数预训练能力。
- [Spatial Mixture-of-Experts, NeurIPS 2022](https://openreview.net/forum?id=Oc_gQJ3I5eK)  
  用空间位置相关的专家处理空间异质性，适合启发空间/参数域分区。
- [Physics-informed machine learning, Nature Reviews Physics 2021](https://www.nature.com/articles/s42254-021-00314-5)  
  不是 MoE 论文，但提供了把物理约束嵌入学习系统的总体框架。
- [Physics-informed neural networks, JCP 2019](https://www.sciencedirect.com/science/article/pii/S0021999118307125)  
  PINN 基础论文，说明残差损失、边界条件损失等可作为监督数据之外的约束。

对 POD-ROM 的启发：

- MoE 不一定直接输出全场解，可以只学习低维系数 alpha。
- 神经算子的专家分工可以迁移到 ROM：一个专家处理光滑低频模式，另一个处理高频/局部瞬态，另一个处理某些参数区间。
- 若能把 gate 的分区和物理 regime 对上，就会比普通黑箱回归更有说服力。

## 4. JCP / CMAME 中相关但不一定叫 MoE 的工作

### 4.1 非侵入式 POD 系数学习

POD-ROM 的常见非侵入式路线是：先通过 SVD 得到 POD basis，再训练一个回归器预测 POD coefficients。

相关工作：

- [Hesthaven & Ubbiali, Non-intrusive reduced order modeling of nonlinear problems using neural networks, JCP 2018](https://www.sciencedirect.com/science/article/pii/S0021999118303942)  
  用神经网络学习参数到 reduced coefficients 的映射，是“POD coefficients + NN 回归”的经典路线。
- [Fresca, Manzoni, Dedè, POD-DL-ROM, CMAME 2021/2022](https://www.sciencedirect.com/science/article/pii/S0045782521004290)  
  用深度学习构造非线性降维/低维动力学，是 ROM 与深度网络结合的重要参考。
- [Non-intrusive reduced-order modeling of unsteady flows using artificial neural networks, JCP 2019](https://www.sciencedirect.com/science/article/pii/S0021999119301164)  
  针对非定常流动，用 ANN 预测低维表示/系数。

这些工作通常使用单一网络或普通多层感知机/时序网络。MoE 的创新点可以放在：让系数映射在参数-时间-mode 空间中自动分片，而不是用一个全局函数硬拟合所有行为。

### 4.2 Domain decomposition / 局部专家思想

PINN 和 SciML 中大量“领域分解”工作，虽然不一定写成 MoE，但思想上非常接近专家模型：

- [Conservative PINNs, CMAME 2020](https://www.sciencedirect.com/science/article/pii/S0045782520302127)  
  把计算域分成子域，在界面上施加守恒/连续条件。
- [XPINNs: generalized space-time domain decomposition, CMAME 2020](https://www.sciencedirect.com/science/article/pii/S0045782520302929)  
  对空间-时间域分解，每个子域可有单独网络。
- [Parallel PINNs / PPINNs, CMAME 2020](https://www.sciencedirect.com/science/article/pii/S0045782520304357)  
  并行化训练多个子域网络。

对 MoE-ROM 的启发：

- 子域网络可以看成“硬路由专家”；MoE 则是可学习软/稀疏路由。
- POD 系数虽然是全局量，但其行为可能仍可按参数区间、时间阶段或物理机制分解。
- 可以把 domain decomposition 的界面一致性思想迁移为 mode group 之间的耦合/一致性损失。

### 4.3 局部 ROM / 聚类 ROM

ROM 社区早有 local ROM、cluster-based ROM、adaptive basis 等思想：对不同参数区域或状态区域建立不同局部基/局部动力系统。这些方法与 MoE 的关系很近：

- local ROM 的“选择局部基”类似 MoE 的 gate。
- 每个 local basis / local reduced operator 类似专家。
- MoE 的优势是路由、专家和输出可以端到端训练，还能做可解释性分析。

因此，如果想把工作定位得更稳，可以说：

> 本研究不是把 MoE 生硬搬到 ROM，而是把 local ROM / domain decomposition ROM 的思想，用现代稀疏门控专家网络重新表达，并用于非侵入式 POD coefficient regression。

## 5. 可解释性 MoE 的常见做法

可解释性不是只看专家编号，而是要证明“专家分工有物理意义”。常见手段包括：

- 专家利用率图：统计不同参数、时间、边界条件下 top-k 专家频率。
- gate heatmap：在 mu-t 平面画出专家分区。
- mutual information：度量专家选择与 Reynolds number、Mach number、shock position、vortex shedding phase 等物理量的相关性。
- surrogate tree：用浅层决策树拟合 router 输出，得到近似规则。
- expert attribution：分析每个专家主要贡献哪些 POD mode、哪些时间频率或哪些空间重构误差。
- ablation：关掉某个专家或冻结 gate，看哪些物理 regime 误差上升。
- uncertainty：当 gate 熵高或专家意见分歧大时，标记为 OOD/低置信度。

## 6. 对 POD-ROM 选题的文献定位

可以把拟议工作放在以下交叉点：

1. **非侵入式 POD-ROM**：已有工作证明神经网络可预测 POD coefficients。
2. **局部 ROM / domain decomposition**：已有工作证明复杂参数空间需要局部模型。
3. **稀疏 MoE**：已有工作证明可用可学习路由实现条件计算与专家分工。
4. **可解释 SciML**：目标不是只降低误差，而是让专家分区对应物理 regime。

一个清晰的论文叙事可以是：

> 传统非侵入式 POD-ROM 使用单一全局网络学习 alpha(mu,t)，容易在多 regime、强非线性、瞬态切换或高阶系数上欠拟合。本文提出 mode-aware sparse MoE，通过条件路由为不同参数-时间区域和不同 POD mode/group 分配专家，并通过物理一致性损失与 router 可解释性分析，构建高精度、可解释、计算量可控的非侵入式 ROM。

## 7. 初步参考文献清单

### MoE 基础/顶会

- Shazeer et al. [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://openreview.net/forum?id=B1ckMDqlg), ICLR 2017.
- Lepikhin et al. [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://openreview.net/forum?id=qrwe7XHTmYb), ICLR 2021.
- Fedus et al. [Switch Transformers](https://www.jmlr.org/papers/v23/21-0998.html), JMLR 2022.
- Zhou et al. [Mixture-of-Experts with Expert Choice Routing](https://openreview.net/forum?id=jdJo1HIVinI), NeurIPS 2022.
- Puigcerver et al. [From Sparse to Soft Mixtures of Experts](https://openreview.net/forum?id=jxpsAj7ltE), ICLR 2024.

### 应用型 MoE

- Ma et al. [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/10.1145/3219819.3220007), KDD 2018.
- Riquelme et al. [Scaling Vision with Sparse Mixture of Experts](https://proceedings.neurips.cc/paper/2021/hash/48237d9f2dea8c74c2a72126cf63d933-Abstract.html), NeurIPS 2021.
- Mustafa et al. [Multimodal Contrastive Learning with LIMoE](https://proceedings.neurips.cc/paper_files/paper/2022/hash/8651b704b0de9d7ffefb91b88d5fd0b1-Abstract-Conference.html), NeurIPS 2022.
- Jiang et al. [Spatial Mixture-of-Experts](https://openreview.net/forum?id=Oc_gQJ3I5eK), NeurIPS 2022.
- Time-MoE authors. [Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts](https://openreview.net/forum?id=e1wDDFmlVu), ICLR 2025.
- Gan et al. [EWMoE: An Effective Model for Global Weather Forecasting with Mixture-of-Experts](https://ojs.aaai.org/index.php/AAAI/article/view/31997), AAAI 2025.
- Wang et al. [Mixture-of-Experts Operator Transformer for Large-Scale PDE Pre-Training](https://openreview.net/forum?id=PNgG4H3q9D), NeurIPS 2025.

### ROM / PINN / SciML

- Hesthaven & Ubbiali. [Non-intrusive reduced order modeling of nonlinear problems using neural networks](https://www.sciencedirect.com/science/article/pii/S0021999118303942), Journal of Computational Physics, 2018.
- Raissi, Perdikaris, Karniadakis. [Physics-informed neural networks](https://www.sciencedirect.com/science/article/pii/S0021999118307125), Journal of Computational Physics, 2019.
- Fresca, Manzoni, Dedè. [POD-DL-ROM](https://www.sciencedirect.com/science/article/pii/S0045782521004290), Computer Methods in Applied Mechanics and Engineering.
- Jagtap et al. [Conservative PINNs](https://www.sciencedirect.com/science/article/pii/S0045782520302127), CMAME, 2020.
- Jagtap & Karniadakis. [XPINNs](https://www.sciencedirect.com/science/article/pii/S0045782520302929), CMAME, 2020.
- Karniadakis et al. [Physics-informed machine learning](https://www.nature.com/articles/s42254-021-00314-5), Nature Reviews Physics, 2021.
