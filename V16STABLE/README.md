# V16STABLE

`V16STABLE` 是 `V16_1_SteadyPressureAnchor32` 的独立整理版本，用于重新检查模型结构、训练配置、已知问题与复现流程。这里不修改原始 V16_1 实验，也不包含大型 checkpoint 或原始 83 MB metrics。

## 内容

- [`MODEL_ARCHITECTURE_PARAMETERS_AND_ISSUES.md`](MODEL_ARCHITECTURE_PARAMETERS_AND_ISSUES.md)：网络、参数、训练流程、真实结果和问题清单。
- [`REPRODUCIBILITY_REPORT.md`](REPRODUCIBILITY_REPORT.md)：原实验的完整数据与复现说明。
- [`train_v16_1_steady_pressure_anchor32.py`](train_v16_1_steady_pressure_anchor32.py)：原始训练/评测实现的逐字节副本。
- [`run_train.sh`](run_train.sh)：冻结为 `V16_1_SteadyPressureAnchor32` 的单案例训练入口。
- [`eval_checkpoint.sh`](eval_checkpoint.sh)：已有 checkpoint 的 eval-only 入口。
- [`evaluate_metrics.py`](evaluate_metrics.py)：metrics JSON 的紧凑打印工具。
- [`results/reference`](results/reference)：已完成实验的 compact summary、逐 Re CSV 和误差曲线。

## 快速复现

默认假设仓库位于 `/root/moe`，数据位于 `/root/moe/ROM_PhysicsGeneralizable/data`：

```bash
cd /root/moe/V16STABLE
export SWANLAB_API_KEY="..."
CUDA_VISIBLE_DEVICES=0 ./run_train.sh
```

路径均可覆盖：

```bash
MOE_ROOT=/path/to/moe \
PYTHON_BIN=/path/to/python \
OUTPUT_DIR=/path/to/output \
SWANLAB_TRACKING_MODE=disabled \
CUDA_VISIBLE_DEVICES=0 \
./run_train.sh
```

已有 checkpoint 时：

```bash
CUDA_VISIBLE_DEVICES=0 ./eval_checkpoint.sh /path/to/checkpoint.pt
```

命令行末尾可以传递覆盖参数，例如做一个 2 epoch 的 wiring smoke test：

```bash
SWANLAB_TRACKING_MODE=disabled V16STABLE_EPOCHS=2 V16STABLE_MIN_EPOCHS=1 \
  ./run_train.sh --patience 1
```

## 环境

已完成实验使用：

- Python 环境：`/root/miniconda3/envs/pt_env`
- PyTorch：`2.11.0+cu126`
- GPU：CUDA，TF32 开启
- 监控：SwanLab，通过 `SWANLAB_API_KEY` 环境变量登录

`requirements.txt` 记录 Python 依赖名称；CUDA 版 PyTorch 应按集群驱动和官方安装方式单独安装。

## 来源校验

训练脚本来自：

```text
V16_1_TrainStableAttractorMoE/test_results_v16_1/train_v16_1_train_stable_attractor_moe.py
```

两个文件的 SHA-256 均为：

```text
c8bd9ade7efc1c80e3b25f4b0e1156b8476ab98480ecc928621e1efa64870470
```

不要在仓库中提交 SwanLab API key、SSH 密码、GitHub token、checkpoint 或完整 raw metrics。
