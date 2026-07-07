# V15_2 Hopf Diagnostic 技术报告

本文件的完整中文分析见仓库根报告：

`V15_2_HopfDiagnostic/V15_2_HOPF_DIAGNOSTIC_REPORT.md`

核心结论：

- 主振荡 POD pair 为 `(a0, a1)`。
- `Re=51.786` 不是单纯相位错。24-step velocity L2 与 phase error 的相关性为 `-0.464`，与 amplitude error 的相关性为 `0.975`。
- 因此 Re=51.786 的直接主因是幅值/能量过冲；相位误差存在但不是窗口 L2 的主导解释；frequency/Strouhal 误差平均约 `20.8%`，属于中等误差。
- Hopf 区域的 POD 截断也很明显：Re=51.786 的 velocity tail relative L2=`0.418`，pressure tail relative L2=`0.382`，说明 `ru=16/rp=16` 在 Hopf transition 附近偏紧。

关键输出：

- `v15_2_per_re_summary.csv`
- `v15_2_per_window_metrics.csv`
- `v15_2_pod_projection_summary.csv`
- `v15_2_hopf_diagnostic_summary.json`
- `figures/Re_51p786450_hopf_diagnostic.svg`
- `timeseries/Re_51p786450_representative.csv`
