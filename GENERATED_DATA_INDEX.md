# OpenFOAM Cylinder Flow Data Index

This file summarizes the generated 2D circular-cylinder flow datasets, converted Python-readable data, and POD outputs.

## Main Result Folder

Recommended dataset:

```text
/home/ray/Desktop/Cylinder_Results_Re500_1000_VTU
```

This folder contains the final Re=500-1000 vortex-shedding time-series data, Python-readable `u/v/p` arrays, and global POD decomposition.

## Simulation Setup

Geometry and physics:

```text
Cylinder diameter D = 1 m
Kinematic viscosity nu = 1e-3 m^2/s
Re = 500, 600, 700, 800, 900, 1000
Inlet velocity U = Re * nu / D
Solver = pimpleFoam, OpenFOAM-13 compatibility wrapper to foamRun -solver incompressibleFluid
Turbulence model = laminar
Mesh = 2D blockMesh with empty front/back patches
```

Time-series settings:

```text
Estimated Strouhal number St = 0.2
Estimated shedding period T = D / (St * U)
Each case covers 12 estimated vortex-shedding periods
Each period has about 20 saved frames
maxCo = 0.8
adjustTimeStep = yes
parallelSubdomains = 8
```

Case summary:

| Re | U (m/s) | Period (s) | End Time (s) | Saved Interval (s) | VTU Frames | Final Max Co |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0.5 | 10 | 120 | 0.5 | 241 | 0.78106843 |
| 600 | 0.6 | 8.333333333 | 100 | 0.4166666667 | 241 | 0.78698273 |
| 700 | 0.7 | 7.142857143 | 85.71428571 | 0.3571428571 | 241 | 0.7754424 |
| 800 | 0.8 | 6.25 | 75 | 0.3125 | 241 | 0.76523966 |
| 900 | 0.9 | 5.555555556 | 66.66666667 | 0.2777777778 | 241 | 0.76615627 |
| 1000 | 1.0 | 5 | 60 | 0.25 | 241 | 0.78103388 |

The source summary table is:

```text
/home/ray/Desktop/Cylinder_Results_Re500_1000_VTU/Summary.csv
```

## VTU Time-Series Files

Each Re has one folder of full-domain binary VTU files:

```text
Re_500_VTU/
Re_600_VTU/
Re_700_VTU/
Re_800_VTU/
Re_900_VTU/
Re_1000_VTU/
```

Each folder contains:

```text
flow_<time>.vtu
```

Example:

```text
/home/ray/Desktop/Cylinder_Results_Re500_1000_VTU/Re_1000_VTU/flow_60.vtu
```

ParaView time-series entry files:

```text
Re_500_timeseries.pvd
Re_600_timeseries.pvd
Re_700_timeseries.pvd
Re_800_timeseries.pvd
Re_900_timeseries.pvd
Re_1000_timeseries.pvd
```

Open one of these `.pvd` files in ParaView to play the corresponding time sequence.

Example:

```bash
paraview /home/ray/Desktop/Cylinder_Results_Re500_1000_VTU/Re_1000_timeseries.pvd
```

Per-case information files:

```text
Re_500_info.txt
Re_600_info.txt
Re_700_info.txt
Re_800_info.txt
Re_900_info.txt
Re_1000_info.txt
```

## Python-Readable U/V/P Files

Each Re has one compressed NumPy file:

```text
Re_500_uvp_pointData.npz
Re_600_uvp_pointData.npz
Re_700_uvp_pointData.npz
Re_800_uvp_pointData.npz
Re_900_uvp_pointData.npz
Re_1000_uvp_pointData.npz
```

Dimensions are identical for all Re:

```text
times:  (241,)
points: (97368, 3)
u:      (241, 97368)
v:      (241, 97368)
p:      (241, 97368)
```

Indexing:

```text
u[t, i], v[t, i], p[t, i]
```

means data at:

```text
time = times[t]
coordinate = points[i]
```

Units:

```text
times: s
points: m
u, v: m/s
p: m^2/s^2, OpenFOAM incompressible kinematic pressure
```

Python loading example:

```python
import numpy as np
import json

path = "/home/ray/Desktop/Cylinder_Results_Re500_1000_VTU/Re_1000_uvp_pointData.npz"
data = np.load(path)

times = data["times"]
points = data["points"]
u = data["u"]
v = data["v"]
p = data["p"]
metadata = json.loads(str(data["metadata"]))
```

Supporting metadata:

```text
README_uvp_npz.txt
uvp_npz_summary.json
Re_XXX_uvp_dimensions.txt
Re_XXX_uvp_metadata.json
```

## Global POD Output

POD folder:

```text
/home/ray/Desktop/Cylinder_Results_Re500_1000_VTU/Global_POD_Unweighted
```

Method:

```text
Unweighted global snapshot POD
Velocity [u, v] POD and pressure p POD are computed separately
For each Re, the first 2 estimated shedding cycles are discarded
For each Re, retained snapshots are centered by that Re time-mean field
Each Re contributes 201 snapshots
Total snapshots = 1206
```

Files:

```text
global_velocity_pod.npz
global_pressure_pod.npz
pod_snapshot_index.csv
pod_metadata.json
README.txt
```

Velocity POD dimensions:

```text
phi_uv:        (80, 194736)
coeff_uv:      (1206, 80)
mean_uv_by_Re: (6, 194736)
points:        (97368, 3)
```

Pressure POD dimensions:

```text
phi_p:        (80, 97368)
coeff_p:      (1206, 80)
mean_p_by_Re: (6, 97368)
```

Energy captured by retained modes:

```text
Velocity POD, 80 modes: 0.9958305400465817
Pressure POD, 80 modes: 0.9988679740596714
```

Snapshot index:

```text
pod_snapshot_index.csv
```

This maps each coefficient row to:

```text
snapshot_id, Re, time, period, cycle, phase, local_snapshot_index
```

POD loading example:

```python
import numpy as np
import pandas as pd

pod_dir = "/home/ray/Desktop/Cylinder_Results_Re500_1000_VTU/Global_POD_Unweighted"

vel = np.load(f"{pod_dir}/global_velocity_pod.npz")
prs = np.load(f"{pod_dir}/global_pressure_pod.npz")
index = pd.read_csv(f"{pod_dir}/pod_snapshot_index.csv")

phi_uv = vel["phi_uv"]
coeff_uv = vel["coeff_uv"]
mean_uv_by_Re = vel["mean_uv_by_Re"]
points = vel["points"]
n_points = int(vel["n_points"])

phi_p = prs["phi_p"]
coeff_p = prs["coeff_p"]
mean_p_by_Re = prs["mean_p_by_Re"]
```

Velocity reconstruction:

```python
snapshot_id = 0
re_index = 0
r = 80

uv_fluct = coeff_uv[snapshot_id, :r] @ phi_uv[:r]
uv = mean_uv_by_Re[re_index] + uv_fluct

u = uv[:n_points]
v = uv[n_points:]
```

Pressure reconstruction:

```python
p_fluct = coeff_p[snapshot_id, :r] @ phi_p[:r]
p = mean_p_by_Re[re_index] + p_fluct
```

## Earlier Result Folder

An earlier final-time-only dataset also exists:

```text
/home/ray/Desktop/Cylinder_Results
```

This contains final-time VTK results for the broader Re=100-1000 sweep, plus Re=1000 visualization PNGs. It is useful for quick inspection, but the recommended time-series and POD dataset is:

```text
/home/ray/Desktop/Cylinder_Results_Re500_1000_VTU
```

## Generated Scripts

The main scripts used to produce and post-process the data are in `/home/ray`:

```text
run_cylinder_re500_1000_cycles_vtu.py
convert_vtu_to_uvp_npz.py
compute_global_pod_unweighted.py
visualize_re1000_paraview.py
run_cylinder_re_sweep.py
run_cylinder_re_sweep_timeseries.py
```

Recommended script meanings:

| Script | Purpose |
|---|---|
| `run_cylinder_re500_1000_cycles_vtu.py` | Re=500-1000 production rerun, writes VTU time series and case info. |
| `convert_vtu_to_uvp_npz.py` | Converts VTU time series to Python-readable `u/v/p` `.npz` files. |
| `compute_global_pod_unweighted.py` | Computes unweighted global snapshot POD for velocity and pressure. |
| `visualize_re1000_paraview.py` | Creates ParaView PNG visualizations for Re=1000. |
| `run_cylinder_re_sweep.py` | Earlier Re=100-1000 final-time sweep. |
| `run_cylinder_re_sweep_timeseries.py` | Earlier fixed-step time-series sweep. |

## OpenFOAM Case Templates

Useful local case folders:

```text
/home/ray/base_case
/home/ray/base_case_cycles
/home/ray/base_case_timeseries
/home/ray/cylinder2D
```

`base_case_cycles` corresponds most closely to the final Re=500-1000 VTU dataset.

