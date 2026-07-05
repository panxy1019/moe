# Regime-specific Area-weighted POD: hopf

## Regime Mapping

- Target regime: `hopf`
- Source labels: `['hopf_transition']`
- Number of Re cases: `17`
- Total snapshots: `2740`
- Re range: `47.0813545644` to `59.2014322659`

## POD Method

- Per-Re mean subtraction is used before POD.
- Inner product uses lumped `point_areas` from `mesh_l2_point_area_weights.npz`.
- Randomized block SVD is applied directly to weighted raw snapshots; no global ROM tensors are reused.

## Outputs

- `phi_uv.shape = (80, 194736)`
- `coeff_uv.shape = (2740, 80)`
- `velocity captured energy = 9.999999991620e-01`
- `phi_p.shape = (80, 97368)`
- `coeff_p.shape = (2740, 80)`
- `pressure captured energy = 9.999999960741e-01`
- elapsed: `169.6 s`

## Cases

- `Re_47p081355`: Re=47.0813545644, source=`hopf_transition`, snapshots=161
- `Re_47p722947`: Re=47.7229474482, source=`hopf_transition`, snapshots=161
- `Re_48p368688`: Re=48.3686884481, source=`hopf_transition`, snapshots=161
- `Re_49p022357`: Re=49.0223566571, source=`hopf_transition`, snapshots=161
- `Re_49p687640`: Re=49.6876404962, source=`hopf_transition`, snapshots=162
- `Re_50p368054`: Re=50.3680543703, source=`hopf_transition`, snapshots=161
- `Re_51p066785`: Re=51.066784903, source=`hopf_transition`, snapshots=161
- `Re_51p786450`: Re=51.7864496836, source=`hopf_transition`, snapshots=161
- `Re_52p528767`: Re=52.5287670834, source=`hopf_transition`, snapshots=162
- `Re_53p294175`: Re=53.2941749584, source=`hopf_transition`, snapshots=161
- `Re_54p081508`: Re=54.0815080552, source=`hopf_transition`, snapshots=161
- `Re_54p887950`: Re=54.887950068, source=`hopf_transition`, snapshots=161
- `Re_55p709610`: Re=55.709610114, source=`hopf_transition`, snapshots=161
- `Re_56p543246`: Re=56.5432463134, source=`hopf_transition`, snapshots=161
- `Re_57p389970`: Re=57.3899704283, source=`hopf_transition`, snapshots=162
- `Re_58p262636`: Re=58.2626356101, source=`hopf_transition`, snapshots=161
- `Re_59p201432`: Re=59.2014322659, source=`hopf_transition`, snapshots=161
