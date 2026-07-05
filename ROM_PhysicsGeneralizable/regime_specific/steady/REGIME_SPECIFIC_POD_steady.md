# Regime-specific Area-weighted POD: steady

## Regime Mapping

- Target regime: `steady`
- Source labels: `['steady_wake', 'pre_hopf_steady']`
- Number of Re cases: `20`
- Total snapshots: `1277`
- Re range: `20` to `46.440071584`

## POD Method

- Per-Re mean subtraction is used before POD.
- Inner product uses lumped `point_areas` from `mesh_l2_point_area_weights.npz`.
- Randomized block SVD is applied directly to weighted raw snapshots; no global ROM tensors are reused.

## Outputs

- `phi_uv.shape = (80, 194736)`
- `coeff_uv.shape = (1277, 80)`
- `velocity captured energy = 9.999999996581e-01`
- `phi_p.shape = (80, 97368)`
- `coeff_p.shape = (1277, 80)`
- `pressure captured energy = 9.999999994277e-01`
- elapsed: `91.6 s`

## Cases

- `Re_20p000000`: Re=20, source=`steady_wake`, snapshots=64
- `Re_22p535676`: Re=22.5356758074, source=`steady_wake`, snapshots=64
- `Re_24p630436`: Re=24.6304355074, source=`steady_wake`, snapshots=64
- `Re_26p667332`: Re=26.6673319278, source=`steady_wake`, snapshots=64
- `Re_28p695138`: Re=28.695137758, source=`steady_wake`, snapshots=64
- `Re_30p720428`: Re=30.7204283434, source=`steady_wake`, snapshots=64
- `Re_32p740068`: Re=32.740068162, source=`steady_wake`, snapshots=64
- `Re_34p737570`: Re=34.7375697369, source=`steady_wake`, snapshots=64
- `Re_36p657767`: Re=36.6577674843, source=`steady_wake`, snapshots=63
- `Re_38p357249`: Re=38.357249335, source=`steady_wake`, snapshots=64
- `Re_39p685479`: Re=39.6854792076, source=`steady_wake`, snapshots=64
- `Re_40p711525`: Re=40.7115247449, source=`pre_hopf_steady`, snapshots=64
- `Re_41p576575`: Re=41.5765746111, source=`pre_hopf_steady`, snapshots=63
- `Re_42p359071`: Re=42.3590705675, source=`pre_hopf_steady`, snapshots=64
- `Re_43p093925`: Re=43.0939251612, source=`pre_hopf_steady`, snapshots=64
- `Re_43p797402`: Re=43.7974019747, source=`pre_hopf_steady`, snapshots=63
- `Re_44p478353`: Re=44.4783532118, source=`pre_hopf_steady`, snapshots=64
- `Re_45p142703`: Re=45.142702578, source=`pre_hopf_steady`, snapshots=64
- `Re_45p795194`: Re=45.7951936696, source=`pre_hopf_steady`, snapshots=64
- `Re_46p440072`: Re=46.440071584, source=`pre_hopf_steady`, snapshots=64
