# Compact Semi-intrusive Galerkin Tensor File

File:

```text
semi_intrusive_galerkin_tensors_allRe30_weightedL2_ru80_rp80_compact.npz
```

This compact file is numerically equivalent to:

```text
semi_intrusive_galerkin_tensors_allRe30_weightedL2_ru80_rp80.npz
```

but avoids repeating Reynolds-independent tensors for every Reynolds number.

## Shared arrays

```text
G_u     shape = (80, 80)
H       shape = (80, 80, 80)
P       shape = (80, 80)
H_raw   shape = (80, 80, 80)
P_raw   shape = (80, 80)
```

These arrays are common to all 30 Reynolds-number mean fields.

## Per-Re arrays

For each label in `Re_labels_computed`, for example `Re_500p000000`:

```text
Re_500p000000_c      shape = (80,)
Re_500p000000_A      shape = (80, 80)
Re_500p000000_c_raw  shape = (80,)
Re_500p000000_A_raw  shape = (80, 80)
```

The explicit ROM is:

```text
da_i/dt = c_i + sum_j A_ij a_j + sum_j sum_k H_ijk a_j a_k + sum_m P_im b_m
```

Use the shared `H` and `P` with the per-Re `c` and `A`.

## Validation

The compact file was checked against the full all-Re file:

```text
max_abs_diff(G_u, H, P, H_raw, P_raw) = 0
max_abs_diff(per-Re c/A/c_raw/A_raw)  = 0
all stored floating arrays are finite
```

