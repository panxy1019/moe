Weighted L2 global POD outputs

This POD uses mesh-derived point control volumes as diagonal mass weights.

Weight file:
  mesh_l2_point_weights.npz
    point_volumes: nodal control volumes, shape (n_points,)
    sqrt_point_volumes: sqrt(point_volumes), shape (n_points,)
    cell_volumes: VTK cell volumes, shape (n_cells,)

Velocity POD:
  global_velocity_pod_weighted_l2.npz
    phi_uv: raw-space weighted POD modes, shape (r_uv, 2*n_points)
    phi_uv_weighted: weighted-coordinate modes, shape (r_uv, 2*n_points)
    coeff_uv: POD coefficients, shape (n_snapshots, r_uv)
    mean_uv_by_Re: per-Re retained-snapshot raw mean, shape (n_Re, 2*n_points)

Pressure POD:
  global_pressure_pod_weighted_l2.npz
    phi_p: raw-space weighted POD modes, shape (r_p, n_points)
    phi_p_weighted: weighted-coordinate modes, shape (r_p, n_points)
    coeff_p: POD coefficients, shape (n_snapshots, r_p)
    mean_p_by_Re: per-Re retained-snapshot raw mean, shape (n_Re, n_points)

Raw-space reconstruction:
  uv = mean_uv_by_Re[re_index] + coeff_uv[snapshot_id, :r] @ phi_uv[:r]
  p = mean_p_by_Re[re_index] + coeff_p[snapshot_id, :r] @ phi_p[:r]

Weighted inner product:
  <a,b>_M = sum_i point_volumes[i] * a[i] * b[i]
  for velocity [u,v], the same point_volumes are applied to both components.
