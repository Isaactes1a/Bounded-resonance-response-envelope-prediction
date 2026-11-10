# -*- coding: utf-8 -*-
"""Where does each model put its field nulls at the antiresonance frequency?"""

import numpy as np

data = np.load("../code/results/dip_field/nominal_dip_field.npz")
grid = int(data["grid"])
nodes = np.asarray(data["nodes"], dtype=float)
x = nodes[:, 0].reshape(grid, grid)
y = nodes[:, 1].reshape(grid, grid)
interior = (x > 0.02) & (x < 0.98) & (y > 0.02) & (y < 0.98)

print(f"frequency {float(data['frequency_hz']):.1f} Hz, grid {grid}x{grid}")
for key in ("direct", "static", "learned"):
    amplitude = np.asarray(data[key], dtype=float).reshape(grid, grid)
    inside = amplitude[interior]
    # local minima far below the field maximum
    threshold = 0.02 * amplitude.max()
    count = int((inside < threshold).sum())
    index = np.unravel_index(int(np.argmin(np.where(interior, amplitude, np.inf))),
                             amplitude.shape)
    print(f"{key:8s} max {amplitude.max():.5f} mm  interior min "
          f"{inside.min():.3e} mm at (x,y)=({x[index]:.3f},{y[index]:.3f})  "
          f"nodes below 2% of max: {count}")
    # null positions along the diagonal and along y = 7/30
    row = int(round(7 / 30 * (grid - 1)))
    line = amplitude[row, :]
    low = np.flatnonzero(line < 0.02 * amplitude.max())
    print(f"         nulls on the row y=7/30 at x = {np.round(x[row, low], 3)}")
com = np.asarray(data["direct_complex"], dtype=complex).reshape(grid, grid)
sta = np.asarray(data["static_complex"], dtype=complex).reshape(grid, grid)
lea = np.asarray(data["learned_complex"], dtype=complex).reshape(grid, grid)
print("\ncomplex residual of the two approximations against the reference")
peak = np.abs(com).max()
print(f"  static   |dw|/max: mean {np.abs(sta - com).mean() / peak:.3e}  "
      f"max {np.abs(sta - com).max() / peak:.3e}")
print(f"  learned  |dw|/max: mean {np.abs(lea - com).mean() / peak:.3e}  "
      f"max {np.abs(lea - com).max() / peak:.3e}")
