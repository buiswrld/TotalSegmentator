"""
Analytic validation of the scipy NSD implementation (no installs). Two concentric solid
spheres, radii 20 mm and 22 mm, on a 1.5 mm isotropic grid, so their surfaces are 2 mm apart.
Expectation: NSD ~ 1.0 at tolerance 3 mm (2 mm < 3 mm), ~ 0 at tolerance 1 mm (2 mm > 1 mm).
Also confirms distances are in millimetres (measured surface separation ~ 2 mm) and prints the
connectivity defining boundary voxels. Uses the exact surf()/nsd() from _nsd.py.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
from experiments.rebuttal._nsd import surf, nsd

SPACING = (1.5, 1.5, 1.5)
N = 64                              # voxels per side; physical extent 96 mm, centre at 48 mm
c = (N - 1) / 2.0
ax = (np.arange(N) - c) * SPACING[0]
X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
R = np.sqrt(X**2 + Y**2 + Z**2)     # mm from centre
inner = R <= 20.0                   # pred: 20 mm sphere
outer = R <= 22.0                   # ref:  22 mm sphere

res = nsd(inner, outer, SPACING)

# measured surface separation (mm) between the two surfaces, to prove mm units
si, so = surf(inner), surf(outer)
d_i_to_o = distance_transform_edt(~so, sampling=SPACING)[si]   # inner surface -> outer surface
d_o_to_i = distance_transform_edt(~si, sampling=SPACING)[so]

st = generate_binary_structure(3, 1)
print("connectivity for boundary voxels: scipy binary_erosion default =")
print("  generate_binary_structure(3, 1) -> 6-connectivity (face-connected); neighbours =",
      int(st.sum()) - 1)
print(f"\nsphere radii: inner 20 mm, outer 22 mm  -> surfaces nominally 2 mm apart")
print(f"inner surface voxels = {int(si.sum())}, outer surface voxels = {int(so.sum())}")
print(f"measured inner->outer surface distance: mean {d_i_to_o.mean():.3f} mm, "
      f"median {np.median(d_i_to_o):.3f} mm, max {d_i_to_o.max():.3f} mm")
print(f"measured outer->inner surface distance: mean {d_o_to_i.mean():.3f} mm, "
      f"median {np.median(d_o_to_i):.3f} mm")
print(f"\nNSD results (expected ~0 @1mm, ~1 @3mm):")
for t in (1.0, 2.0, 3.0):
    print(f"  NSD @ {int(t)} mm = {res[t]:.4f}")

# sanity: identical masks -> NSD exactly 1.0 at every tolerance
same = nsd(outer, outer, SPACING)
print(f"\nidentical-mask control: NSD @1/2/3 mm = "
      f"{same[1.0]:.4f} / {same[2.0]:.4f} / {same[3.0]:.4f}  (expected 1.0)")

ok1 = res[1.0] < 0.15
ok3 = res[3.0] > 0.90
okmm = abs(d_i_to_o.mean() - 2.0) < 0.6
print(f"\nPASS checks: NSD@1mm near 0 -> {ok1} ; NSD@3mm near 1 -> {ok3} ; "
      f"mean distance ~2mm -> {okmm}")
