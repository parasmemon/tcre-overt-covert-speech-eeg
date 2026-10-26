"""
Figure 1: positions of the four TCRE arrays (F4, P6, F3, P5 of the 10-10 system), top view of the head.
Midline and temporal positions are drawn for orientation only. No data are needed.

Usage:
    python make_fig1.py /path/to/output_dir
"""
import sys
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Ellipse
import mne
from mne.channels.layout import _find_topomap_coords

INK, INK_2, GRID, TRI = "#0b0b0b", "#52514e", "#d9d8d3", "#5b3fb0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
ref = ["Fz", "Cz", "Pz", "T7", "T8", "Oz"]; sites = ["F4", "P6", "F3", "P5"]
names = ref + sites; info = mne.create_info(names, 250, "eeg"); info.set_montage("standard_1005")
pos = _find_topomap_coords(info, picks=list(range(len(names))), sphere=None); R = 0.095

fig = plt.figure(figsize=(3.4, 3.2)); ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
ax.set_aspect("equal"); ax.axis("off")
ax.add_patch(Circle((0, 0), R, fc="white", ec=INK_2, lw=1.0))
ax.plot([-0.013, 0, 0.013], [R * 0.99, R * 1.13, R * 0.99], color=INK_2, lw=1.0)
for s in (-1, 1): ax.add_patch(Ellipse((s * R * 1.04, 0), 0.012, 0.032, fc="white", ec=INK_2, lw=1.0))
for (x, y), n in zip(pos[:len(ref)], ref):
    ax.plot(x, y, "o", ms=2.4, color=GRID, mec=INK_2, mew=0.4)
    ax.text(x + 0.006, y, n, ha="left", va="center", fontsize=6.5, color=INK_2)
for i, ((x, y), n) in enumerate(zip(pos[len(ref):], sites)):
    ax.add_patch(Rectangle((x - 0.012, y - 0.012), 0.024, 0.024, fc=TRI, ec="none"))
    ax.text(x, y, str(i + 1), ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    below = n.startswith("F")
    ax.text(x, y - 0.016 if below else y + 0.016, n, ha="center", va="top" if below else "bottom", fontsize=9, color=INK, fontweight="bold")
ax.set_xlim(-0.12, 0.12); ax.set_ylim(-0.102, 0.118)
out = Path(sys.argv[1] if len(sys.argv) > 1 else ".").expanduser(); out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "fig1_array_positions.png", dpi=600, facecolor="white")
print("saved", out / "fig1_array_positions.png")
