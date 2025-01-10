"""Layers"""

import matplotlib.pylab as plt
from einops import rearrange
import numpy as np

SIZE = 1000
grid = np.ones(shape=(SIZE, SIZE))
rng = np.random.default_rng(1701)

# Create distance array for drawing circles
x, y = np.indices(grid.shape)
distance_array = np.sqrt((x - SIZE // 2) ** 2 + (y - SIZE // 2) ** 2)

# Draw random circles
for y, x in rng.integers(10, SIZE - 10, size=(100, 2)):
    circle_mask = distance_array < rng.integers(20, 100, size=1)[0]
    circle_mask = np.roll(
        np.roll(circle_mask, y - SIZE // 2, axis=0), x - SIZE // 2, axis=1
    )
    grid[circle_mask] = rng.random()

# Offset colours
offset = 10
grid_red = grid.copy()
grid_green = np.roll(np.roll(grid.copy(), offset, axis=0), offset, axis=1)
grid_blue = np.roll(np.roll(grid.copy(), -offset, axis=0), -offset, axis=1)
grid = rearrange([grid_red, grid_green, grid_blue], "b h w -> h w b")

# Apply copies of top-left wuarter to other quarter
mid = SIZE // 2
quart = SIZE // 4
corner = grid[:mid, :mid]
grid[:mid, mid:] *= 1 / corner
grid[mid:, :mid] *= corner
grid[mid:, mid:] *= 1 / corner

# Spin concentric squaures from outside-in
shade_size = 5
for inset in np.linspace(0, 0.5, 15):
    dist = int(inset * SIZE)
    back_dist = SIZE - dist
    rotated_section = np.rot90(grid[dist:back_dist, dist:back_dist].copy())
    # Apply a darkening just wider than the patch, to give a shadow
    grid[
        (dist - 1) : (back_dist + shade_size),
        (dist - 1) : (back_dist + shade_size),
    ] *= 0.8
    # Pare the rotated section
    grid[dist:back_dist, dist:back_dist] = rotated_section

# Show and save
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(grid, cmap="gray")
ax.set_xticks([])
ax.set_yticks([])
plt.savefig("out/day02.png", bbox_inches="tight")
plt.show()
