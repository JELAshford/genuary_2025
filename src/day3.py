"""42 Lines"""

import matplotlib.pylab as plt
from einops import rearrange
import numpy as np


def draw_circles(y, x, centres, radii):
    grid = np.ones(shape=(SIZE, SIZE))
    for (y_off, x_off), radius in zip(centres, radii):
        circle_grid = np.sqrt((y - y_off) ** 2 + (x - x_off) ** 2) - radius
        grid = np.minimum(grid, circle_grid)
    return grid


SIZE = 1000
NUM_CIRCLES = 100
OFFSET = 20
rng = np.random.default_rng(1701)
half_size = SIZE // 2

# Create a centred distance grid
y, x = np.indices((SIZE, SIZE))
y, x = y - half_size, x - half_size

# Draw circles at and offset from start pos
centres = rng.integers(-half_size, half_size, size=(NUM_CIRCLES, 2))
radii = rng.integers(20, 100, size=NUM_CIRCLES).astype(float)
r_grid = draw_circles(y, x, centres, radii)
b_grid = draw_circles(y, x, centres + OFFSET, radii * 0.9)
g_grid = draw_circles(y, x, centres + OFFSET * 2, radii * 0.5)

# Show and save
show_grid = rearrange([r_grid, b_grid, g_grid], "c h w -> h w c")
show_grid = (show_grid - np.min(show_grid)) / (np.max(show_grid) - np.min(show_grid))

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(show_grid, cmap="gray")
ax.set_xticks([])
ax.set_yticks([])
plt.savefig("out/day3.png", bbox_inches="tight")
plt.show()
