"""Textile design patterns of public transport seating - niche!"""

import matplotlib.pylab as plt
from einops import rearrange
import numpy as np


def create_pattern(rng_obj):
    y, x = np.indices((SIZE, SIZE))
    grid = np.sin(x * 0.05) * np.cos(y * 0.05)
    grid = (grid > 0.3).astype(float)
    r_grid = grid.copy()
    g_grid = np.roll(np.roll(grid.copy(), 25, axis=0), 25, axis=1)
    grid = rearrange([r_grid, g_grid, np.ones(shape=(SIZE, SIZE))], "c h w -> h w c")

    step = SIZE // 10
    grid_slice = grid[:, -step:, :]
    print(grid_slice.shape)
    for idx, start in enumerate(np.arange(0, SIZE, step)):
        print(start)
        start = int(start)
        this_slice = grid_slice if idx % 2 == 0 else np.fliplr(grid_slice)
        grid[:, start : (start + step), :] = this_slice

    grid *= np.roll(np.roll(grid, 1, axis=2), 30, axis=0)

    return grid


SIZE = 1000
rng = np.random.default_rng(1701)

fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
for ax in axs.flatten():
    ax.imshow(create_pattern(rng), cmap="gray", vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.savefig("out/day09.png", bbox_inches="tight")
plt.show()
