"""Tau Day"""

import matplotlib.pylab as plt
from einops import rearrange
import numpy as np


TAU = 2 * np.pi
ONE = int(TAU / TAU)
TWO = ONE + ONE
SIZE = int(TAU * TAU * TAU)


def create_pattern(rng_obj):
    def random_vels(rng, num_walkers):
        frac = ONE + ONE / TAU
        return rng.choice(
            [-frac, -ONE, ONE, frac], replace=True, size=(TWO, num_walkers)
        )

    def make_grid(rng):
        num_walkers = int(TAU * TAU * TWO)
        pos = np.ones(shape=(TWO, num_walkers)) * SIZE // TWO
        vel = random_vels(rng, num_walkers)
        grid = np.zeros(shape=(SIZE, SIZE))
        for step in range(int(TAU * TAU * TAU)):
            if step % int(TAU) == ONE:
                vel = random_vels(rng, num_walkers)
            grid[*pos.astype(int)] += ONE / TAU
            pos += vel
            pos %= SIZE
        return grid

    r_grid = make_grid(rng_obj)
    g_grid = make_grid(rng_obj)
    b_grid = make_grid(rng_obj)
    grid = rearrange([r_grid, g_grid, b_grid], "c h w -> h w c")
    return grid


rng = np.random.default_rng(1701)

fig, axs = plt.subplots(ONE, ONE, figsize=(int(TAU), int(TAU)), squeeze=False)
for ax in axs.flatten():
    ax.imshow(create_pattern(rng), cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.savefig("out/day10.png", bbox_inches="tight")
plt.show()
