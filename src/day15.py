"""Design a Carpet"""

import matplotlib.pylab as plt
from einops import rearrange
import numpy as np

SIZE = 256


def create_pattern(rng_obj):
    yy, xx = np.indices((SIZE, SIZE))
    combo = (xx + yy) ^ abs(yy - xx)
    r_grid = combo % 12
    g_grid = combo % 9
    b_grid = combo % 15
    show_grid = rearrange([r_grid, g_grid, b_grid], "c h w -> h w c") > 6
    show_grid[:, :, 0] *= show_grid[:, :, 1]
    show_grid[:, :, 2] *= show_grid[:, :, 1]
    return show_grid.astype(float)


rng = np.random.default_rng(3)
fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
for ax in axs.flatten():
    ax.imshow(create_pattern(rng), cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig("out/day15.png", bbox_inches="tight")
plt.show()
