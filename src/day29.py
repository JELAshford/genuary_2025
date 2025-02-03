"""Grid-based graphic design"""

from skimage.draw import rectangle
import matplotlib.pylab as plt
from einops import rearrange
import numpy as np


SIZE = 1000


def create_pattern(rng_obj, num_rects=1_000, grid_size=64):
    random_positions = rng_obj.integers(0, SIZE, size=(num_rects, 2))
    random_sizes = rng_obj.integers(30, 100, size=(num_rects, 2))

    # Generate image
    show_grid = np.zeros(shape=(SIZE, SIZE, 3)).astype(int)
    for pos, sizes in zip(random_positions, random_sizes):
        colour = rng.integers(0, 255, size=(3,))
        colour[0] = 0
        yy, xx = rectangle(pos, extent=sizes, shape=(SIZE, SIZE))
        show_grid[yy, xx, :] = colour

    # Apply random rotations in a grid
    grid_width = SIZE // grid_size
    positions = rearrange(
        np.meshgrid(np.arange(grid_size + 2), np.arange(grid_size + 2)),
        "b w h -> (w h) b",
    )
    num_rotations = rng_obj.choice(np.arange(1, 4), size=(len(positions)))
    for (y, x), rots in zip(positions, num_rotations):
        gx, gy = int(y * grid_width), int(x * grid_width)
        segment = show_grid[gy : (gy + grid_width), gx : (gx + grid_width), :].copy()
        show_grid[gy : (gy + grid_width), gx : (gx + grid_width), :] = np.rot90(
            segment, k=rots
        )

    return show_grid


rng = np.random.default_rng(1701)
fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
for ax in axs.flatten():
    seed = rng.integers(0, 1e5)
    grid = create_pattern(np.random.default_rng(seed))
    ax.imshow(grid, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig("out/day29.png", bbox_inches="tight", transparent=True, dpi=300)
plt.show()
