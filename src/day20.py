"""Generative Architecture"""

from skimage.draw import ellipse_perimeter, line
from skimage.measure import label
import matplotlib.pylab as plt
from einops import rearrange
import numpy as np

SIZE = 1000


def create_pattern(rng_obj):
    show_grid = np.ones(shape=(SIZE, SIZE))

    # Draw the top and bottom parts of the column
    top_points = ellipse_perimeter(
        int((1 / 6) * SIZE), SIZE // 2, 100, 40, -1, shape=(SIZE, SIZE)
    )
    base_points = ellipse_perimeter(
        int((5 / 6) * SIZE), SIZE // 2, 200, 10, np.pi / 2, shape=(SIZE, SIZE)
    )
    show_grid[*top_points] = np.arange(len(top_points[0])) / len(top_points[0])
    show_grid[*base_points] = np.arange(len(base_points[0])) / len(base_points[0])

    # Connect points from top to bottom
    base_points = (base_points[0][::-1], base_points[1][::-1])
    for frac in np.arange(0, 80, 3) / 80:
        ind_top, int_bottom = (
            int(frac * len(top_points[0])),
            int(frac * len(base_points[0])),
        )

        line_points = line(
            top_points[0][ind_top],
            top_points[1][ind_top],
            base_points[0][int_bottom],
            base_points[1][int_bottom],
        )
        if len(line_points[0]) < 740:
            show_grid[*line_points] = 0

    # Detect regions in the image
    show_grid = show_grid.astype(int)
    areas = label(show_grid, connectivity=1)
    # outline_mask = areas == 0
    # background_mask = areas == 1
    show_grid = show_grid.astype(float)
    for index in range(2, np.max(areas)):
        show_grid[areas == index] = (rng.random() * (1 / 2)) + 0.5

    r_grid = show_grid.copy()
    g_grid = np.roll(show_grid.copy(), 10, axis=1)
    # b_grid = np.roll(show_grid.copy(), 10, axis=1)
    vis_grid = rearrange(
        [r_grid, g_grid, np.ones(shape=(SIZE, SIZE))], "c h w -> h w c"
    )
    return vis_grid


rng = np.random.default_rng(3)
fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
for ax in axs.flatten():
    seed = rng.integers(0, 1e5)
    ax.imshow(create_pattern(np.random.default_rng(seed)), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.savefig("out/day20.png", bbox_inches="tight", transparent=True)
plt.show()
