"""Make a Collision Detection System"""

from skimage.draw import circle_perimeter, disk
from einops import repeat
import matplotlib.pylab as plt
import numpy as np

SIZE = 1000


def create_pattern(rng_obj, num_shapes: int = 100, focus_ind: int = 20):
    location_grids = np.zeros(shape=(num_shapes, SIZE, SIZE))
    centres = rng_obj.integers(0, SIZE, size=(num_shapes, 2))
    pixels = []
    for ind, (y, x) in enumerate(centres):
        radius = rng_obj.integers(50, 100)
        shape_pixels = np.stack(circle_perimeter(y, x, radius, shape=(SIZE, SIZE)))
        if ind == focus_ind:
            shape_pixels = np.stack(disk((y, x), radius, shape=(SIZE, SIZE)))
        location_grids[ind, *shape_pixels] = 1
        pixels.append(shape_pixels)

    # Work out the indices of shapes that overlap with focus shape
    overlaps = np.sum(location_grids, axis=0)
    intersect_indices = np.argwhere(overlaps[*pixels[focus_ind]] > 1)
    intersect_pos = pixels[focus_ind][:, intersect_indices].squeeze()
    intersecting_circles = np.unique(
        np.argwhere(location_grids[:, *intersect_pos])[:, 0]
    )

    # Draw the overlapping and focus shapes in distinct colours
    show_grid = repeat(np.max(location_grids, axis=0), "w h -> w h c", c=3) + 0.2
    show_grid[*pixels[focus_ind], :] = [0, 0.8, 0.8]
    for circle_ind in intersecting_circles:
        if circle_ind != focus_ind:
            show_grid[*pixels[circle_ind], :] = [1, 0, 0]
    return show_grid


rng = np.random.default_rng(2)
fig, axs = plt.subplots(3, 3, figsize=(10, 10), squeeze=False)
for ax in axs.flatten():
    seed = rng.integers(0, 1e5)
    ax.imshow(create_pattern(np.random.default_rng(seed)), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.savefig("out/day21.png", bbox_inches="tight", transparent=True)
plt.show()
