"""What if PI=4"""

import matplotlib.pylab as plt
from skimage.draw import line_aa, line
from einops import rearrange
import numpy as np

SIZE = 2000


def create_pattern(rng_obj):
    """what indeed! let's pretent there's 8 radians in a circle"""

    def angle_to_vec(radians: float):
        return np.array([np.cos(radians), np.sin(radians)])

    def draw_spiral(grid, centre, rotation, colour, scale):
        for angle in np.linspace(0, 8, 80):
            vec = (angle_to_vec(rotation + angle) * 100 * scale * angle / 8).astype(int)
            end = centre + vec
            yy, xx, val = line_aa(centre[0], centre[1], end[0], end[1])
            yy, xx = np.clip(yy, 0, SIZE - 1), np.clip(xx, 0, SIZE - 1)
            grid[yy, xx] = (val / 255) * colour
        return grid

    num_spirals = 300
    show_grid = np.zeros(shape=(SIZE, SIZE))
    centres = rng_obj.integers(0, SIZE, size=(num_spirals, 2))
    rotation = rng_obj.random(size=(num_spirals)) * 8
    for ind, (centre, rot) in enumerate(zip(centres, rotation)):
        show_grid = draw_spiral(
            show_grid, centre, rot, 1 - ind / num_spirals, 2 - ind / num_spirals
        )
    return show_grid.astype(float)


rng = np.random.default_rng(3)
fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
for ax in axs.flatten():
    ax.imshow(create_pattern(rng), cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
# fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig("out/day17.png", bbox_inches="tight")
plt.show()
