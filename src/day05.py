"""Isometric Art - redblobgames.com/grids/hexagons/ as reference"""

import matplotlib.pylab as plt
from itertools import product
from einops import rearrange
import skimage as ski
import numpy as np


def pointy_top_hex_coords(centre: np.array, hex_size: int):
    half_height = 0.5 * hex_size
    half_width = np.sqrt(3) * hex_size * 0.5
    rot_matrix = np.array([[-0.5, -np.sqrt(3) / 2], [np.sqrt(3) / 2, -0.5]])
    centred_top_coords = np.array(
        [
            [0, 0],
            [0 - half_height, 0 + half_width],
            [0 - hex_size, 0],
            [0 - half_height, 0 - half_width],
        ]
    )
    top_coords = (centred_top_coords) + centre
    right_coords = (centred_top_coords @ rot_matrix) + centre
    left_coords = (centred_top_coords @ rot_matrix @ rot_matrix) + centre
    return top_coords, left_coords, right_coords


def draw_cube(grid, centre, hex_size, side_cols=(1.0, 0.7, 0.4), inner=False):
    flip_matrix = np.array([[-1, 0], [0, -1]])
    for coords, col in zip(pointy_top_hex_coords(centre, hex_size), side_cols):
        if inner:
            coords = ((coords - centre) @ flip_matrix) + centre
        grid[ski.draw.polygon(*coords.T)] = col
    return grid


def draw_grid(rng_obj):
    grid = np.zeros(shape=(SIZE, SIZE))
    hex_size = int(0.1 * SIZE)
    x_centre = range(150, 900, int(np.sqrt(3) * hex_size))
    y_centre = range(150, 900, int(3 / 2) * hex_size)
    orient = rng_obj.integers(0, 2, size=(len(x_centre) * len(y_centre)))
    for (y, x), flip in zip(product(y_centre, x_centre), orient):
        grid = draw_cube(grid, (y, x), hex_size, inner=flip)
    return grid


def create_chrom_grid(rng_obj):
    """Draw a grid of cubes, repeated slightly offset on each channel"""

    def offset(grid, x, y):
        return np.roll(np.roll(grid, x, axis=0), y, axis=1)

    grid = draw_grid(rng_obj)
    return rearrange(
        [offset(grid, 0, 0), offset(grid, 15, 10), offset(grid, 25, 20)],
        "c h w -> h w c",
    )


SIZE = 1000
rng = np.random.default_rng(1701)

fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
for ax in axs.flatten():
    ax.imshow(create_chrom_grid(rng), cmap="gray", vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.savefig("out/day05.png", bbox_inches="tight")
plt.show()
