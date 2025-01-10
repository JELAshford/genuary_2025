"""Landscape with a Primitive Shape"""

import matplotlib.pylab as plt
from shapely import Polygon, Point
from typing import Tuple
import skimage as ski
import numpy as np

SIZE = 1000


def sample_points_in_poly(rng_obj, poly, n: int = 100, max_attempts=10_000):
    (minx, miny, maxx, maxy) = poly.bounds
    points = []
    attempts = 0
    while (len(points) < n) or (attempts > max_attempts):
        x = (rng_obj.random() * (maxx - minx)) + minx
        y = (rng_obj.random() * (maxy - miny)) + miny
        if poly.contains(Point(x, y)):
            points.append((x, y))
        attempts += 1
    return points


def random_rotated_squares(
    rng_obj, area_poly: Polygon, scale_range: Tuple[int, int], num_points: int
):
    square_centres = sample_points_in_poly(rng_obj, area_poly, n=num_points)
    square_rotations = rng_obj.random(size=(num_points)) * 2 * np.pi
    square_scales = rng_obj.integers(*scale_range, size=(num_points))

    y_indices, x_indices = [], []
    for centre, rot, scale in zip(square_centres, square_rotations, square_scales):
        half = scale // 2
        rot_matrix = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
        square_coords = np.array(
            [[-half, -half], [-half, half], [half, half], [half, -half]]
        )
        square_coords = square_coords @ rot_matrix
        square_coords = square_coords + centre
        yy, xx = ski.draw.polygon(*square_coords.T, shape=(SIZE, SIZE))
        y_indices.append(yy)
        x_indices.append(xx)
    return y_indices, x_indices


def draw_squares(
    rng_obj,
    grid,
    area: Polygon,
    size_range: Tuple[int, int],
    gray_range: Tuple[float, float],
    num_squares: int,
):
    (square_y, square_x) = random_rotated_squares(
        rng_obj,
        area,
        size_range,
        num_squares,
    )
    colours = rng_obj.uniform(*gray_range, size=num_squares)
    for yy, xx, col in zip(square_y, square_x, colours):
        grid[yy, xx] = col
    return grid


def create_image(rng_obj: np.random.Generator):
    def quickdraw(poly, sizes, grays, n):
        return draw_squares(rng_obj, grid, poly, sizes, grays, n)

    def scale(point_list):
        return np.array(point_list) * SIZE

    grid = np.ones(shape=(SIZE, SIZE))
    # Sky
    grid = quickdraw(
        Polygon(scale([[0, 0], [0, 1], [0.5, 1], [0.5, 0]])),
        (200, 250),
        (0.9, 0.95),
        100,
    )
    # Ground
    grid = quickdraw(
        Polygon(scale([[0.55, 0], [0.55, 1], [1, 1], [1, 0]])),
        (50, 110),
        (0.1, 0.12),
        800,
    )
    # Lake
    grid = quickdraw(
        Polygon(scale([[1, 0.4], [0.55, 0.25], [0.55, 0.7], [1, 0.6]])),
        (80, 100),
        (0.5, 0.55),
        200,
    )
    # Mountain reflection
    grid = quickdraw(
        Polygon(
            scale([[0.55, 0.2], [0.65, 0.4], [0.85, 0.5], [0.65, 0.6], [0.55, 0.8]])
        ),
        (40, 50),
        (0.45, 0.5),
        400,
    )
    # Mountain
    grid = quickdraw(
        Polygon(scale([[0.55, 0.2], [0.4, 0.4], [0.2, 0.5], [0.4, 0.6], [0.55, 0.8]])),
        (40, 50),
        (0.15, 0.2),
        500,
    )
    # Mountain Peak
    grid = quickdraw(
        Polygon(scale([[0.27, 0.45], [0.15, 0.5], [0.25, 0.55]])),
        (20, 25),
        (0.6, 0.65),
        100,
    )

    grid = quickdraw(
        Polygon(
            scale(
                [
                    [0.5, 0.15],
                    [0.55, 0.4],
                    [0.55, 0.6],
                    [0.5, 0.85],
                    [0.57, 0.85],
                    [0.57, 0.15],
                ]
            )
        ),
        (40, 50),
        (0.1, 0.12),
        200,
    )

    return grid


rng = np.random.default_rng(1705)
fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
for ax in axs.flatten():
    ax.imshow(create_image(rng), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.savefig("out/day06.png", bbox_inches="tight")
plt.show()
