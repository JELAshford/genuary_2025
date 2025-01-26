"""Symmetry"""

from skimage.draw import circle_perimeter
import matplotlib.pylab as plt
from einops import rearrange
import numpy as np


SIZE = 1000


def create_pattern(rng_obj):
    # Generate set of starting points (central circle)
    mid, off = SIZE // 2, SIZE // 10
    points = np.stack(circle_perimeter(mid, mid, off))
    points = rearrange(points, "c l -> l c")

    # Get points after mirroring around lines
    for val in range(6):
        # Define a random line
        line_point = rng_obj.choice(points)
        angle = rng_obj.random() * 2 * np.pi
        # Get nearest points on line
        M_normalized = np.array([np.cos(angle), np.sin(angle)])
        proj_lengths = np.dot(points - line_point, M_normalized)
        orthog_points = line_point + proj_lengths[:, np.newaxis] * M_normalized
        # Generate new points on line
        new_points = points + ((orthog_points - points) * 2)
        points = np.unique(np.vstack([points, new_points]), axis=0)

    # Scale all the points
    largest_range = max(np.max(points, axis=0) - np.min(points, axis=0))
    if largest_range > int(SIZE * (3 / 4)):
        scale_factor = SIZE / (largest_range + SIZE)
        points = (np.array([[scale_factor, 0], [0, scale_factor]]) @ points.T).T
    # Centre all the points
    centering_vec = np.array([mid, mid]) - np.mean(points, axis=0)
    points += centering_vec
    # Display all the generated points
    show_grid = np.zeros(shape=(SIZE, SIZE))
    show_grid[*points.astype(int).T] = 1

    return show_grid


rng = np.random.default_rng(1701)
fig, axs = plt.subplots(8, 8, figsize=(10, 10), squeeze=False)
for ax in axs.flatten():
    seed = rng.integers(0, 1e5)
    ax.imshow(create_pattern(np.random.default_rng(seed)), cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig("out/day26.png", bbox_inches="tight", transparent=True, dpi=300)
plt.show()
