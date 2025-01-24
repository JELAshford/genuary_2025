"""Geometric art - pick either a circle, rectangle, or triangle and use only that geometric shape."""

from skimage.draw import rectangle
import matplotlib.pylab as plt
import numpy as np

SIZE = 1000


def create_pattern(
    rng_obj,
    num_points: int = 250,
    step: float = 0.1,
    scale: float = 1.5,
):
    # out = np.ones(shape=(SIZE, SIZE))
    yy, xx = np.indices((SIZE, SIZE))
    out = yy / SIZE
    out[: SIZE // 2, :] = np.rot90(np.rot90(out[SIZE // 2 :, :]))
    # Generate spirals
    t = np.arange(0, num_points, step=step)
    for y, x in np.stack([scale * t * np.sin(t), scale * t * np.cos(t)]).T:
        width = 3
        colour = rng_obj.random()
        params = {"extent": (width, width), "shape": (SIZE, SIZE)}

        y, x = y + SIZE // 2, x + SIZE // 2
        yy, xx = rectangle((y - width // 2, x - width // 2), **params)
        out[yy, xx] = colour

        x = SIZE - x
        yy, xx = rectangle((y - width // 2, x - width // 2), **params)
        out[yy, xx] = colour

        y = SIZE - y
        yy, xx = rectangle((y - width // 2, x - width // 2), **params)
        out[yy, xx] = colour

        x = SIZE - x
        yy, xx = rectangle((y - width // 2, x - width // 2), **params)
        out[yy, xx] = colour

    # Rotate random sections
    num_squares = 500
    top_left = rng_obj.integers(0, SIZE - 100, size=(num_squares, 2))
    dims = rng_obj.integers(4, 100, size=(num_squares))

    show_grid = np.zeros(shape=(SIZE, SIZE, 3))
    for (y, x), (d) in zip(top_left, dims):
        segment = out[y : y + d, x : x + d].copy()
        show_grid[y : y + d, x : x + d, rng.integers(0, 3)] = np.rot90(segment)
    return show_grid


rng = np.random.default_rng(2)
fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
for ax in axs.flatten():
    seed = rng.integers(0, 1e5)
    ax.imshow(create_pattern(np.random.default_rng(seed)), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.savefig("out/day24.png", bbox_inches="tight", transparent=True)
plt.show()
