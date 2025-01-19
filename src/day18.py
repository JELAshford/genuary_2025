"""What does the wind look like?"""

from perlin_numpy import generate_perlin_noise_2d
from einops import rearrange
import matplotlib.pylab as plt
import numpy as np

SIZE = 1000


def create_pattern(rng_obj, steps: int = 30, points: int = 3000):
    def create_grid():
        np.random.seed(rng_obj.integers(0, 1e5))
        x_grid = np.ones(shape=(SIZE, SIZE))
        y_grid = generate_perlin_noise_2d((SIZE, SIZE), (5, 5))
        show_grid = np.ones(shape=(SIZE, SIZE))
        positions = rng_obj.integers(0, SIZE, size=(2, points)).astype(float)
        for step in range(steps):
            show_grid[*positions.astype(int)] = np.minimum(0.9, 1 - step / steps)
            x_vel, y_vel = (
                x_grid[*positions.astype(int)],
                y_grid[*positions.astype(int)],
            )
            vels = np.stack([y_vel, x_vel]).astype(float)
            positions += vels * 1
            positions %= SIZE
        return show_grid

    r_grid = create_grid()
    g_grid = create_grid()
    b_grid = create_grid()
    show_grid = rearrange([r_grid * g_grid, g_grid, b_grid], "c h w -> h w c")
    return show_grid.astype(float)


rng = np.random.default_rng(3)
fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
for ax in axs.flatten():
    seed = rng.integers(0, 1e5)
    ax.imshow(create_pattern(np.random.default_rng(seed)), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.savefig("out/day18.png", bbox_inches="tight")
plt.show()
