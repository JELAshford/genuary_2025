"""Abstract Map"""

from perlin_numpy import generate_perlin_noise_2d
import matplotlib.pylab as plt
import numpy as np


SIZE = 1000


def create_pattern(seed):
    # Generate noise map
    np.random.seed(seed)
    noises = np.array(
        [
            generate_perlin_noise_2d((SIZE, SIZE), (2, 2)),
            generate_perlin_noise_2d((SIZE, SIZE), (5, 5)),
            generate_perlin_noise_2d((SIZE, SIZE), (10, 10)),
        ]
    )
    combo = np.sum(noises, axis=0)
    combo = (combo**4) * 10
    combo = np.round(combo, 2)
    # Generate lines
    return np.isin(combo, np.arange(0, 10, step=1) / 10)


rng = np.random.default_rng(1701)
fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
for ax in axs.flatten():
    seed = rng.integers(0, 1e5)
    grid = create_pattern(seed)
    ax.imshow(grid, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig("out/day30.png", bbox_inches="tight", transparent=True, dpi=300)
plt.show()
