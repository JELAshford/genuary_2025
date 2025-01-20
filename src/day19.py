"""Op Art"""

import matplotlib.pylab as plt

import numpy as np

SIZE = 1000


def create_pattern(rng_obj, steps: int = 30, points: int = 3000):
    def step_scale(x, minima):
        return min([np.abs(np.tanh((x - m) / (CHECK_SIDE * 5))) ** 0.8 for m in minima])

    IM_HEIGHT = int((4 / 5) * SIZE)
    CHECK_SIDE = SIZE // 20
    v_lines = [int(0.1 * SIZE), int(0.3 * SIZE), int(0.8 * SIZE)]
    h_lines = [int(0.1 * SIZE), int(0.5 * SIZE)]
    # Generate a slice
    show_slice = np.zeros(shape=(1, SIZE))
    x, step, fill = 0, 0, CHECK_SIDE
    while x < SIZE:
        show_slice[:, x : (x + step)] = fill
        fill = 1 - fill
        next_step = CHECK_SIDE * step_scale(x, v_lines)
        next_step = int(max(min(CHECK_SIDE, next_step), 2))
        step = int((step + next_step) / 2)
        x += step
    # Repeat vertically with generated heights
    show_grid = np.zeros(shape=(IM_HEIGHT, SIZE))
    y, step, ind = 0, CHECK_SIDE, 0
    while y < IM_HEIGHT:
        this_slice = show_slice.copy()
        this_slice = this_slice if ind % 2 == 0 else 1 - this_slice
        show_grid[y : (y + step), :] = this_slice
        ind += 1

        next_step = CHECK_SIDE * step_scale(y, h_lines)
        next_step = int(max(min(CHECK_SIDE, next_step), 2))
        step = int((step + next_step) / 2)
        y += step

    return show_grid


rng = np.random.default_rng(3)
fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
for ax in axs.flatten():
    seed = rng.integers(0, 1e5)
    ax.imshow(create_pattern(np.random.default_rng(seed)), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.savefig("out/day19.png", bbox_inches="tight")
plt.show()
