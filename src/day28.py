"""Infinite Scroll: random walks redux"""

import matplotlib.animation as animation
import matplotlib.pylab as plt
import numpy as np


SIZE = 50


def batch_returning_walkers(rng_obj, num_steps: int, step_scale: int, num_walkers: int):
    steps = (rng.random(size=(num_walkers, num_steps)) - 0.5) * step_scale
    steps[:, 0] = 0
    steps[:, 1:] -= np.sum(steps, axis=1, keepdims=True) / (SIZE - 1)
    return steps.T


def init_func():
    ax.set_xticks([])
    ax.set_yticks([])
    im.set_data(show_grid)
    return (im,)


def data_func():
    for step in walker_steps:
        yield step


def update_func(step):
    global im, show_grid, positions
    show_grid[*positions.astype(int).T] = colours
    show_grid *= 0.8
    positions[:, 0] = (positions[:, 0] + 1) % SIZE
    positions[:, 1] = (positions[:, 1] + step) % SIZE
    im.set_data(show_grid)
    return (im,)


NUM_WALKERS = 30
STEP_SCALE = 3

# Setup initial conditons
rng = np.random.default_rng(1701)
show_grid = np.zeros(shape=(SIZE, SIZE))
positions = rng.integers(0, SIZE, size=(NUM_WALKERS, 2)).astype(float)
colours = rng.random(size=(NUM_WALKERS))

# Define random walker behaviour
walker_steps = batch_returning_walkers(
    rng,
    num_steps=SIZE,
    step_scale=STEP_SCALE,
    num_walkers=NUM_WALKERS,
)
walker_steps = np.vstack([walker_steps, walker_steps])

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
im = ax.imshow(show_grid, cmap="gray", vmin=0, vmax=1)

# Pre_run
for ind, data in enumerate(data_func()):
    _ = update_func(data)
    if ind >= (SIZE - 1):
        break

# Generate animation
ani = animation.FuncAnimation(
    fig,
    func=update_func,
    frames=data_func,
    interval=10,
    init_func=init_func,
    save_count=SIZE,
    blit=True,
)
ani.save("out/day28.gif", writer="pillow")
plt.show()
