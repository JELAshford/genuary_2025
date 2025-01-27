"""Make something interesting with no randomness or noise or trig."""

import matplotlib.pylab as plt
import numpy as np


SIZE = 1000


def generate_rule_dictionary(rule_number: int = 30, window_size: int = 5):
    num_entries = 2**window_size
    rule_as_binary = f"{rule_number:b}".zfill(num_entries)
    return {
        tuple(map(int, f"{state:b}".zfill(window_size))): int(value)
        for state, value in zip(reversed(range(num_entries)), rule_as_binary)
    }


def create_pattern(rng_obj, window_size=5):
    automata_rule = generate_rule_dictionary(
        rule_number=int(0b10100101111110010011111111011110),
        window_size=window_size,
    )
    print(automata_rule)
    show_grid = np.zeros(shape=(SIZE, SIZE)).astype(int)
    show_grid[0, SIZE // 2] = 1
    for row in range(1, SIZE):
        row_data = np.pad(show_grid[row - 1, :], window_size // 2)
        windows = np.lib.stride_tricks.sliding_window_view(row_data, (window_size,))
        show_grid[row, :] = np.array([automata_rule[tuple(item)] for item in windows])
    return 1 - show_grid


rng = np.random.default_rng(1701)
fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
for ax in axs.flatten():
    seed = rng.integers(0, 1e5)
    grid = create_pattern(np.random.default_rng(seed))
    ax.imshow(grid[SIZE // 2 :, SIZE // 2 :], cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig("out/day27.png", bbox_inches="tight", transparent=True, dpi=300)
plt.show()
