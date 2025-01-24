"""Gradients Only"""

import matplotlib.pylab as plt
import numpy as np

SIZE = 500


def create_pattern(rng_obj, num_blocks: int = 500):
    background = np.ones(shape=(SIZE, SIZE)) * 0.1
    foreground = np.zeros(shape=(SIZE, SIZE))

    # Add floor
    foreground[-1, :] = 1.0
    for block in range(num_blocks):
        # Parameterise the block
        width = rng_obj.integers(5, 50)
        height = int(width * 0.2)
        if rng.random() > 0.9:
            width, height = height, width

        x_from_centre = rng_obj.integers(-SIZE // 2, SIZE // 2)
        x_offset = int((SIZE // 2) + x_from_centre)

        # Find height to draw it at
        block_column = foreground[:, x_offset : (x_offset + width)]
        highest_point = np.argmax(np.sum(block_column, axis=1) > 0)
        set_down_height = highest_point if highest_point != 0 else SIZE

        # Draw
        foreground[
            set_down_height - height : set_down_height, x_offset : (x_offset + width)
        ] = (rng.random() / 2) + 0.5

    # Combine
    out = foreground.copy()
    out[foreground == 0] = background[foreground == 0]
    return out


rng = np.random.default_rng(2)
fig, axs = plt.subplots(2, 2, figsize=(10, 10), squeeze=False)
for ax in axs.flatten():
    seed = rng.integers(0, 1e5)
    ax.imshow(create_pattern(np.random.default_rng(seed)), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.savefig("out/day23.png", bbox_inches="tight", transparent=True)
plt.show()
