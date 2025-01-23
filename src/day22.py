"""Gradients Only"""

import matplotlib.pylab as plt
import numpy as np

SIZE = 500


def fl_dithering(image_array, quant_levels=1):
    fl_quantised = image_array.copy()
    height, width = fl_quantised.shape
    for y in range(height):
        for x in range(width):
            old_pixel = fl_quantised[y, x]
            new_pixel = np.floor(old_pixel * quant_levels) / quant_levels
            fl_quantised[y, x] = new_pixel
            error = old_pixel - new_pixel
            if (x != 0) and (x < (width - 1)) and (y < (height - 1)):
                fl_quantised[y, x + 1] += error * (2 / 4)
                fl_quantised[y + 1, x] += error * (1 / 4)
                fl_quantised[y + 1, x - 1] += error * (1 / 4)
    return fl_quantised


def mae_dithering(image_array, quant_levels=1):
    mae_quantised = image_array.copy()
    height, width = mae_quantised.shape
    for y in range(height):
        for x in range(width):
            old_pixel = mae_quantised[y, x]
            new_pixel = np.floor(old_pixel * quant_levels) / quant_levels
            mae_quantised[y, x] = new_pixel
            error = old_pixel - new_pixel
            if (x > 1) and (x < (width - 3)) and (y < (height - 3)):
                mae_quantised[y, x + 1] += error * (7 / 48)
                mae_quantised[y, x + 2] += error * (5 / 48)
                mae_quantised[y + 1, x - 2] += error * (3 / 48)
                mae_quantised[y + 1, x - 1] += error * (5 / 48)
                mae_quantised[y + 1, x] += error * (7 / 48)
                mae_quantised[y + 1, x + 1] += error * (5 / 48)
                mae_quantised[y + 1, x + 2] += error * (4 / 48)
                mae_quantised[y + 2, x - 2] += error * (1 / 48)
                mae_quantised[y + 2, x - 1] += error * (3 / 48)
                mae_quantised[y + 2, x] += error * (5 / 48)
                mae_quantised[y + 2, x + 1] += error * (3 / 48)
                mae_quantised[y + 2, x + 2] += error * (1 / 48)
    return mae_quantised


def create_pattern(rng_obj):
    def lerp(v1, v2, t):
        return t * (v2) + (1 - t) * v1

    yy, xx = np.indices((SIZE, SIZE))
    lower_tri = np.tril_indices(SIZE)

    gradient_grid = yy / SIZE
    gradient_grid *= xx / SIZE

    mae = mae_dithering(gradient_grid, quant_levels=10)
    fl = fl_dithering(gradient_grid, quant_levels=10)

    show_image = mae.copy()
    show_image[lower_tri] = fl[lower_tri]
    # np.fill_diagonal(show_image, 0)
    return show_image


rng = np.random.default_rng(2)
fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
for ax in axs.flatten():
    seed = rng.integers(0, 1e5)
    ax.imshow(create_pattern(np.random.default_rng(seed)), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.savefig("out/day22.png", bbox_inches="tight", transparent=True)
plt.show()
