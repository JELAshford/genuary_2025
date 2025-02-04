"""Pixel sorting"""

import matplotlib.pylab as plt
from PIL import Image
import numpy as np


SIZE = 1000


def create_pattern():
    day2 = Image.open("out/day02.png").resize((SIZE, SIZE))
    colour_array = np.array(day2.convert("RGB"))
    im = np.array(day2.convert("L"))
    return colour_array[np.arange(SIZE), np.argsort(im, axis=0), :]


fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
for ax in axs.flatten():
    ax.imshow(create_pattern())
    ax.set_xticks([])
    ax.set_yticks([])
fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig("out/day31.png", bbox_inches="tight", transparent=True, dpi=300)
plt.show()
