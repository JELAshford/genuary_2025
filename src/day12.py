"""Subdivision"""

from skimage.draw import line, rectangle
from dataclasses import dataclass
import matplotlib.pylab as plt
from collections import deque
import numpy as np

SIZE = 1000
SUBDIV_LEVEL = 9


@dataclass
class Rect:
    x: int
    y: int
    width: int
    height: int
    level: int


def create_pattern(rng_obj):
    def lerp(v1, v2, t):
        return int(v1 + t * (v2 - v1))

    background = np.zeros(shape=(SIZE, SIZE))
    foreground = np.zeros(shape=(SIZE, SIZE))
    rects = deque([Rect(0, 0, SIZE - 1, SIZE - 1, 0)])
    while rects:
        rect = rects.popleft()
        nlevel = rect.level + 1
        if rect.level > SUBDIV_LEVEL:
            continue
        # Randomly fill some
        if rect.level > 3 and rng.random() > 0.9:
            yy, xx = rectangle((rect.y, rect.x), extent=(rect.height, rect.width))
            background[yy, xx] = rng.random()
        # Split at a random point in a random orientation
        split_frac = rng.random()
        if rng.random() > 0.5:  # Vertical
            mid_x = lerp(rect.x, rect.x + rect.width, split_frac)
            # Line data
            yy, xx = line(rect.y, mid_x, rect.y + rect.height, mid_x)
            # Append new rects
            nw = int(rect.width * split_frac)
            r1 = Rect(rect.x, rect.y, nw, rect.height, nlevel)
            r2 = Rect(rect.x + nw, rect.y, rect.width - nw, rect.height, nlevel)
        else:  # Horizontal
            mid_y = lerp(rect.y, rect.y + rect.height, split_frac)
            # Line data
            yy, xx = line(mid_y, rect.x, mid_y, rect.x + rect.width)
            # Append new rects
            nh = int(rect.height * split_frac)
            r1 = Rect(rect.x, rect.y, rect.width, nh, nlevel)
            r2 = Rect(rect.x, rect.y + nh, rect.width, rect.height - nh, nlevel)
        # Draw and store
        foreground[yy, xx] = 1.0
        rects.append(r1)
        rects.append(r2)

    return background + foreground


rng = np.random.default_rng(3)
fig, axs = plt.subplots(3, 3, figsize=(10, 10), squeeze=False)
for ax in axs.flatten():
    ax.imshow(create_pattern(rng), cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig("out/day12.png", bbox_inches="tight")
plt.show()
