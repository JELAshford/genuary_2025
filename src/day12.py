"""Subdivision"""

from skimage.draw import line, rectangle
import matplotlib.pylab as plt
from collections import deque
import numpy as np

SIZE = 1000
SUBDIV_LEVEL = 8


def create_pattern(rng_obj):
    def lerp(v1, v2, t):
        return int(v1 + t * (v2 - v1))

    background = np.zeros(shape=(SIZE, SIZE))
    foreground = np.zeros(shape=(SIZE, SIZE))
    rects = deque([(((0, 0), SIZE - 1, SIZE - 1), 0)])
    while rects:
        this_rect, this_level = rects.popleft()
        ((this_y, this_x), this_height, this_width) = this_rect
        if this_level > SUBDIV_LEVEL:
            continue
        # Randomly fill some
        if this_level > 3 and rng.random() > 0.9:
            yy, xx = rectangle((this_y, this_x), extent=(this_height, this_width))
            background[yy, xx] = rng.random()

        # Split at a random point in a random orientation
        split_frac = rng.random()
        if rng.random() > 0.5:  # Vertical
            mid_x = lerp(this_x, this_x + this_width, split_frac)
            # Draw line
            yy, xx = line(this_y, mid_x, this_y + this_height, mid_x)
            foreground[yy, xx] = 1.0
            # Append new rects
            first_width = int(this_width * split_frac)
            rects.append((((this_y, this_x), this_height, first_width), this_level + 1))
            rects.append(
                (
                    (
                        (this_y, this_x + first_width),
                        this_height,
                        this_width - first_width,
                    ),
                    this_level + 1,
                )
            )
        else:  # Horizontal
            mid_y = lerp(this_y, this_y + this_height, split_frac)
            # Draw line
            yy, xx = line(mid_y, this_x, mid_y, this_x + this_width)
            foreground[yy, xx] = 1.0
            # Append new rects
            first_height = int(this_height * split_frac)
            rects.append((((this_y, this_x), first_height, this_width), this_level + 1))
            rects.append(
                (
                    (
                        (this_y + first_height, this_x),
                        this_height - first_height,
                        this_width,
                    ),
                    this_level + 1,
                )
            )
    return background + foreground


rng = np.random.default_rng(2)
fig, axs = plt.subplots(3, 3, figsize=(10, 10), squeeze=False)
for ax in axs.flatten():
    ax.imshow(create_pattern(rng), cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig("out/day12.png", bbox_inches="tight")
plt.show()
