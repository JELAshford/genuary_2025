"""Black and White"""

import matplotlib.pylab as plt
import numpy as np

SIZE = 256


def create_pattern(rng_obj):
    yy, xx = np.indices((SIZE, SIZE))
    grid = ((xx ^ abs(yy - xx)) * xx + 6) % 24
    return (grid > 6).astype(int)


rng = np.random.default_rng(3)
fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
for ax in axs.flatten():
    ax.imshow(create_pattern(rng), cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig("out/day14.png", bbox_inches="tight")
plt.show()
