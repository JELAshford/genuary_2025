"""Only Triangles"""

from skimage.draw import polygon
import matplotlib.pylab as plt
from tqdm import trange
import numpy as np

SIZE = 1000
BOID_SCALE = 10
NUM_BOIDS = 1000
STEPS = 200


def create_pattern(rng_obj):
    def vec_to_angle(array):
        a = np.arctan2(array[:, 1], array[:, 0])  # returns -180 to 180
        a[a < 0] += 2 * np.pi  # convert to 0-360
        return a

    def angle_to_vec(radians: float):
        return np.array([np.cos(radians), np.sin(radians)])

    # Setup triangles
    positions = rng_obj.random(size=(NUM_BOIDS, 2))
    bearings = rng_obj.random(size=(NUM_BOIDS)) * 2 * np.pi

    # Simulate
    grid = np.ones(shape=(SIZE, SIZE))
    for step in trange(STEPS):
        # Update positions
        directions = angle_to_vec(bearings) * 0.01
        positions += directions.T

        # Draw triangles
        int_pos = (positions * SIZE).astype(int)
        for pos, bearing in zip(int_pos, bearings):
            fwd = angle_to_vec(bearing)
            cw = angle_to_vec(bearing + (2 * np.pi) / 3)
            ccw = angle_to_vec(bearing - (2 * np.pi) / 3)
            points = np.array(
                [
                    pos + fwd * BOID_SCALE,
                    pos + cw * BOID_SCALE,
                    pos + ccw * BOID_SCALE,
                ]
            ).astype(int)
            yy, xx = polygon(*points.T, shape=(SIZE, SIZE))
            grid[yy, xx] = (STEPS - step) / STEPS
    return grid


rng = np.random.default_rng(3)
fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
for ax in axs.flatten():
    ax.imshow(create_pattern(rng), cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig("out/day13.png", bbox_inches="tight")
plt.show()
