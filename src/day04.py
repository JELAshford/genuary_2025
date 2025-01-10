"""Black on Black"""

from perlin_numpy import generate_perlin_noise_2d
import matplotlib.pylab as plt
import numpy as np
import random

SIZE = 1000

# Generate perlin noise structure
np.random.seed(1701)
noise1 = generate_perlin_noise_2d((SIZE, SIZE), (5, 5))
noise2 = generate_perlin_noise_2d((SIZE, SIZE), (8, 8))
grid = noise1 * noise2
grid = np.maximum(grid, 0)

# Sample from the noise into channels of image
show_img = np.zeros(shape=(SIZE, SIZE))
for channel in range(3):
    sample_points = np.array(
        random.choices(np.arange(SIZE * SIZE), weights=grid.flatten(), k=500_000)
    )
    sample_idxs = np.unravel_index(sample_points, shape=(SIZE, SIZE))
    show_img[*sample_idxs] += 0.1

# Create edges from the original grid
gradient_x = np.gradient(grid, axis=0)
gradient_y = np.gradient(grid, axis=1)
edges = np.sqrt(gradient_x**2 + gradient_y**2)

# Normalize and threshold the edges
edges = edges / edges.max()
edges = np.where(edges > 0.2, edges * 0.2, 0)

# Add edges to the final image
show_img = np.clip(show_img + edges, 0, 1)

# Show
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(show_img, cmap="gray", vmax=1)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig("out/day04.png", bbox_inches="tight")
plt.show()
