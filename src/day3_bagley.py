"""Day 3: Joshua Bagley's Differential Growth"""

from skimage.draw import rectangle
from collections import namedtuple
import matplotlib.pylab as plt
import numpy as np


Node = namedtuple("Node", ["x", "y", "vx", "vy"])


SIZE = 500
ITERATION_STEPS = 150
half_size = SIZE // 2


# Setup initial node population in circle
nodes = [
    Node(x=10 * np.sin(pos) + half_size, y=10 * np.cos(pos) + half_size, vx=0, vy=0)
    for pos in np.linspace(0, 2 * np.pi, 10)
]

# Run simulation
for _ in range(ITERATION_STEPS):
    # Pairwise interaction
    for i1 in range(len(nodes)):
        # Compute forces
        c = nodes[i1]
        p = nodes[(i1 + len(nodes) - 1) % len(nodes)]
        n = nodes[(i1 + 1) % len(nodes)]
        mp = Node(x=(p.x + n.x) / 2, y=(p.y + n.y) / 2, vx=0, vy=0)
        count = 3
        force = [
            (p.x - c.x) - (c.x - n.x) - 10 * (c.x - mp.x),
            (p.y - c.y) - (c.y - n.y) - 10 * (c.y - mp.y),
        ]
        for i2 in range(len(nodes)):
            if i1 != i2:
                distance = (nodes[i2].x - c.x) ** 2 + (nodes[i2].y - c.y) ** 2
                mapped_distance = np.interp(distance, [0, 50**2], [1, 0])
                if distance < 50**2:
                    force[0] += (c.x - nodes[i2].x) * mapped_distance
                    force[1] += (c.y - nodes[i2].y) * mapped_distance
                    count += 1
        # Compute velocities
        nodes[i1] = Node(
            c.x,
            c.y,
            c.vx + force[0] / count,
            c.vy + force[1] / count,
        )

    # Update positions
    for ind, node in enumerate(nodes):
        nodes[ind] = Node(
            x=node.x + node.vx, y=node.y + node.vy, vx=node.vx * 0.75, vy=node.vy * 0.75
        )

    # Add new nodes
    insert_pos = np.random.randint(len(nodes))
    that_node = nodes[insert_pos]
    next_node = nodes[(insert_pos + 1) % len(nodes)]
    nodes.insert(
        insert_pos + 1,
        Node(
            x=(that_node.x + next_node.x) / 2,
            y=(that_node.y + next_node.y) / 2,
            vx=0,
            vy=0,
        ),
    )

# Draw population
show_img = np.zeros(shape=(SIZE, SIZE))
for node in nodes:
    rr, cc = rectangle((node.x, node.y), extent=20, shape=show_img.shape)
    show_img[rr, cc] = 1

# Show
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(show_img, cmap="gray", vmax=1)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig("out/day3_bagley.png", bbox_inches="tight")
plt.show()
