import matplotlib.pyplot as plt
from functools import partial
import jax.numpy as jnp
from jax import random, vmap, jit
import jax


@jit
def two_peak_density(x):
    mu1 = jnp.array([-1.0, -1.0])
    mu2 = jnp.array([1.0, 1.0])
    std = 0.5

    d1 = jnp.linalg.norm(x - mu1, axis=-1)
    d2 = jnp.linalg.norm(x - mu2, axis=-1)
    d = jnp.minimum(d1, d2)

    p = (d < std).astype(jnp.float32)
    return jnp.clip(p, 1e-9, 1)


def estimate_origin(x_t, sample_x, sample_fitness, alpha):
    """
    Args:
        x_t: single point (dim,)
        sample_x: batch of points (batch_size, dim)
        sample_fitness: fitness values (batch_size,)
        alpha: scalar
    """

    # Compute diffusion probability
    mu = sample_x * (alpha**0.5)  # (batch_size, dim)
    sigma = (1 - alpha) ** 0.5

    # Broadcast x_t to match sample_x shape
    x_t_expanded = jnp.expand_dims(x_t, 0)  # (1, dim)
    dist = jnp.linalg.norm(x_t_expanded - mu, axis=-1)
    p_diffusion = jnp.exp(-(dist**2) / (2 * sigma**2))  # (batch_size,)

    # Estimate origin
    weights = (sample_fitness + 1e-9) * (p_diffusion + 1e-9)  # (batch_size,)
    weights = weights / (jnp.sum(weights) + 1e-9)  # Normalize weights

    # Weighted sum
    origin = jnp.sum(jnp.expand_dims(weights, 1) * sample_x, axis=0)  # (dim,)
    return origin


def ddim_step(xt, x0, alphas, key, noise):
    alphat, alphatp = alphas
    sigma = ((1 - alphatp) / (1 - alphat) * (1 - alphat / alphatp)) ** 0.5 * noise
    eps = (xt - (alphat**0.5) * x0) / (1.0 - alphat) ** 0.5
    x_next = (
        (alphatp**0.5) * x0
        + ((1 - alphatp - sigma**2) ** 0.5) * eps
        + sigma * random.normal(key, shape=x0.shape)
    )
    return x_next


def generate_step(x, fitness, alphas, key, noise):
    # Vectorize origin estimation over all points
    x0_est = vmap(lambda x_t: estimate_origin(x_t, x, fitness, alphas[0]))(x)
    return x0_est, ddim_step(x, x0_est, alphas, key, noise)


@partial(jit, static_argnums=(2,))
def diffusion_step(state, t, num_steps, power=1, eps=1e-4):
    x, key, scaling, noise = state

    # Calculate alphas for current step
    curr_t = num_steps - t - 1
    alpha = jnp.linspace(1 - eps, (eps * eps) ** (1 / power), num_steps) ** power
    alphas = (alpha[curr_t], alpha[curr_t - 1])

    # Calculate fitness and generate next points
    fitness = two_peak_density(x * scaling)
    key, step_key = random.split(key)
    x0_est, x_next = generate_step(x, fitness, alphas, step_key, noise)

    return (x_next, key, scaling, noise), (x_next * scaling, x0_est * scaling, fitness)


@partial(jit, static_argnums=(2,))
def optimize(key, initial_population, num_steps, scaling=1.0, noise=1.0):
    state = (initial_population, key, scaling, noise)
    # Run the diffusion process
    _, (population_trace, x0_trace, fitness_trace) = jax.lax.scan(
        lambda state, t: diffusion_step(state, t, num_steps),
        state,
        jnp.arange(num_steps - 1),
    )
    return population_trace, x0_trace, fitness_trace


def plot_evolution(
    population_trace,
    population_x0_trace,
    fitnesses,
    timesteps=[20, 45, 70, 95],
    focus_id=20,
):
    """Plot the evolution of the diffusion process at specified timesteps.

    Args:
        population_trace: array of shape (n_steps, n_particles, 2)
        population_x0_trace: array of shape (n_steps, n_particles, 2)
        fitnesses: array of shape (n_steps, n_particles)
        timesteps: list of timesteps to visualize
        focus_id: index of the particle to focus on
    """
    fig, axes = plt.subplots(1, len(timesteps), figsize=(10, 10), squeeze=False)

    # Style settings
    colors = {
        "background": "#000000",
        "trace": "#FFFFFF",
        "focus": "#F5851E",
        "selected": "#46B3D5",
        "unselected": "#C6C6C6",
        "target": "white",
        "circle": "white",
        "x0": "red",
    }

    for idx, (ax, t) in enumerate(zip(axes.flatten(), timesteps)):
        # Plot particle traces (gray lines)
        for particle in population_trace[:t, :100].transpose(1, 0, 2):
            ax.plot(*particle.T, "-", color=colors["trace"], alpha=0.5, zorder=1)

        # Plot focus particle trajectory
        focus_trajectory = population_trace[: t + 1, focus_id]
        ax.plot(*focus_trajectory.T, "-", color=colors["focus"], linewidth=2, zorder=4)

        # Plot current population
        current_pos = population_trace[t]
        current_fit = fitnesses[t]

        # Plot all particles, scaled by fitness
        sizes = 20 * (current_fit / current_fit.max()) + 5
        ax.scatter(*current_pos.T, c=colors["unselected"], s=sizes, alpha=0.6, zorder=2)

        # Plot target regions (dashed circles)
        for center in [(-1, -1), (1, 1)]:
            circle = plt.Circle(
                center, 0.5, fill=False, linestyle="--", color=colors["circle"]
            )
            ax.add_artist(circle)

        # Previous focus points
        if idx > 0:
            for prev_t in timesteps[:idx]:
                prev_pos = population_trace[prev_t, focus_id]
                ax.scatter(*prev_pos, color="gray", marker="*", s=100, zorder=5)

        # Formatting
        ax.set_facecolor(colors["background"])
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    return fig


# Main execution
if __name__ == "__main__":
    key = random.key(1701)
    noise = 0.1
    name = "ddpm"

    key, init_key = random.split(key)
    x0 = random.normal(init_key, shape=(1024, 2))
    trace, x0_trace, fitnesses = optimize(
        key, x0, num_steps=1000, scaling=4, noise=noise
    )
    fig = plot_evolution(trace, x0_trace, fitnesses, timesteps=[1000])
    plt.savefig("out/day08.png", dpi=300, bbox_inches="tight")
    plt.close()
