# %% [markdown]
# # Gradients
#

# %%
import jax
import jax.numpy as jnp
import numpy as np
import jaxvis
import fad
from jaxvis.fields import grid

fad.day(__file__)

# %% [markdown]
# ## A circle out of noise
#
# $$z \leftarrow z + \eta\,(y - \sigma(z)) + \eta\lambda\,\nabla^2 z$$
#
# Start from pure noise and descend on two terms at once. One pulls pixels to the edges, one punishes disagreement with neighbours.

# %%
gx, gy = grid(256, 1.0)
target = jnp.asarray((np.hypot(np.asarray(gx), np.asarray(gy)) < 0.55)
                     .astype(float))
noise = jnp.asarray(np.random.default_rng(0).normal(size=(256, 256)) * 2.5)

lr, smooth = 0.06, 3.0


def lap(a):
    return (jnp.roll(a, 1, 0) + jnp.roll(a, -1, 0) +
            jnp.roll(a, 1, 1) + jnp.roll(a, -1, 1) - 4 * a)


@jaxvis.draw_sim(init=noise, frames=130, every=1, tween=2, palette="ultra",
                 show=jax.nn.sigmoid, size=680, duration=45, grain=0.0)
def a_circle_from_noise(z):
    return z + lr * (target - jax.nn.sigmoid(z)) + lr * smooth * lap(z)


# %% [markdown]
# ## Cross-entropy
#
# $$z \leftarrow z + \eta\,(y - \sigma(z))$$
#
# Drop the neighbour term.
#
# %%
@jaxvis.draw_sim(init=noise, frames=130, every=1, tween=2, palette="ultra",
                 show=jax.nn.sigmoid, size=680, duration=45, grain=0.0)
def cross_entropy(z):
    return z + lr * (target - jax.nn.sigmoid(z))




# %% [markdown]
# ## No target at all
#
# $$z \leftarrow z + \eta\,(z - z^3) + \eta\lambda\,\nabla^2 z$$
#
# Swap the label for a term that pushes each pixel away from zero, toward
# either $+1$ or $-1$, and keep the smoothing. Nothing is being fitted now.

# %%
@jaxvis.draw_sim(init=noise * 0.04, frames=130, every=4, tween=2,
                 palette="ultra", show=jax.nn.sigmoid, size=680, duration=45,
                 grain=0.0)
def no_target_at_all(z):
    return z + lr * (z - z ** 3) + lr * smooth * lap(z)
