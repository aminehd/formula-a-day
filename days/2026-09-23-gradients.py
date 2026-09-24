# %% [markdown]
# # Gradients
#
# _Write-up goes here._

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
# Start from pure noise and descend on two terms at once: one pulls every
# pixel toward its label, the other punishes disagreement with its neighbours.
# The speckle coarsens, a disc surfaces, and the edge tightens. The same two
# forces that separate oil from water, aimed at a circle.
#
# _Notes._

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
