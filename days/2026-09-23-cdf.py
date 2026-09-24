# %% [markdown]
# # Between uniform and Gaussian
#
#

# %%
import jax
import jax.numpy as jnp
import numpy as np
import jaxvis
import fad
from jaxvis.render import ramp

fad.day(__file__)

n = 80_000
rng = np.random.default_rng(0)
cloud = dict(rep={(n, 2): "points"}, only="points", structural=True,
             size=640, tween=30, hold=8, duration=55, blend="mean")


def by_value(v):
    u = (v - v.min()) / (np.ptp(v) + 1e-9)
    c = ramp("ultra")[(u * 255).astype(int)] / 255.0
    return c / c.max(1, keepdims=True)


# %% [markdown]
# ## Gaussian to uniform
#
# $$u = \Phi(x) = \tfrac{1}{2}\left(1 + \operatorname{erf}(x/\sqrt{2})\right)$$
#
# Push any distribution through its own CDF to a uniform distribution.
#
#

# %%
gauss = rng.normal(size=(n, 2))


@jaxvis.draw(jnp.asarray(gauss), palette="bloom",
             colors=by_value(np.hypot(gauss[:, 0], gauss[:, 1])),
             frame="fixed", **cloud)
def to_uniform(x):
    return jax.scipy.special.erf(x / jnp.sqrt(2.0))


# %% [markdown]
# ## Uniform to Gaussian
#
# $$r = \sqrt{-2\ln u_1}, \quad \theta = 2\pi u_2, \quad z = r(\cos\theta, \sin\theta)$$
#
#

# %%
unif = rng.uniform(1e-4, 1.0, size=(n, 2))


@jaxvis.draw(jnp.asarray(unif), palette="bloom", colors=by_value(unif[:, 0]),
             **cloud)
def box_muller(u):
    r = jnp.sqrt(-2.0 * jnp.log(u[:, :1]))
    th = 2.0 * jnp.pi * u[:, 1:]
    return jnp.concatenate([jnp.cos(th), jnp.sin(th)], 1) * r
