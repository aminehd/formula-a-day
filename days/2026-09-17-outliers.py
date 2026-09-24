# %% [markdown]
# # Outliers
#

# %%
import jax.numpy as jnp
import numpy as np
import jaxvis
import fad

fad.day(__file__)

n = 80_000
rng = np.random.default_rng(1)
cloud = dict(rep={(n, 2): "points"}, only="points", structural=True,
             size=640, tween=30, hold=8, duration=55, blend="mean",
             frame=(0.0, 0.0, 5.5))

clean = rng.normal(size=(n, 2))
bad = rng.integers(0, 100, n) < 3
data = jnp.asarray(np.where(bad[:, None], clean * 9.0, clean))
flag = np.where(bad[:, None], [1.0, 0.35, 0.5], [0.3, 0.8, 1.0])

# %% [markdown]
# ## What the outliers do to the scale
#
# $$z = \frac{x - \bar{x}}{s}$$
#
# Standardising with the mean and standard deviation.
#
# %%
mean = np.asarray(data).mean(0)
std = np.asarray(data).std(0)


@jaxvis.draw(data, palette="bloom", colors=flag, **cloud)
def z_score(x):
    return (x - mean) / std


# %% [markdown]
# ## Median and MAD instead
#
# $$z = \frac{x - \mathrm{med}(x)}{1.4826\,\mathrm{MAD}}$$
#
#

# %%
med = np.median(np.asarray(data), 0)
mad = 1.4826 * np.median(np.abs(np.asarray(data) - med), 0)


@jaxvis.draw(data, palette="bloom", colors=flag, **cloud)
def robust_scale(x):
    return (x - med) / mad


# %% [markdown]
# ## Winsorising
#
# $$x \mapsto \mathrm{clip}(x,\; -3,\; 3)$$
#
# Rather than dropping outliers, pin them to the edge.
#
# %%
@jaxvis.draw(data, palette="bloom", colors=flag, **cloud)
def winsorise(x):
    return jnp.clip((x - med) / mad, -3.0, 3.0)
