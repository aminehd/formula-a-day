# %% [markdown]
# # Distributions
#
#
# <!-- page: fun -->

# %%
import jax
import jax.numpy as jnp
import numpy as np
import jaxvis
import fad
from jaxvis import data, tint

fad.day(__file__)

n = 80_000
rng = np.random.default_rng(0)

# %% [markdown]
# ## Adding uniforms
#
# $$S_k = \frac{R + \sum_{i=1}^{k} U_i}{\sqrt{1 + k\,\sigma_U^2}}$$
#
# Simulate the central limit theorem by starting with any distribution and adding plain uniform noise.
#
# %%
steps, amp = 200, 0.275
su = amp / np.sqrt(3.0)

dirs = np.array([[0.0, 1.0], [-0.866, -0.5], [0.866, -0.5]]) * 2.6
squares = jnp.asarray(dirs[rng.integers(0, 3, n)]
                      + rng.uniform(-0.85, 0.85, size=(n, 2)))
key = jax.random.PRNGKey(1)


@jaxvis.draw_sim(init=(squares, jnp.asarray(0)), frames=steps, every=1,
                 tween=1, rep="points", palette="vapor", size=640,
                 duration=45,
                 show=lambda s: s[0] / jnp.sqrt(1.0 + s[1] * su ** 2))
def central_limit(state):
    total, k = state
    draw = jax.random.uniform(jax.random.fold_in(key, k), total.shape,
                              minval=-amp, maxval=amp)
    return total + draw, k + 1


# %% [markdown]
# ## Preprocessing normal data
#
# $$z = L^{-1}(x - \mu), \qquad LL^{\top} = \Sigma$$
# %%

tilted = data.tilted(n)
mu = np.asarray(tilted).mean(0)
unmix = np.linalg.inv(np.linalg.cholesky(np.cov(np.asarray(tilted).T))).T


@jaxvis.draw(tilted, colors=tint.by_angle(tilted), frame="fixed",
             **jaxvis.cloud(tilted))
def whiten(x):
    return (x - mu) @ unmix
