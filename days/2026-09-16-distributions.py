# %% [markdown]
# # Distributions
#
# _Write-up goes here._

# %%
import jax
import jax.numpy as jnp
import numpy as np
import jaxvis
import fad

fad.day(__file__)

n = 80_000
rng = np.random.default_rng(0)
cloud = dict(rep={(n, 2): "points"}, only="points", structural=True,
             size=640, tween=30, hold=8, duration=55)

# %% [markdown]
# ## Adding uniforms
#
# $$S_k = \frac{R + \sum_{i=1}^{k} U_i}{\sqrt{1 + k\,\sigma_U^2}}$$
#
# Start from a ring -- nothing like a Gaussian, and obviously not one to look
# at. Then add plain uniform noise, one draw at a time. The hole fills, the
# edge softens, and it settles into a Gaussian. Whatever you begin with, sums
# end up here.
#
# _Notes._

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
# ## Whitening
#
# $$z = L^{-1}(x - \mu), \qquad LL^{\top} = \Sigma$$
#
# Data rarely arrives centred and round. Subtracting the mean slides the cloud
# to the origin; undoing the covariance unstretches it. A tilted oval becomes
# a plain Gaussian. Colour is the angle each point started at.
#
# _Notes._

# %%
tilted = rng.multivariate_normal([0.9, -0.5], [[2.2, 1.6], [1.6, 1.4]], n)
mu = tilted.mean(0)
chol = np.linalg.cholesky(np.cov(tilted.T))
unmix = np.linalg.inv(chol).T

ang = (np.arctan2(tilted[:, 1] - mu[1], tilted[:, 0] - mu[0]) + np.pi) / (2 * np.pi)
wheel = np.stack([np.sin(np.pi * ang), np.sin(np.pi * (ang + 1 / 3)),
                  np.sin(np.pi * (ang + 2 / 3))], 1) ** 2
wheel = wheel / wheel.max(1, keepdims=True)


@jaxvis.draw(jnp.asarray(tilted), palette="bloom", colors=wheel, blend="mean",
             frame="fixed", **cloud)
def whiten(x):
    return (x - mu) @ unmix
