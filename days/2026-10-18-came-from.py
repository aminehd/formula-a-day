# %% [markdown]
# # Where it came from
#
# <!-- order: 9 -->
#
# Colour means **which input**, not value. Each input is a $9\times9$
# block of grains, split among the outputs in proportion to
# $|M_{ij}x_j|$ — the share of $j$ that output $i$ used.
#
# What piles up at the top is the provenance. One source, one colour.
# Everything averaged, grey.
#

# %%
import jax.numpy as jnp
import numpy as np
import jaxvis
import fad

fad.day(__file__)

n = 8
i = jnp.arange(n)
x = jnp.asarray([1.0, 2.0, 1.5, 3.0, 0.8, 2.4, 1.2, 2.8])


# %% [markdown]
# ## A permutation
#
# $$M_{ij} = [\,i + j = n - 1\,]$$
#
# Every input goes to exactly one output. Eight pure hues, reversed.
#

# %%
jaxvis.grains((i[:, None] + i[None, :] == n - 1) * 1.0, x,
              name="permutation")


# %% [markdown]
# ## A running total
#
# $$M_{ij} = [\,i \ge j\,]$$
#
# Pure on the left, every colour on the right. That gradient is the
# prefix sum.
#

# %%
jaxvis.grains((i[:, None] >= i[None, :]) * 1.0, x, name="running_total")


# %% [markdown]
# ## Local averaging
#
# $$M_{ij} = \tfrac{1}{3}\,[\,|i - j| \le 1\,]$$
#
# Three colours per pile, always its own and its neighbours'. That is
# what local means.
#

# %%
jaxvis.grains((jnp.abs(i[:, None] - i[None, :]) <= 1) / 3.0, x,
              name="local_average")


# %% [markdown]
# ## A learned weight matrix
#
# $$M_{ij} \sim \mathcal{N}(0, 1)$$
#
# Grey everywhere. "Where did this activation come from" has no short
# answer in a dense layer, and this is what no short answer looks like.
#

# %%
W = jnp.asarray(np.random.default_rng(7).normal(size=(n, n)) / 2.4)

jaxvis.grains(W, x, name="dense", spread=0.7)


# %% [markdown]
# ## Try it
#
# ```python
# jaxvis.grains(M, x, name="mine")
# ```
#
# [github.com/aminehd/jaxvis](https://github.com/aminehd/jaxvis)
#
