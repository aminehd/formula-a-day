# %% [markdown]
# # Where it came from
#
# The wiring view shows the structure of a matrix, but every wire looks
# like every other wire. Stand at an output square and you still cannot
# say which inputs built it.
#
# So stop using colour for value and use it for **identity**. Each input
# is a small $9 \times 9$ block of grains with its own hue. Input $j$'s
# grains are divided among the outputs in proportion to
# $|M_{ij}x_j|$ — the share of $j$ that output $i$ actually used — and
# each grain crosses on its own path, at its own moment.
#
# What piles up at the top is a mixture, and the colour of that mixture
# *is* the provenance of the output. An output that comes from one
# place is one clean colour. An output that averages everything is
# grey. The number of grains in a pile is how much landed there, so
# magnitude comes along for free.
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
# The cleanest case, and the one to look at first. Every input sends
# all $81$ of its grains to exactly one output, so each pile at the top
# is a single pure hue — and the hues come out in reverse order. No
# mixing anywhere, which is precisely what a permutation promises.
#

# %%
jaxvis.grains((i[:, None] + i[None, :] == n - 1) * 1.0, x,
              name="permutation")


# %% [markdown]
# ## A running total
#
# $$M_{ij} = [\,i \ge j\,]$$
#
# Now watch the piles differ. The leftmost output holds grains of one
# colour, because only the first input reaches it. The rightmost holds
# every colour there is. The gradient from pure to mixed, left to
# right, is the prefix sum — said in a way no list of numbers says it.
#

# %%
jaxvis.grains((i[:, None] >= i[None, :]) * 1.0, x, name="running_total")


# %% [markdown]
# ## Local averaging
#
# $$M_{ij} = \tfrac{1}{3}\,[\,|i - j| \le 1\,]$$
#
# Three colours in every pile, always its own and its two neighbours',
# never more. Each input splits its grains three ways and they do not
# travel far. A blur is a local mixture, and this is what local looks
# like when you can see the sources.
#

# %%
jaxvis.grains((jnp.abs(i[:, None] - i[None, :]) <= 1) / 3.0, x,
              name="local_average")


# %% [markdown]
# ## A learned weight matrix
#
# $$M_{ij} \sim \mathcal{N}(0, 1)$$
#
# Every pile gets grains from everywhere, so every pile is the same
# muddy grey. That greyness is not a failure of the picture — it is the
# honest answer. Asking *where did this activation come from* in a
# dense layer has no short answer, and this is what no short answer
# looks like.
#

# %%
W = jnp.asarray(np.random.default_rng(7).normal(size=(n, n)) / 2.4)

jaxvis.grains(W, x, name="dense", spread=0.7)


# %% [markdown]
# ## Try it
#
# ```python
# import jaxvis
# jaxvis.grains(M, x, name="mine")
# ```
#
# `k` sets the block size (9 means 81 grains per input), `spread` how
# staggered the departures are, `travel` how fast a grain crosses, and
# `seed` reshuffles which grain goes where.
#
# [github.com/aminehd/jaxvis](https://github.com/aminehd/jaxvis)
#
